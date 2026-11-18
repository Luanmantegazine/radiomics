"""Local prognostic models: reference implementations, censoring rules, splits."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from client.data import LocalDataset
from client.models.cox import (
    Standardization,
    breslow_cumulative_hazard,
    fit_cox,
    partial_likelihood,
)
from client.models.forest import RandomSurvivalForestModel
from client.models.logistic import (
    _evaluate,
    censoring_survival,
    fit_horizon_logistic,
    horizon_labels,
    profile_logistic,
    weighted_logistic,
)
from client.models.prediction import predict_conversion
from client.splits import partition
from client.splits import test_mask as split_test_mask
from shared.newton import NewtonOptimizer, NewtonSettings
from shared.protocol import DataOrigin, PartitionSpec
from shared.training import RandomSurvivalForestSpec
from tests.federated import phantoms

EXACT = NewtonSettings(ridge=0.0, max_iterations=50, tolerance=1e-13)
HORIZONS = (12.0, 24.0, 36.0)


@pytest.fixture(scope="module")
def cohort():
    names, x, time, event = phantoms.survival_cohort(n=300, p=4, seed=1)
    return names, x, np.ceil(time), event  # integer months: many ties


def _sksurv_cox(z, time, event, alpha):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CoxPHSurvivalAnalysis(alpha=alpha, ties="breslow").fit(
            z, Surv.from_arrays(event=event, time=time)
        )


@pytest.mark.parametrize("ridge", [0.0, 2.0])
def test_cox_matches_scikit_survival(cohort, ridge):
    names, x, time, event = cohort
    settings = NewtonSettings(ridge, 50, 1e-13)
    model = fit_cox(names, x, time, event, settings, HORIZONS, min_events=5)
    z = Standardization.fit(x).apply(x)
    reference = _sksurv_cox(z, time, event, ridge)
    np.testing.assert_allclose(model.coefficients, reference.coef_, atol=1e-8)
    assert model.fit.converged and model.fit.n_clients == 1
    # Absolute risk: survival at each horizon equals scikit-survival's.
    _, probabilities = predict_conversion(model, names, x[:20])
    for j, horizon in enumerate(HORIZONS):
        expected = [1 - f(horizon) for f in reference.predict_survival_function(z[:20])]
        np.testing.assert_allclose(probabilities[:, j], expected, atol=1e-8)


def test_partial_likelihood_gradient_and_information_are_derivatives(cohort):
    _, x, time, event = cohort
    z = Standardization.fit(x).apply(x)
    beta = np.array([0.3, -0.2, 0.1, 0.05])
    loglik, gradient, information = partial_likelihood(z, time, event, beta)
    step = 1e-6
    for j in range(4):
        e = np.zeros(4)
        e[j] = step
        plus, g_plus, _ = partial_likelihood(z, time, event, beta + e)
        minus, g_minus, _ = partial_likelihood(z, time, event, beta - e)
        assert gradient[j] == pytest.approx((plus - minus) / (2 * step), rel=1e-5)
        np.testing.assert_allclose(-(g_plus - g_minus) / (2 * step), information[j], rtol=1e-4)
    assert np.isfinite(loglik)


def test_stratified_partial_likelihood_is_a_sum(cohort):
    _, x, time, event = cohort
    strata = np.arange(len(time)) % 3
    beta = np.array([0.2, 0.1, 0.0, -0.1])
    total = partial_likelihood(x, time, event, beta, strata=strata)
    parts = [partial_likelihood(x[strata == s], time[strata == s], event[strata == s], beta)
             for s in range(3)]
    assert total[0] == pytest.approx(sum(p[0] for p in parts))
    np.testing.assert_allclose(total[1], sum(p[1] for p in parts))


def test_baseline_is_not_extrapolated(cohort):
    _, x, time, event = cohort
    hazard = breslow_cumulative_hazard(x, time, event, np.zeros(4), (12.0, 1e6))
    assert hazard[0] > 0 and hazard[1] is None
    model = fit_cox(("a", "b", "c", "d"), x, time, event, EXACT, (12.0, 1e6), min_events=5)
    _, probabilities = predict_conversion(model, ("a", "b", "c", "d"), x)
    assert np.isfinite(probabilities[:, 0]).all() and np.isnan(probabilities[:, 1]).all()


def test_cox_probabilities_are_valid_and_increase_with_horizon(cohort):
    names, x, time, event = cohort
    model = fit_cox(names, x, time, event, EXACT, HORIZONS, min_events=5)
    _, p = predict_conversion(model, names, x)
    assert ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p, axis=1) >= 0).all()
    with pytest.raises(ValueError, match="missing features"):
        predict_conversion(model, names[:2], x[:, :2])
    with pytest.raises(ValueError, match="events observed"):
        fit_cox(names, x, time, event, EXACT, HORIZONS, min_events=10_000)


# --------------------------------------------------------------------------- #
# Censoring
# --------------------------------------------------------------------------- #
def test_censoring_distribution_matches_kaplan_meier():
    _, _, time, event = phantoms.survival_cohort(n=300, p=3, seed=2)  # continuous
    times, values = censoring_survival(time, event)
    reference_times, reference = kaplan_meier_estimator(~event, time)
    np.testing.assert_allclose(_evaluate(times, values, reference_times, left=False), reference)


@pytest.mark.parametrize("rounding", [None, 1.0, 6.0])
def test_ipcw_reproduces_the_kaplan_meier_cumulative_incidence(rounding):
    """Intercept-only IPCW = 1 - KM(tau), exactly, also with tied times."""
    _, _, time, event = phantoms.survival_cohort(n=300, p=3, seed=3)
    if rounding:
        time = np.ceil(time / rounding) * rounding
    km_times, km = kaplan_meier_estimator(event, time)
    for horizon in HORIZONS:
        y, w = horizon_labels(time, event, horizon, "ipcw")
        incidence = 1 - km[np.searchsorted(km_times, horizon, side="right") - 1]
        assert (w * y).sum() / w.sum() == pytest.approx(incidence, abs=1e-12)
        assert w.sum() == pytest.approx(len(time))


def test_censored_before_the_horizon_is_never_a_non_converter():
    time = np.array([3.0, 6.0, 12.0, 15.0, 30.0, 8.0, 40.0])
    event = np.array([True, False, False, True, False, False, True])
    for censoring in ("ipcw", "complete_case"):
        y, w = horizon_labels(time, event, 12.0, censoring)
        assert y.tolist() == [1, 0, 0, 0, 0, 0, 0]
        # Censored at 6, 8 and exactly at 12: unknown status, weight 0.
        assert w[[1, 2, 5]].tolist() == [0, 0, 0]
        # Followed beyond 12 without conversion by 12 (event at 15 counts as 0 at 12).
        assert (w[[3, 4, 6]] > 0).all()


def test_logistic_matches_scikit_learn(cohort):
    names, x, time, event = cohort
    model = fit_horizon_logistic(names, x, time, event, (24.0,), "ipcw", EXACT, min_events=5)
    y, w = horizon_labels(time, event, 24.0, "ipcw")
    z = Standardization.fit(x).apply(x)
    reference = LogisticRegression(penalty=None, max_iter=10_000, tol=1e-12).fit(
        z, y, sample_weight=w
    )
    fit = model.horizons[0]
    assert fit.intercept == pytest.approx(reference.intercept_[0], abs=1e-6)
    np.testing.assert_allclose(fit.coefficients, reference.coef_[0], atol=1e-6)
    assert fit.fit.n_events == int(((w > 0) & (y == 1)).sum())


def test_profile_likelihood_equals_joint_site_intercept_fit(cohort):
    _, x, time, event = cohort
    z = Standardization.fit(x).apply(x)
    site = np.arange(len(time)) % 3
    z = z + np.array([2.0, 0.0, -1.0])[site][:, None]  # site offsets in the features
    y, w = horizon_labels(time, event, 24.0, "ipcw")
    # Centralized joint fit: one intercept per site, shared slopes.
    design = np.column_stack([(site == s).astype(float) for s in range(3)] + [z])
    joint = NewtonOptimizer(design.shape[1], EXACT)
    while not joint.finished:
        joint.observe(*weighted_logistic(design, y, w, joint.proposal))
    # Federated: sites profile out their intercepts; only slopes are optimised.
    profile = NewtonOptimizer(z.shape[1], EXACT)
    while not profile.finished:
        parts = [profile_logistic(z[site == s], y[site == s], w[site == s], profile.proposal)
                 for s in range(3)]
        profile.observe(sum(p[0] for p in parts), sum(p[1] for p in parts),
                        sum(p[2] for p in parts))
    np.testing.assert_allclose(profile.best, joint.best[3:], atol=1e-7)
    intercepts = [profile_logistic(z[site == s], y[site == s], w[site == s], profile.best)[3]
                  for s in range(3)]
    np.testing.assert_allclose(intercepts, joint.best[:3], atol=1e-7)
    # Standard errors of the slopes: profile information = joint information block.
    np.testing.assert_allclose(profile.standard_errors(), joint.standard_errors()[3:],
                               rtol=1e-6)


def test_random_survival_forest_behind_the_same_interface(cohort):
    _, x, time, event = cohort
    spec = RandomSurvivalForestSpec(kind="random_survival_forest", name="rsf",
                                    n_estimators=20, min_samples_leaf=10, seed=1)
    model = RandomSurvivalForestModel(spec, (12.0, 1e6)).fit(x, time, event)
    p = model.predict_conversion(x[:10])
    assert p.shape == (10, 2) and ((p[:, 0] >= 0) & (p[:, 0] <= 1)).all()
    assert np.isnan(p[:, 1]).all()  # beyond follow-up


# --------------------------------------------------------------------------- #
# Optimiser
# --------------------------------------------------------------------------- #
def test_newton_halves_overshooting_steps_and_reports_failures():
    # Concave 1-D objective f(b) = -log cosh(3 (b - 1)): Newton from 0 overshoots.
    def evaluate(b):
        u = 3 * (b[0] - 1)
        return -np.log(np.cosh(u)), np.array([-3 * np.tanh(u)]), np.array([[9 / np.cosh(u) ** 2]])

    optimizer = NewtonOptimizer(1, NewtonSettings(0.0, 50, 1e-12))
    while not optimizer.finished:
        optimizer.observe(*evaluate(optimizer.proposal))
    assert optimizer.converged and optimizer.best[0] == pytest.approx(1.0, abs=1e-5)
    assert optimizer.evaluations > len(optimizer.history)  # some steps were halved

    capped = NewtonOptimizer(1, NewtonSettings(0.0, 2, 1e-15))
    while not capped.finished:
        capped.observe(*evaluate(capped.proposal))
    assert capped.failed == "max_iterations"

    singular = NewtonOptimizer(2, NewtonSettings(0.0, 10, 1e-12))
    singular.observe(0.0, np.ones(2), np.zeros((2, 2)))
    assert singular.failed == "singular_information"


# --------------------------------------------------------------------------- #
# Train/test split
# --------------------------------------------------------------------------- #
def _dataset(n=200, seed=0) -> LocalDataset:
    names, x, time, event = phantoms.survival_cohort(n=n, p=3, seed=seed)
    return LocalDataset(tuple(f"P{i:04d}" for i in range(n)), names, x, time, event,
                        DataOrigin.SYNTHETIC)


def test_split_is_deterministic_stratified_and_order_free():
    dataset = _dataset()
    mask = split_test_mask(dataset, 0.3, seed=5)
    assert mask.sum() == round(0.3 * dataset.event.sum()) + round(0.3 * (~dataset.event).sum())
    assert dataset.event[mask].mean() == pytest.approx(dataset.event.mean(), abs=0.02)
    order = np.random.default_rng(0).permutation(dataset.n_participants)
    shuffled = LocalDataset(tuple(np.array(dataset.participant_ids)[order]),
                            dataset.feature_names, dataset.features[order],
                            dataset.time_months[order], dataset.event[order], dataset.origin)
    assert set(np.array(dataset.participant_ids)[mask]) == set(
        np.array(shuffled.participant_ids)[split_test_mask(shuffled, 0.3, seed=5)]
    )
    assert not np.array_equal(mask, split_test_mask(dataset, 0.3, seed=6))
    train = partition(dataset, PartitionSpec(name="train", test_fraction=0.3, seed=5))
    test = partition(dataset, PartitionSpec(name="test", test_fraction=0.3, seed=5))
    assert set(train.participant_ids).isdisjoint(test.participant_ids)
    assert train.n_participants + test.n_participants == dataset.n_participants
    assert partition(dataset, None) is dataset


def test_local_training_cli(tmp_path, capsys):
    """Mechanics only, on an extracted phantom table with random follow-up."""
    from client.imaging.pipeline import run_extraction
    from client.models.__main__ import main as models_cli

    manifest = phantoms.write_site(tmp_path / "site", 40, seed=12, rois={"box": 1},
                                   vary_masks=True)
    run_extraction(manifest, phantoms.config(), tmp_path / "out")
    config = __import__("yaml").safe_load(
        (phantoms.REPO / "experiments/configs/models/local_v1.yaml").read_text()
    )
    config["min_events"] = 1  # tiny phantom cohort; the shipped threshold is 5
    (tmp_path / "local.yaml").write_text(json.dumps(config))
    args = ["--features", str(tmp_path / "out/features.csv"), "--manifest", str(manifest),
            "--data-origin", "synthetic", "--config", str(tmp_path / "local.yaml"),
            "--feature-names", "box__original_firstorder_Mean,box__original_firstorder_Entropy",
            "--output-dir", str(tmp_path / "models"), "--log-level", "ERROR"]
    assert models_cli(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert set(summary["models"]) == {"cox_local", "logistic_ipcw_local", "rsf_local"}
    assert summary["n_train"] == 28
    predictions = __import__("pandas").read_csv(tmp_path / "models/predictions.csv")
    assert set(predictions["partition"]) == {"train", "test"}
    assert list(predictions.columns[-3:]) == ["p_12", "p_24", "p_36"]
    cox = json.loads((tmp_path / "models/models/cox_local.json").read_text())
    assert cox["kind"] == "cox" and cox["fit"]["n_samples"] == 28
