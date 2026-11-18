"""Evaluation metrics: reference implementations, pooling and disclosure control."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

from evaluation.censoring import horizon_labels
from evaluation.metrics import (
    HorizonStatistics,
    decision_curve,
    horizon_statistics,
    merge_bins,
    pool,
    summarize,
)
from tests.federated import phantoms

GRID = np.round(np.linspace(0, 1, 21), 12)


def _cohort(seed=4, n=400):
    _, x, time, event = phantoms.survival_cohort(n=n, p=3, seed=seed)
    risk = 0.9 * x[:, 0] + 0.7 * x[:, 1]
    return time, event, risk


def _probability(risk, horizon):
    return 1 - np.exp(-0.03 * horizon * np.exp(risk))


@pytest.mark.parametrize("horizon", [12.0, 24.0, 36.0])
def test_metrics_match_scikit_survival(horizon):
    time, event, risk = _cohort()
    p = _probability(risk, horizon)
    metrics = summarize(horizon_statistics(time, event, p, horizon, GRID), ())
    y = Surv.from_arrays(event=event, time=time)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auc, _ = cumulative_dynamic_auc(y, y, p, [horizon])
        _, brier = brier_score(y, y, (1 - p)[:, None], [horizon])
        c = concordance_index_ipcw(y, y, p, tau=horizon, tied_tol=0)[0]
    assert metrics["auc"] == pytest.approx(auc[0], abs=1e-12)
    assert metrics["brier"] == pytest.approx(brier[0], abs=1e-12)
    assert metrics["c_index"] == pytest.approx(c, abs=1e-12)


def _brute_force_pooled_auc(sites, horizon):
    """Within-site pairs only, each pair weighted w_i w_j."""
    num = den = 0.0
    for time, event, p in sites:
        y, w = horizon_labels(time, event, horizon, "ipcw")
        for i in np.flatnonzero((w > 0) & (y == 1)):
            for j in np.flatnonzero((w > 0) & (y == 0)):
                weight = w[i] * w[j]
                num += weight * (1.0 if p[i] > p[j] else 0.5 if p[i] == p[j] else 0.0)
                den += weight
    return num / den


def test_pooling_is_site_stratified():
    sites = []
    for seed, shift in ((1, 0.0), (2, 1.5)):
        time, event, risk = _cohort(seed=seed, n=120)
        sites.append((time, event, _probability(risk + shift, 24.0)))
    pooled = pool([horizon_statistics(t, e, p, 24.0, GRID) for t, e, p in sites])
    metrics = summarize(pooled, ())
    assert metrics["auc"] == pytest.approx(_brute_force_pooled_auc(sites, 24.0), abs=1e-12)
    assert pooled.n == 240
    # Brier pools exactly: weighted squared errors over all participants.
    expected = sum(
        (horizon_labels(t, e, 24.0, "ipcw")[1] * (horizon_labels(t, e, 24.0, "ipcw")[0] - p) ** 2
         ).sum() for t, e, p in sites
    ) / 240
    assert metrics["brier"] == pytest.approx(expected, abs=1e-12)


def test_well_specified_model_is_calibrated_and_useful():
    time, event, risk = _cohort(seed=7, n=4000)
    # The generating model of phantoms.survival_cohort: hazard 0.03 exp(0.9 f0 + 0.7 f1).
    p = _probability(risk, 24.0)
    metrics = summarize(horizon_statistics(time, event, p, 24.0, GRID), (0.2, 0.5))
    assert metrics["o_e"] == pytest.approx(1.0, abs=0.06)
    assert metrics["calibration_error"] < 0.05
    curve = {d["threshold"]: d for d in metrics["decision_curve"]}
    assert curve[0.2]["model"] > max(curve[0.2]["treat_all"], 0.0)
    assert metrics["scaled_brier"] > 0.1


def test_decision_curve_matches_direct_computation():
    time, event, risk = _cohort(seed=8)
    p = _probability(risk, 24.0)
    stats = horizon_statistics(time, event, p, 24.0, GRID)
    y, w = horizon_labels(time, event, 24.0, "ipcw")
    for entry in decision_curve(stats, (0.3, 0.33)):
        t = entry["threshold"]
        if t == 0.33:
            assert entry["model"] is None  # not a bin edge: not evaluable exactly
            continue
        treated = p >= t
        expected = ((w * y)[treated].sum() - (w * (1 - y))[treated].sum() * t / (1 - t)) / w.sum()
        assert entry["model"] == pytest.approx(expected, abs=1e-12)


def test_small_bins_are_merged_and_totals_kept():
    time, event, risk = _cohort(seed=9, n=150)
    stats = horizon_statistics(time, event, _probability(risk, 24.0), 24.0, GRID)
    merged = merge_bins(stats, 10)
    assert all(n == 0 or n >= 10 for n in merged.bin_n)
    assert set(merged.edges) <= set(stats.edges)
    assert merged.edges[0] == 0.0 and merged.edges[-1] == 1.0
    for name in ("bin_n", "bin_w", "bin_wy", "bin_wp"):
        assert getattr(merged, name).sum() == pytest.approx(getattr(stats, name).sum())
    other = merge_bins(horizon_statistics(*_cohort(seed=10, n=150)[:2],
                                          _probability(_cohort(seed=10, n=150)[2], 24.0),
                                          24.0, GRID), 25)
    pooled = pool([merged, other])
    assert set(pooled.edges) == set(merged.edges) & set(other.edges)
    assert pooled.bin_n.sum() == 300


def test_invalid_predictions_are_rejected():
    time, event, _ = _cohort()
    with pytest.raises(ValueError, match="within"):
        horizon_statistics(time, event, np.full(len(time), 1.2), 24.0, GRID)
    empty = summarize(HorizonStatistics(horizon=24.0, edges=(0.0, 1.0),
                                        bin_n=np.zeros(1), bin_w=np.zeros(1),
                                        bin_wy=np.zeros(1), bin_wp=np.zeros(1)), ())
    assert empty["auc"] is None and empty["brier"] is None


def test_merging_never_leaves_a_small_nonempty_bin():
    rng = np.random.default_rng(0)
    for _ in range(3000):
        counts = rng.integers(0, 8, size=rng.integers(1, 15)).astype(float)
        if counts.sum() < 10:
            continue
        k = len(counts)
        stats = HorizonStatistics(horizon=1.0, edges=tuple(np.linspace(0, 1, k + 1)),
                                  bin_n=counts, bin_w=counts, bin_wy=counts / 2,
                                  bin_wp=counts / 3)
        merged = merge_bins(stats, 10)
        assert not any(0 < n < 10 for n in merged.bin_n)
        assert merged.bin_n.sum() == counts.sum()
        assert merged.bin_wy.sum() == pytest.approx(counts.sum() / 2)
