"""Prognostic relevance (censoring-aware), redundancy and selection stability."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sksurv.compare import compare_survival
from sksurv.util import Surv

from client.redundancy import correlation_matrix, redundancy_groups
from client.relevance import RelevanceUnavailable, concordance, cox_score_test, get_relevance
from evaluation.stability import jaccard, nogueira_stability, selection_frequency
from tests.federated import phantoms


def _brute_force_score(x, time, event):
    score = information = 0.0
    for i in np.flatnonzero(event):
        risk = x[time >= time[i]]
        score += x[i] - risk.mean()
        information += risk.var()
    return score / np.sqrt(information)


def test_score_test_matches_definition_and_handles_ties():
    _, x, time, event = phantoms.survival_cohort(n=120, seed=1)
    time = np.ceil(time)  # heavy ties
    result = cox_score_test(x, time, event, min_events=5)
    for j in range(x.shape[1]):
        assert result.statistic[j] == pytest.approx(_brute_force_score(x[:, j], time, event))
    assert result.n_events == int(event.sum())


def test_score_test_equals_logrank_for_a_binary_feature():
    _, x, time, event = phantoms.survival_cohort(n=200, seed=2)
    group = (x[:, 0] > 0).astype(float)
    chi2, _ = compare_survival(Surv.from_arrays(event, time), group)
    z = cox_score_test(group[:, None], time, event, min_events=5).statistic[0]
    # Continuous times (no ties): identical up to the logrank's finite-sample variance.
    assert z**2 == pytest.approx(chi2, rel=0.02)


def test_relevance_recovers_planted_signal_and_is_scale_invariant():
    names, x, time, event = phantoms.survival_cohort(seed=3)
    for method in ("cox_score", "concordance"):
        result = get_relevance(method)(x, time, event, min_events=10)
        ranking = [names[i] for i in np.argsort(-result.score)]
        assert set(ranking[:3]) == {"f0", "f1", "f2"}, method
    base = cox_score_test(x, time, event, min_events=10)
    rescaled = cox_score_test(x * 1000 + 7, time, event, min_events=10)
    np.testing.assert_allclose(base.statistic, rescaled.statistic, rtol=1e-8)
    assert base.statistic[0] > 0  # higher f0 -> higher hazard
    assert concordance(x, time, event, min_events=10).statistic[0] > 0.6
    assert base.p_value[0] < 1e-6 and base.p_value[5] > 1e-4


def test_censored_participants_are_not_treated_as_non_converters():
    """Shortening the follow-up of censored participants must change the result:
    they only count while they are at risk."""
    _, x, time, event = phantoms.survival_cohort(n=200, seed=4)
    full = cox_score_test(x[:, :1], time, event, min_events=5).statistic[0]
    truncated_time = np.where(event, time, time * 0.05)
    truncated = cox_score_test(x[:, :1], truncated_time, event, min_events=5).statistic[0]
    assert truncated != pytest.approx(full)
    # And a naive binary "converted vs not" comparison ignores time altogether.
    naive = stats.pointbiserialr(event, x[:, 0]).statistic
    assert np.isfinite(naive)


def test_relevance_requires_enough_events():
    _, x, time, event = phantoms.survival_cohort(n=40, seed=5)
    with pytest.raises(RelevanceUnavailable, match="events observed"):
        cox_score_test(x, time, event, min_events=int(event.sum()) + 1)
    with pytest.raises(RelevanceUnavailable):
        concordance(x, time, np.zeros_like(event), min_events=1)
    with pytest.raises(ValueError, match="unknown relevance method"):
        get_relevance("t_test")


def test_constant_feature_has_undefined_relevance():
    _, x, time, event = phantoms.survival_cohort(n=80, seed=6)
    x[:, 4] = 1.0
    assert np.isnan(cox_score_test(x, time, event, min_events=5).score[4])
    assert np.isnan(concordance(x, time, event, min_events=5).score[4])


def test_correlation_matrix_and_groups():
    _, x, _, _ = phantoms.survival_cohort(seed=7)
    x[:, 5] = -np.exp(x[:, 4])  # monotone, non-linear, decreasing
    spearman = correlation_matrix(x, "spearman")
    np.testing.assert_allclose(spearman, np.abs(stats.spearmanr(x).statistic), atol=1e-12)
    np.testing.assert_allclose(
        correlation_matrix(x, "pearson"), np.abs(np.corrcoef(x, rowvar=False)), atol=1e-12
    )
    assert spearman[4, 5] == pytest.approx(1.0)
    assert correlation_matrix(x, "pearson")[4, 5] < 0.95
    groups = redundancy_groups(spearman, 0.8)
    assert groups[0] == groups[2] and groups[4] == groups[5]
    assert len(set(groups)) == x.shape[1] - 2
    x[:, 6] = 3.0
    assert correlation_matrix(x, "spearman")[6, :6].max() == 0.0


def test_selection_stability_measures():
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    same = np.tile([True, True, False, False, False, False], (20, 1))
    assert nogueira_stability(same) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    random = np.array([rng.permutation(same[0]) for _ in range(2000)])
    assert abs(nogueira_stability(random)) < 0.02
    np.testing.assert_allclose(selection_frequency(same), [1, 1, 0, 0, 0, 0])
    assert np.isnan(nogueira_stability(same[:1]))
    assert np.isnan(nogueira_stability(np.ones((5, 4), dtype=bool)))
