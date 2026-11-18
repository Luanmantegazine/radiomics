"""Federated statistics equal their centralized counterparts (method checks)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from client.relevance import cox_score_components, cox_score_test
from coordinator.selection import statistics as st
from tests.federated import phantoms


def _sites(seed: int = 0, offsets=(0.0, 3.0, -2.0), sizes=(150, 120, 200)):
    """Three sites with feature offsets and different baseline hazards."""
    out = []
    for k, (offset, n) in enumerate(zip(offsets, sizes, strict=True)):
        names, x, time, event = phantoms.survival_cohort(n=n, p=6, seed=seed * 10 + k)
        out.append((f"client-0{k + 1}", x + offset, time * (1 + k), event))
    return names, out


def _relevance(sites):
    return [
        st.SiteRelevance(cid, len(t), int(e.sum()), *cox_score_components(x, t, e))
        for cid, x, t, e in sites
    ]


def _centralized_stratified_z(sites):
    """Brute force: one risk set per site and event time, summed over sites."""
    p = sites[0][1].shape[1]
    score, information = np.zeros(p), np.zeros(p)
    for _, x, time, event in sites:
        for i in np.flatnonzero(event):
            risk = x[time >= time[i]]
            score += x[i] - risk.mean(axis=0)
            information += risk.var(axis=0)
    return score / np.sqrt(information)


def test_stratified_score_equals_centralized_stratified_computation():
    _, sites = _sites()
    np.testing.assert_allclose(
        st.stratified_z(_relevance(sites)), _centralized_stratified_z(sites), rtol=1e-10
    )


def test_stratification_protects_against_site_offsets():
    """Pooling raw data across sites without strata confounds site and outcome."""
    _, sites = _sites(offsets=(0.0, 4.0, -4.0))
    unstratified = cox_score_test(
        np.concatenate([x for _, x, _, _ in sites]),
        np.concatenate([t for _, _, t, _ in sites]),
        np.concatenate([e for _, _, _, e in sites]),
        min_events=5,
    ).statistic
    stratified = st.stratified_z(_relevance(sites))
    assert not np.allclose(unstratified, stratified, rtol=1e-3)
    # Noise features stay null under stratification.
    assert np.all(np.abs(stratified[3:]) < 4)


def test_stouffer_and_mean_rank():
    _, sites = _sites(seed=1)
    relevance = _relevance(sites)
    z = np.stack([st.site_z(s) for s in relevance])
    w = np.sqrt([s.n_events for s in relevance])
    expected = (w[:, None] * z).sum(0) / np.sqrt((w**2).sum())
    np.testing.assert_allclose(st.stouffer_z(relevance), expected)
    eligible = np.ones(z.shape[1], dtype=bool)
    ranks = st.mean_rank_score(relevance, eligible)
    expected_ranks = np.mean([stats.rankdata(np.abs(row)) / len(row) for row in z], axis=0)
    np.testing.assert_allclose(ranks, expected_ranks)
    assert np.argmax(ranks) in (0, 2)  # planted features rank first everywhere


def test_i_squared_known_answers():
    def site(u, i):
        return st.SiteRelevance("c", 100, 20, np.array([u]), np.array([i]))

    identical = [site(4.0, 16.0), site(4.0, 16.0), site(4.0, 16.0)]
    assert st.i_squared(identical)[0] == 0.0
    # One-step estimates b = (+1, -1) with weights 25 each: Q = 50, I^2 = 49/50.
    opposite = [site(25.0, 25.0), site(-25.0, 25.0)]
    assert st.i_squared(opposite)[0] == pytest.approx(0.98)
    assert st.sign_consistency(opposite)[0] != st.sign_consistency(opposite)[0]  # NaN: sum 0
    mixed = [site(5.0, 25.0), site(5.0, 25.0), site(-1.0, 25.0)]
    assert st.sign_consistency(mixed)[0] == pytest.approx(2 / 3)
    # Q from the formula equals the textbook inverse-variance definition.
    b, w = np.array([0.2, 0.2, -0.04]), np.full(3, 25.0)
    q = (w * (b - (w * b).sum() / w.sum()) ** 2).sum()
    assert st.i_squared(mixed)[0] == pytest.approx(max(0.0, (q - 2) / q))


def test_feature_constant_at_one_site_is_not_estimable():
    _, sites = _sites(seed=2)
    sites[1][1][:, 4] = 7.0
    relevance = _relevance(sites)
    assert not st.estimable(relevance)[4] and st.estimable(relevance)[:4].all()
    assert np.isnan(st.i_squared(relevance)[4])


@pytest.mark.parametrize("method", ["pearson", "spearman"])
def test_pooled_correlation_equals_centralized_within_site_correlation(method):
    _, sites = _sites(seed=3)
    scatters, centred = [], []
    for cid, x, _, _ in sites:
        values = stats.rankdata(x, axis=0) if method == "spearman" else x
        values = values - values.mean(axis=0)
        centred.append(values)
        scatters.append(st.SiteScatter(cid, len(x), values.T @ values))
    expected = np.abs(np.corrcoef(np.concatenate(centred), rowvar=False))
    np.testing.assert_allclose(st.pooled_correlation(scatters), expected, atol=1e-12)
    # Without within-site centring, the site offsets create spurious correlation.
    assert not np.allclose(
        np.abs(np.corrcoef(np.concatenate([x for _, x, _, _ in sites]), rowvar=False)),
        expected, atol=0.05,
    )


def test_reliability_aggregation():
    sites = [
        st.SiteReliability("a", 20, np.array([0.9, 0.5, np.nan])),
        st.SiteReliability("b", 60, np.array([0.7, 0.9, 0.8])),
    ]
    np.testing.assert_allclose(st.reliability_minimum(sites), [0.7, 0.5, np.nan])
    np.testing.assert_allclose(st.reliability_weighted_mean(sites), [0.75, 0.8, np.nan])


def test_scatter_round_trip_and_percentiles():
    matrix = np.array([[4.0, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]])
    np.testing.assert_array_equal(st.unpack_upper(matrix[np.triu_indices(3)], 3), matrix)
    np.testing.assert_allclose(st.percentile_rank(np.array([3.0, 1.0, 3.0, 2.0])),
                               [0.875, 0.25, 0.875, 0.5])
    p = st.two_sided_p(np.array([1.959963984540054, np.nan]))
    assert p[0] == pytest.approx(0.05) and np.isnan(p[1])
