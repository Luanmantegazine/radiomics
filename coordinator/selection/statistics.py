"""Federated statistics for the four selection dimensions.

Pure functions over per-site sufficient statistics (NumPy only). Each one is
exactly equal to a centralized computation stated in its docstring, which the
tests verify; none needs individual-level data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Per-site inputs (held only in coordinator memory during a selection run)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SiteReliability:
    client_id: str
    n_subjects: int
    values: np.ndarray  # (p,), NaN = undefined at this site


@dataclass(frozen=True, slots=True)
class SiteRelevance:
    client_id: str
    n_samples: int
    n_events: int
    score: np.ndarray  # U, (p,)
    information: np.ndarray  # I, (p,)


@dataclass(frozen=True, slots=True)
class SiteScatter:
    client_id: str
    n_samples: int
    scatter: np.ndarray  # (q, q) within-site scatter of the candidate features


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def reliability_minimum(sites: Sequence[SiteReliability]) -> np.ndarray:
    """Worst site per feature; undefined at any site -> undefined."""
    return np.min(np.stack([s.values for s in sites]), axis=0)


def reliability_weighted_mean(sites: Sequence[SiteReliability]) -> np.ndarray:
    """Mean weighted by reliability-study size; undefined at any site -> undefined."""
    weights = np.array([s.n_subjects for s in sites], dtype=float)
    values = np.stack([s.values for s in sites])
    return (weights[:, None] * values).sum(axis=0) / weights.sum()


# --------------------------------------------------------------------------- #
# Relevance and agreement
# --------------------------------------------------------------------------- #
def estimable(sites: Sequence[SiteRelevance]) -> np.ndarray:
    """Features with positive information at every site.

    A feature that is constant within a site has I = 0 there; such a feature
    cannot be assessed for between-site agreement and is excluded.
    """
    return np.all(np.stack([s.information for s in sites]) > 0, axis=0)


def stratified_z(sites: Sequence[SiteRelevance]) -> np.ndarray:
    """Score test of a Cox model stratified by site: sum(U) / sqrt(sum(I)).

    Equal to the score test computed centrally with one baseline hazard per
    site (each site's risk sets contain only its own participants).
    """
    u = np.sum([s.score for s in sites], axis=0)
    i = np.sum([s.information for s in sites], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(i > 0, u / np.sqrt(i), np.nan)


def site_z(site: SiteRelevance) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(site.information > 0, site.score / np.sqrt(site.information), np.nan)


def stouffer_z(sites: Sequence[SiteRelevance]) -> np.ndarray:
    """Weighted Stouffer combination of per-site z, weights sqrt(events)."""
    weights = np.sqrt([s.n_events for s in sites])
    z = np.stack([site_z(s) for s in sites])
    return (weights[:, None] * z).sum(axis=0) / np.sqrt((weights**2).sum())


def mean_rank_score(sites: Sequence[SiteRelevance], eligible: np.ndarray) -> np.ndarray:
    """Mean over sites of the percentile rank of |z| among eligible features.

    1 = best at every site. Ties get average ranks.
    """
    scores = np.full(len(eligible), np.nan)
    if not eligible.any():
        return scores
    ranks = [percentile_rank(np.abs(site_z(s))[eligible]) for s in sites]
    scores[eligible] = np.mean(ranks, axis=0)
    return scores


def i_squared(sites: Sequence[SiteRelevance]) -> np.ndarray:
    """Higgins' I^2 of the one-step per-site log-hazard estimates b_k = U_k / I_k.

    With inverse-variance weights I_k: Q = sum U_k^2/I_k - (sum U)^2 / sum I,
    I^2 = max(0, (Q - (K - 1)) / Q). Hypothesis: one-step estimates are good
    approximations of the per-site Cox coefficients (small effects).
    Q = 0 (perfect agreement) gives I^2 = 0.
    """
    u = np.stack([s.score for s in sites])
    i = np.stack([s.information for s in sites])
    k = len(sites)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (u**2 / i).sum(axis=0) - u.sum(axis=0) ** 2 / i.sum(axis=0)
        q = np.clip(q, 0.0, None)
        result = np.where(q > 0, np.clip((q - (k - 1)) / q, 0.0, None), 0.0)
    result[~np.all(i > 0, axis=0)] = np.nan
    return result


def sign_consistency(sites: Sequence[SiteRelevance]) -> np.ndarray:
    """Fraction of sites whose score has the sign of the pooled score."""
    u = np.stack([s.score for s in sites])
    pooled = np.sign(u.sum(axis=0))
    result = (np.sign(u) == pooled).mean(axis=0)
    result[pooled == 0] = np.nan
    return result


def two_sided_p(z: np.ndarray) -> np.ndarray:
    return np.array([math.erfc(abs(v) / math.sqrt(2)) if np.isfinite(v) else np.nan for v in z])


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile in (0, 1]; larger value -> larger percentile."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):  # average ties
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks / len(values)


# --------------------------------------------------------------------------- #
# Redundancy
# --------------------------------------------------------------------------- #
def unpack_upper(upper: Sequence[float], p: int) -> np.ndarray:
    matrix = np.zeros((p, p))
    matrix[np.triu_indices(p)] = upper
    return matrix + np.triu(matrix, 1).T


def pooled_correlation(sites: Sequence[SiteScatter]) -> np.ndarray:
    """Absolute pooled within-site correlation from summed scatter matrices.

    Equal to the correlation of the concatenated data after centring each site
    on its own means (site offsets do not create spurious correlation).
    Features with zero pooled variance get 0 off the diagonal.
    """
    scatter = np.sum([s.scatter for s in sites], axis=0)
    sd = np.sqrt(np.clip(np.diag(scatter), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(scatter / np.outer(sd, sd))
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, 0.0, 1.0)
