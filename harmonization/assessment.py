"""How much do sites differ? Feature site effects and clinical/follow-up differences.

All quantities come from per-site sums, so they can be computed federatedly.

``eta_squared``  between-site share of the variance of each feature
                 (one-way ANOVA): SS_between / SS_total.
``f_statistic``  (SS_between / (K-1)) / (SS_within / (n-K)).
``event_rate_chi2``  Pearson chi-square of site x converted (crude; ignores
                 censoring and follow-up differences).

Limitations: eta^2 is descriptive; no p-values are given (they would need the
F distribution and multiple-testing control across hundreds of features).
A site effect after harmonization close to 0 does not prove the adjustment is
right: if sites differ clinically, harmonization may have removed real signal.
"""

from __future__ import annotations

import numpy as np


def site_effect(n: np.ndarray, sums: np.ndarray, sums_sq: np.ndarray) -> dict:
    """Per-feature eta^2 and F from per-site counts, sums and sums of squares.

    ``n`` (K,), ``sums``/``sums_sq`` (K, p).
    """
    n = np.asarray(n, dtype=float)
    sums, sums_sq = np.atleast_2d(sums).astype(float), np.atleast_2d(sums_sq).astype(float)
    k, total = len(n), n.sum()
    grand = sums.sum(0) / total
    ss_total = sums_sq.sum(0) - total * grand**2
    ss_between = ((sums / n[:, None] - grand) ** 2 * n[:, None]).sum(0)
    ss_within = ss_total - ss_between
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(ss_total > 0, ss_between / ss_total, np.nan)
        f = np.where(ss_within > 0, (ss_between / (k - 1)) / (ss_within / (total - k)), np.nan)
    return {"eta_squared": eta, "f_statistic": f}


def event_rate_chi2(n: np.ndarray, events: np.ndarray) -> float:
    n, events = np.asarray(n, dtype=float), np.asarray(events, dtype=float)
    rate = events.sum() / n.sum()
    if rate in (0.0, 1.0):
        return 0.0
    expected = np.stack([n * rate, n * (1 - rate)])
    observed = np.stack([events, n - events])
    return float(((observed - expected) ** 2 / expected).sum())


def summarize_effect(eta: np.ndarray) -> dict:
    finite = eta[np.isfinite(eta)]
    if finite.size == 0:
        return {"median_eta_squared": None, "max_eta_squared": None,
                "fraction_above_0_1": None}
    return {
        "median_eta_squared": float(np.median(finite)),
        "max_eta_squared": float(finite.max()),
        "fraction_above_0_1": float((finite > 0.1).mean()),
    }
