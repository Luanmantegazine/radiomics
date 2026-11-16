"""Redundancy between features: absolute pairwise correlation and its graph.

Hypotheses and limitations
--------------------------
* Pairwise only: a feature that is a combination of several others is not
  detected (no multicollinearity / VIF analysis).
* Spearman captures monotone, Pearson linear, dependence; neither captures
  non-monotone relations.
* With few participants, correlations are noisy and a hard threshold makes the
  resulting groups unstable.
* Correlation is estimated without using the outcome, so it does not leak
  outcome information, but it is still cohort-specific.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse, stats
from scipy.sparse.csgraph import connected_components

CorrelationMethod = Literal["spearman", "pearson"]


def correlation_matrix(features: np.ndarray, method: CorrelationMethod) -> np.ndarray:
    """Absolute correlation between features, shape (p, p), diagonal = 1.

    Features with no variation have undefined correlation; they are reported as
    uncorrelated with everything (0) and should be removed beforehand.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError("need a (n, p) matrix with at least 3 rows")
    if method == "spearman":
        x = stats.rankdata(x, axis=0)
    elif method != "pearson":
        raise ValueError(f"unknown correlation method {method!r}")
    centred = x - x.mean(axis=0)
    norm = np.sqrt((centred**2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(centred.T @ centred / np.outer(norm, norm))
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, 0.0, 1.0)


def redundancy_groups(correlation: np.ndarray, threshold: float) -> np.ndarray:
    """Group label per feature: connected components of the graph whose edges
    join features with absolute correlation >= ``threshold``.

    Single linkage: two weakly correlated features end up together when a chain
    of strongly correlated features connects them.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    adjacency = sparse.csr_matrix(np.asarray(correlation) >= threshold)
    return connected_components(adjacency, directed=False)[1]
