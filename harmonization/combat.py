"""ComBat with parametric empirical Bayes, and its federated decomposition.

Model, for feature g of participant j at site i::

    y_ijg = alpha_g + x_ij' beta_g + gamma_ig + delta_ig * eps_ijg

``x`` are biological covariates to preserve (e.g. age); ``gamma``/``delta``
are additive/multiplicative site effects. Estimation (as neuroCombat):

1. OLS of y on [site indicators, covariates]; grand mean = site intercepts
   averaged with weights n_i / n; pooled variance = mean squared residual.
2. Standardise: s = (y - grand_mean - x' beta) / sqrt(var_pooled).
3. Per site: gamma_hat = mean(s), delta_hat = var(s); empirical Bayes priors
   are estimated *across features within the site*, so step 3 is local.
4. Adjust: y* = (s - gamma*) / sqrt(delta*) * sqrt(var_pooled) + grand_mean + x' beta.

Federated (d-ComBat): sites send sums for step 1 (n, sum x, sum x x', sum y,
sum x y'); the site intercept that solves the OLS normal equations is the site
mean of (y - x' beta), so each site computes it itself; sites send residual
sums of squares for the pooled variance; step 3 and 4 run locally.

Hypotheses and limitations
--------------------------
* Site effects are additive and multiplicative per feature; features within a
  site share the empirical Bayes priors.
* **Everything that differs between sites and is not a listed covariate is
  removed**, including genuine clinical differences between cohorts (e.g. a
  site with more advanced disease). Covariates must capture them; the
  assessment module checks how much sites differ clinically.
* The outcome is never a covariate here (it would leak into prognostic models).
* Parameters are estimated on the training partition and applied unchanged to
  held-out participants of the same site; a new site needs its own estimation.
* Only the parametric EB variant is implemented (no non-parametric EB, no
  reference batch, no longitudinal ComBat).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SiteSums:
    """Additive regression statistics of one site (step 1)."""

    n: int
    sum_x: np.ndarray  # (c,)
    sum_xx: np.ndarray  # (c, c)
    sum_y: np.ndarray  # (p,)
    sum_yy: np.ndarray  # (p,)  (for the site-effect assessment)
    sum_xy: np.ndarray  # (c, p)


def site_sums(y: np.ndarray, x: np.ndarray) -> SiteSums:
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float).reshape(len(y), -1)
    return SiteSums(len(y), x.sum(0), x.T @ x, y.sum(0), (y**2).sum(0), x.T @ y)


def solve_regression(sites: list[SiteSums]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS on [site indicators, covariates] from site sums.

    Returns ``(intercepts (K, p), beta (c, p), grand_mean (p,))``.
    """
    k, c = len(sites), sites[0].sum_x.shape[0]
    xtx = np.zeros((k + c, k + c))
    xty = np.zeros((k + c, sites[0].sum_y.shape[0]))
    for i, s in enumerate(sites):
        xtx[i, i] = s.n
        xtx[i, k:] = xtx[k:, i] = s.sum_x
        xtx[k:, k:] += s.sum_xx
        xty[i] = s.sum_y
        xty[k:] += s.sum_xy
    solution = np.linalg.solve(xtx, xty)
    n = np.array([s.n for s in sites], dtype=float)
    intercepts, beta = solution[:k], solution[k:]
    return intercepts, beta, (n[:, None] * intercepts).sum(0) / n.sum()


def site_intercept(y: np.ndarray, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """The site's OLS intercept given the shared covariate effects."""
    return (np.asarray(y) - np.asarray(x).reshape(len(y), -1) @ beta).mean(0)


def residual_sum_of_squares(y: np.ndarray, x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(len(y), -1)
    residual = np.asarray(y) - x @ beta
    return ((residual - residual.mean(0)) ** 2).sum(0)


@dataclass(frozen=True, slots=True)
class GlobalCombat:
    """Parameters shared by all sites (released to the sites, not individual)."""

    grand_mean: np.ndarray  # (p,)
    beta: np.ndarray  # (c, p)
    var_pooled: np.ndarray  # (p,)


@dataclass(frozen=True, slots=True)
class SiteCombat:
    """A site's own location/scale adjustment (kept at the site)."""

    globals: GlobalCombat
    gamma_star: np.ndarray  # (p,)
    delta_star: np.ndarray  # (p,)

    def apply(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        g = self.globals
        x = np.asarray(x, dtype=float).reshape(len(y), -1)
        mean = g.grand_mean + x @ g.beta
        s = (np.asarray(y, dtype=float) - mean) / np.sqrt(g.var_pooled)
        return (s - self.gamma_star) / np.sqrt(self.delta_star) * np.sqrt(g.var_pooled) + mean


def _priors(gamma_hat: np.ndarray, delta_hat: np.ndarray):
    gamma_bar, t2 = gamma_hat.mean(), gamma_hat.var(ddof=1)
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return gamma_bar, t2, (2 * s2 + m**2) / s2, (m * s2 + m**3) / s2


def fit_site(
    y: np.ndarray, x: np.ndarray, globals_: GlobalCombat, *, empirical_bayes: bool,
    tolerance: float,
) -> SiteCombat:
    """Steps 2-3 at one site: standardise, estimate (and shrink) site effects."""
    x = np.asarray(x, dtype=float).reshape(len(y), -1)
    s = (np.asarray(y, dtype=float) - globals_.grand_mean - x @ globals_.beta) / np.sqrt(
        globals_.var_pooled
    )
    n = s.shape[0]
    if n < 2:
        raise ValueError("a site needs at least two participants")
    gamma_hat = s.mean(0)
    delta_hat = s.var(0, ddof=1)
    delta_hat[delta_hat == 0] = 1.0
    if not empirical_bayes:
        return SiteCombat(globals_, gamma_hat, delta_hat)
    if s.shape[1] < 2:
        raise ValueError("empirical Bayes needs at least two features")
    gamma_bar, t2, a, b = _priors(gamma_hat, delta_hat)
    g_old, d_old = gamma_hat.copy(), delta_hat.copy()
    for _ in range(10_000):
        g_new = (t2 * n * gamma_hat + d_old * gamma_bar) / (t2 * n + d_old)
        sum2 = ((s - g_new) ** 2).sum(0)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
        change = max(np.max(np.abs(g_new - g_old) / np.abs(g_old)),
                     np.max(np.abs(d_new - d_old) / d_old))
        g_old, d_old = g_new, d_new
        if change <= tolerance:
            return SiteCombat(globals_, g_new, d_new)
    raise ValueError("empirical Bayes iteration did not converge")


def fit_combat(
    y: np.ndarray, batch: np.ndarray, x: np.ndarray | None = None, *,
    empirical_bayes: bool = True, tolerance: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """Centralized ComBat (reference). Returns adjusted data and site models."""
    y = np.asarray(y, dtype=float)
    x = np.zeros((len(y), 0)) if x is None else np.asarray(x, dtype=float).reshape(len(y), -1)
    labels = list(dict.fromkeys(batch))
    rows = {label: np.asarray(batch) == label for label in labels}
    sums = [site_sums(y[r], x[r]) for r in rows.values()]
    _, beta, grand_mean = solve_regression(sums)
    rss = sum(residual_sum_of_squares(y[r], x[r], beta) for r in rows.values())
    globals_ = GlobalCombat(grand_mean, beta, rss / len(y))
    if (globals_.var_pooled <= 0).any():
        raise ValueError("a feature has no residual variance")
    adjusted = np.empty_like(y)
    models = {}
    for label, r in rows.items():
        models[label] = fit_site(y[r], x[r], globals_, empirical_bayes=empirical_bayes,
                                 tolerance=tolerance)
        adjusted[r] = models[label].apply(y[r], x[r])
    return adjusted, models
