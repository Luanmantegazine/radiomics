"""Intraclass correlation coefficients (Shrout & Fleiss 1979; McGraw & Wong 1996).

Single-measurement forms, computed from the two-way ANOVA mean squares of an
``n subjects x k conditions`` table, vectorised over features:

``icc1``  ICC(1,1): one-way random effects.
``icc2``  ICC(2,1): two-way, absolute agreement (systematic differences between
          conditions count as disagreement).
``icc3``  ICC(3,1): two-way, consistency (systematic offsets are ignored).

Hypotheses and limitations
--------------------------
* Complete, balanced data: every subject observed under every condition.
* ANOVA assumptions (additive effects, approximately normal residuals,
  homogeneous variances). Skewed radiomic features violate them; no transform
  is applied here.
* The ICC depends on between-subject variance: the same measurement error
  yields a lower ICC in a more homogeneous cohort. Values are not portable
  across populations.
* Confidence intervals are the exact-F intervals of Shrout & Fleiss; with few
  subjects they are wide and the point estimate alone should not be trusted.
* A feature that does not vary at all has an undefined ICC (reported as NaN).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy import stats


class IccForm(StrEnum):
    ICC1 = "icc1"
    ICC2 = "icc2"
    ICC3 = "icc3"


class ReliabilityUnavailable(Exception):
    """Reliability cannot be estimated from the observations provided."""


@dataclass(frozen=True, slots=True)
class IccResult:
    form: IccForm
    estimate: np.ndarray  # shape (n_features,)
    ci_low: np.ndarray
    ci_high: np.ndarray
    confidence_level: float
    n_subjects: int
    n_conditions: int


def icc(
    measurements: np.ndarray,
    form: IccForm,
    *,
    confidence_level: float = 0.95,
    min_subjects: int = 2,
) -> IccResult:
    """ICC per feature.

    ``measurements`` has shape ``(n_subjects, n_conditions, n_features)`` (a 2-D
    array is treated as a single feature).
    """
    y = np.asarray(measurements, dtype=float)
    if y.ndim == 2:
        y = y[:, :, None]
    if y.ndim != 3:
        raise ValueError("measurements must have shape (n_subjects, n_conditions, n_features)")
    n, k, _ = y.shape
    if k < 2:
        raise ReliabilityUnavailable(
            "reliability needs at least two observations per subject (repeated or alternative)"
        )
    if n < max(min_subjects, 2):
        raise ReliabilityUnavailable(f"only {n} subjects with complete repeated observations")
    if not np.isfinite(y).all():
        raise ValueError("measurements contain non-finite values")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    grand = y.mean(axis=(0, 1))
    ss_rows = k * ((y.mean(axis=1) - grand) ** 2).sum(axis=0)
    ss_cols = n * ((y.mean(axis=0) - grand) ** 2).sum(axis=0)
    ss_error = np.clip(((y - grand) ** 2).sum(axis=(0, 1)) - ss_rows - ss_cols, 0.0, None)
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))
    msw = (ss_cols + ss_error) / (n * (k - 1))
    q = 1 - (1 - confidence_level) / 2

    with np.errstate(divide="ignore", invalid="ignore"):
        if form is IccForm.ICC1:
            estimate = (msr - msw) / (msr + (k - 1) * msw)
            f = msr / msw
            f_low = f / stats.f.ppf(q, n - 1, n * (k - 1))
            f_high = f * stats.f.ppf(q, n * (k - 1), n - 1)
            low, high = (f_low - 1) / (f_low + k - 1), (f_high - 1) / (f_high + k - 1)
        elif form is IccForm.ICC3:
            estimate = (msr - mse) / (msr + (k - 1) * mse)
            f = msr / mse
            f_low = f / stats.f.ppf(q, n - 1, (n - 1) * (k - 1))
            f_high = f * stats.f.ppf(q, (n - 1) * (k - 1), n - 1)
            low, high = (f_low - 1) / (f_low + k - 1), (f_high - 1) / (f_high + k - 1)
        elif form is IccForm.ICC2:
            estimate = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
            fj = msc / mse
            a = n * (1 + (k - 1) * estimate) - k * estimate
            v = ((k - 1) * (n - 1) * (k * estimate * fj + a) ** 2) / (
                (n - 1) * k**2 * estimate**2 * fj**2 + a**2
            )
            f1, f2 = stats.f.ppf(q, n - 1, v), stats.f.ppf(q, v, n - 1)
            common = k * msc + (k * n - k - n) * mse
            low = n * (msr - f1 * mse) / (f1 * common + n * msr)
            high = n * (f2 * msr - mse) / (common + n * f2 * msr)
        else:  # pragma: no cover
            raise ValueError(f"unknown ICC form {form!r}")

    # Degenerate cases, decided relative to the data scale (floating point).
    ss_total = ((y - grand) ** 2).sum(axis=(0, 1))
    undefined = ss_total <= 1e-18 * (y**2).sum(axis=(0, 1))  # no variation at all
    disagreement = ss_error + (ss_cols if form is not IccForm.ICC3 else 0.0)
    perfect = ~undefined & (disagreement <= 1e-14 * ss_total)
    for array in (estimate, low, high):
        array[perfect] = 1.0
        array[undefined] = np.nan
    return IccResult(form, estimate, low, high, confidence_level, n, k)
