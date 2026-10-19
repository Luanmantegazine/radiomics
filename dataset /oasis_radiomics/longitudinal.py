"""Bilateral derivation and longitudinal modelling of radiomic features.

Three transformations live here, applied in this order:

1. :func:`build_wide_table` - collapse the long ``session x ROI`` table into one
   row per ``subject x session``, adding ``_left`` / ``_right`` / ``_mean`` /
   ``_total`` / ``_asymmetry`` columns. This is where bilateral information is
   produced, **tabularly**, instead of by feeding PyRadiomics a disconnected
   union mask.
2. :func:`compute_deltas` - per-subject differences between timepoints,
   annualised.
3. :func:`compute_slopes` - per ``subject x feature`` ordinary least squares fit
   of the feature against time in years.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import BilateralConfig, LongitudinalConfig

logger = logging.getLogger(__name__)

#: Identifier columns carried through every table.
ID_COLUMNS = ("subject_id", "session_id", "days_from_reference")

#: Columns in the long table that describe the ROI rather than the image content.
LONG_METADATA_COLUMNS = ("roi", "image_path", "mask_path")

SUFFIX_LEFT = "_left"
SUFFIX_RIGHT = "_right"
SUFFIX_MEAN = "_mean"
SUFFIX_TOTAL = "_total"
SUFFIX_ASYMMETRY = "_asymmetry"

#: Convenience alias required by the study design: total hippocampal volume.
HIPPOCAMPAL_VOLUME_TOTAL = "hippocampal_volume_total"
MASK_VOLUME_FEATURE = "mask_volume_mm3"


class LongitudinalError(ValueError):
    """Raised when the input tables cannot be reshaped or modelled."""


# ---------------------------------------------------------------------------
# scalar helpers (unit-tested directly)
# ---------------------------------------------------------------------------
def calculate_mean(left: float, right: float) -> float:
    """Bilateral mean ``(left + right) / 2``; NaN if either side is missing."""
    left_value, right_value = float(left), float(right)
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        return float("nan")
    return (left_value + right_value) / 2.0


def calculate_total(left: float, right: float) -> float:
    """Bilateral sum ``left + right``; NaN if either side is missing."""
    left_value, right_value = float(left), float(right)
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        return float("nan")
    return left_value + right_value


def calculate_asymmetry(left: float, right: float, mode: str = "positive_only") -> float:
    """Normalised left-right asymmetry index ``(left - right) / (left + right)``.

    Parameters
    ----------
    left, right:
        Feature values for the two hemispheres.
    mode:
        ``"positive_only"`` (default) returns NaN unless both sides are strictly
        positive. The index is only interpretable for positive quantities:
        around zero the denominator collapses and the ratio explodes, which is
        exactly what happens for signed texture features such as
        ``firstorder_Skewness`` or ``glcm_ClusterShade``.
        ``"always"`` computes the index whenever the denominator is non-zero.

    Returns
    -------
    float
        The index, or NaN when it is undefined for the requested mode.
    """
    if mode not in ("positive_only", "always"):
        raise LongitudinalError(f"Unknown asymmetry mode: {mode!r}")

    left_value, right_value = float(left), float(right)
    if not (math.isfinite(left_value) and math.isfinite(right_value)):
        return float("nan")

    if mode == "positive_only" and not (left_value > 0.0 and right_value > 0.0):
        return float("nan")

    denominator = left_value + right_value
    if denominator == 0.0:
        return float("nan")

    return (left_value - right_value) / denominator


def calculate_delta(value_t0: float, value_t1: float) -> float:
    """Absolute change ``value_t1 - value_t0``; NaN if either endpoint is missing."""
    start, end = float(value_t0), float(value_t1)
    if not (math.isfinite(start) and math.isfinite(end)):
        return float("nan")
    return end - start


def calculate_annualised_rate(delta: float, delta_years: float) -> float:
    """Annualised rate of change ``delta / delta_years``.

    Returns NaN when the interval is zero or non-finite, so two scans acquired
    on the same day never produce an infinite slope.
    """
    delta_value, years = float(delta), float(delta_years)
    if not (math.isfinite(delta_value) and math.isfinite(years)) or years == 0.0:
        return float("nan")
    return delta_value / years


def calculate_slope(
    times: Sequence[float], values: Sequence[float]
) -> dict[str, float]:
    """Ordinary least squares fit ``value = intercept + slope * time``.

    Uses :func:`scipy.stats.linregress` when SciPy is available and falls back to
    an equivalent NumPy implementation otherwise. Only finite ``(time, value)``
    pairs are used, and at least two *distinct* times are required.

    Returns
    -------
    dict
        ``slope``, ``intercept``, ``r2``, ``pvalue``, ``stderr``, ``n_points``,
        ``time_span``. Statistics that cannot be estimated are NaN.
    """
    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if time_array.shape != value_array.shape:
        raise LongitudinalError("times and values must have the same length.")

    keep = np.isfinite(time_array) & np.isfinite(value_array)
    time_array, value_array = time_array[keep], value_array[keep]

    empty = {
        "slope": float("nan"),
        "intercept": float("nan"),
        "r2": float("nan"),
        "pvalue": float("nan"),
        "stderr": float("nan"),
        "n_points": float(time_array.size),
        "time_span": float("nan"),
    }

    if time_array.size < 2 or np.unique(time_array).size < 2:
        return empty

    empty["time_span"] = float(time_array.max() - time_array.min())

    try:
        from scipy.stats import linregress

        fit = linregress(time_array, value_array)
        return {
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
            "r2": float(fit.rvalue) ** 2,
            "pvalue": float(fit.pvalue),
            "stderr": float(fit.stderr),
            "n_points": float(time_array.size),
            "time_span": empty["time_span"],
        }
    except ImportError:  # pragma: no cover - SciPy is a pinned dependency
        logger.warning("SciPy unavailable; falling back to a NumPy least-squares fit.")
        return _numpy_linregress(time_array, value_array, empty["time_span"])


def _numpy_linregress(
    times: np.ndarray, values: np.ndarray, time_span: float
) -> dict[str, float]:
    """SciPy-free least squares fit with the same output contract."""
    slope, intercept = np.polyfit(times, values, 1)
    predicted = intercept + slope * times
    residual_ss = float(np.sum((values - predicted) ** 2))
    total_ss = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
        "pvalue": float("nan"),
        "stderr": float("nan"),
        "n_points": float(times.size),
        "time_span": time_span,
    }


# ---------------------------------------------------------------------------
# long -> wide
# ---------------------------------------------------------------------------
def _feature_columns(row: Mapping[str, Any]) -> list[str]:
    """Numeric feature columns of a long-format row."""
    skip = set(ID_COLUMNS) | set(LONG_METADATA_COLUMNS)
    return [key for key in row if key not in skip]


def build_wide_table(
    long_rows: Sequence[Mapping[str, Any]], config: BilateralConfig
) -> list[dict[str, Any]]:
    """Reshape the long ``session x ROI`` table into one row per session.

    For every feature present on both sides the output carries::

        <feature>_left       <feature>_right
        <feature>_mean       (all features)
        <feature>_total      (features listed in bilateral.additive_features)
        <feature>_asymmetry  (subject to bilateral.asymmetry_mode)

    ROIs other than the configured left/right pair (for instance a legacy
    bilateral union mask) are kept with a plain ``<feature>_<roi>`` prefix so no
    information is lost, but they take part in no derivation.
    """
    if not long_rows:
        raise LongitudinalError("Cannot build a wide table from an empty long table.")

    by_session: dict[str, dict[str, Mapping[str, Any]]] = {}
    session_meta: dict[str, dict[str, Any]] = {}
    for row in long_rows:
        session_id = str(row["session_id"])
        roi = str(row["roi"])
        by_session.setdefault(session_id, {})[roi] = row
        session_meta.setdefault(
            session_id,
            {column: row.get(column) for column in ID_COLUMNS},
        )

    wide_rows: list[dict[str, Any]] = []
    for session_id, rois in by_session.items():
        wide_rows.append(_build_wide_row(session_id, rois, session_meta[session_id], config))

    wide_rows.sort(key=lambda row: (row["subject_id"], row["days_from_reference"], row["session_id"]))
    logger.info("Wide table: %d session row(s), %d column(s).", len(wide_rows), len(wide_rows[0]))
    return wide_rows


def _build_wide_row(
    session_id: str,
    rois: Mapping[str, Mapping[str, Any]],
    meta: Mapping[str, Any],
    config: BilateralConfig,
) -> dict[str, Any]:
    """Assemble the wide row for a single session."""
    row: dict[str, Any] = dict(meta)
    row["session_id"] = session_id

    left_row = rois.get(config.left_roi)
    right_row = rois.get(config.right_roi)

    if left_row is None or right_row is None:
        missing = [
            name
            for name, value in ((config.left_roi, left_row), (config.right_roi, right_row))
            if value is None
        ]
        logger.warning(
            "Session %s is missing ROI(s) %s; bilateral columns will be NaN.",
            session_id,
            ", ".join(missing),
        )

    features = _paired_feature_names(left_row, right_row)
    additive = set(config.additive_features)

    for feature in features:
        left_value = _as_float(left_row.get(feature) if left_row else None)
        right_value = _as_float(right_row.get(feature) if right_row else None)

        row[f"{feature}{SUFFIX_LEFT}"] = left_value
        row[f"{feature}{SUFFIX_RIGHT}"] = right_value
        row[f"{feature}{SUFFIX_MEAN}"] = calculate_mean(left_value, right_value)
        if feature in additive:
            row[f"{feature}{SUFFIX_TOTAL}"] = calculate_total(left_value, right_value)
        row[f"{feature}{SUFFIX_ASYMMETRY}"] = calculate_asymmetry(
            left_value, right_value, config.asymmetry_mode
        )

    if f"{MASK_VOLUME_FEATURE}{SUFFIX_TOTAL}" in row:
        row[HIPPOCAMPAL_VOLUME_TOTAL] = row[f"{MASK_VOLUME_FEATURE}{SUFFIX_TOTAL}"]

    for roi_name, roi_row in rois.items():
        if roi_name in (config.left_roi, config.right_roi):
            continue
        for feature in _feature_columns(roi_row):
            row[f"{feature}_{roi_name}"] = _as_float(roi_row.get(feature))

    return row


def _paired_feature_names(
    left_row: Mapping[str, Any] | None, right_row: Mapping[str, Any] | None
) -> list[str]:
    """Ordered union of the feature names present on either side."""
    names: list[str] = []
    for row in (left_row, right_row):
        if row is None:
            continue
        for feature in _feature_columns(row):
            if feature not in names:
                names.append(feature)
    return names


def derived_feature_columns(wide_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Numeric columns of the wide table, i.e. everything but the identifiers."""
    if not wide_rows:
        return []
    return [column for column in wide_rows[0] if column not in ID_COLUMNS]


# ---------------------------------------------------------------------------
# deltas
# ---------------------------------------------------------------------------
def compute_deltas(
    wide_rows: Sequence[Mapping[str, Any]], config: LongitudinalConfig
) -> list[dict[str, Any]]:
    """Per-subject differences between timepoints.

    Two comparison schemes, selected in the configuration:

    ``consecutive``
        every ``t(i) -> t(i+1)`` pair;
    ``baseline``
        every ``t(0) -> t(i)`` pair, i > 0.

    Each output row carries ``delta_<feature>`` and ``slope_<feature>`` columns,
    where ``slope`` is the annualised rate ``delta / delta_years`` for that pair.
    Subjects with a single session produce no rows.
    """
    grouped = _group_rows_by_subject(wide_rows)
    features = derived_feature_columns(wide_rows)
    rows: list[dict[str, Any]] = []

    for subject_id, sessions in grouped.items():
        if len(sessions) < 2:
            logger.info("Subject %s has a single session; no deltas computed.", subject_id)
            continue

        for comparison in config.delta_comparisons:
            for start, end in _pairs(sessions, comparison):
                rows.append(_delta_row(subject_id, start, end, comparison, features, config))

    logger.info("Delta table: %d row(s) from %d subject(s).", len(rows), len(grouped))
    return rows


def _pairs(
    sessions: Sequence[Mapping[str, Any]], comparison: str
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Timepoint pairs for one comparison scheme."""
    if comparison == "consecutive":
        return list(zip(sessions[:-1], sessions[1:]))
    if comparison == "baseline":
        return [(sessions[0], session) for session in sessions[1:]]
    raise LongitudinalError(f"Unknown delta comparison scheme: {comparison!r}")


def _delta_row(
    subject_id: str,
    start: Mapping[str, Any],
    end: Mapping[str, Any],
    comparison: str,
    features: Sequence[str],
    config: LongitudinalConfig,
) -> dict[str, Any]:
    """Build one delta row for a pair of sessions."""
    delta_days = int(end["days_from_reference"]) - int(start["days_from_reference"])
    delta_years = delta_days / config.days_per_year

    row: dict[str, Any] = {
        "subject_id": subject_id,
        "comparison": comparison,
        "session_id_t0": start["session_id"],
        "session_id_t1": end["session_id"],
        "days_t0": start["days_from_reference"],
        "days_t1": end["days_from_reference"],
        "delta_days": delta_days,
        "delta_years": delta_years,
    }

    if delta_days == 0:
        logger.warning(
            "Sessions %s and %s share the same day; annualised rates will be NaN.",
            start["session_id"],
            end["session_id"],
        )

    for feature in features:
        delta = calculate_delta(_as_float(start.get(feature)), _as_float(end.get(feature)))
        row[f"delta_{feature}"] = delta
        row[f"slope_{feature}"] = calculate_annualised_rate(delta, delta_years)

    return row


# ---------------------------------------------------------------------------
# slopes
# ---------------------------------------------------------------------------
def compute_slopes(
    wide_rows: Sequence[Mapping[str, Any]], config: LongitudinalConfig
) -> list[dict[str, Any]]:
    """Least-squares slope per ``subject x feature`` over all timepoints.

    For a subject with more than two visits this is preferable to
    ``(last - first) / span``: every timepoint contributes, and ``r2`` reports
    how linear the trajectory actually is. With exactly two visits the fit is
    exact (``r2 = 1``) and reduces to the annualised delta, which keeps the
    two-session pilot comparable with later multi-visit runs.

    Output is long: one row per ``(subject_id, feature)``.

    ``n_sessions`` counts the timepoints that contributed a **finite** value for
    that particular feature, so it can be smaller than the subject's session
    count - and 0 for a column that is NaN everywhere, such as the asymmetry of
    a signed feature under ``asymmetry_mode='positive_only'``. ``followup_years``
    always describes the subject's full follow-up span.
    """
    grouped = _group_rows_by_subject(wide_rows)
    features = derived_feature_columns(wide_rows)
    rows: list[dict[str, Any]] = []

    for subject_id, sessions in grouped.items():
        if len(sessions) < config.min_sessions_for_slope:
            logger.info(
                "Subject %s has %d session(s); %d required for a slope.",
                subject_id,
                len(sessions),
                config.min_sessions_for_slope,
            )
            continue

        times = [
            int(session["days_from_reference"]) / config.days_per_year for session in sessions
        ]
        followup_years = max(times) - min(times)

        for feature in features:
            values = [_as_float(session.get(feature)) for session in sessions]
            fit = calculate_slope(times, values)
            rows.append(
                {
                    "subject_id": subject_id,
                    "feature": feature,
                    "feature_slope": fit["slope"],
                    "feature_intercept": fit["intercept"],
                    "feature_r2": fit["r2"],
                    "feature_pvalue": fit["pvalue"],
                    "feature_stderr": fit["stderr"],
                    "n_sessions": int(fit["n_points"]),
                    "followup_years": followup_years,
                    "baseline_value": values[0],
                    "last_value": values[-1],
                }
            )

    logger.info("Slope table: %d row(s) from %d subject(s).", len(rows), len(grouped))
    return rows


def _group_rows_by_subject(
    rows: Iterable[Mapping[str, Any]]
) -> dict[str, list[Mapping[str, Any]]]:
    """Group wide rows per subject, ordered by ``days_from_reference``."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["subject_id"]), []).append(row)
    for sessions in grouped.values():
        sessions.sort(key=lambda row: (int(row["days_from_reference"]), str(row["session_id"])))
    return dict(sorted(grouped.items()))


def _as_float(value: Any) -> float:
    """Best-effort float conversion; non-numeric values become NaN."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return result
