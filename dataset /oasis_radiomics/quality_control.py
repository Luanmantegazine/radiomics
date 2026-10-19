"""Quality control for OASIS-3 sessions and hippocampal masks.

Two independent layers:

1. **Per-session checks** (:func:`check_session`) - geometry, intensity content
   and mask sanity for a single session, judged against fixed bounds from the
   configuration.
2. **Cohort-relative outlier flagging** (:func:`flag_outliers`) - robust
   statistics (MAD z-score or IQR) over all sessions in the run.

Nothing is ever excluded automatically. QC only annotates; the decision to drop
a session is a scientific one and stays with the investigator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .config import OutlierConfig, QualityControlConfig
from .discovery import DiscoveredSession
from .masks import LoadedVolume, RoiMask

logger = logging.getLogger(__name__)

QC_PASS = "pass"
QC_WARN = "warn"
QC_FAIL = "fail"

#: Relative tolerance when comparing the T1 and aseg affine matrices.
AFFINE_TOLERANCE = 1e-3

#: 0.6745 = Phi^-1(0.75); scales the MAD to a standard-deviation equivalent.
MAD_SCALE = 0.6745


@dataclass
class SessionQC:
    """Quality control outcome for one session."""

    subject_id: str
    session_id: str
    days_from_reference: int
    left_voxels: int = 0
    right_voxels: int = 0
    left_volume_mm3: float = float("nan")
    right_volume_mm3: float = float("nan")
    total_volume_mm3: float = float("nan")
    volume_asymmetry: float = float("nan")
    image_shape: str = ""
    voxel_spacing: str = ""
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """``fail`` if the session is unusable, ``warn`` if suspicious, else ``pass``."""
        if self.errors:
            return QC_FAIL
        if self.warnings:
            return QC_WARN
        return QC_PASS

    @property
    def usable(self) -> bool:
        """Whether radiomics extraction should be attempted for this session."""
        return not self.errors

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for ``quality_control.csv``."""
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "days_from_reference": self.days_from_reference,
            "left_voxels": self.left_voxels,
            "right_voxels": self.right_voxels,
            "left_volume_mm3": self.left_volume_mm3,
            "right_volume_mm3": self.right_volume_mm3,
            "total_volume_mm3": self.total_volume_mm3,
            "volume_asymmetry": self.volume_asymmetry,
            "image_shape": self.image_shape,
            "voxel_spacing": self.voxel_spacing,
            "qc_status": self.status,
            "qc_warning": "; ".join(self.warnings),
            "qc_error": "; ".join(self.errors),
        }


# ---------------------------------------------------------------------------
# per-session checks
# ---------------------------------------------------------------------------
def check_image_geometry(t1: LoadedVolume, aseg: LoadedVolume) -> tuple[list[str], list[str]]:
    """Compare T1 and ``aseg`` geometry and check the image carries signal.

    Returns
    -------
    tuple
        ``(errors, warnings)``. A shape mismatch is an error: the masks would
        not correspond to the image at all. An affine mismatch is a warning,
        because masks are re-saved with the image affine downstream.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if t1.shape[:3] != aseg.shape[:3]:
        errors.append(f"shape_mismatch(T1={t1.shape[:3]}, aseg={aseg.shape[:3]})")

    if not np.allclose(t1.affine, aseg.affine, atol=AFFINE_TOLERANCE):
        warnings.append("affine_mismatch(T1 vs aseg)")

    spacing = np.asarray(t1.spacing, dtype=float)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        errors.append(f"invalid_voxel_spacing({t1.spacing})")

    t1_data = np.asarray(t1.data)
    if t1_data.size == 0 or not np.any(t1_data):
        errors.append("t1_all_zero")

    return errors, warnings


def check_mask(roi: RoiMask, image_shape: Sequence[int], config: QualityControlConfig) -> list[str]:
    """Check one ROI mask against the configured sanity bounds.

    Returns the list of problems found (empty when the mask looks fine). An
    empty mask is reported here as a message; whether that becomes an error is
    decided by :func:`check_session`.
    """
    problems: list[str] = []

    if roi.is_empty:
        return [f"{roi.name}_empty"]

    if roi.data.shape[:3] != tuple(image_shape)[:3]:
        problems.append(f"{roi.name}_shape_mismatch({roi.data.shape[:3]})")

    box = roi.bounding_box
    if box is not None:
        for axis, (low, high) in enumerate(zip(box[0::2], box[1::2])):
            if low < 0 or high >= image_shape[axis]:
                problems.append(f"{roi.name}_out_of_bounds(axis={axis})")

    if roi.n_voxels < config.min_hippocampus_voxels:
        problems.append(f"{roi.name}_few_voxels({roi.n_voxels})")

    if roi.volume_mm3 < config.min_hippocampus_volume_mm3:
        problems.append(f"{roi.name}_volume_below_{config.min_hippocampus_volume_mm3:g}mm3")
    elif roi.volume_mm3 > config.max_hippocampus_volume_mm3:
        problems.append(f"{roi.name}_volume_above_{config.max_hippocampus_volume_mm3:g}mm3")

    return problems


def check_session(
    session: DiscoveredSession,
    t1: LoadedVolume,
    aseg: LoadedVolume,
    masks: Mapping[str, RoiMask],
    config: QualityControlConfig,
    left_roi: str = "left_hippocampus",
    right_roi: str = "right_hippocampus",
) -> SessionQC:
    """Run every per-session check and collect the result.

    A missing or empty left/right hippocampus is an **error** (the session
    cannot contribute a bilateral row); everything else is a warning.
    """
    qc = SessionQC(
        subject_id=session.subject_id,
        session_id=session.session_id,
        days_from_reference=session.days_from_reference,
        image_shape="x".join(str(dim) for dim in t1.shape[:3]),
        voxel_spacing="x".join(f"{value:g}" for value in t1.spacing),
    )

    errors, warnings = check_image_geometry(t1, aseg)
    qc.errors.extend(errors)
    qc.warnings.extend(warnings)

    for roi_name in (left_roi, right_roi):
        roi = masks.get(roi_name)
        if roi is None:
            qc.errors.append(f"{roi_name}_missing")
            continue
        problems = check_mask(roi, t1.shape, config)
        if roi.is_empty:
            qc.errors.extend(problems)
        else:
            qc.warnings.extend(problems)

    left = masks.get(left_roi)
    right = masks.get(right_roi)
    if left is not None:
        qc.left_voxels = left.n_voxels
        qc.left_volume_mm3 = left.volume_mm3
    if right is not None:
        qc.right_voxels = right.n_voxels
        qc.right_volume_mm3 = right.volume_mm3

    if left is not None and right is not None and not (left.is_empty or right.is_empty):
        qc.total_volume_mm3 = left.volume_mm3 + right.volume_mm3
        qc.volume_asymmetry = (left.volume_mm3 - right.volume_mm3) / qc.total_volume_mm3
        if abs(qc.volume_asymmetry) > config.max_absolute_volume_asymmetry:
            qc.warnings.append(f"high_volume_asymmetry({qc.volume_asymmetry:+.3f})")

    _log_session_qc(qc)
    return qc


def _log_session_qc(qc: SessionQC) -> None:
    """Emit the QC outcome at a level matching its severity."""
    if qc.errors:
        logger.error("QC %s: FAIL (%s)", qc.session_id, "; ".join(qc.errors))
    elif qc.warnings:
        logger.warning("QC %s: WARN (%s)", qc.session_id, "; ".join(qc.warnings))
    else:
        logger.info(
            "QC %s: pass (L=%d vox, R=%d vox, asym=%+.3f)",
            qc.session_id,
            qc.left_voxels,
            qc.right_voxels,
            qc.volume_asymmetry,
        )


# ---------------------------------------------------------------------------
# cohort-relative outlier flagging
# ---------------------------------------------------------------------------
def robust_z_scores(values: Sequence[float]) -> np.ndarray:
    """Modified z-score based on the median absolute deviation.

    ``z = 0.6745 * (x - median) / MAD``. Returns an array of NaN when the MAD is
    zero (every finite value identical), because no scale can be estimated.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full(array.shape, np.nan)

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad == 0.0:
        return np.full(array.shape, np.nan)

    return MAD_SCALE * (array - median) / mad


def iqr_bounds(values: Sequence[float], multiplier: float) -> tuple[float, float]:
    """Tukey fences ``(Q1 - k*IQR, Q3 + k*IQR)``; NaNs when undeterminable."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")

    q1, q3 = (float(value) for value in np.percentile(finite, [25, 75]))
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def flag_outliers(rows: Sequence[Mapping[str, Any]], config: OutlierConfig) -> list[dict[str, Any]]:
    """Add ``qc_outlier`` / ``qc_outlier_reason`` to a list of QC rows.

    Cohort statistics are only computed when at least ``config.min_samples``
    sessions are available; below that the reason column records
    ``insufficient_samples(n=...)`` so a small pilot run is never mistaken for a
    clean one. Nothing is removed - the flag is advisory.
    """
    flagged = [dict(row) for row in rows]
    for row in flagged:
        row["qc_outlier"] = False
        row["qc_outlier_reason"] = ""

    if config.method == "none":
        logger.info("Outlier flagging disabled (quality_control.outliers.method='none').")
        for row in flagged:
            row["qc_outlier_reason"] = "disabled"
        return flagged

    if len(flagged) < config.min_samples:
        reason = f"insufficient_samples(n={len(flagged)}, required={config.min_samples})"
        logger.warning(
            "Skipping cohort outlier detection: %d session(s) available, %d required.",
            len(flagged),
            config.min_samples,
        )
        for row in flagged:
            row["qc_outlier_reason"] = reason
        return flagged

    for column in config.columns:
        if column not in flagged[0]:
            logger.warning("Outlier column %r is not present in the QC table; skipping.", column)
            continue
        values = [_as_float(row.get(column)) for row in flagged]
        reasons = _column_outlier_reasons(column, values, config)
        for row, reason in zip(flagged, reasons):
            if reason:
                row["qc_outlier"] = True
                row["qc_outlier_reason"] = _append_reason(row["qc_outlier_reason"], reason)

    n_flagged = sum(1 for row in flagged if row["qc_outlier"])
    logger.info("Outlier flagging (%s): %d/%d session(s) flagged.", config.method, n_flagged, len(flagged))
    return flagged


def _column_outlier_reasons(
    column: str, values: Sequence[float], config: OutlierConfig
) -> list[str]:
    """Per-row outlier reason for one column (empty string when not an outlier)."""
    if config.method == "mad":
        scores = robust_z_scores(values)
        if not np.any(np.isfinite(scores)):
            logger.warning("MAD is zero or undefined for %r; no flags from this column.", column)
            return ["" for _ in values]
        return [
            f"{column}_mad_z={score:+.2f}"
            if np.isfinite(score) and abs(score) > config.threshold
            else ""
            for score in scores
        ]

    low, high = iqr_bounds(values, config.iqr_multiplier)
    if not (np.isfinite(low) and np.isfinite(high)):
        return ["" for _ in values]
    return [
        f"{column}_outside_iqr[{low:.1f},{high:.1f}]"
        if np.isfinite(value) and (value < low or value > high)
        else ""
        for value in values
    ]


def _append_reason(existing: str, reason: str) -> str:
    """Join outlier reasons with a separator, skipping empties."""
    return f"{existing}; {reason}" if existing else reason


def _as_float(value: Any) -> float:
    """Best-effort float conversion; non-numeric values become NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
