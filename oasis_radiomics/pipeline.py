"""Orchestration of the OASIS-3 longitudinal radiomics pipeline.

Stages
------
``run_extraction``
    discover sessions -> QC -> masks -> NIfTI -> PyRadiomics -> long table
``run_longitudinal``
    long table -> wide table -> deltas -> slopes
``run_pipeline``
    both, plus ``quality_control.csv`` and ``run_metadata.json``

Every stage works on data already present on disk. No download is ever
triggered from here; see :mod:`oasis_radiomics.download_oasis`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PipelineConfig
from .discovery import DiscoveredSession, DiscoveryError, discover_sessions
from .longitudinal import (
    ID_COLUMNS,
    build_wide_table,
    compute_deltas,
    compute_slopes,
)
from .masks import LoadedVolume, build_roi_masks, load_volume, write_session_nifti
from .metadata import build_run_metadata, write_run_metadata
from .quality_control import SessionQC, check_session, flag_outliers
from .radiomics_extractor import ExtractionError, build_extractor, extract_roi_features
from .tables import read_csv_rows, write_csv

logger = logging.getLogger(__name__)

LONG_CSV = "radiomics_features_long.csv"
WIDE_CSV = "radiomics_features_wide.csv"
DELTAS_CSV = "radiomics_longitudinal_deltas.csv"
SLOPES_CSV = "radiomics_longitudinal_slopes.csv"
QC_CSV = "quality_control.csv"

PREPARED_DIRNAME = "prepared_nifti"

LONG_LEADING_COLUMNS = (
    "subject_id",
    "session_id",
    "days_from_reference",
    "roi",
    "mask_voxels",
    "mask_volume_mm3",
)
WIDE_LEADING_COLUMNS = ID_COLUMNS
DELTA_LEADING_COLUMNS = (
    "subject_id",
    "comparison",
    "session_id_t0",
    "session_id_t1",
    "days_t0",
    "days_t1",
    "delta_days",
    "delta_years",
)
SLOPE_LEADING_COLUMNS = (
    "subject_id",
    "feature",
    "feature_slope",
    "feature_intercept",
    "feature_r2",
    "n_sessions",
    "followup_years",
)
QC_LEADING_COLUMNS = (
    "subject_id",
    "session_id",
    "days_from_reference",
    "left_voxels",
    "right_voxels",
    "left_volume_mm3",
    "right_volume_mm3",
    "total_volume_mm3",
    "volume_asymmetry",
    "qc_status",
    "qc_outlier",
    "qc_warning",
)


class PipelineError(RuntimeError):
    """Raised when a pipeline stage cannot produce any usable output."""


@dataclass
class ExtractionResult:
    """Outcome of the per-session extraction stage."""

    long_rows: list[dict[str, Any]] = field(default_factory=list)
    qc_rows: list[dict[str, Any]] = field(default_factory=list)
    sessions: list[DiscoveredSession] = field(default_factory=list)
    failed_sessions: list[str] = field(default_factory=list)

    @property
    def n_subjects(self) -> int:
        return len({row["subject_id"] for row in self.long_rows})

    @property
    def n_sessions(self) -> int:
        return len({row["session_id"] for row in self.long_rows})


@dataclass
class LongitudinalResult:
    """Outcome of the longitudinal derivation stage."""

    wide_rows: list[dict[str, Any]] = field(default_factory=list)
    delta_rows: list[dict[str, Any]] = field(default_factory=list)
    slope_rows: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def process_session(
    session: DiscoveredSession,
    config: PipelineConfig,
    prepared_dir: Path,
    extractor,
) -> tuple[list[dict[str, Any]], SessionQC]:
    """Run QC, mask construction and radiomics extraction for one session.

    Returns
    -------
    tuple
        ``(long_rows, session_qc)``. ``long_rows`` is empty when QC failed, so a
        broken session is recorded in ``quality_control.csv`` instead of
        silently disappearing.
    """
    logger.info("Processing %s (subject %s, day %d)", session.session_id, session.subject_id, session.days_from_reference)
    logger.debug("  image: %s", session.t1_path)
    logger.debug("  aseg : %s", session.aseg_path)

    t1 = load_volume(session.t1_path)
    aseg = load_volume(session.aseg_path)
    masks = build_roi_masks(aseg, config.extraction_roi_labels)

    qc = check_session(
        session=session,
        t1=t1,
        aseg=aseg,
        masks=masks,
        config=config.quality_control,
        left_roi=config.bilateral.left_roi,
        right_roi=config.bilateral.right_roi,
    )
    if not qc.usable:
        logger.error("Skipping extraction for %s: QC failed.", session.session_id)
        return [], qc

    image_path, mask_paths = write_session_nifti(session.session_id, t1, masks, prepared_dir)

    rows: list[dict[str, Any]] = []
    for roi_name, mask_path in mask_paths.items():
        roi = masks[roi_name]
        logger.info("  PyRadiomics %s: %d voxels", roi_name, roi.n_voxels)
        features = extract_roi_features(extractor, image_path, mask_path)
        logger.info("  extracted %d radiomic features for %s", len(features), roi_name)

        row: dict[str, Any] = {
            "subject_id": session.subject_id,
            "session_id": session.session_id,
            "days_from_reference": session.days_from_reference,
            "roi": roi_name,
            "mask_voxels": roi.n_voxels,
            "mask_volume_mm3": roi.volume_mm3,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
        row.update(features)
        rows.append(row)

    return rows, qc


def run_extraction(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    max_sessions: int | None = None,
    prepared_dir: Path | None = None,
    write_outputs: bool = True,
) -> ExtractionResult:
    """Extract radiomic features for every session found under ``input_dir``.

    A session that fails is logged and recorded in the QC table; the run
    continues with the remaining sessions.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    prepared_dir = Path(prepared_dir) if prepared_dir else output_dir / PREPARED_DIRNAME

    sessions = discover_sessions(input_dir)
    if not sessions:
        raise DiscoveryError(
            f"No paired {input_dir}/**/T1.mgz + aseg.mgz were found. "
            "Check that the selected OASIS-3 sessions have FreeSurfer outputs."
        )

    if max_sessions is not None:
        if max_sessions < 1:
            raise PipelineError("max_sessions must be >= 1")
        if len(sessions) > max_sessions:
            logger.warning(
                "Limiting the run to the first %d of %d discovered session(s).",
                max_sessions,
                len(sessions),
            )
        sessions = sessions[:max_sessions]

    extractor = build_extractor(config)
    result = ExtractionResult(sessions=list(sessions))

    for session in sessions:
        try:
            rows, qc = process_session(session, config, prepared_dir, extractor)
        except (ExtractionError, OSError, ValueError) as exc:
            logger.error("Session %s failed: %s", session.session_id, exc)
            result.failed_sessions.append(session.session_id)
            continue

        result.long_rows.extend(rows)
        result.qc_rows.append(qc.as_row())

    result.qc_rows = flag_outliers(result.qc_rows, config.quality_control.outliers)

    if not result.long_rows:
        raise PipelineError(
            "No radiomic features were extracted. Inspect the quality control "
            "table for the reason each session was rejected."
        )

    logger.info(
        "Extraction complete: %d row(s) covering %d session(s) from %d subject(s).",
        len(result.long_rows),
        result.n_sessions,
        result.n_subjects,
    )

    if write_outputs:
        write_csv(result.long_rows, output_dir / LONG_CSV, LONG_LEADING_COLUMNS)
        write_csv(result.qc_rows, output_dir / QC_CSV, QC_LEADING_COLUMNS)

    return result


# ---------------------------------------------------------------------------
# longitudinal
# ---------------------------------------------------------------------------
def run_longitudinal(
    long_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: PipelineConfig,
    write_outputs: bool = True,
) -> LongitudinalResult:
    """Derive the wide, delta and slope tables from a long feature table."""
    output_dir = Path(output_dir)

    wide_rows = build_wide_table(long_rows, config.bilateral)
    delta_rows = compute_deltas(wide_rows, config.longitudinal)
    slope_rows = compute_slopes(wide_rows, config.longitudinal)

    if not delta_rows:
        logger.warning(
            "No delta rows produced: every subject has a single session. "
            "Longitudinal analysis needs at least two visits per subject."
        )

    if write_outputs:
        write_csv(wide_rows, output_dir / WIDE_CSV, WIDE_LEADING_COLUMNS)
        write_csv(delta_rows, output_dir / DELTAS_CSV, DELTA_LEADING_COLUMNS)
        write_csv(slope_rows, output_dir / SLOPES_CSV, SLOPE_LEADING_COLUMNS)

    return LongitudinalResult(
        wide_rows=wide_rows, delta_rows=delta_rows, slope_rows=slope_rows
    )


def run_longitudinal_from_csv(
    features_csv: Path, output_dir: Path, config: PipelineConfig
) -> LongitudinalResult:
    """Run the longitudinal stage on a previously written long feature table.

    Long tables produced by the legacy script carry no ``subject_id`` /
    ``days_from_reference`` columns; both are re-derived from ``session_id``.
    """
    rows = read_csv_rows(Path(features_csv))
    if not rows:
        raise PipelineError(f"Feature table is empty: {features_csv}")
    return run_longitudinal(_ensure_identifier_columns(rows), output_dir, config)


def _ensure_identifier_columns(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Re-derive ``subject_id`` and ``days_from_reference`` from ``session_id``."""
    from .ids import parse_session_id

    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if "session_id" not in item:
            raise PipelineError("The feature table has no 'session_id' column.")
        if "subject_id" not in item or "days_from_reference" not in item:
            key = parse_session_id(str(item["session_id"]))
            item.setdefault("subject_id", key.subject_id)
            item.setdefault("days_from_reference", key.days_from_reference)
        enriched.append(item)
    return enriched


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    config: PipelineConfig,
    max_sessions: int | None = None,
) -> dict[str, Path]:
    """Run extraction, longitudinal derivation and metadata writing.

    Returns
    -------
    dict
        Mapping of output name to the path that was written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extraction = run_extraction(
        input_dir=input_dir, output_dir=output_dir, config=config, max_sessions=max_sessions
    )
    longitudinal = run_longitudinal(extraction.long_rows, output_dir, config)

    outputs = {
        "features_long": output_dir / LONG_CSV,
        "features_wide": output_dir / WIDE_CSV,
        "deltas": output_dir / DELTAS_CSV,
        "slopes": output_dir / SLOPES_CSV,
        "quality_control": output_dir / QC_CSV,
    }

    counts = {
        "subjects": extraction.n_subjects,
        "sessions": extraction.n_sessions,
        "roi_rows": len(extraction.long_rows),
        "wide_rows": len(longitudinal.wide_rows),
        "delta_rows": len(longitudinal.delta_rows),
        "slope_rows": len(longitudinal.slope_rows),
        "features_per_roi": _features_per_roi(extraction.long_rows),
        "wide_columns": len(longitudinal.wide_rows[0]) if longitudinal.wide_rows else 0,
        "failed_sessions": extraction.failed_sessions,
    }

    metadata = build_run_metadata(config, input_dir, output_dir, counts, outputs)
    outputs["run_metadata"] = write_run_metadata(metadata, output_dir)

    _log_summary(counts, outputs)
    return outputs


def _features_per_roi(long_rows: Sequence[Mapping[str, Any]]) -> int:
    """Number of radiomic features in a long row (identifiers excluded)."""
    if not long_rows:
        return 0
    skip = set(LONG_LEADING_COLUMNS) | {"image_path", "mask_path"}
    return len([column for column in long_rows[0] if column not in skip])


def _log_summary(counts: Mapping[str, Any], outputs: Mapping[str, Path]) -> None:
    """Log the end-of-run summary."""
    logger.info("---- run summary ----")
    for key, value in counts.items():
        logger.info("  %-18s %s", key, value)
    for name, path in outputs.items():
        logger.info("  %-18s -> %s", name, path)
    logger.info(
        "Reminder: bin width, normalisation, resampling and ROI definitions are "
        "still provisional and must be fixed before the real study."
    )
