"""Local extraction pipeline: manifest -> validation -> features -> local files.

Outputs (all **local to the site**, they contain individual-level data):

``features.csv``
    One row per participant with every ROI extracted, columns
    ``<roi>__<pyradiomics feature name>``. Complete cases only.
``cases.csv``
    One row per (participant, ROI): status, issues, ROI size, geometry, hashes.
``provenance.json``
    Deterministic record: configuration, library versions, manifest hash,
    feature schema, counts and the SHA-256 of ``features.csv``. Two runs on the
    same inputs, configuration and environment produce identical files.
``run_info.json``
    Non-deterministic record: timestamps, durations, memory, platform.

Policy: a case with a validation error or an extraction failure is skipped and
reported; a participant missing any ROI is left out of ``features.csv`` (no
imputation here).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from client.imaging.extraction import (
    ExtractionConfig,
    ExtractionError,
    PyRadiomicsExtractor,
)
from client.imaging.manifest import Manifest, load_manifest
from client.imaging.validation import validate_pair
from shared.costs import peak_memory_bytes
from shared.logging import get_logger

FEATURES_FILE = "features.csv"
CASES_FILE = "cases.csv"
PROVENANCE_FILE = "provenance.json"
RUN_INFO_FILE = "run_info.json"
SEPARATOR = "__"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_dir: Path
    n_cases: int
    n_extracted: int
    n_participants: int
    n_complete_participants: int
    feature_names: tuple[str, ...]
    features_sha256: str
    duration_s: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_extraction(
    manifest: Manifest | str | Path,
    config: ExtractionConfig,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> PipelineResult:
    """Validate and extract every case of a manifest, writing local results."""
    log = get_logger("client.imaging")
    started = time.perf_counter()
    started_at = datetime.now(UTC)
    if not isinstance(manifest, Manifest):
        manifest = load_manifest(manifest)
    output_dir = Path(output_dir)
    if (output_dir / PROVENANCE_FILE).exists() and not overwrite:
        raise FileExistsError(f"{output_dir} already holds results; pass overwrite=True")
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = PyRadiomicsExtractor(config)
    schema: tuple[str, ...] | None = None
    case_rows: list[dict] = []
    extracted: dict[str, dict[str, dict[str, float]]] = {}
    validation_s = extraction_s = 0.0

    for case in manifest.cases:
        tick = time.perf_counter()
        report = validate_pair(
            case.image_path, case.mask_path, case.mask_label, config.validation
        )
        validation_s += time.perf_counter() - tick
        geometry = report.image_geometry
        row = {
            "participant_id": case.participant_id,
            "roi": case.roi,
            "mask_label": case.mask_label,
            "status": "extracted",
            "errors": ";".join(i.code.value for i in report.errors),
            "warnings": ";".join(i.code.value for i in report.warnings),
            "roi_voxels": report.roi_voxels,
            "roi_components": report.roi_components,
            "spacing": "x".join(f"{s:g}" for s in geometry.spacing) if geometry else "",
            "size": "x".join(map(str, geometry.size)) if geometry else "",
            "orientation": geometry.orientation if geometry else "",
            "image_hash": "",
            "mask_hash": "",
        }
        case_rows.append(row)
        if not report.ok:
            row["status"] = "invalid"
            continue
        tick = time.perf_counter()
        try:
            result = extractor.extract(case.image_path, case.mask_path, case.mask_label)
            names = tuple(result.features)
            if schema is None:
                schema = names
            elif names != schema:
                raise ExtractionError("feature_schema_mismatch")
        except ExtractionError as exc:
            row["status"], row["errors"] = "failed", exc.code
            log.warning("extraction_failed", code=exc.code)  # no participant id in logs
            continue
        finally:
            extraction_s += time.perf_counter() - tick
        row["image_hash"] = result.diagnostics.get("Image-original_Hash", "")
        row["mask_hash"] = result.diagnostics.get("Mask-original_Hash", "")
        extracted.setdefault(case.participant_id, {})[case.roi] = result.features

    rois = manifest.rois
    columns = [f"{roi}{SEPARATOR}{name}" for roi in rois for name in (schema or ())]
    complete = sorted(p for p, by_roi in extracted.items() if set(by_roi) == set(rois))
    table = pd.DataFrame(
        [[extracted[p][roi][name] for roi in rois for name in (schema or ())] for p in complete],
        columns=columns,
        dtype=float,
    )
    table.insert(0, "participant_id", complete)
    features_path = output_dir / FEATURES_FILE
    table.to_csv(features_path, index=False, lineterminator="\n")
    pd.DataFrame(case_rows).to_csv(output_dir / CASES_FILE, index=False, lineterminator="\n")

    n_extracted = sum(row["status"] == "extracted" for row in case_rows)
    followup = manifest.followup_summary()
    result = PipelineResult(
        output_dir=output_dir,
        n_cases=len(case_rows),
        n_extracted=n_extracted,
        n_participants=len(manifest.participants),
        n_complete_participants=len(complete),
        feature_names=tuple(columns),
        features_sha256=sha256_file(features_path),
        duration_s=time.perf_counter() - started,
    )
    _write_json(
        output_dir / PROVENANCE_FILE,
        {
            **extractor.provenance(),
            "manifest_sha256": manifest.sha256,
            "rois": list(rois),
            "feature_names": columns,
            "features_sha256": result.features_sha256,
            "counts": {
                "cases": result.n_cases,
                "cases_extracted": n_extracted,
                "cases_invalid": sum(r["status"] == "invalid" for r in case_rows),
                "cases_failed": sum(r["status"] == "failed" for r in case_rows),
                "participants": result.n_participants,
                "participants_complete": result.n_complete_participants,
                "participants_with_followup": followup.n_with_followup,
                "participants_without_followup": followup.n_without_followup,
            },
        },
    )
    _write_json(
        output_dir / RUN_INFO_FILE,
        {
            "started_at": started_at.isoformat(),
            "duration_s": result.duration_s,
            "validation_time_s": validation_s,
            "extraction_time_s": extraction_s,
            "peak_memory_bytes": peak_memory_bytes(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    )
    log.info(
        "extraction_finished",
        config_id=config.config_id,
        cases=result.n_cases,
        extracted=n_extracted,
        participants_complete=result.n_complete_participants,
        duration_s=round(result.duration_s, 3),
    )
    return result
