"""Utilities that close the OASIS-3 acquisition stage.

This module deliberately contains no machine learning.  It handles two tasks:

1. build a reproducible download manifest from a catalogue of OASIS-3
   ``freesurfer_id`` values;
2. validate that the extracted long table satisfies the frozen protocol:
   16 ROIs/session x 107 radiomic features/ROI = 1,712 raw features/session.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import (
    ALZHEIMER_ROIS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_RAW_FEATURES_PER_SESSION,
    PROTOCOL_VERSION,
    expected_feature_keys,
)

FREESURFER_ID_RE = re.compile(
    r"^(?P<subject>OAS\d+)_Freesurfer(?P<version>\d+)_d(?P<day>\d+)$"
)


class AcquisitionError(ValueError):
    """Raised when a catalogue or extracted dataset violates the acquisition contract."""


@dataclass(frozen=True)
class CatalogueSession:
    freesurfer_id: str
    subject_id: str
    days_from_reference: int
    freesurfer_version: str


@dataclass(frozen=True)
class ValidationIssue:
    session_id: str
    severity: str
    code: str
    detail: str


@dataclass(frozen=True)
class AcquisitionValidationSummary:
    protocol_version: str
    expected_rois_per_session: int
    expected_features_per_roi: int
    expected_raw_features_per_session: int
    sessions_seen: int
    valid_sessions: int
    invalid_sessions: int
    subjects_seen: int
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


def parse_freesurfer_id(value: str) -> CatalogueSession:
    """Parse ``OAS30001_Freesurfer53_d0129`` into subject/time metadata."""
    match = FREESURFER_ID_RE.fullmatch(value.strip())
    if not match:
        raise AcquisitionError(f"Invalid OASIS FreeSurfer id: {value!r}")
    return CatalogueSession(
        freesurfer_id=value.strip(),
        subject_id=match.group("subject"),
        days_from_reference=int(match.group("day")),
        freesurfer_version=match.group("version"),
    )


def read_catalogue(path: Path) -> list[CatalogueSession]:
    """Read a NITRC/OASIS catalogue containing a ``freesurfer_id`` column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "freesurfer_id" not in reader.fieldnames:
            raise AcquisitionError(
                f"{path} must contain a column named 'freesurfer_id'."
            )
        parsed = [parse_freesurfer_id(row["freesurfer_id"]) for row in reader if row.get("freesurfer_id")]

    # Exact duplicate IDs are removed. Multiple sessions for a subject are retained.
    unique = {session.freesurfer_id: session for session in parsed}
    return sorted(unique.values(), key=lambda item: (item.subject_id, item.days_from_reference, item.freesurfer_id))


def eligible_subjects(
    sessions: Sequence[CatalogueSession], min_sessions: int = 2
) -> dict[str, list[CatalogueSession]]:
    """Group longitudinally eligible subjects and sort their sessions in time."""
    if min_sessions < 1:
        raise AcquisitionError("min_sessions must be >= 1")

    grouped: dict[str, list[CatalogueSession]] = defaultdict(list)
    for session in sessions:
        grouped[session.subject_id].append(session)

    return {
        subject_id: sorted(items, key=lambda item: item.days_from_reference)
        for subject_id, items in sorted(grouped.items())
        if len(items) >= min_sessions
    }


def select_subjects(
    eligible: Mapping[str, Sequence[CatalogueSession]],
    target_subjects: int | None = None,
    oversample: float = 1.20,
    seed: int = 2026,
) -> list[str]:
    """Select subjects reproducibly, optionally with a QC attrition margin.

    ``target_subjects`` is the desired *final* participant count from the study
    design. During acquisition we select ``ceil(target * oversample)`` subjects,
    capped by availability. The default 20% margin is not treated as scientific
    sample-size inflation; it only protects against missing files/QC attrition.
    """
    subject_ids = sorted(eligible)
    if target_subjects is None:
        return subject_ids
    if target_subjects < 1:
        raise AcquisitionError("target_subjects must be >= 1")
    if oversample < 1.0:
        raise AcquisitionError("oversample must be >= 1.0")

    requested = min(len(subject_ids), int(math.ceil(target_subjects * oversample)))
    rng = random.Random(seed)
    selected = rng.sample(subject_ids, requested)
    return sorted(selected)


def write_download_manifest(
    eligible: Mapping[str, Sequence[CatalogueSession]],
    selected_subjects: Sequence[str],
    ids_path: Path,
    subjects_path: Path,
) -> tuple[Path, Path]:
    """Write downloader-compatible IDs plus an auditable subject manifest."""
    ids_path = Path(ids_path)
    subjects_path = Path(subjects_path)
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    subjects_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[CatalogueSession] = []
    for subject_id in selected_subjects:
        rows.extend(eligible[subject_id])
    rows.sort(key=lambda item: (item.subject_id, item.days_from_reference))

    with ids_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["freesurfer_id"])
        for session in rows:
            writer.writerow([session.freesurfer_id])

    with subjects_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "subject_id",
                "n_sessions",
                "first_day",
                "last_day",
                "followup_days",
                "freesurfer_versions",
            ]
        )
        for subject_id in selected_subjects:
            items = list(eligible[subject_id])
            days = [item.days_from_reference for item in items]
            versions = sorted({item.freesurfer_version for item in items})
            writer.writerow(
                [subject_id, len(items), min(days), max(days), max(days) - min(days), ";".join(versions)]
            )

    return ids_path, subjects_path


def read_long_feature_rows(path: Path) -> list[dict[str, str]]:
    """Read ``radiomics_features_long.csv`` as dictionaries."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_long_feature_rows(
    rows: Sequence[Mapping[str, Any]],
) -> AcquisitionValidationSummary:
    """Validate the final raw radiomics table at session/ROI/schema level."""
    if not rows:
        raise AcquisitionError("The long feature table is empty.")

    expected_rois = set(ALZHEIMER_ROIS)
    expected_features = set(expected_feature_keys())
    by_session: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    subjects: set[str] = set()

    for row in rows:
        session_id = str(row.get("session_id", ""))
        if not session_id:
            raise AcquisitionError("A long-table row has no session_id.")
        by_session[session_id].append(row)
        if row.get("subject_id"):
            subjects.add(str(row["subject_id"]))

    issues: list[ValidationIssue] = []
    invalid_sessions: set[str] = set()

    for session_id, session_rows in sorted(by_session.items()):
        roi_names = [str(row.get("roi", "")) for row in session_rows]
        actual_rois = set(roi_names)
        missing_rois = sorted(expected_rois - actual_rois)
        unexpected_rois = sorted(actual_rois - expected_rois)
        duplicates = sorted({roi for roi in roi_names if roi_names.count(roi) > 1})

        if missing_rois:
            invalid_sessions.add(session_id)
            issues.append(ValidationIssue(session_id, "error", "missing_rois", ",".join(missing_rois)))
        if unexpected_rois:
            invalid_sessions.add(session_id)
            issues.append(ValidationIssue(session_id, "error", "unexpected_rois", ",".join(unexpected_rois)))
        if duplicates:
            invalid_sessions.add(session_id)
            issues.append(ValidationIssue(session_id, "error", "duplicate_rois", ",".join(duplicates)))

        for row in session_rows:
            roi = str(row.get("roi", ""))
            radiomic_keys = {key for key in row if key.startswith("original_")}
            missing_features = expected_features - radiomic_keys
            unexpected_features = radiomic_keys - expected_features
            if missing_features or unexpected_features or len(radiomic_keys) != EXPECTED_FEATURE_COUNT:
                invalid_sessions.add(session_id)
                detail = (
                    f"roi={roi}; count={len(radiomic_keys)}; "
                    f"missing={len(missing_features)}; unexpected={len(unexpected_features)}"
                )
                issues.append(ValidationIssue(session_id, "error", "feature_schema", detail))

            nonfinite: list[str] = []
            for key in expected_features & radiomic_keys:
                try:
                    value = float(row[key])
                except (TypeError, ValueError):
                    nonfinite.append(key)
                    continue
                if not math.isfinite(value):
                    nonfinite.append(key)
            if nonfinite:
                invalid_sessions.add(session_id)
                issues.append(
                    ValidationIssue(
                        session_id,
                        "error",
                        "nonfinite_features",
                        f"roi={roi}; n={len(nonfinite)}",
                    )
                )

    sessions_seen = len(by_session)
    return AcquisitionValidationSummary(
        protocol_version=PROTOCOL_VERSION,
        expected_rois_per_session=len(expected_rois),
        expected_features_per_roi=EXPECTED_FEATURE_COUNT,
        expected_raw_features_per_session=EXPECTED_RAW_FEATURES_PER_SESSION,
        sessions_seen=sessions_seen,
        valid_sessions=sessions_seen - len(invalid_sessions),
        invalid_sessions=len(invalid_sessions),
        subjects_seen=len(subjects),
        issues=tuple(issues),
    )


def write_validation_summary(summary: AcquisitionValidationSummary, path: Path) -> Path:
    """Persist a machine-readable acquisition gate report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path
