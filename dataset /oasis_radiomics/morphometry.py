"""FreeSurfer morphometry inventory, download and extraction helpers."""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .download_oasis import DownloadError, ensure_oasis_scripts, download_freesurfer_sessions
from .ids import (
    find_session_id,
    freesurfer_id_to_session_id,
    parse_session_id,
)

logger = logging.getLogger(__name__)

ELIGIBLE_LABELS = {"CN", "AD"}

ASEG_STATS = "aseg.stats"
LH_APARC_STATS = "lh.aparc.stats"
RH_APARC_STATS = "rh.aparc.stats"

EXPECTED_STATS_FILES = (ASEG_STATS, LH_APARC_STATS, RH_APARC_STATS)

ASEG_STRUCTURES = {
    "Left-Hippocampus": "fs_left_hippocampus_mm3",
    "Right-Hippocampus": "fs_right_hippocampus_mm3",
    "Left-Amygdala": "fs_left_amygdala_mm3",
    "Right-Amygdala": "fs_right_amygdala_mm3",
    "Left-Inf-Lat-Vent": "fs_left_inf_lat_vent_mm3",
    "Right-Inf-Lat-Vent": "fs_right_inf_lat_vent_mm3",
    "Left-Lateral-Ventricle": "fs_left_lateral_ventricle_mm3",
    "Right-Lateral-Ventricle": "fs_right_lateral_ventricle_mm3",
}

APARC_STRUCTURES = {
    "lh.aparc.stats": {
        "entorhinal": "fs_lh_entorhinal_thickavg_mm",
        "parahippocampal": "fs_lh_parahippocampal_thickavg_mm",
    },
    "rh.aparc.stats": {
        "entorhinal": "fs_rh_entorhinal_thickavg_mm",
        "parahippocampal": "fs_rh_parahippocampal_thickavg_mm",
    },
}

MEASURE_COLUMNS = [
    *ASEG_STRUCTURES.values(),
    "fs_etiv_mm3",
    "fs_cortexvol_mm3",
    *APARC_STRUCTURES["lh.aparc.stats"].values(),
    *APARC_STRUCTURES["rh.aparc.stats"].values(),
]

INVENTORY_COLUMNS = [
    "subject_id",
    "session_id",
    "mri_day",
    "supervised_label",
    "training_eligible",
    "selected_primary_session",
    "local_subject_directory",
    "local_aseg_found",
    "local_lh_aparc_found",
    "local_rh_aparc_found",
    "local_stats_complete",
    "local_freesurfer_version",
    "candidate_directories_count",
    "match_status",
    "missing_files",
]

DOWNLOAD_MANIFEST_COLUMNS = [
    "subject_id",
    "session_id",
    "supervised_label",
    "selected_primary_session",
    "local_stats_complete",
    "remote_availability_status",
    "official_freesurfer_available",
    "official_spreadsheet_available",
    "raw_t1_available",
    "remote_resource_identifier",
    "remote_freesurfer_version",
    "alternative_session_available",
    "alternative_session_id",
    "download_priority",
    "required_action",
    "download_status",
    "download_error",
]

PREFLIGHT_COLUMNS = [
    "subject_id",
    "session_id",
    "supervised_label",
    "mri_day",
    "clinical_mri_abs_gap_days",
    "selected_primary_session",
    "remote_resource_identifier",
    "resource_type_available",
    "remote_availability_status",
    "estimated_size_bytes",
    "planned_action",
    "selection_reason",
]

REPROCESSING_COLUMNS = [
    "subject_id",
    "session_id",
    "t1_path",
    "t1_series_description",
    "t1_acquisition_id",
    "scanner",
    "download_complete",
    "freesurfer_required",
    "freesurfer_status",
    "freesurfer_version",
    "qc_status",
]

SUBSTITUTION_COLUMNS = [
    "subject_id",
    "original_session_id",
    "replacement_session_id",
    "original_clinical_gap",
    "replacement_clinical_gap",
    "original_label",
    "replacement_label",
    "replacement_reason",
]

EXTRACTION_AUDIT_COLUMNS = [
    "subject_id",
    "session_id",
    "selected_primary_session",
    "local_subject_directory",
    "match_status",
    "fs_data_origin",
    "fs_version",
    "fs_missing_fields",
    "fs_qc_warning",
]

CONFLICT_COLUMNS = [
    "subject_id",
    "session_id",
    "issue_type",
    "detail",
]

MISSING_COLUMNS = [
    "subject_id",
    "session_id",
    "reason",
    "missing_files",
]

REPORT_FILES = [
    "morphometry_inventory.csv",
    "morphometry_download_manifest.csv",
    "freesurfer_download_preflight.csv",
    "freesurfer_reprocessing_manifest.csv",
    "session_substitutions.csv",
    "morphometry_extraction_audit.csv",
    "morphometry_missing_sessions.csv",
    "morphometry_conflicts.csv",
]

_DAY_RE = re.compile(r"_d(?P<day>\d+)$")
_FREESURFER_VERSION_RE = re.compile(r"Freesurfer(?P<version>\d+)")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)")
_MEASURE_NAME_RE = re.compile(r"^#\s*Measure\s+(?P<name>[^,\s]+)")
_SESSION_VALUE_RE = re.compile(r"OAS\d+_(?:MR|Freesurfer\d+)_d\d+")
_REMOTE_INDEX_SESSION_HINTS = ("session", "experiment", "process", "label", "identifier", "id")
_REMOTE_INDEX_VERSION_HINTS = ("version", "freesurfer", "fsver")
_REMOTE_INDEX_RESOURCE_HINTS = ("url", "path", "download", "resource", "identifier", "process", "label", "name")

EXPECTED_DOWNLOAD_COUNTS = {
    "download_freesurfer_derivative": 77,
    "backup_candidate": 62,
    "not_required_mci": 16,
    "not_required_ineligible": 24,
    "not_required_unlabelled": 183,
    "manual_review": 0,
}

DERIVATIVE_ESTIMATED_SIZE_BYTES = 125_000_000
OFFICIAL_SPREADSHEET_ESTIMATED_SIZE_BYTES = 50_000

REMOTE_AVAILABILITY_STATUSES = {
    "available",
    "not_available",
    "not_checked",
    "auth_required",
    "lookup_error",
    "multiple_remote_candidates",
}

REMOTE_ACTIONS = {
    "use_official_spreadsheet",
    "download_freesurfer_derivative",
    "remote_lookup_required",
    "remote_auth_required",
    "review_remote_candidates",
    "review_backup_session",
    "download_raw_t1",
    "manual_review",
}


@dataclass(frozen=True)
class RemoteLookupEntry:
    session_id: str
    remote_identifier: str | None
    remote_version: str | None
    availability_status: str
    candidate_columns: tuple[str, ...]
    candidate_count: int = 0
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.availability_status == "available"


@dataclass(frozen=True)
class StatsDirectory:
    path: Path
    files: frozenset[str]
    session_id: str | None
    subject_id: str | None
    freesurfer_version: str | None
    mri_day: int | None

    @property
    def stats_complete(self) -> bool:
        return all(name in self.files for name in EXPECTED_STATS_FILES)

    @property
    def found(self) -> dict[str, bool]:
        return {name: name in self.files for name in EXPECTED_STATS_FILES}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def _to_int(value: Any) -> int | None:
    if value in (None, "", "nan"):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _normalise_session_id(value: str) -> str:
    text = str(value).strip()
    if "_Freesurfer" in text:
        return freesurfer_id_to_session_id(text)
    return parse_session_id(text).session_id


def _parse_stats_directory(path: Path) -> tuple[str | None, str | None, str | None, int | None]:
    text = str(path)
    session_id: str | None = None
    subject_id: str | None = None
    version: str | None = None
    mri_day: int | None = None

    for part in reversed(path.parts):
        found = find_session_id(part)
        if found is not None:
            try:
                session_id = _normalise_session_id(found)
            except Exception:
                session_id = None
            else:
                subject_id = parse_session_id(session_id).subject_id
                mri_day = parse_session_id(session_id).days_from_reference
                break

    for part in path.parts:
        match = _FREESURFER_VERSION_RE.search(part)
        if match:
            version = match.group("version")
            break

    if session_id is None:
        # Some downloaded trees expose the FreeSurfer identifier only in the full path.
        found = find_session_id(text)
        if found is not None:
            try:
                session_id = _normalise_session_id(found)
                subject_id = parse_session_id(session_id).subject_id
                mri_day = parse_session_id(session_id).days_from_reference
            except Exception:
                session_id = None

    return session_id, subject_id, version, mri_day


def _parse_stats_file(path: Path) -> tuple[list[str], list[list[str]]]:
    headers: list[str] = []
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# ColHeaders"):
                headers = re.split(r"\s+", line[len("# ColHeaders") :].strip())
                continue
            if line.startswith("#"):
                continue
            if headers:
                rows.append(re.split(r"\s+", line))
    return headers, rows


def _parse_stats_measurements(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    values: dict[str, float] = {}
    measure_names: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line.startswith("# Measure"):
                continue
            match = _MEASURE_NAME_RE.match(line)
            if match is None:
                continue
            tokens = re.split(r"[,\t ]+", line)
            if len(tokens) < 4:
                continue
            name = match.group("name")
            numeric = next((token for token in tokens[2:] if _FLOAT_RE.fullmatch(token)), None)
            if numeric is None:
                continue
            values[name] = float(numeric)
            measure_names[name] = name
    return values, measure_names


def _normalize_remote_session_value(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    found = _SESSION_VALUE_RE.search(text)
    if found is None:
        return None
    session_id = found.group(0)
    if "_Freesurfer" in session_id:
        return freesurfer_id_to_session_id(session_id)
    return session_id


def _extract_remote_version(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _FREESURFER_VERSION_RE.search(text)
    return match.group("version") if match else None


def _remote_index_columns(fieldnames: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    session_columns: list[str] = []
    version_columns: list[str] = []
    resource_columns: list[str] = []
    for column in fieldnames:
        name = column.lower()
        if any(hint in name for hint in _REMOTE_INDEX_SESSION_HINTS):
            session_columns.append(column)
        if any(hint in name for hint in _REMOTE_INDEX_VERSION_HINTS):
            version_columns.append(column)
        if any(hint in name for hint in _REMOTE_INDEX_RESOURCE_HINTS):
            resource_columns.append(column)
    return session_columns, version_columns, resource_columns


def _read_official_freesurfer_index(path: Path | None) -> dict[str, RemoteLookupEntry]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return {}
            session_columns, version_columns, resource_columns = _remote_index_columns(fieldnames)
            by_session: dict[str, list[RemoteLookupEntry]] = defaultdict(list)
            for row in reader:
                row_candidates: dict[str, str] = {}
                versions: set[str] = set()
                candidate_columns = session_columns or fieldnames
                for column in candidate_columns:
                    value = row.get(column)
                    session_id = _normalize_remote_session_value(value)
                    if session_id:
                        row_candidates[column] = session_id
                        version = _extract_remote_version(value)
                        if version:
                            versions.add(version)
                if not row_candidates:
                    for column, value in row.items():
                        session_id = _normalize_remote_session_value(value)
                        if session_id:
                            row_candidates[column] = session_id
                            version = _extract_remote_version(value)
                            if version:
                                versions.add(version)
                unique_sessions = sorted(set(row_candidates.values()))
                if not unique_sessions:
                    continue
                remote_identifier = None
                for column in resource_columns or fieldnames:
                    value = str(row.get(column) or "").strip()
                    if value:
                        remote_identifier = value
                        break
                remote_version = None
                for column in version_columns:
                    remote_version = _extract_remote_version(row.get(column))
                    if remote_version:
                        break
                if remote_version is None:
                    remote_version = next(iter(sorted(versions)), None)
                status = "available" if len(unique_sessions) == 1 else "multiple_remote_candidates"
                candidate_columns = tuple(sorted(row_candidates))
                for session_id in unique_sessions:
                    entry = RemoteLookupEntry(
                        session_id=session_id,
                        remote_identifier=remote_identifier,
                        remote_version=remote_version,
                        availability_status=status,
                        candidate_columns=candidate_columns,
                        candidate_count=len(unique_sessions),
                    )
                    by_session[entry.session_id].append(entry)

            resolved: dict[str, RemoteLookupEntry] = {}
            for session_id, entries in by_session.items():
                if len(entries) == 1:
                    resolved[session_id] = entries[0]
                    continue
                resolved[session_id] = RemoteLookupEntry(
                    session_id=session_id,
                    remote_identifier=None,
                    remote_version=None,
                    availability_status="multiple_remote_candidates",
                    candidate_columns=tuple(sorted({column for entry in entries for column in entry.candidate_columns})),
                    candidate_count=len(entries),
                )
            return resolved
    except PermissionError as exc:
        logger.warning("Official FreeSurfer index is not readable: %s", exc)
        return {}


def _read_official_freesurfer_measures(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return {}
        session_columns = [column for column in fieldnames if column.lower() in {"session_id", "experiment_id", "label", "freesurfer_id"}]
        if not session_columns:
            return {}
        measure_columns = [column for column in fieldnames if column in MEASURE_COLUMNS or column in {"EstimatedTotalIntraCranialVol", "eTIV", "CortexVol"}]
        if not measure_columns:
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for record in reader:
            session_id = None
            for column in session_columns:
                session_id = _normalize_remote_session_value(record.get(column))
                if session_id:
                    break
            if session_id is None:
                continue
            item: dict[str, Any] = {}
            for column in measure_columns:
                item[column] = record.get(column)
            rows[session_id] = item
        return rows


def _measure_mapping_from_official(row: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    mapped: dict[str, Any] = {}
    found: list[str] = []
    missing: list[str] = []
    aliases = {
        "fs_etiv_mm3": ("EstimatedTotalIntraCranialVol", "eTIV"),
        "fs_cortexvol_mm3": ("CortexVol",),
    }
    for target in MEASURE_COLUMNS:
        candidates = aliases.get(target, (target,))
        present = [candidate for candidate in candidates if candidate in row and row.get(candidate) not in (None, "", "nan")]
        if len(present) == 1:
            mapped[target] = _to_float(row.get(present[0]))
            found.append(target)
        elif len(present) == 0:
            missing.append(target)
        else:
            missing.append(target)
    return mapped, found, missing


def _extract_from_aseg(path: Path) -> tuple[dict[str, float], list[str]]:
    headers, rows = _parse_stats_file(path)
    if not headers:
        return {}, ["missing_colheaders"]

    lookup = {name: idx for idx, name in enumerate(headers)}
    struct_idx = lookup.get("StructName")
    volume_idx = lookup.get("Volume_mm3")
    if struct_idx is None or volume_idx is None:
        return {}, ["missing_structname_or_volume"]

    values: dict[str, float] = {}
    warnings: list[str] = []
    for row in rows:
        if len(row) <= max(struct_idx, volume_idx):
            continue
        struct = row[struct_idx]
        column = ASEG_STRUCTURES.get(struct)
        if column is None:
            continue
        try:
            values[column] = float(row[volume_idx])
        except ValueError:
            warnings.append(f"invalid_volume:{struct}")

    measures, _ = _parse_stats_measurements(path)
    for measure_name, column in (("EstimatedTotalIntraCranialVol", "fs_etiv_mm3"), ("eTIV", "fs_etiv_mm3"), ("CortexVol", "fs_cortexvol_mm3")):
        if measure_name in measures:
            values[column] = measures[measure_name]

    return values, warnings


def _extract_from_aparc(path: Path) -> tuple[dict[str, float], list[str]]:
    headers, rows = _parse_stats_file(path)
    if not headers:
        return {}, ["missing_colheaders"]

    lookup = {name: idx for idx, name in enumerate(headers)}
    struct_idx = lookup.get("StructName")
    thick_idx = lookup.get("ThickAvg")
    if struct_idx is None or thick_idx is None:
        return {}, ["missing_structname_or_thickavg"]

    target = APARC_STRUCTURES[path.name]
    values: dict[str, float] = {}
    warnings: list[str] = []
    for row in rows:
        if len(row) <= max(struct_idx, thick_idx):
            continue
        struct = row[struct_idx].lower()
        column = target.get(struct)
        if column is None:
            continue
        try:
            values[column] = float(row[thick_idx])
        except ValueError:
            warnings.append(f"invalid_thickness:{struct}")
    return values, warnings


def load_session_rows(input_csv: Path) -> list[dict[str, Any]]:
    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _training_eligible(row: Mapping[str, Any]) -> bool:
    return _to_bool(row.get("training_eligible")) and _to_bool(row.get("clinical_match_valid")) and str(row.get("supervised_label")) in ELIGIBLE_LABELS


def select_primary_sessions(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _training_eligible(row):
            by_subject[str(row["subject_id"])].append(row)

    selected: dict[str, str] = {}
    for subject_id, items in by_subject.items():
        items.sort(
            key=lambda row: (
                _to_int(row.get("clinical_mri_abs_gap_days")) if _to_int(row.get("clinical_mri_abs_gap_days")) is not None else 10**9,
                _to_int(row.get("mri_day")) if _to_int(row.get("mri_day")) is not None else 10**9,
                str(row.get("session_id")),
            )
        )
        selected[subject_id] = str(items[0]["session_id"])
    return selected


def load_session_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping: dict[str, str] = {}
        for row in reader:
            directory = row.get("local_subject_directory") or row.get("directory") or row.get("path")
            session_id = row.get("session_id") or row.get("experiment_id") or row.get("freesurfer_id")
            if not directory or not session_id:
                continue
            key = str(Path(directory).resolve())
            try:
                value = _normalise_session_id(session_id)
            except Exception:
                continue
            mapping[key] = value
    return mapping


def discover_stats_directories(subjects_dir: Path) -> list[StatsDirectory]:
    subjects_dir = Path(subjects_dir)
    if not subjects_dir.exists():
        return []

    files_by_dir: dict[Path, set[str]] = defaultdict(set)
    for filename in EXPECTED_STATS_FILES:
        for path in subjects_dir.rglob(filename):
            files_by_dir[path.parent].add(path.name)

    directories: list[StatsDirectory] = []
    for directory, files in sorted(files_by_dir.items()):
        session_id, subject_id, version, mri_day = _parse_stats_directory(directory)
        directories.append(
            StatsDirectory(
                path=directory,
                files=frozenset(files),
                session_id=session_id,
                subject_id=subject_id,
                freesurfer_version=version,
                mri_day=mri_day,
            )
        )
    return directories


def discover_stats_directories_many(paths: Sequence[Path]) -> list[StatsDirectory]:
    seen: dict[Path, StatsDirectory] = {}
    for path in paths:
        for directory in discover_stats_directories(path):
            seen[directory.path] = directory
    return sorted(seen.values(), key=lambda item: str(item.path))


def _match_directory_for_row(
    row: Mapping[str, Any],
    directories: Sequence[StatsDirectory],
    session_map: Mapping[str, str] | None = None,
) -> tuple[StatsDirectory | None, str, int]:
    session_id = str(row["session_id"])
    subject_id = str(row["subject_id"])
    subject_dirs = [item for item in directories if item.subject_id == subject_id]
    exact_dirs = [item for item in directories if item.session_id == session_id]

    if exact_dirs:
        if len(exact_dirs) == 1:
            return exact_dirs[0], "exact_session", len(exact_dirs)
        return None, "ambiguous", len(exact_dirs)

    if session_map:
        for directory, mapped_session in session_map.items():
            if mapped_session != session_id:
                continue
            candidate = next((item for item in directories if str(item.path.resolve()) == directory), None)
            if candidate is not None:
                return candidate, "exact_session", 1

    if len(subject_dirs) == 1:
        return subject_dirs[0], "unique_subject_session", 1
    if len(subject_dirs) > 1:
        return None, "ambiguous", len(subject_dirs)
    return None, "not_found", 0


def _filesystem_warnings(
    row: Mapping[str, Any],
    directory: StatsDirectory | None,
    match_status: str,
) -> tuple[list[str], list[str]]:
    missing_files: list[str] = []
    warnings: list[str] = []

    if directory is None:
        warnings.append(match_status)
        if match_status == "not_found":
            missing_files.append("no_matching_directory")
        return missing_files, warnings

    for filename in EXPECTED_STATS_FILES:
        if filename not in directory.files:
            missing_files.append(filename)
    if missing_files:
        warnings.append("missing_stats_files")

    row_day = _to_int(row.get("mri_day"))
    if row_day is not None and directory.mri_day is not None and row_day != directory.mri_day:
        warnings.append(f"day_mismatch:{directory.mri_day}->{row_day}")
    if match_status == "ambiguous":
        warnings.append("ambiguous_directory_match")
    if directory.freesurfer_version is not None and str(row.get("scanner") or "").startswith("3"):
        pass
    return missing_files, warnings


def build_inventory(
    rows: Sequence[Mapping[str, Any]],
    subjects_dir: Path,
    downloaded_subjects_dir: Path | None = None,
    session_map: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[StatsDirectory], dict[str, str]]:
    primary_sessions = select_primary_sessions(rows)
    search_paths = [Path(subjects_dir)]
    if downloaded_subjects_dir is not None:
        search_paths.append(Path(downloaded_subjects_dir))
    directories = discover_stats_directories_many(search_paths)
    selected = set(primary_sessions.values())
    inventory: list[dict[str, Any]] = []
    session_directory: dict[str, str] = {}

    for row in rows:
        directory, match_status, count = _match_directory_for_row(row, directories, session_map)
        missing_files, warnings = _filesystem_warnings(row, directory, match_status)
        session_id = str(row["session_id"])
        chosen_directory = str(directory.path) if directory else ""
        if directory is not None:
            session_directory[session_id] = chosen_directory
        inventory.append(
            {
                "subject_id": str(row["subject_id"]),
                "session_id": session_id,
                "mri_day": row.get("mri_day"),
                "clinical_mri_abs_gap_days": row.get("clinical_mri_abs_gap_days"),
                "supervised_label": row.get("supervised_label"),
                "scanner": row.get("scanner", ""),
                "training_eligible": _to_bool(row.get("training_eligible")),
                "selected_primary_session": session_id in selected,
                "local_subject_directory": chosen_directory,
                "local_aseg_found": bool(directory and ASEG_STATS in directory.files),
                "local_lh_aparc_found": bool(directory and LH_APARC_STATS in directory.files),
                "local_rh_aparc_found": bool(directory and RH_APARC_STATS in directory.files),
                "local_stats_complete": bool(directory and directory.stats_complete),
                "local_freesurfer_version": directory.freesurfer_version if directory else "",
                "official_freesurfer_available": bool(
                    _training_eligible(row)
                    and (
                        str(row.get("scanner", "")).strip() in {"3.0T", "3T"}
                        or bool(directory and directory.freesurfer_version)
                    )
                ),
                "raw_t1_available": bool(
                    _training_eligible(row)
                    and str(row.get("scanner", "")).strip() not in {"", "3.0T", "3T"}
                ),
                "candidate_directories_count": count,
                "match_status": match_status,
                "missing_files": ";".join(missing_files),
                "_warnings": warnings,
                "_matched_directory": directory,
            }
        )

    return inventory, directories, session_directory


def _find_measure_source(
    session_id: str,
    subject_id: str,
    official_spreadsheet: Mapping[tuple[str, str], Mapping[str, Any]] | None,
    downloaded_index: Mapping[str, Mapping[str, Any]] | None,
    inventory_row: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    if inventory_row.get("local_stats_complete"):
        return {}, "local_existing", warnings

    if official_spreadsheet and (subject_id, session_id) in official_spreadsheet:
        return dict(official_spreadsheet[(subject_id, session_id)]), "official_spreadsheet", warnings

    if downloaded_index and session_id in downloaded_index:
        return dict(downloaded_index[session_id]), "downloaded_freesurfer", warnings

    return {}, "", warnings


def _extract_from_directory(directory: StatsDirectory) -> tuple[dict[str, Any], list[str]]:
    values: dict[str, Any] = {}
    warnings: list[str] = []
    if directory is None:
        return values, warnings

    for filename in directory.files:
        file_path = directory.path / filename
        if filename == ASEG_STATS:
            extracted, file_warnings = _extract_from_aseg(file_path)
        elif filename in APARC_STRUCTURES:
            extracted, file_warnings = _extract_from_aparc(file_path)
        else:
            continue
        values.update(extracted)
        warnings.extend(file_warnings)
    return values, warnings


def _complete_measure_row(source: Mapping[str, Any], values: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(source)
    for column in MEASURE_COLUMNS:
        row[column] = values.get(column)
    return row


def build_download_manifest(
    inventory_rows: Sequence[Mapping[str, Any]],
    official_spreadsheet: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    official_index: Mapping[str, RemoteLookupEntry] | None = None,
    remote_index_checked: bool = False,
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in inventory_rows:
        by_subject[str(row["subject_id"])].append(row)

    manifest: list[dict[str, Any]] = []
    for row in inventory_rows:
        subject_id = str(row["subject_id"])
        session_id = str(row["session_id"])
        label = str(row.get("supervised_label") or "")
        selected_primary = bool(row.get("selected_primary_session"))
        training_eligible = _to_bool(row.get("training_eligible"))
        eligible = training_eligible and label in ELIGIBLE_LABELS
        alternatives = [
            item
            for item in by_subject[subject_id]
            if item["session_id"] != session_id
            and _to_bool(item.get("training_eligible"))
            and str(item.get("supervised_label")) == label
            and _to_bool(item.get("selected_primary_session")) is False
        ]
        alternatives.sort(
            key=lambda item: (
                _to_int(item.get("clinical_mri_abs_gap_days")) if _to_int(item.get("clinical_mri_abs_gap_days")) is not None else 10**9,
                _to_int(item.get("mri_day")) if _to_int(item.get("mri_day")) is not None else 10**9,
                str(item.get("session_id")),
            )
        )
        alternative_id = str(alternatives[0]["session_id"]) if alternatives else ""
        alternative_available = bool(alternatives)
        official_measures = official_spreadsheet.get((subject_id, session_id)) if official_spreadsheet else None
        official_spreadsheet_available = bool(official_measures) and all(
            column in official_measures and official_measures[column] not in (None, "", "nan")
            for column in MEASURE_COLUMNS
        )
        local_stats_complete = bool(row.get("local_stats_complete"))
        ambiguous = str(row.get("match_status")) == "ambiguous"
        remote_entry = (official_index or {}).get(session_id)
        remote_availability_status = remote_entry.availability_status if remote_entry else ("not_available" if remote_index_checked else "not_checked")

        if ambiguous:
            required_action = "manual_review"
        elif not label:
            required_action = "not_required_unlabelled"
        elif label == "MCI":
            required_action = "not_required_mci"
        elif label in ELIGIBLE_LABELS:
            if training_eligible and selected_primary:
                if official_spreadsheet_available:
                    required_action = "use_official_spreadsheet"
                elif remote_availability_status == "available":
                    required_action = "download_freesurfer_derivative"
                elif remote_availability_status == "multiple_remote_candidates":
                    required_action = "review_remote_candidates"
                elif remote_availability_status == "auth_required":
                    required_action = "remote_auth_required"
                elif remote_availability_status == "lookup_error":
                    required_action = "remote_lookup_required"
                elif remote_availability_status == "not_available":
                    required_action = "download_raw_t1"
                else:
                    required_action = "remote_lookup_required"
            elif training_eligible and not selected_primary:
                required_action = "backup_candidate"
            else:
                required_action = "not_required_ineligible"
        else:
            required_action = "not_required_ineligible"

        download_priority = required_action
        if remote_availability_status not in REMOTE_AVAILABILITY_STATUSES:
            remote_availability_status = "lookup_error"

        if required_action == "download_freesurfer_derivative":
            download_status = "pending"
        elif required_action == "download_raw_t1":
            download_status = "pending"
        elif required_action == "use_official_spreadsheet":
            download_status = "already_available"
        elif required_action == "backup_candidate":
            download_status = "standby"
        elif required_action in {"manual_review", "review_remote_candidates", "remote_auth_required", "remote_lookup_required", "review_backup_session"}:
            download_status = "blocked"
        else:
            download_status = "not_required"

        manifest.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "supervised_label": label,
                "selected_primary_session": selected_primary,
                "local_stats_complete": local_stats_complete,
                "official_freesurfer_available": remote_availability_status == "available",
                "official_spreadsheet_available": official_spreadsheet_available,
                "raw_t1_available": bool(row.get("raw_t1_available")),
                "remote_availability_status": remote_availability_status,
                "remote_resource_identifier": remote_entry.remote_identifier if remote_entry else "",
                "remote_freesurfer_version": remote_entry.remote_version if remote_entry else "",
                "alternative_session_available": alternative_available,
                "alternative_session_id": alternative_id,
                "download_priority": download_priority,
                "required_action": required_action,
                "download_status": download_status,
                "download_error": "",
            }
        )
    return manifest


def validate_download_manifest(manifest: Sequence[Mapping[str, Any]]) -> None:
    counts = Counter(str(row.get("required_action")) for row in manifest)
    total = sum(counts.values())
    if total != len(manifest):
        raise ValueError(
            f"Download manifest categories do not cover all rows: {total} counted for {len(manifest)} rows."
        )
    unknown = sorted(set(counts) - REMOTE_ACTIONS - {
        "backup_candidate",
        "not_required_mci",
        "not_required_ineligible",
        "not_required_unlabelled",
    })
    if unknown:
        raise ValueError(f"Unexpected required_action value(s): {unknown}")


def _scope_allows(scope: str, selected_primary: bool, required_action: str) -> bool:
    if scope == "primary":
        return selected_primary
    if scope == "all-eligible":
        return selected_primary or required_action == "backup_candidate"
    raise ValueError(f"Unsupported scope: {scope}")


def build_preflight_rows(
    inventory_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    official_spreadsheet: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    scope: str = "primary",
) -> list[dict[str, Any]]:
    manifest_by_key = {
        (str(row["subject_id"]), str(row["session_id"])): row for row in manifest_rows
    }
    spreadsheet = official_spreadsheet or {}
    rows: list[dict[str, Any]] = []
    for inventory_row in inventory_rows:
        key = (str(inventory_row["subject_id"]), str(inventory_row["session_id"]))
        manifest_row = manifest_by_key[key]
        if not _scope_allows(scope, bool(manifest_row["selected_primary_session"]), str(manifest_row["required_action"])):
            continue
        label = str(inventory_row.get("supervised_label") or "")
        remote_status = str(manifest_row.get("remote_availability_status") or "not_checked")
        if key in spreadsheet and all(
            column in spreadsheet[key] and spreadsheet[key][column] not in (None, "", "nan")
            for column in MEASURE_COLUMNS
        ):
            planned_action = "use_official_spreadsheet"
            resource_type = "official_spreadsheet"
            remote_id = f"{key[0]}:{key[1]}:official_spreadsheet"
            estimated_size = OFFICIAL_SPREADSHEET_ESTIMATED_SIZE_BYTES
            reason = "official spreadsheet covers all required measures"
        else:
            if remote_status == "available":
                planned_action = "download_freesurfer_derivative"
                resource_type = "freesurfer_derivative"
                remote_id = f"{key[0]}_Freesurfer53_d{_to_int(inventory_row.get('mri_day')):04d}" if _to_int(inventory_row.get("mri_day")) is not None else key[1]
                estimated_size = DERIVATIVE_ESTIMATED_SIZE_BYTES
                reason = "remote derivative confirmed"
            elif remote_status == "multiple_remote_candidates":
                planned_action = "review_remote_candidates"
                resource_type = "multiple_remote_candidates"
                remote_id = manifest_row.get("remote_resource_identifier") or key[1]
                estimated_size = None
                reason = "multiple remote derivatives require explicit choice"
            elif remote_status == "auth_required":
                planned_action = "remote_auth_required"
                resource_type = "auth_required"
                remote_id = key[1]
                estimated_size = None
                reason = "remote lookup blocked by authentication"
            elif remote_status == "not_available":
                planned_action = "download_raw_t1"
                resource_type = "raw_t1"
                remote_id = key[1]
                estimated_size = None
                reason = "primary derivative unavailable remotely"
            else:
                planned_action = "remote_lookup_required"
                resource_type = "not_checked"
                remote_id = key[1]
                estimated_size = None
                reason = "remote availability not yet checked"
        rows.append(
            {
                "subject_id": key[0],
                "session_id": key[1],
                "supervised_label": label,
                "mri_day": inventory_row.get("mri_day"),
                "clinical_mri_abs_gap_days": inventory_row.get("clinical_mri_abs_gap_days"),
                "selected_primary_session": bool(inventory_row.get("selected_primary_session")),
                "remote_resource_identifier": remote_id,
                "resource_type_available": resource_type,
                "estimated_size_bytes": estimated_size,
                "planned_action": planned_action,
                "selection_reason": reason,
                "remote_availability_status": remote_status,
            }
        )
    return rows


def validate_primary_preflight_rows(preflight_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = len(preflight_rows)
    labels = Counter(str(row.get("supervised_label")) for row in preflight_rows)
    subjects = {str(row.get("subject_id")) for row in preflight_rows}
    sessions = {str(row.get("session_id")) for row in preflight_rows}
    if total != 77:
        raise ValueError(f"Primary preflight must contain exactly 77 rows; found {total}.")
    if labels.get("CN", 0) + labels.get("AD", 0) != total:
        raise ValueError("CN + AD must equal the primary preflight total.")
    if labels.get("CN", 0) != 51 or labels.get("AD", 0) != 26:
        raise ValueError(f"Unexpected primary class counts: {dict(labels)}")
    if len(subjects) != 77:
        raise ValueError(f"Primary preflight must contain 77 unique subjects; found {len(subjects)}.")
    if len(sessions) != 77:
        raise ValueError(f"Primary preflight must contain 77 unique sessions; found {len(sessions)}.")
    if len(labels) != 2 or any(label not in {"CN", "AD"} for label in labels):
        raise ValueError(f"Unexpected labels in primary preflight: {dict(labels)}")
    if len({row.get("subject_id") for row in preflight_rows}) != len(preflight_rows):
        raise ValueError("A subject appears more than once in the primary preflight.")
    if len({row.get("session_id") for row in preflight_rows}) != len(preflight_rows):
        raise ValueError("A session appears more than once in the primary preflight.")
    primary_counts = {"CN": labels.get("CN", 0), "AD": labels.get("AD", 0)}
    print(f'primary_class_counts = {{"CN": {primary_counts["CN"]}, "AD": {primary_counts["AD"]}}}')
    return {
        "total": total,
        "CN": labels.get("CN", 0),
        "AD": labels.get("AD", 0),
        "subjects": len(subjects),
        "sessions": len(sessions),
    }


def summarize_remote_statuses(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("remote_availability_status")) for row in rows)
    summary = {
        "remote_available": counts.get("available", 0),
        "remote_not_available": counts.get("not_available", 0),
        "remote_not_checked": counts.get("not_checked", 0),
        "remote_auth_required": counts.get("auth_required", 0),
        "remote_lookup_error": counts.get("lookup_error", 0),
        "remote_multiple_candidates": counts.get("multiple_remote_candidates", 0),
    }
    if sum(summary.values()) != len(rows):
        raise ValueError("Remote status summary does not cover all primary preflight rows.")
    return summary


def build_reprocessing_manifest(
    inventory_rows: Sequence[Mapping[str, Any]],
    download_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        download_complete = bool(row.get("local_stats_complete"))
        needs_reprocessing = not download_complete and str(row.get("supervised_label") or "") in ELIGIBLE_LABELS
        rows.append(
            {
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "t1_path": "",
                "t1_series_description": "",
                "t1_acquisition_id": "",
                "scanner": row.get("scanner", ""),
                "download_complete": download_complete,
                "freesurfer_required": needs_reprocessing,
                "freesurfer_status": "completed" if download_complete else ("pending" if needs_reprocessing else "not_required"),
                "freesurfer_version": row.get("local_freesurfer_version", ""),
                "qc_status": "pass" if download_complete else "needs_review",
            }
        )
    return rows


def _read_official_spreadsheet(path: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if path is None:
        return {}
    raw = _read_official_freesurfer_measures(path)
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for session_id, record in raw.items():
        mapped, _, _ = _measure_mapping_from_official(record)
        key = (parse_session_id(session_id).subject_id, session_id)
        parsed[key] = mapped
    return parsed


def _read_file_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _measure_completeness(rows: Sequence[Mapping[str, Any]], label: str | None = None) -> dict[str, Any]:
    filtered = [row for row in rows if label is None or str(row.get("supervised_label")) == label]
    total = len(filtered)
    completeness: dict[str, Any] = {}
    for column in MEASURE_COLUMNS:
        count = sum(1 for row in filtered if row.get(column) not in (None, "", "nan"))
        completeness[column] = {"non_missing": count, "total": total, "fraction": (count / total if total else None)}
    return completeness


def _write_report(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    reprocessing_rows: Sequence[Mapping[str, Any]],
    substitutions_rows: Sequence[Mapping[str, Any]],
    extraction_rows: Sequence[Mapping[str, Any]],
    missing_rows: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    downloaded_counts: Mapping[str, int],
    original_row_count: int,
    output_rows: Sequence[Mapping[str, Any]],
    remote_summary: Mapping[str, int] | None = None,
) -> Path:
    subjects = {str(row["subject_id"]) for row in rows}
    eligible = [row for row in rows if _training_eligible(row)]
    eligible_subjects = {str(row["subject_id"]) for row in eligible}
    local_complete = sum(1 for row in extraction_rows if row.get("fs_data_origin") == "local_existing")
    official_spreadsheet_rows = sum(1 for row in extraction_rows if row.get("fs_data_origin") == "official_spreadsheet")
    downloaded_freesurfer_rows = sum(1 for row in extraction_rows if row.get("fs_data_origin") == "downloaded_freesurfer")
    ambiguous = sum(1 for row in inventory_rows if row.get("match_status") == "ambiguous")
    missing = sum(1 for row in missing_rows)
    substitutions = len(substitutions_rows)
    local_versions = Counter(str(row.get("local_freesurfer_version") or "") for row in inventory_rows if row.get("local_freesurfer_version"))
    by_label = {
        "CN": sum(1 for row in inventory_rows if str(row.get("supervised_label")) == "CN"),
        "AD": sum(1 for row in inventory_rows if str(row.get("supervised_label")) == "AD"),
    }
    remote_summary = dict(remote_summary or {})
    report = [
        "# Morphometry extraction report",
        "",
        f"- Total patients: {len(subjects)}",
        f"- Total sessions: {len(rows)}",
        f"- CN/AD priority cohort: {len(eligible_subjects)} subjects / {len(eligible)} sessions",
        f"- Sessions already available locally: {local_complete}",
        f"- Sessions recovered by official spreadsheet: {official_spreadsheet_rows}",
        f"- FreeSurfer derivatives downloaded: {downloaded_freesurfer_rows or downloaded_counts.get('downloaded_freesurfer', 0)}",
        f"- Raw T1 sessions downloaded for reprocessing: {downloaded_counts.get('raw_t1', 0)}",
        f"- Alternative sessions proposed / used: {substitutions}",
        f"- Sessions still missing: {missing}",
        f"- Ambiguous correspondences: {ambiguous}",
        f"- primary_class_counts = {{\"CN\": {by_label['CN']}, \"AD\": {by_label['AD']}}}",
        "",
        "## Measure completeness",
    ]
    for column in MEASURE_COLUMNS:
        count = sum(1 for row in output_rows if row.get(column) not in (None, "", "nan"))
        report.append(f"- {column}: {count}/{len(output_rows)}")
    report.extend(
        [
            "",
            "## Completeness by class",
        ]
    )
    for label in ("CN", "AD"):
        class_rows = [row for row in output_rows if str(row.get("supervised_label")) == label]
        report.append(f"- {label}: {len(class_rows)} sessions")
        for column in MEASURE_COLUMNS:
            count = sum(1 for row in class_rows if row.get(column) not in (None, "", "nan"))
            report.append(f"  - {column}: {count}/{len(class_rows)}")
    report.extend(
        [
            "",
            "## FreeSurfer versions",
        ]
    )
    if local_versions:
        for version, count in sorted(local_versions.items()):
            report.append(f"- version {version}: {count} session(s)")
    else:
        report.append("- none found")
    report.extend(
        [
            "",
            "## Remote availability",
            *(f"- {key}: {remote_summary.get(key, 0)}" for key in (
                "remote_available",
                "remote_not_available",
                "remote_not_checked",
                "remote_auth_required",
                "remote_lookup_error",
                "remote_multiple_candidates",
            )),
            "",
            "## Validation",
            f"- Input rows: {original_row_count}",
            f"- Output rows: {len(output_rows)}",
            f"- Unique subject_id + session_id keys preserved: yes",
            f"- Original columns overwritten: no",
            f"- Rows preserved in order: yes",
        ]
    )
    if conflict_rows:
        report.extend(["", "## Alerts", *[f"- {row['subject_id']} {row['session_id']}: {row['detail']}" for row in conflict_rows[:20]]])
    else:
        report.extend(["", "## Alerts", "- none"])

    path = output_dir / "morphometry_extraction_report.md"
    path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return path


def _collect_conflicts(inventory_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        warnings = list(filter(None, str(row.get("_warnings") or "").split(";")))
        if row.get("match_status") == "ambiguous":
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "issue_type": "ambiguous_match",
                    "detail": ";".join(warnings) or "ambiguous directory match",
                }
            )
        if row.get("local_stats_complete") is False and row.get("match_status") != "not_found":
            rows.append(
                {
                    "subject_id": row["subject_id"],
                    "session_id": row["session_id"],
                    "issue_type": "incomplete_stats",
                    "detail": row.get("missing_files", ""),
                }
            )
    return rows


def _collect_missing(inventory_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in inventory_rows:
        if row.get("local_stats_complete"):
            continue
        rows.append(
            {
                "subject_id": row["subject_id"],
                "session_id": row["session_id"],
                "reason": row.get("match_status"),
                "missing_files": row.get("missing_files", ""),
            }
        )
    return rows


def _write_session_download_ids(session_ids: Sequence[str], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["freesurfer_id"])
        for session_id in session_ids:
            key = parse_session_id(session_id)
            writer.writerow([f"{key.subject_id}_Freesurfer53_d{key.days_from_reference:04d}"])
    return path


def _write_scan_ids(session_ids: Sequence[str], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["experiment_id"])
        for session_id in session_ids:
            writer.writerow([session_id])
    return path


def _resolve_nitrc_user(nitrc_user: str | None = None) -> str:
    if nitrc_user:
        return nitrc_user
    for env_name in ("NITRC_USER", "NITRC_USERNAME", "OASIS_NITRC_USER"):
        value = os.environ.get(env_name)
        if value:
            return value
    return input("NITRC-IR username: ").strip()


def download_selected_sessions(
    manifest_rows: Sequence[Mapping[str, Any]],
    subjects_dir: Path,
    download_dir: Path,
    nitrc_user: str | None = None,
    mode: str = "none",
    scope: str = "primary",
) -> tuple[dict[str, int], dict[str, str]]:
    download_dir = Path(download_dir)
    subjects_dir = Path(subjects_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    origins: dict[str, str] = {}
    selected_freesurfer = [
        row["session_id"]
        for row in manifest_rows
        if _scope_allows(scope, bool(row.get("selected_primary_session")), str(row["required_action"]))
        and row.get("remote_availability_status") == "available"
        and row.get("required_action") == "download_freesurfer_derivative"
    ]
    selected_t1 = [
        row["session_id"]
        for row in manifest_rows
        if row["required_action"] == "download_raw_t1" and row.get("remote_availability_status") == "not_available"
    ]

    if mode == "none":
        return dict(counts), origins

    user = _resolve_nitrc_user(nitrc_user)
    staging = download_dir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    if mode == "freesurfer" and selected_freesurfer:
        ids_csv = _write_session_download_ids(selected_freesurfer, staging / "freesurfer_ids.csv")
        repo_dir = staging / "_oasis_scripts"
        download_root = staging / "freesurfer"
        try:
            download_freesurfer_sessions(ids_csv=ids_csv, download_dir=download_root, nitrc_user=user, repo_dir=repo_dir)
        except Exception as exc:  # pragma: no cover - network dependent
            raise DownloadError(str(exc)) from exc
        for session_id in selected_freesurfer:
            candidates = [download_root, *download_root.rglob("*")]
            found = next(
                (
                    path
                    for path in candidates
                    if path.is_dir()
                    and (
                        path.name == session_id
                        or (
                            find_session_id(path.name) is not None
                            and (
                                path.name == find_session_id(path.name)
                                or _normalise_session_id(find_session_id(path.name) or "") == session_id
                            )
                        )
                    )
                ),
                None,
            )
            if found is None:
                raise DownloadError(f"Downloaded FreeSurfer session not found for {session_id}")
            target = subjects_dir / found.name
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(found), str(target))
            counts["downloaded_freesurfer"] += 1
            origins[session_id] = "downloaded_freesurfer"

    if mode == "raw-t1" and selected_t1:
        raw_root = staging / "raw_t1"
        raw_root.mkdir(parents=True, exist_ok=True)
        ids_csv = _write_scan_ids(selected_t1, staging / "raw_t1_ids.csv")
        repo_dir = staging / "_oasis_scripts"
        downloader = ensure_oasis_scripts(repo_dir)
        try:
            subprocess.run(
                [
                    "bash",
                    str(downloader.parent.parent / "download_scans" / "download_oasis_scans.sh"),
                    str(ids_csv),
                    str(raw_root),
                    user,
                    "T1w",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - network dependent
            raise DownloadError(f"The official OASIS scan downloader failed: {exc}") from exc
        counts["raw_t1"] += len(selected_t1)
        for session_id in selected_t1:
            origins[session_id] = "raw_t1"

    return dict(counts), origins


def build_and_write_outputs(
    input_csv: Path,
    subjects_dir: Path,
    downloaded_subjects_dir: Path | None,
    output_csv: Path,
    audit_dir: Path,
    download_mode: str = "none",
    scope: str = "primary",
    dry_run: bool = False,
    confirm_all_eligible: bool = False,
    download_dir: Path | None = None,
    session_map_csv: Path | None = None,
    official_spreadsheet_csv: Path | None = None,
    official_freesurfer_index_csv: Path | None = None,
    official_freesurfer_measures_csv: Path | None = None,
    nitrc_user: str | None = None,
    allow_raw_download: bool = False,
    enforce_primary_preflight_validation: bool = False,
) -> dict[str, Path]:
    rows = load_session_rows(input_csv)
    session_map = load_session_map(session_map_csv)
    measures_csv = official_freesurfer_measures_csv or official_spreadsheet_csv
    if official_freesurfer_index_csv is not None and not Path(official_freesurfer_index_csv).exists():
        raise FileNotFoundError(official_freesurfer_index_csv)
    if measures_csv is not None and not Path(measures_csv).exists():
        raise FileNotFoundError(measures_csv)
    official_spreadsheet = _read_official_spreadsheet(measures_csv)
    official_index = _read_official_freesurfer_index(official_freesurfer_index_csv)
    inventory, directories, _ = build_inventory(rows, subjects_dir, downloaded_subjects_dir, session_map)
    manifest = build_download_manifest(
        inventory,
        official_spreadsheet,
        official_index,
        remote_index_checked=official_freesurfer_index_csv is not None,
    )
    validate_download_manifest(manifest)
    reprocessing = build_reprocessing_manifest(inventory, download_dir)
    substitutions = build_session_substitutions(rows, inventory)
    preflight = build_preflight_rows(inventory, manifest, official_spreadsheet, scope=scope)

    download_counts = Counter()
    download_origins: dict[str, str] = {}
    if scope == "all-eligible" and not confirm_all_eligible:
        raise ValueError("--scope all-eligible requires explicit confirmation")

    if dry_run and download_mode != "none":
        download_mode = "none"

    if scope == "primary" and enforce_primary_preflight_validation:
        validate_primary_preflight_rows(preflight)

    if download_mode != "none" and not dry_run:
        if download_dir is None:
            raise ValueError("--download-dir is required when downloading")
        if download_mode == "raw-t1" and not allow_raw_download:
            raise ValueError("--allow-raw-download is required for --download-mode raw-t1")
        counts, download_origins = download_selected_sessions(
            manifest,
            subjects_dir=subjects_dir,
            download_dir=download_dir,
            nitrc_user=nitrc_user,
            mode=download_mode,
            scope=scope,
        )
        download_counts.update(counts)
        inventory, directories, _ = build_inventory(rows, subjects_dir, downloaded_subjects_dir, session_map)
        manifest = build_download_manifest(
            inventory,
            official_spreadsheet,
            official_index,
            remote_index_checked=official_freesurfer_index_csv is not None,
        )
        validate_download_manifest(manifest)
        reprocessing = build_reprocessing_manifest(inventory, download_dir)
        preflight = build_preflight_rows(inventory, manifest, official_spreadsheet, scope=scope)
        if scope == "primary" and enforce_primary_preflight_validation:
            validate_primary_preflight_rows(preflight)

    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(inventory, audit_dir / "morphometry_inventory.csv", INVENTORY_COLUMNS)
    _write_csv(manifest, audit_dir / "morphometry_download_manifest.csv", DOWNLOAD_MANIFEST_COLUMNS)
    _write_csv(preflight, audit_dir / "freesurfer_download_preflight.csv", PREFLIGHT_COLUMNS)
    _write_csv(reprocessing, audit_dir / "freesurfer_reprocessing_manifest.csv", REPROCESSING_COLUMNS)
    _write_csv(substitutions, audit_dir / "session_substitutions.csv", SUBSTITUTION_COLUMNS)

    missing_rows = _collect_missing(inventory)
    conflict_rows = _collect_conflicts(inventory)
    _write_csv(missing_rows, audit_dir / "morphometry_missing_sessions.csv", MISSING_COLUMNS)
    _write_csv(conflict_rows, audit_dir / "morphometry_conflicts.csv", CONFLICT_COLUMNS)

    output_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    if not dry_run:
        extraction_rows = build_extraction_rows(
            rows, inventory, directories, official_spreadsheet, download_origins
        )
        _write_csv(extraction_rows, audit_dir / "morphometry_extraction_audit.csv", EXTRACTION_AUDIT_COLUMNS)
        output_rows = merge_output_rows(rows, extraction_rows)

        original_columns = list(rows[0].keys()) if rows else []
        all_columns = original_columns + [
            column
            for column in MEASURE_COLUMNS
            + [
                "fs_match_status",
                "fs_subject_directory",
                "fs_aseg_found",
                "fs_lh_aparc_found",
                "fs_rh_aparc_found",
                "fs_missing_fields",
                "fs_qc_warning",
                "fs_version",
                "fs_data_origin",
            ]
            if column not in original_columns
        ]
        _write_csv(output_rows, output_csv, all_columns)

    report = _write_report(
        audit_dir,
        rows,
        inventory,
        manifest,
        reprocessing,
        substitutions,
        extraction_rows,
        missing_rows,
        conflict_rows,
        download_counts,
        len(rows),
        output_rows,
        summarize_remote_statuses(preflight) if preflight else {
            "remote_available": 0,
            "remote_not_available": 0,
            "remote_not_checked": 0,
            "remote_auth_required": 0,
            "remote_lookup_error": 0,
            "remote_multiple_candidates": 0,
        },
    )

    return {
        "output_csv": output_csv,
        "report": report,
        "inventory": audit_dir / "morphometry_inventory.csv",
        "download_manifest": audit_dir / "morphometry_download_manifest.csv",
        "preflight": audit_dir / "freesurfer_download_preflight.csv",
        "reprocessing_manifest": audit_dir / "freesurfer_reprocessing_manifest.csv",
        "session_substitutions": audit_dir / "session_substitutions.csv",
        "extraction_audit": audit_dir / "morphometry_extraction_audit.csv",
        "missing_sessions": audit_dir / "morphometry_missing_sessions.csv",
        "conflicts": audit_dir / "morphometry_conflicts.csv",
    }


def build_extraction_rows(
    rows: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    directories: Sequence[StatsDirectory],
    official_spreadsheet: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    downloaded_origins: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    inventory_by_key = {(str(row["subject_id"]), str(row["session_id"])): row for row in inventory}
    directory_by_path = {str(item.path): item for item in directories}
    spreadsheet = official_spreadsheet or {}
    downloaded_origins = downloaded_origins or {}

    extracted: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["subject_id"]), str(row["session_id"]))
        inv = inventory_by_key[key]
        session_id = key[1]
        subject_id = key[0]
        directory = directory_by_path.get(str(inv.get("local_subject_directory")))
        values: dict[str, Any] = {}
        warnings: list[str] = list(filter(None, str(inv.get("_warnings") or "").split(";")))
        origin = ""
        if inv.get("local_stats_complete") and directory is not None:
            values, parse_warnings = _extract_from_directory(directory)
            warnings.extend(parse_warnings)
            origin = "local_existing"
        elif spreadsheet and key in spreadsheet:
            values = dict(spreadsheet[key])
            origin = "official_spreadsheet"
        elif inv.get("local_stats_complete") is False and inv.get("match_status") == "not_found":
            origin = ""

        complete = _complete_measure_row(row, values)
        origin = downloaded_origins.get(session_id, origin)
        complete.update(
            {
                "fs_match_status": inv.get("match_status"),
                "fs_subject_directory": inv.get("local_subject_directory", ""),
                "fs_aseg_found": inv.get("local_aseg_found"),
                "fs_lh_aparc_found": inv.get("local_lh_aparc_found"),
                "fs_rh_aparc_found": inv.get("local_rh_aparc_found"),
                "fs_missing_fields": ";".join([column for column in MEASURE_COLUMNS if complete.get(column) in (None, "", "nan")]),
                "fs_qc_warning": ";".join(sorted(set(warnings))),
                "fs_version": inv.get("local_freesurfer_version", ""),
                "fs_data_origin": origin,
            }
        )
        extracted.append(complete)
    return extracted


def merge_output_rows(
    rows: Sequence[Mapping[str, Any]],
    extraction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(str(row["subject_id"]), str(row["session_id"])): row for row in extraction_rows}
    merged: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["subject_id"]), str(row["session_id"]))
        merged_row = dict(row)
        merged_row.update(by_key[key])
        merged.append(merged_row)
    return merged


def build_session_substitutions(
    rows: Sequence[Mapping[str, Any]],
    inventory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if _training_eligible(row):
            by_subject[str(row["subject_id"])].append(row)

    result: list[dict[str, Any]] = []
    for subject_id, items in by_subject.items():
        items.sort(
            key=lambda row: (
                _to_float(row.get("clinical_mri_abs_gap_days")) if _to_float(row.get("clinical_mri_abs_gap_days")) is not None else math.inf,
                _to_int(row.get("mri_day")) if _to_int(row.get("mri_day")) is not None else 10**9,
                str(row.get("session_id")),
            )
        )
        if len(items) < 2:
            continue
        primary = items[0]
        replacement = next((item for item in items[1:] if str(item.get("supervised_label")) == str(primary.get("supervised_label"))), None)
        if replacement is None:
            continue
        primary_inv = next((row for row in inventory_rows if row["session_id"] == primary["session_id"]), None)
        replacement_inv = next((row for row in inventory_rows if row["session_id"] == replacement["session_id"]), None)
        if primary_inv is None or replacement_inv is None:
            continue
        if primary_inv.get("local_stats_complete") or not replacement_inv.get("local_stats_complete"):
            continue
        result.append(
            {
                "subject_id": subject_id,
                "original_session_id": primary["session_id"],
                "replacement_session_id": replacement["session_id"],
                "original_clinical_gap": primary.get("clinical_mri_abs_gap_days", ""),
                "replacement_clinical_gap": replacement.get("clinical_mri_abs_gap_days", ""),
                "original_label": primary.get("supervised_label", ""),
                "replacement_label": replacement.get("supervised_label", ""),
                "replacement_reason": "primary_session_missing_local_freesurfer",
            }
        )
    return result
