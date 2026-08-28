"""Assembly of the clinical-imaging master table and its radiomic joins.

Two stages, mirroring the two CLI commands:

``build_clinical_linkage`` / ``write_linkage_outputs``
    MRI catalogue + D1 + B4 (+ optional C1) -> ``clinical_visits.csv``,
    ``clinical_imaging_master.csv``, ``clinical_linkage_validation.json``.

``build_clinical_radiomics``
    master table + the frozen radiomics outputs -> ``clinical_radiomics_sessions.csv``,
    ``clinical_radiomics_deltas.csv``, ``clinical_radiomics_subjects.csv``.

The radiomics inputs are read-only. Nothing in this module changes the 16-ROI /
107-feature protocol or rewrites any existing radiomics file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__
from .clinical import (
    CLINICAL_LINKAGE_VERSION,
    DEFAULT_CLINICAL_WINDOW_DAYS,
    MATCHING_STRATEGY,
)
from .clinical.classification import ClassificationCodebook, classify_clinical_visit
from .clinical.matching import (
    gap_statistics,
    group_visits_by_subject,
    match_mri_to_clinical,
    match_mri_to_cognitive,
    merge_d1_b4,
)
from .clinical.models import ClinicalMatch, ClinicalVisit, MriSession, SubjectTrajectory
from .clinical.readers import read_b4, read_c1, read_d1, read_mri_catalogue
from .clinical.trajectories import annotate_session, build_trajectories, conversion_between
from .clinical.validation import (
    VALIDATION_FILENAME,
    ValidationIssue,
    check_master_rows,
    check_matches,
    check_mri_sessions,
    check_raw_day_values,
    log_issue_summary,
    summarise,
    write_validation_report,
)
from .tables import read_csv_rows, write_csv

logger = logging.getLogger(__name__)

VISITS_CSV = "clinical_visits.csv"
MASTER_CSV = "clinical_imaging_master.csv"
SESSIONS_CSV = "clinical_radiomics_sessions.csv"
DELTAS_CSV = "clinical_radiomics_deltas.csv"
SUBJECTS_CSV = "clinical_radiomics_subjects.csv"

#: Join keys shared by the clinical and radiomics tables.
JOIN_KEYS = ("subject_id", "session_id")

#: Clinical instrument columns already surfaced under a normalised name; they
#: are not repeated in the raw pass-through.
_KEY_COLUMNS = frozenset(
    {"OASISID", "OASIS_session_label", "days_to_visit", "age at visit"}
)

MASTER_LEADING_COLUMNS = (
    "subject_id",
    "session_id",
    "mri_day",
    "sex",
    "age_at_mri",
    "scanner",
    "clinical_session_id",
    "clinical_day",
    "clinical_mri_gap_days",
    "clinical_mri_abs_gap_days",
    "clinical_match_found",
    "clinical_match_valid",
    "clinical_match_reason",
    "clinical_match_ambiguous",
    "clinical_source",
    "age_at_clinical_visit",
    "cognitive_status",
    "ad_etiology",
    "classification_status",
    "classification_reason",
    "diagnosis_at_mri",
    "future_diagnosis",
    "conversion_event",
    "conversion_day",
    "days_to_conversion",
    "clinical_trajectory",
    "MMSE",
    "CDRSUM",
    "CDRTOT",
    "dx1",
    "dx2",
    "dx3",
    "dx4",
    "dx5",
)

VISITS_LEADING_COLUMNS = (
    "subject_id",
    "clinical_day",
    "d1_session_id",
    "b4_session_id",
    "clinical_source",
    "age_at_clinical_visit",
    "cognitive_status",
    "ad_etiology",
    "classification_status",
)

SUBJECTS_LEADING_COLUMNS = (
    "subject_id",
    "baseline_diagnosis",
    "last_diagnosis",
    "clinical_trajectory",
    "conversion_event",
    "conversion_day",
    "n_mri_sessions",
    "n_clinical_visits",
    "followup_years",
    "feature",
    "feature_slope",
    "feature_r2",
)


class ClinicalDatasetError(RuntimeError):
    """Raised when a clinical dataset stage cannot produce a usable output."""


@dataclass
class ClinicalLinkageResult:
    """Everything the linkage stage produced."""

    master_rows: list[dict[str, Any]] = field(default_factory=list)
    visit_rows: list[dict[str, Any]] = field(default_factory=list)
    matches: list[ClinicalMatch] = field(default_factory=list)
    trajectories: dict[str, SubjectTrajectory] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# stage 1: clinical linkage
# ---------------------------------------------------------------------------
def build_clinical_linkage(
    mri_catalog: Path,
    d1_path: Path,
    b4_path: Path,
    c1_path: Path | None = None,
    window_days: int = DEFAULT_CLINICAL_WINDOW_DAYS,
    cognitive_window_days: int | None = None,
    codebook: ClassificationCodebook | None = None,
) -> ClinicalLinkageResult:
    """Link every MRI session in the catalogue to its nearest clinical visit.

    C1 is optional; when supplied it is matched independently, with its own
    window (defaulting to ``window_days``), because psychometric testing rarely
    falls on the diagnostic visit day.
    """
    codebook = codebook or ClassificationCodebook.load(None)
    cognitive_window_days = (
        window_days if cognitive_window_days is None else cognitive_window_days
    )

    sessions = read_mri_catalogue(mri_catalog)
    if not sessions:
        raise ClinicalDatasetError(f"No usable MRI sessions in {mri_catalog}")

    d1_records = read_d1(d1_path)
    b4_records = read_b4(b4_path)
    c1_records = read_c1(c1_path) if c1_path else []

    issues: list[ValidationIssue] = []
    issues.extend(check_mri_sessions(sessions))
    for path in (d1_path, b4_path, c1_path):
        if path:
            issues.extend(check_raw_day_values(Path(path)))

    visits, merge_issues = merge_d1_b4(d1_records, b4_records)
    issues.extend(merge_issues)

    classifications = {
        (visit.subject_id, visit.clinical_day): classify_clinical_visit(visit, codebook)
        for visit in visits
    }

    visits_by_subject = group_visits_by_subject(visits)
    classifications_by_subject = {
        subject_id: {
            visit.clinical_day: classifications[(subject_id, visit.clinical_day)]
            for visit in subject_visits
        }
        for subject_id, subject_visits in visits_by_subject.items()
    }
    trajectories, trajectory_issues = build_trajectories(
        visits_by_subject, classifications_by_subject
    )
    issues.extend(trajectory_issues)

    matches = match_mri_to_clinical(sessions, visits, window_days)
    issues.extend(check_matches(matches))

    cognitive_matches = (
        {
            match.session_id: match
            for match in match_mri_to_cognitive(sessions, c1_records, cognitive_window_days)
        }
        if c1_records
        else {}
    )

    sessions_by_id = {session.session_id: session for session in sessions}
    # A session with no clinical visit still gets an explicit classification
    # ('no_clinical_data'); leaving those columns empty would be indistinguishable
    # from a value that simply failed to be written.
    no_data = classify_clinical_visit(None, codebook)
    master_rows = [
        _build_master_row(
            sessions_by_id[match.session_id],
            match,
            classifications.get((match.subject_id, match.visit.clinical_day), no_data)
            if match.visit is not None
            else no_data,
            trajectories.get(match.subject_id),
            cognitive_matches.get(match.session_id),
        )
        for match in matches
    ]
    issues.extend(check_master_rows(master_rows))

    visit_rows = [
        _build_visit_row(visit, classifications[(visit.subject_id, visit.clinical_day)])
        for visit in visits
    ]

    parameters = {
        "clinical_linkage_version": CLINICAL_LINKAGE_VERSION,
        "oasis_radiomics_version": __version__,
        "clinical_window_days": window_days,
        "cognitive_window_days": cognitive_window_days,
        "matching_strategy": MATCHING_STRATEGY,
        "classification_version": codebook.version,
        "classification_frozen": codebook.frozen,
        "mri_catalog": str(Path(mri_catalog).resolve()),
        "d1": str(Path(d1_path).resolve()),
        "b4": str(Path(b4_path).resolve()),
        "c1": str(Path(c1_path).resolve()) if c1_path else None,
    }

    log_issue_summary(issues)
    return ClinicalLinkageResult(
        master_rows=master_rows,
        visit_rows=visit_rows,
        matches=matches,
        trajectories=trajectories,
        issues=issues,
        parameters=parameters,
        summary=summarise(matches, issues, gap_statistics(matches)),
    )


def _raw_passthrough(record: Any) -> dict[str, Any]:
    """Every raw variable of a clinical record, minus the join keys."""
    if record is None:
        return {}
    return {
        column: value
        for column, value in record.raw.items()
        if column not in _KEY_COLUMNS
    }


def _build_master_row(
    session: MriSession,
    match: ClinicalMatch,
    classification: Any,
    trajectory: SubjectTrajectory | None,
    cognitive: Any,
) -> dict[str, Any]:
    """One row of ``clinical_imaging_master.csv``."""
    row: dict[str, Any] = dict(session.as_row())
    row.update(match.as_row())

    if classification is not None:
        row.update(classification.as_row())
    row.update(annotate_session(match, classification, trajectory))

    if cognitive is not None:
        row.update(cognitive.as_row())
        row.update(_raw_passthrough(cognitive.record))

    # Raw D1/B4 variables last: they are preserved verbatim and never
    # overwritten by a derived column (the name spaces are disjoint).
    if match.visit is not None:
        row.update(_raw_passthrough(match.visit.d1))
        row.update(_raw_passthrough(match.visit.b4))

    return row


def _build_visit_row(visit: ClinicalVisit, classification: Any) -> dict[str, Any]:
    """One row of ``clinical_visits.csv`` (the merged D1+B4 visit table)."""
    row: dict[str, Any] = {
        "subject_id": visit.subject_id,
        "clinical_day": visit.clinical_day,
        "d1_session_id": visit.d1_session_id,
        "b4_session_id": visit.b4_session_id,
        "clinical_source": visit.source,
        "age_at_clinical_visit": visit.age_at_clinical_visit,
    }
    row.update(classification.as_row())
    row.update(_raw_passthrough(visit.d1))
    row.update(_raw_passthrough(visit.b4))
    return row


def write_linkage_outputs(result: ClinicalLinkageResult, output_dir: Path) -> dict[str, Path]:
    """Write the three linkage artefacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "clinical_visits": write_csv(
            result.visit_rows, output_dir / VISITS_CSV, VISITS_LEADING_COLUMNS
        ),
        "clinical_imaging_master": write_csv(
            result.master_rows, output_dir / MASTER_CSV, MASTER_LEADING_COLUMNS
        ),
        "validation": write_validation_report(
            output_dir / VALIDATION_FILENAME,
            result.summary,
            result.issues,
            result.parameters,
        ),
    }
    return outputs


# ---------------------------------------------------------------------------
# stage 2: joins with the frozen radiomics outputs
# ---------------------------------------------------------------------------
def join_sessions(
    master_rows: Sequence[Mapping[str, Any]], wide_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    """Attach clinical columns to every session-level radiomics row.

    Driven by the radiomics table: exactly one output row per input radiomics
    row. A radiomics session with no entry in the clinical master is **kept**,
    with ``clinical_master_found = False``, rather than dropped.
    """
    master_index = {
        (str(row["subject_id"]), str(row["session_id"])): row for row in master_rows
    }
    issues: list[ValidationIssue] = []
    joined: list[dict[str, Any]] = []

    for radiomics_row in wide_rows:
        key = (str(radiomics_row["subject_id"]), str(radiomics_row["session_id"]))
        clinical = master_index.get(key)
        if clinical is None:
            issues.append(
                ValidationIssue(
                    code="no_clinical_visit",
                    severity="warning",
                    subject_id=key[0],
                    identifier=key[1],
                    detail="Radiomics session is absent from the clinical imaging master.",
                )
            )
        row: dict[str, Any] = dict(clinical) if clinical else {
            "subject_id": key[0],
            "session_id": key[1],
        }
        row["clinical_master_found"] = clinical is not None
        # Radiomic values last so a clinical column can never shadow a feature.
        row.update({k: v for k, v in radiomics_row.items() if k not in JOIN_KEYS})
        joined.append(row)

    issues.extend(check_master_rows(joined))
    logger.info(
        "clinical x radiomics sessions: %d row(s), %d with clinical data.",
        len(joined),
        sum(1 for row in joined if row["clinical_master_found"]),
    )
    return joined, issues


def join_deltas(
    master_rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, SubjectTrajectory],
) -> list[dict[str, Any]]:
    """Annotate radiomic deltas with the clinical status at ``t0`` and ``t1``."""
    by_session = {str(row["session_id"]): row for row in master_rows}
    joined: list[dict[str, Any]] = []

    for delta in delta_rows:
        t0 = by_session.get(str(delta.get("session_id_t0")), {})
        t1 = by_session.get(str(delta.get("session_id_t1")), {})
        subject_id = str(delta.get("subject_id"))

        row: dict[str, Any] = {
            "subject_id": subject_id,
            "comparison": delta.get("comparison"),
            "session_id_t0": delta.get("session_id_t0"),
            "session_id_t1": delta.get("session_id_t1"),
            "days_t0": delta.get("days_t0"),
            "days_t1": delta.get("days_t1"),
            "delta_days": delta.get("delta_days"),
            "delta_years": delta.get("delta_years"),
            "diagnosis_t0": t0.get("diagnosis_at_mri"),
            "diagnosis_t1": t1.get("diagnosis_at_mri"),
            "cdr_t0": t0.get("CDRTOT"),
            "cdr_t1": t1.get("CDRTOT"),
            "cdrsum_t0": t0.get("CDRSUM"),
            "cdrsum_t1": t1.get("CDRSUM"),
            "mmse_t0": t0.get("MMSE"),
            "mmse_t1": t1.get("MMSE"),
            "clinical_match_valid_t0": t0.get("clinical_match_valid"),
            "clinical_match_valid_t1": t1.get("clinical_match_valid"),
            "conversion_between_visits": conversion_between(
                trajectories.get(subject_id),
                # Conversion days live on the CLINICAL axis, so the interval must
                # too. Using the MRI days would miss a conversion recorded at the
                # visit that t1 was matched to, whenever that visit falls a few
                # days after the scan - exactly the common case.
                _interval_day(t0, "clinical_day", delta.get("days_t0")),
                _interval_day(t1, "clinical_day", delta.get("days_t1")),
            ),
        }
        row.update(
            {
                key: value
                for key, value in delta.items()
                if key.startswith(("delta_", "slope_")) and key not in row
            }
        )
        joined.append(row)

    logger.info("clinical x radiomics deltas: %d row(s).", len(joined))
    return joined


def join_subjects(
    master_rows: Sequence[Mapping[str, Any]],
    slope_rows: Sequence[Mapping[str, Any]],
    trajectories: Mapping[str, SubjectTrajectory],
    mri_session_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Attach each subject's clinical trajectory to its radiomic slopes."""
    counts = dict(mri_session_counts or {})
    joined: list[dict[str, Any]] = []

    for slope in slope_rows:
        subject_id = str(slope.get("subject_id"))
        trajectory = trajectories.get(subject_id)
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "baseline_diagnosis": trajectory.baseline_diagnosis if trajectory else "UNKNOWN",
            "last_diagnosis": trajectory.last_diagnosis if trajectory else "UNKNOWN",
            "clinical_trajectory": trajectory.trajectory if trajectory else "UNKNOWN",
            "conversion_event": trajectory.conversion_event if trajectory else None,
            "conversion_day": trajectory.conversion_day if trajectory else None,
            "n_mri_sessions": counts.get(subject_id),
            # None (not 0) when no visit history was available: "unknown" and
            # "zero visits" are different facts.
            "n_clinical_visits": (
                trajectory.n_clinical_visits if trajectory and trajectory.points else None
            ),
            "clinical_followup_years": trajectory.followup_years if trajectory else None,
        }
        row.update({key: value for key, value in slope.items() if key not in row})
        joined.append(row)

    logger.info("clinical x radiomics subjects: %d row(s).", len(joined))
    return joined


def build_clinical_radiomics(
    clinical_master: Path,
    radiomics_wide: Path,
    output_dir: Path,
    deltas: Path | None = None,
    slopes: Path | None = None,
    trajectories: Mapping[str, SubjectTrajectory] | None = None,
    clinical_visits: Path | None = None,
) -> dict[str, Path]:
    """Join the clinical master with the frozen radiomics outputs.

    Subject-level trajectories are resolved in order of fidelity:

    1. ``trajectories`` passed in by the caller (in-process, richest);
    2. ``clinical_visits.csv`` - explicitly given, or auto-detected next to
       ``clinical_master``. This is the only source with one row per *visit*,
       so it is the only one that can yield a true ``n_clinical_visits`` and a
       full ``CN -> MCI -> AD`` path;
    3. the master table alone, which has one row per MRI session and therefore
       yields only a degraded, per-session view.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_rows = read_csv_rows(Path(clinical_master))
    if not master_rows:
        raise ClinicalDatasetError(f"Clinical master table is empty: {clinical_master}")

    wide_rows = read_csv_rows(Path(radiomics_wide))
    if not wide_rows:
        raise ClinicalDatasetError(f"Radiomics wide table is empty: {radiomics_wide}")

    if trajectories is None:
        visits_path = _resolve_visits_path(clinical_visits, Path(clinical_master))
        if visits_path is not None:
            trajectories = _trajectories_from_visits(read_csv_rows(visits_path))
            logger.info(
                "Rebuilt %d subject trajectory/ies from %s.", len(trajectories), visits_path
            )
        else:
            logger.warning(
                "No clinical_visits.csv found next to %s; subject-level columns are "
                "derived from the per-session master only and n_clinical_visits will "
                "be empty.",
                clinical_master,
            )
            trajectories = _trajectories_from_master(master_rows)
    trajectories = dict(trajectories)

    session_rows, issues = join_sessions(master_rows, wide_rows)
    outputs = {
        "clinical_radiomics_sessions": write_csv(
            session_rows, output_dir / SESSIONS_CSV, MASTER_LEADING_COLUMNS
        )
    }

    counts: dict[str, int] = {}
    for row in wide_rows:
        counts[str(row["subject_id"])] = counts.get(str(row["subject_id"]), 0) + 1

    if deltas is not None:
        delta_rows = read_csv_rows(Path(deltas))
        outputs["clinical_radiomics_deltas"] = write_csv(
            join_deltas(master_rows, delta_rows, trajectories),
            output_dir / DELTAS_CSV,
            (
                "subject_id",
                "comparison",
                "session_id_t0",
                "session_id_t1",
                "diagnosis_t0",
                "diagnosis_t1",
                "cdr_t0",
                "cdr_t1",
                "delta_days",
                "delta_years",
                "conversion_between_visits",
            ),
        )

    if slopes is not None:
        slope_rows = read_csv_rows(Path(slopes))
        outputs["clinical_radiomics_subjects"] = write_csv(
            join_subjects(master_rows, slope_rows, trajectories, counts),
            output_dir / SUBJECTS_CSV,
            SUBJECTS_LEADING_COLUMNS,
        )

    if issues:
        log_issue_summary(issues)
    return outputs


def _resolve_visits_path(explicit: Path | None, master: Path) -> Path | None:
    """The clinical visits table to use, if one can be found."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise ClinicalDatasetError(f"Clinical visits table not found: {path}")
        return path
    candidate = Path(master).parent / VISITS_CSV
    return candidate if candidate.exists() else None


def _trajectories_from_visits(
    visit_rows: Sequence[Mapping[str, Any]]
) -> dict[str, SubjectTrajectory]:
    """Rebuild full subject trajectories from ``clinical_visits.csv``.

    This reuses :func:`~oasis_radiomics.clinical.trajectories.build_trajectory`,
    so a trajectory reconstructed from CSV is identical to one built in-process.
    """
    from .clinical.models import ClinicalClassification
    from .clinical.trajectories import build_trajectory

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in visit_rows:
        grouped.setdefault(str(row.get("subject_id")), []).append(row)

    rebuilt: dict[str, SubjectTrajectory] = {}
    for subject_id, rows in grouped.items():
        visits: list[ClinicalVisit] = []
        classifications: dict[int, ClinicalClassification] = {}
        for row in rows:
            day = _as_optional_int(row.get("clinical_day"))
            if day is None:
                continue
            visits.append(ClinicalVisit(subject_id=subject_id, clinical_day=day))
            classifications[day] = ClinicalClassification(
                cognitive_status=str(row.get("cognitive_status") or "UNKNOWN"),
                ad_etiology=str(row.get("ad_etiology") or "UNKNOWN"),
                status=str(row.get("classification_status") or "unresolved_codebook"),
            )
        trajectory, _ = build_trajectory(subject_id, visits, classifications)
        rebuilt[subject_id] = trajectory
    return rebuilt


def _trajectories_from_master(
    master_rows: Sequence[Mapping[str, Any]]
) -> dict[str, SubjectTrajectory]:
    """Rebuild minimal subject trajectories from a written master table.

    Only the columns the subject-level join needs are reconstructed; the full
    per-visit history stays in ``clinical_visits.csv``.
    """
    rebuilt: dict[str, SubjectTrajectory] = {}
    for row in master_rows:
        subject_id = str(row.get("subject_id"))
        if subject_id in rebuilt:
            continue
        conversion_day = _as_optional_int(row.get("conversion_day"))
        event = row.get("conversion_event") or None
        rebuilt[subject_id] = SubjectTrajectory(
            subject_id=subject_id,
            trajectory=str(row.get("clinical_trajectory") or "UNKNOWN"),
            baseline_diagnosis=str(row.get("diagnosis_at_mri") or "UNKNOWN"),
            last_diagnosis=str(row.get("future_diagnosis") or row.get("diagnosis_at_mri") or "UNKNOWN"),
            conversions=((str(event), conversion_day),) if event and conversion_day else (),
            conversion_event=str(event) if event else None,
            conversion_day=conversion_day,
        )
    return rebuilt


def _interval_day(
    master_row: Mapping[str, Any], column: str, fallback: Any
) -> int:
    """Day to use as an interval endpoint, preferring the matched clinical day.

    Falls back to the MRI day when the session has no clinical match, so an
    unmatched endpoint still yields a usable (if wider) interval.
    """
    day = _as_optional_int(master_row.get(column)) if master_row else None
    if day is None:
        day = _as_optional_int(fallback)
    return 0 if day is None else day


def _as_optional_int(value: Any) -> int | None:
    """Best-effort int, or ``None``."""
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
