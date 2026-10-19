"""Assembly of the supervised-learning datasets and their audits.

Produces four artefacts from the clinical-radiomics session table plus the
clinical visit history:

``supervised_radiomics_sessions.csv``
    Target A - one row per MRI session, CN/MCI/AD plus every radiomic feature.
``supervised_mci_progression.csv``
    Target B - MCI sessions with MCI_TO_AD / MCI_STABLE / CENSORED.
``diagnosis_vocabulary.csv``
    every unique dx1..dx5 string, its count and how the policy mapped it.
``supervised_label_audit.json``
    session **and subject** counts, plus the run's parameters.

No machine learning, no feature selection: features are preserved as they are.
Raw clinical and radiomic inputs are read-only.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import __version__
from .clinical.labels import (
    ALL_LABELS,
    LABEL_CONFLICTING,
    LABEL_DEMENTIA_UNKNOWN_ETIOLOGY,
    LABEL_IMPAIRED_NOT_MCI,
    FUTURE_INFORMATION_COLUMNS,
    LABEL_AD,
    LABEL_CN,
    LABEL_MCI,
    LABEL_NON_DIAGNOSTIC,
    LABEL_OTHER_DEMENTIA,
    LABEL_UNCERTAIN,
    LABEL_UNMAPPED,
    PROGRESSION_CENSORED,
    PROGRESSION_STABLE,
    PROGRESSION_TO_AD,
    STATUS_CONFLICTING,
    LabelPolicy,
    derive_current_label,
    derive_progression_label,
    leaking_columns,
    training_eligibility,
)
from .tables import read_csv_rows, write_csv

logger = logging.getLogger(__name__)

SESSIONS_CSV = "supervised_radiomics_sessions.csv"
PROGRESSION_CSV = "supervised_mci_progression.csv"
VOCABULARY_CSV = "diagnosis_vocabulary.csv"
AUDIT_JSON = "supervised_label_audit.json"

DEFAULT_CLINICAL_WINDOW_DAYS = 180
#: ~3 years. The horizon over which MCI->AD conversion is considered observable.
DEFAULT_PROGRESSION_HORIZON_DAYS = 1095

#: Session identity and clinical context. Never predictors, never leakage.
METADATA_COLUMNS = (
    "subject_id",
    "session_id",
    "mri_day",
    "clinical_session_id",
    "clinical_day",
    "clinical_mri_gap_days",
    "clinical_mri_abs_gap_days",
    "clinical_match_valid",
    "clinical_match_found",
    "clinical_match_reason",
    "clinical_match_ambiguous",
    "clinical_source",
    "clinical_master_found",
    "days_from_reference",
    "age_at_clinical_visit",
    "cognitive_session_id",
    "cognitive_day",
    "cognitive_mri_gap_days",
    "cognitive_mri_abs_gap_days",
    "cognitive_match_valid",
    "cognitive_match_reason",
)

#: Derived-label bookkeeping.
LABEL_COLUMNS = (
    "supervised_label",
    "label_source",
    "label_rule_id",
    "label_status",
    "label_confidence",
    "label_reason",
    "label_policy_version",
    "dx1_normalized",
    "ad_etiology",
    "ad_etiology_role",
    "mci_subtype",
    "mci_domains",
    "b4_label",
    "b4_agreement",
    "b4_disagreement_reason",
    "label_warnings",
    "training_eligible",
    "training_exclusion_reason",
)

#: Clinical covariates: candidate model inputs, kept separately identifiable so
#: radiomics-only / clinical-only / combined experiments can be compared.
COVARIATE_COLUMNS = ("age_at_mri", "sex", "scanner", "CDRTOT", "CDRSUM", "MMSE")

#: Raw diagnosis strings, preserved next to the derived label.
DIAGNOSIS_COLUMNS = (
    "dx1", "dx2", "dx3", "dx4", "dx5",
    "dx1_code", "dx2_code", "dx3_code", "dx4_code", "dx5_code",
)

SESSIONS_LEADING_COLUMNS = (
    "subject_id", "session_id", "mri_day",
    "clinical_day", "clinical_mri_gap_days", "clinical_match_valid",
    "supervised_label", "training_eligible", "training_exclusion_reason",
    "label_status", "label_source", "label_rule_id", "label_reason",
    "label_policy_version", "ad_etiology", "ad_etiology_role",
    "mci_subtype", "mci_domains",
    "b4_label", "b4_agreement", "b4_disagreement_reason",
    *DIAGNOSIS_COLUMNS,
    "CDRTOT", "CDRSUM", "MMSE", "age_at_mri", "sex", "scanner",
)

PROGRESSION_LEADING_COLUMNS = (
    "subject_id", "session_id", "mri_day",
    "diagnosis_at_mri", "progression_label",
    "progression_eligible", "progression_exclusion_reason",
    "conversion_event", "conversion_day", "days_to_conversion",
    "last_followup_day", "followup_days_after_mri", "progression_horizon_days",
    "progression_reason",
)

VOCABULARY_LEADING_COLUMNS = (
    "source_column", "raw_diagnosis", "normalized_diagnosis",
    "count_sessions", "count_subjects", "mapped_label", "rule_id", "mapping_status",
)


class SupervisedDatasetError(RuntimeError):
    """Raised when the supervised datasets cannot be produced."""


@dataclass
class SupervisedDatasetResult:
    """Everything the supervised-label stage produced."""

    session_rows: list[dict[str, Any]] = field(default_factory=list)
    progression_rows: list[dict[str, Any]] = field(default_factory=list)
    vocabulary_rows: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# column roles
# ---------------------------------------------------------------------------
def predictor_columns(columns: Iterable[str]) -> list[str]:
    """Columns a model may legitimately consume as ``x``.

    Everything that is not session metadata, a derived label, a raw diagnosis
    string, or post-scan information. In practice this is the radiomic feature
    block plus the clinical covariates.
    """
    excluded = (
        set(METADATA_COLUMNS)
        | set(LABEL_COLUMNS)
        | set(DIAGNOSIS_COLUMNS)
        | set(FUTURE_INFORMATION_COLUMNS)
        | {"diagnosis_at_mri"}
    )
    return [column for column in columns if column not in excluded]


def subject_groups(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Grouping vector for subject-wise cross-validation.

    This project is longitudinal: one participant contributes several MRI
    sessions. Splitting by row would place ``OAS30001`` scans in both train and
    test - subject leakage. Pass this to ``GroupKFold``/``GroupShuffleSplit`` as
    ``groups``.
    """
    return [str(row["subject_id"]) for row in rows]


def assert_no_subject_leakage(
    train_subjects: Iterable[str], test_subjects: Iterable[str]
) -> None:
    """Raise if any subject appears on both sides of a split."""
    overlap = sorted(set(train_subjects) & set(test_subjects))
    if overlap:
        raise SupervisedDatasetError(
            f"Subject leakage: {len(overlap)} subject(s) in both train and test "
            f"(e.g. {overlap[:5]}). Split by subject_id, never by session row."
        )


# ---------------------------------------------------------------------------
# Target A
# ---------------------------------------------------------------------------
def build_session_labels(
    session_rows: Sequence[Mapping[str, Any]],
    policy: LabelPolicy,
    window_days: int = DEFAULT_CLINICAL_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Attach Target-A labels to every clinical-radiomics session row.

    Every input row produces exactly one output row, including sessions that are
    excluded from training: exclusions are recorded, never dropped.
    """
    if not session_rows:
        raise SupervisedDatasetError("No clinical-radiomics session rows to label.")

    labelled: list[dict[str, Any]] = []
    for source in session_rows:
        label, warnings = derive_current_label(source, policy)
        eligible, exclusion = training_eligibility(
            label,
            clinical_match_valid=_as_bool(source.get("clinical_match_valid")),
            abs_gap_days=_as_float(source.get("clinical_mri_abs_gap_days")),
            window_days=window_days,
            training_labels=policy.training_labels,
        )

        row = dict(source)
        row.update(label.as_row())
        row["label_warnings"] = "; ".join(warnings)
        row["training_eligible"] = eligible
        row["training_exclusion_reason"] = exclusion
        labelled.append(row)

    _log_label_distribution(labelled)
    return labelled


def _log_label_distribution(rows: Sequence[Mapping[str, Any]]) -> None:
    """Log the resulting class distribution."""
    counts = Counter(row["supervised_label"] for row in rows)
    eligible = sum(1 for row in rows if row["training_eligible"])
    logger.info(
        "Target A: %d session(s) -> %s; %d training-eligible.",
        len(rows),
        dict(sorted(counts.items())),
        eligible,
    )


# ---------------------------------------------------------------------------
# Target B
# ---------------------------------------------------------------------------
def build_progression_labels(
    session_rows: Sequence[Mapping[str, Any]],
    visit_rows: Sequence[Mapping[str, Any]],
    policy: LabelPolicy,
    horizon_days: int = DEFAULT_PROGRESSION_HORIZON_DAYS,
) -> list[dict[str, Any]]:
    """Derive Target B for the MCI sessions.

    Only sessions whose Target-A label is ``MCI`` and which are training
    eligible take part. Future clinical visits define ``y`` and are deliberately
    **not** carried into the predictor block.
    """
    visits_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for visit in visit_rows:
        subject = str(visit.get("subject_id"))
        if visit.get("clinical_day") not in (None, ""):
            visits_by_subject[subject].append(visit)

    rows: list[dict[str, Any]] = []
    for source in session_rows:
        if source.get("supervised_label") != LABEL_MCI or not source.get("training_eligible"):
            continue

        subject = str(source["subject_id"])
        mri_day = int(float(source["mri_day"]))
        progression = derive_progression_label(
            mri_day=mri_day,
            current_label=LABEL_MCI,
            future_visits=visits_by_subject.get(subject, []),
            policy=policy,
            horizon_days=horizon_days,
        )

        row: dict[str, Any] = {
            "subject_id": subject,
            "session_id": source["session_id"],
            "mri_day": mri_day,
            "diagnosis_at_mri": LABEL_MCI,
            "clinical_day": source.get("clinical_day"),
            "label_policy_version": policy.version,
        }
        row.update(progression.as_row())
        # Predictors travel with the row; future-information columns do not.
        for column in predictor_columns(source.keys()):
            row.setdefault(column, source[column])
        rows.append(row)

    counts = Counter(row["progression_label"] for row in rows)
    logger.info(
        "Target B: %d MCI session(s) -> %s.", len(rows), dict(sorted(counts.items(), key=str))
    )
    return rows


# ---------------------------------------------------------------------------
# vocabulary audit
# ---------------------------------------------------------------------------
def build_diagnosis_vocabulary(
    rows: Sequence[Mapping[str, Any]], policy: LabelPolicy
) -> list[dict[str, Any]]:
    """Every unique dx1..dx5 value with its counts and mapping outcome.

    Exists so a reviewer can audit the whole label policy against the data in
    one table. No observed string may be absent from it.
    """
    columns = (policy.primary_column, *policy.secondary_columns)
    sessions: dict[tuple[str, str], int] = Counter()
    subjects: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        subject = str(row.get("subject_id"))
        for column in columns:
            raw = row.get(column)
            if policy.normalise(raw) is None:
                continue
            key = (column, str(raw))
            sessions[key] += 1
            subjects[key].add(subject)

    vocabulary: list[dict[str, Any]] = []
    for (column, raw), count in sessions.items():
        normalized = policy.normalise(raw)
        is_primary = column == policy.primary_column
        matched = (
            policy.primary_map.get(normalized)
            if is_primary
            else policy.secondary_map.get(normalized)
        )
        vocabulary.append(
            {
                "source_column": column,
                "raw_diagnosis": raw,
                "normalized_diagnosis": normalized,
                "count_sessions": count,
                "count_subjects": len(subjects[(column, raw)]),
                "mapped_label": matched[0] if matched else LABEL_UNMAPPED,
                "rule_id": matched[1] if matched else "unmapped",
                "mapping_status": "mapped" if matched else "UNMAPPED",
                "is_primary_column": is_primary,
            }
        )

    vocabulary.sort(key=lambda item: (item["source_column"], -item["count_sessions"]))
    unmapped = [item for item in vocabulary if item["mapping_status"] == "UNMAPPED"]
    if unmapped:
        logger.warning(
            "%d diagnosis string(s) are UNMAPPED by policy %s; review "
            "diagnosis_vocabulary.csv before treating the dataset as final.",
            len(unmapped),
            policy.version,
        )
        for item in unmapped[:20]:
            logger.warning(
                "  UNMAPPED %s=%r (%d session(s), %d subject(s))",
                item["source_column"],
                item["raw_diagnosis"],
                item["count_sessions"],
                item["count_subjects"],
            )
    else:
        logger.info("All %d diagnosis string(s) are mapped by the policy.", len(vocabulary))
    return vocabulary


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _subjects_with(rows: Sequence[Mapping[str, Any]], predicate) -> int:
    """Number of distinct subjects having at least one row satisfying ``predicate``."""
    return len({str(row["subject_id"]) for row in rows if predicate(row)})


def build_audit(
    session_rows: Sequence[Mapping[str, Any]],
    progression_rows: Sequence[Mapping[str, Any]],
    vocabulary_rows: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble ``supervised_label_audit.json``, by session **and** by subject."""
    by_label = Counter(row["supervised_label"] for row in session_rows)
    by_progression = Counter(row["progression_label"] for row in progression_rows)

    def label_is(label: str):
        return lambda row: row["supervised_label"] == label

    def progression_is(label: str):
        return lambda row: row["progression_label"] == label

    unmapped_strings = [
        {
            "source_column": item["source_column"],
            "raw_diagnosis": item["raw_diagnosis"],
            "count_sessions": item["count_sessions"],
            "count_subjects": item["count_subjects"],
        }
        for item in vocabulary_rows
        if item["mapping_status"] == "UNMAPPED"
    ]

    audit = {
        "parameters": dict(parameters),
        "sessions_total": len(session_rows),
        "subjects_total": len({str(row["subject_id"]) for row in session_rows}),
        "valid_clinical_match": sum(
            1 for row in session_rows if _as_bool(row.get("clinical_match_valid"))
        ),
        "CN_sessions": by_label.get(LABEL_CN, 0),
        "MCI_sessions": by_label.get(LABEL_MCI, 0),
        "AD_sessions": by_label.get(LABEL_AD, 0),
        "other_dementia_sessions": by_label.get(LABEL_OTHER_DEMENTIA, 0),
        "impaired_not_mci_sessions": by_label.get(LABEL_IMPAIRED_NOT_MCI, 0),
        "dementia_unknown_etiology_sessions": by_label.get(LABEL_DEMENTIA_UNKNOWN_ETIOLOGY, 0),
        "uncertain_sessions": by_label.get(LABEL_UNCERTAIN, 0),
        "unmapped_sessions": by_label.get(LABEL_UNMAPPED, 0),
        "non_diagnostic_sessions": by_label.get(LABEL_NON_DIAGNOSTIC, 0),
        "missing_diagnosis_sessions": by_label.get("MISSING", 0),
        "conflicting_sessions": sum(
            1 for row in session_rows if row.get("label_status") == STATUS_CONFLICTING
        ),
        "b4_agreement": dict(
            sorted(Counter(str(row.get("b4_agreement")) for row in session_rows).items())
        ),
        "ad_etiology": dict(
            sorted(Counter(str(row.get("ad_etiology")) for row in session_rows).items())
        ),
        "ad_etiology_role": dict(
            sorted(Counter(str(row.get("ad_etiology_role")) for row in session_rows).items())
        ),
        "mci_subtype": dict(
            sorted(
                Counter(
                    str(row.get("mci_subtype"))
                    for row in session_rows
                    if row.get("mci_subtype")
                ).items()
            )
        ),
        "b4_disagreements": dict(
            sorted(
                Counter(
                    f"D1={row.get('supervised_label')} vs B4={row.get('b4_label')}"
                    for row in session_rows
                    if row.get("b4_agreement") == "disagree"
                ).items(),
                key=lambda item: -item[1],
            )
        ),
        "CN_subjects": _subjects_with(session_rows, label_is(LABEL_CN)),
        "MCI_subjects": _subjects_with(session_rows, label_is(LABEL_MCI)),
        "AD_subjects": _subjects_with(session_rows, label_is(LABEL_AD)),
        "other_dementia_subjects": _subjects_with(session_rows, label_is(LABEL_OTHER_DEMENTIA)),
        "impaired_not_mci_subjects": _subjects_with(session_rows, label_is(LABEL_IMPAIRED_NOT_MCI)),
        "dementia_unknown_etiology_subjects": _subjects_with(
            session_rows, label_is(LABEL_DEMENTIA_UNKNOWN_ETIOLOGY)
        ),
        "uncertain_subjects": _subjects_with(session_rows, label_is(LABEL_UNCERTAIN)),
        "unmapped_subjects": _subjects_with(session_rows, label_is(LABEL_UNMAPPED)),
        "training_eligible_sessions": sum(
            1 for row in session_rows if row.get("training_eligible")
        ),
        "training_eligible_subjects": _subjects_with(
            session_rows, lambda row: bool(row.get("training_eligible"))
        ),
        "exclusion_reasons": dict(
            sorted(
                Counter(
                    row["training_exclusion_reason"]
                    for row in session_rows
                    if row.get("training_exclusion_reason")
                ).items()
            )
        ),
        "label_warnings": dict(
            sorted(
                Counter(
                    warning
                    for row in session_rows
                    for warning in str(row.get("label_warnings") or "").split("; ")
                    if warning
                ).items()
            )
        ),
        "mci_progression_candidates": len(progression_rows),
        "mci_to_ad": by_progression.get(PROGRESSION_TO_AD, 0),
        "mci_stable": by_progression.get(PROGRESSION_STABLE, 0),
        "censored": by_progression.get(PROGRESSION_CENSORED, 0),
        "mci_to_ad_subjects": _subjects_with(progression_rows, progression_is(PROGRESSION_TO_AD)),
        "mci_stable_subjects": _subjects_with(progression_rows, progression_is(PROGRESSION_STABLE)),
        "censored_subjects": _subjects_with(progression_rows, progression_is(PROGRESSION_CENSORED)),
        "diagnosis_strings_total": len(vocabulary_rows),
        "diagnosis_strings_unmapped": len(unmapped_strings),
        "unmapped_diagnosis_strings": unmapped_strings,
        "dataset_is_final": not unmapped_strings and by_label.get(LABEL_MCI, 0) > 0,
        "notes": [
            "Split by subject_id, never by session row: one participant contributes "
            "several MRI sessions and row-wise splitting leaks.",
            "CENSORED is not MCI_STABLE. A participant lost to follow-up before the "
            "horizon has an unobservable outcome and is ineligible.",
            "Radiomic deltas (t0->t1) may only predict outcomes after t1, and "
            "full-history subject slopes must not be used for conversion "
            "prediction: they may include sessions recorded after the conversion.",
        ],
    }
    return audit


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def build_supervised_datasets(
    clinical_radiomics: Path,
    clinical_visits: Path,
    output_dir: Path,
    policy: LabelPolicy | None = None,
    window_days: int = DEFAULT_CLINICAL_WINDOW_DAYS,
    horizon_days: int = DEFAULT_PROGRESSION_HORIZON_DAYS,
) -> tuple[SupervisedDatasetResult, dict[str, Path]]:
    """Build and write all four supervised-label artefacts."""
    policy = policy or LabelPolicy.load(None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_source = read_csv_rows(Path(clinical_radiomics))
    if not session_source:
        raise SupervisedDatasetError(
            f"Clinical-radiomics session table is empty: {clinical_radiomics}"
        )
    visit_source = read_csv_rows(Path(clinical_visits))
    if not visit_source:
        raise SupervisedDatasetError(f"Clinical visits table is empty: {clinical_visits}")

    session_rows = build_session_labels(session_source, policy, window_days)
    progression_rows = build_progression_labels(session_rows, visit_source, policy, horizon_days)
    vocabulary_rows = build_diagnosis_vocabulary(session_source, policy)

    leaks = leaking_columns(predictor_columns(session_rows[0].keys()))
    if leaks:
        raise SupervisedDatasetError(
            f"Post-scan columns reached the predictor block: {leaks}. This would leak "
            "the outcome into x."
        )

    parameters = {
        "label_policy_version": policy.version,
        "label_policy_path": str(policy.source_path) if policy.source_path else None,
        "policy_defines_mci": policy.defines_mci,
        "clinical_window_days": window_days,
        "progression_horizon_days": horizon_days,
        "oasis_radiomics_version": __version__,
        "clinical_radiomics": str(Path(clinical_radiomics).resolve()),
        "clinical_visits": str(Path(clinical_visits).resolve()),
        "n_predictor_columns": len(predictor_columns(session_rows[0].keys())),
    }
    audit = build_audit(session_rows, progression_rows, vocabulary_rows, parameters)

    outputs = {
        "supervised_radiomics_sessions": write_csv(
            session_rows, output_dir / SESSIONS_CSV, SESSIONS_LEADING_COLUMNS
        ),
        "supervised_mci_progression": write_csv(
            progression_rows, output_dir / PROGRESSION_CSV, PROGRESSION_LEADING_COLUMNS
        ),
        "diagnosis_vocabulary": write_csv(
            vocabulary_rows, output_dir / VOCABULARY_CSV, VOCABULARY_LEADING_COLUMNS
        ),
    }
    audit_path = output_dir / AUDIT_JSON
    with audit_path.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, default=str)
        handle.write("\n")
    outputs["supervised_label_audit"] = audit_path
    logger.info("Wrote supervised label audit: %s", audit_path)

    if not audit["dataset_is_final"]:
        logger.warning(
            "Dataset is NOT scientifically final: %d unmapped diagnosis string(s), "
            "%d MCI session(s).",
            audit["diagnosis_strings_unmapped"],
            audit["MCI_sessions"],
        )

    result = SupervisedDatasetResult(
        session_rows=session_rows,
        progression_rows=progression_rows,
        vocabulary_rows=vocabulary_rows,
        audit=audit,
        parameters=parameters,
    )
    return result, outputs


def _as_bool(value: Any) -> bool:
    """Parse the booleans pandas writes as ``True``/``False`` strings."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def _as_float(value: Any) -> float | None:
    """Float or ``None``."""
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None
