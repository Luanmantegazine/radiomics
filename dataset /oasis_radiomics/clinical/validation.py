"""Auditable validation of the clinical linkage.

Nothing here removes or repairs data. Every check produces a
:class:`ValidationIssue`, and the whole set is written to
``clinical_linkage_validation.json`` so a reviewer can audit exactly which
sessions were linked, which were not, and why.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

VALIDATION_FILENAME = "clinical_linkage_validation.json"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"

#: Issue codes this module can emit, documented in CLINICAL_LINKAGE_PROTOCOL.md.
ISSUE_CODES = (
    "duplicate_clinical_visit",
    "duplicate_mri_session",
    "mri_subject_mismatch",
    "invalid_day_value",
    "implausible_day_value",
    "missing_d1",
    "missing_b4",
    "no_clinical_visit",
    "outside_window",
    "ambiguous_equal_distance",
    "non_monotonic_trajectory",
    "conflicting_diagnosis_same_day",
    "duplicate_master_row",
)


@dataclass(frozen=True)
class ValidationIssue:
    """A single auditable finding."""

    code: str
    severity: str
    detail: str
    subject_id: str | None = None
    identifier: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_mri_sessions(sessions: Sequence[Any]) -> list[ValidationIssue]:
    """Check MRI catalogue integrity: duplicate ids and subject/label mismatch."""
    issues: list[ValidationIssue] = []
    seen: dict[str, Any] = {}

    for session in sessions:
        if session.session_id in seen:
            issues.append(
                ValidationIssue(
                    code="duplicate_mri_session",
                    severity=SEVERITY_WARNING,
                    subject_id=session.subject_id,
                    identifier=session.session_id,
                    detail="MRI session id appears more than once in the catalogue.",
                )
            )
        seen[session.session_id] = session

        declared = session.raw.get("Subject") if session.raw else None
        if declared and str(declared) != session.subject_id:
            issues.append(
                ValidationIssue(
                    code="mri_subject_mismatch",
                    severity=SEVERITY_ERROR,
                    subject_id=session.subject_id,
                    identifier=session.session_id,
                    detail=(
                        f"Catalogue 'Subject' column says {declared!r} but the session "
                        f"label encodes {session.subject_id!r}."
                    ),
                )
            )
    return issues


def check_raw_day_values(path: Path, day_column: str = "days_to_visit") -> list[ValidationIssue]:
    """Report rows of a clinical CSV whose day value is unusable.

    Two distinct problems, kept apart because they have different consequences:

    ``invalid_day_value``
        the value cannot be parsed at all, so the row never becomes a visit;
    ``implausible_day_value``
        the value parses but is negative. ``days_to_visit`` is measured forward
        from the participant's entry into the study, so a negative day cannot be
        real. Observed in OASIS-3 (e.g. ``OAS30753_UDSb4_d-39520``). Such rows
        are **kept** - they are flagged here and will simply never fall inside a
        matching window.
    """
    import csv

    from .readers import parse_day

    path = Path(path)
    issues: list[ValidationIssue] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            if day_column not in row:
                break
            day = parse_day(row.get(day_column))
            if day is None:
                issues.append(
                    ValidationIssue(
                        code="invalid_day_value",
                        severity=SEVERITY_WARNING,
                        subject_id=row.get("OASISID"),
                        identifier=f"{path.name}:{line_number}",
                        detail=f"Unparsable {day_column}={row.get(day_column)!r}; row skipped.",
                    )
                )
            elif day < 0:
                issues.append(
                    ValidationIssue(
                        code="implausible_day_value",
                        severity=SEVERITY_WARNING,
                        subject_id=row.get("OASISID"),
                        identifier=row.get("OASIS_session_label") or f"{path.name}:{line_number}",
                        detail=(
                            f"{day_column}={day} is negative; days are measured forward from "
                            "study entry. Row kept and flagged, not removed."
                        ),
                    )
                )
    return issues


def check_matches(matches: Sequence[Any]) -> list[ValidationIssue]:
    """Turn unmatched, out-of-window and ambiguous links into issues."""
    issues: list[ValidationIssue] = []
    for match in matches:
        if match.reason == "no_clinical_visit":
            issues.append(
                ValidationIssue(
                    code="no_clinical_visit",
                    severity=SEVERITY_WARNING,
                    subject_id=match.subject_id,
                    identifier=match.session_id,
                    detail="Subject has no clinical visit in D1 or B4.",
                )
            )
        elif match.reason == "outside_window":
            issues.append(
                ValidationIssue(
                    code="outside_window",
                    severity=SEVERITY_WARNING,
                    subject_id=match.subject_id,
                    identifier=match.session_id,
                    detail=(
                        f"Nearest clinical visit is {match.abs_gap_days} day(s) away, "
                        "beyond the configured window; kept with valid=False."
                    ),
                )
            )
        if match.ambiguous:
            issues.append(
                ValidationIssue(
                    code="ambiguous_equal_distance",
                    severity=SEVERITY_WARNING,
                    subject_id=match.subject_id,
                    identifier=match.session_id,
                    detail=(
                        f"Two clinical visits are {match.abs_gap_days} day(s) from the MRI; "
                        "the earlier one was selected deterministically."
                    ),
                )
            )
    return issues


def check_master_rows(rows: Sequence[Mapping[str, Any]]) -> list[ValidationIssue]:
    """Ensure the master table has exactly one row per subject + session."""
    counts = Counter((row.get("subject_id"), row.get("session_id")) for row in rows)
    return [
        ValidationIssue(
            code="duplicate_master_row",
            severity=SEVERITY_ERROR,
            subject_id=subject_id,
            identifier=session_id,
            detail=f"{count} rows share this subject_id + session_id.",
        )
        for (subject_id, session_id), count in sorted(counts.items(), key=lambda item: str(item[0]))
        if count > 1
    ]


def summarise(
    matches: Sequence[Any],
    issues: Sequence[ValidationIssue],
    gap_stats: Mapping[str, float | None],
) -> dict[str, Any]:
    """Build the machine-readable summary block."""
    return {
        "mri_sessions": len(matches),
        "subjects": len({match.subject_id for match in matches}),
        "matched_sessions": sum(1 for match in matches if match.found),
        "valid_matches": sum(1 for match in matches if match.valid),
        "outside_window": sum(1 for match in matches if match.reason == "outside_window"),
        "without_clinical_data": sum(
            1 for match in matches if match.reason == "no_clinical_visit"
        ),
        "ambiguous_matches": sum(1 for match in matches if match.ambiguous),
        "median_gap_days": gap_stats.get("median_gap_days"),
        "p95_gap_days": gap_stats.get("p95_gap_days"),
        "issues_by_code": dict(sorted(Counter(issue.code for issue in issues).items())),
        "issues_by_severity": dict(
            sorted(Counter(issue.severity for issue in issues).items())
        ),
    }


def write_validation_report(
    path: Path,
    summary: Mapping[str, Any],
    issues: Sequence[ValidationIssue],
    parameters: Mapping[str, Any],
    max_issues: int = 5000,
) -> Path:
    """Write ``clinical_linkage_validation.json``.

    The issue list is capped at ``max_issues`` entries to keep the file
    reviewable; the untruncated counts always remain in ``summary``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "parameters": dict(parameters),
        "summary": dict(summary),
        "issues_truncated": len(issues) > max_issues,
        "issues": [issue.as_dict() for issue in issues[:max_issues]],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")

    logger.info("Wrote clinical linkage validation report: %s", path)
    return path


def log_issue_summary(issues: Iterable[ValidationIssue]) -> None:
    """Log issue counts by code at a level matching their worst severity."""
    issues = list(issues)
    if not issues:
        logger.info("Clinical validation: no issues found.")
        return
    for code, count in sorted(Counter(issue.code for issue in issues).items()):
        severity = max(
            (issue.severity for issue in issues if issue.code == code),
            key=lambda value: (SEVERITY_INFO, SEVERITY_WARNING, SEVERITY_ERROR).index(value),
        )
        log = logger.error if severity == SEVERITY_ERROR else logger.warning
        log("Clinical validation: %s x%d", code, count)
