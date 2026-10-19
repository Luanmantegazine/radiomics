"""Per-subject clinical trajectories and leakage-safe MRI annotation.

The single most important rule in this module:

    **A diagnosis recorded after an MRI never relabels that MRI.**

An MRI acquired while a participant was MCI stays MCI even if the participant
converts to AD three years later. The later information is exposed separately
(``future_diagnosis``, ``conversion_event``, ``days_to_conversion``) so that a
downstream analysis can *predict* conversion without having been trained on the
answer.

Module name
-----------
Called ``trajectories`` rather than ``longitudinal`` to avoid confusion with
:mod:`oasis_radiomics.longitudinal`, which handles radiomic feature deltas and
slopes and is a different concern.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from .models import (
    ClinicalMatch,
    ClinicalVisit,
    SubjectTrajectory,
    TrajectoryPoint,
)
from .validation import SEVERITY_WARNING, ValidationIssue

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 365.25

#: Severity ordering used to decide what counts as a progression.
#: UNKNOWN is deliberately absent: it is not a point on this scale.
SEVERITY_ORDER = ("CN", "MCI", "DEMENTIA")

UNKNOWN = "UNKNOWN"


def diagnosis_label(cognitive_status: str, ad_etiology: str) -> str:
    """Combine status and etiology into the label used in trajectory strings.

    This is a **naming convention**, not a clinical inference: it only decides
    how an already-derived pair is written down.

    ==========================  ===============
    (status, etiology)          label
    ==========================  ===============
    ``DEMENTIA`` + ``AD``       ``AD``
    ``DEMENTIA`` + other        ``DEMENTIA``
    anything else               the status itself
    ==========================  ===============
    """
    if cognitive_status == "DEMENTIA" and ad_etiology == "AD":
        return "AD"
    return cognitive_status


def _severity(label: str) -> int | None:
    """Position of a label on the severity scale, or ``None`` if off-scale."""
    if label == "AD":
        return SEVERITY_ORDER.index("DEMENTIA")
    return SEVERITY_ORDER.index(label) if label in SEVERITY_ORDER else None


def build_trajectory(
    subject_id: str,
    visits: Sequence[ClinicalVisit],
    classifications: Mapping[int, Any],
) -> tuple[SubjectTrajectory, list[ValidationIssue]]:
    """Build one subject's ordered clinical history.

    Parameters
    ----------
    subject_id:
        The subject the visits belong to.
    visits:
        That subject's clinical visits, in any order.
    classifications:
        ``{clinical_day: ClinicalClassification}`` for those visits.

    Returns
    -------
    tuple
        ``(trajectory, issues)``. Issues flag non-monotonic histories (an
        improvement in recorded severity), which are reported but never
        "corrected".
    """
    ordered = sorted(visits, key=lambda visit: visit.clinical_day)
    points: list[TrajectoryPoint] = []
    for visit in ordered:
        classification = classifications.get(visit.clinical_day)
        points.append(
            TrajectoryPoint(
                clinical_day=visit.clinical_day,
                cognitive_status=getattr(classification, "cognitive_status", UNKNOWN),
                ad_etiology=getattr(classification, "ad_etiology", UNKNOWN),
                classification_status=getattr(classification, "status", "no_clinical_data"),
            )
        )

    labels = [diagnosis_label(point.cognitive_status, point.ad_etiology) for point in points]
    issues = _check_monotonic(subject_id, points, labels)

    conversions = _all_conversions(points, labels)
    days = [point.clinical_day for point in points]

    trajectory = SubjectTrajectory(
        subject_id=subject_id,
        points=tuple(points),
        baseline_diagnosis=labels[0] if labels else UNKNOWN,
        last_diagnosis=labels[-1] if labels else UNKNOWN,
        trajectory=_trajectory_string(labels),
        conversions=tuple(conversions),
        conversion_event=conversions[0][0] if conversions else None,
        conversion_day=conversions[0][1] if conversions else None,
        n_clinical_visits=len(points),
        followup_years=(max(days) - min(days)) / DAYS_PER_YEAR if days else None,
    )
    return trajectory, issues


def _trajectory_string(labels: Sequence[str]) -> str:
    """Collapse repeats into a readable path, e.g. ``CN -> MCI -> AD``."""
    if not labels:
        return UNKNOWN
    collapsed = [labels[0]]
    for label in labels[1:]:
        if label != collapsed[-1]:
            collapsed.append(label)
    return " -> ".join(collapsed)


def _all_conversions(
    points: Sequence[TrajectoryPoint], labels: Sequence[str]
) -> list[tuple[str, int]]:
    """Every progression to a more severe status, ordered by day.

    A subject can convert more than once (CN->MCI, later MCI->AD), so all of
    them are kept: an MRI taken between the two needs the *second* one, not the
    first. Only transitions between two on-scale labels count, so an ``UNKNOWN``
    gap in the middle of a history never fabricates a conversion.
    """
    conversions: list[tuple[str, int]] = []
    last_label: str | None = None
    last_severity: int | None = None

    for point, label in zip(points, labels):
        severity = _severity(label)
        if severity is None:
            continue
        if last_severity is not None and severity > last_severity:
            conversions.append((f"{last_label}_to_{label}", point.clinical_day))
        last_label, last_severity = label, severity

    return conversions


def _check_monotonic(
    subject_id: str, points: Sequence[TrajectoryPoint], labels: Sequence[str]
) -> list[ValidationIssue]:
    """Flag histories where recorded severity decreases over time."""
    issues: list[ValidationIssue] = []
    previous_label: str | None = None
    previous_severity: int | None = None
    previous_day: int | None = None

    for point, label in zip(points, labels):
        severity = _severity(label)
        if severity is None:
            continue
        if previous_severity is not None and severity < previous_severity:
            issues.append(
                ValidationIssue(
                    code="non_monotonic_trajectory",
                    severity=SEVERITY_WARNING,
                    subject_id=subject_id,
                    identifier=f"d{point.clinical_day:04d}",
                    detail=(
                        f"Recorded status improves from {previous_label} (day {previous_day}) "
                        f"to {label} (day {point.clinical_day}). Reported, not altered."
                    ),
                )
            )
        previous_label, previous_severity, previous_day = label, severity, point.clinical_day

    return issues


def build_trajectories(
    visits_by_subject: Mapping[str, Sequence[ClinicalVisit]],
    classifications_by_subject: Mapping[str, Mapping[int, Any]],
) -> tuple[dict[str, SubjectTrajectory], list[ValidationIssue]]:
    """Build every subject's trajectory."""
    trajectories: dict[str, SubjectTrajectory] = {}
    issues: list[ValidationIssue] = []

    for subject_id, visits in visits_by_subject.items():
        trajectory, subject_issues = build_trajectory(
            subject_id, visits, classifications_by_subject.get(subject_id, {})
        )
        trajectories[subject_id] = trajectory
        issues.extend(subject_issues)

    logger.info("Built clinical trajectories for %d subject(s).", len(trajectories))
    return trajectories, issues


def annotate_session(
    match: ClinicalMatch,
    classification: Any,
    trajectory: SubjectTrajectory | None,
) -> dict[str, Any]:
    """Leakage-safe clinical annotation of one MRI session.

    ``diagnosis_at_mri`` comes **only** from the visit this MRI was linked to.
    Everything prefixed ``future_`` looks strictly at clinical visits occurring
    *after* the MRI day, and is provided so conversion can be predicted - never
    so it can be back-projected onto the scan.
    """
    diagnosis_at_mri = (
        diagnosis_label(
            getattr(classification, "cognitive_status", UNKNOWN),
            getattr(classification, "ad_etiology", UNKNOWN),
        )
        if match.found
        else UNKNOWN
    )

    row: dict[str, Any] = {
        "diagnosis_at_mri": diagnosis_at_mri,
        "future_diagnosis": None,
        "conversion_event": None,
        "conversion_day": None,
        "days_to_conversion": None,
        "clinical_trajectory": trajectory.trajectory if trajectory else UNKNOWN,
    }

    if trajectory is None or not trajectory.points:
        return row

    future_points = [
        point for point in trajectory.points if point.clinical_day > match.mri_day
    ]
    if future_points:
        last = future_points[-1]
        row["future_diagnosis"] = diagnosis_label(last.cognitive_status, last.ad_etiology)

    # Report the NEXT conversion after the scan, not the subject's first one:
    # for an MRI taken after a CN->MCI transition, the predictable event is the
    # later MCI->AD conversion.
    upcoming = trajectory.next_conversion_after(match.mri_day)
    if upcoming is not None:
        event, day = upcoming
        row["conversion_event"] = event
        row["conversion_day"] = day
        row["days_to_conversion"] = day - match.mri_day

    return row


def conversion_between(
    trajectory: SubjectTrajectory | None, day_t0: int, day_t1: int
) -> str | None:
    """Conversion event occurring in the interval ``(day_t0, day_t1]``, if any.

    Used to annotate radiomic deltas between two MRI sessions.
    """
    if trajectory is None:
        return None
    for event, day in trajectory.conversions:
        if day_t0 < day <= day_t1:
            return event
    return None
