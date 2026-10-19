"""Typed records for the clinical layer.

Every structure keeps its ``raw`` source mapping so that no OASIS variable is
lost on the way to the output tables. Derived values live in dedicated fields;
raw values are never overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

# --- controlled vocabularies ----------------------------------------------
#: Derived cognitive status. UNKNOWN is used whenever the codebook cannot
#: resolve the record - it is never a synonym for "normal".
COGNITIVE_STATUS_VALUES = ("CN", "MCI", "DEMENTIA", "UNKNOWN")

#: Derived aetiology of an impairment, independent of its severity.
AD_ETIOLOGY_VALUES = ("AD", "NON_AD", "UNCERTAIN", "UNKNOWN")

#: Outcome of :func:`~oasis_radiomics.clinical.classification.classify_clinical_visit`.
CLASSIFICATION_STATUS_VALUES = (
    "classified",           # a frozen codebook resolved the record
    "unresolved_codebook",  # no frozen codebook: raw values preserved, no derivation
    "no_clinical_data",     # there is no clinical visit to classify
    "conflicting",          # the codebook produced contradictory results
)

#: Why an MRI session did or did not receive a clinical match.
MATCH_REASON_VALUES = (
    "matched",
    "outside_window",
    "no_clinical_visit",
    "ambiguous_equal_distance",
)

SOURCE_D1 = "d1"
SOURCE_B4 = "b4"
SOURCE_C1 = "c1"
SOURCE_D1_B4 = "d1+b4"


@dataclass(frozen=True)
class MriSession:
    """One MRI session from the OASIS demographic catalogue."""

    subject_id: str
    session_id: str
    mri_day: int
    sex: str | None = None
    age_at_mri: float | None = None
    scanner: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the master table."""
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "mri_day": self.mri_day,
            "sex": self.sex,
            "age_at_mri": self.age_at_mri,
            "scanner": self.scanner,
        }


@dataclass(frozen=True)
class ClinicalRecord:
    """One row of a single clinical instrument (D1, B4 or C1).

    ``clinical_day`` is the instrument's ``days_to_visit``, normalised to an
    integer; the source files store it inconsistently (D1 zero-pads it, B4 does
    not).
    """

    subject_id: str
    clinical_session_id: str
    clinical_day: int
    source: str
    age_at_visit: float | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def value(self, column: str) -> Any:
        """Raw value of ``column``, or ``None`` when absent/blank."""
        return self.raw.get(column)


@dataclass(frozen=True)
class ClinicalVisit:
    """A clinical visit after D1 and B4 have been merged on subject + day.

    Either side may be missing: a D1 diagnosis is never discarded because B4 is
    absent, and vice versa.
    """

    subject_id: str
    clinical_day: int
    d1_session_id: str | None = None
    b4_session_id: str | None = None
    age_at_clinical_visit: float | None = None
    d1: ClinicalRecord | None = field(default=None, repr=False)
    b4: ClinicalRecord | None = field(default=None, repr=False)

    @property
    def source(self) -> str:
        """Which instruments contributed to this visit."""
        if self.d1 is not None and self.b4 is not None:
            return SOURCE_D1_B4
        if self.d1 is not None:
            return SOURCE_D1
        if self.b4 is not None:
            return SOURCE_B4
        return "none"

    @property
    def has_d1(self) -> bool:
        return self.d1 is not None

    @property
    def has_b4(self) -> bool:
        return self.b4 is not None

    def d1_value(self, column: str) -> Any:
        """Raw D1 variable, or ``None`` when D1 is missing for this visit."""
        return self.d1.value(column) if self.d1 is not None else None

    def b4_value(self, column: str) -> Any:
        """Raw B4 variable, or ``None`` when B4 is missing for this visit."""
        return self.b4.value(column) if self.b4 is not None else None


@dataclass(frozen=True)
class ClinicalClassification:
    """Derived interpretation of a clinical visit.

    ``status`` records *how* the values were obtained; ``UNKNOWN`` fields paired
    with ``status='unresolved_codebook'`` mean "not interpretable yet", never
    "normal".
    """

    cognitive_status: str = "UNKNOWN"
    ad_etiology: str = "UNKNOWN"
    status: str = "unresolved_codebook"
    confidence: str = "none"
    reason: str = ""
    codebook_version: str | None = None

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the master table."""
        return {
            "cognitive_status": self.cognitive_status,
            "ad_etiology": self.ad_etiology,
            "classification_status": self.status,
            "classification_confidence": self.confidence,
            "classification_reason": self.reason,
            "classification_version": self.codebook_version,
        }


@dataclass(frozen=True)
class ClinicalMatch:
    """Result of linking one MRI session to at most one clinical visit."""

    session_id: str
    subject_id: str
    mri_day: int
    visit: ClinicalVisit | None = field(default=None, repr=False)
    gap_days: int | None = None
    abs_gap_days: int | None = None
    reason: str = "no_clinical_visit"
    valid: bool = False
    ambiguous: bool = False
    candidates_considered: int = 0

    @property
    def found(self) -> bool:
        """Whether any clinical visit was selected, in or out of the window."""
        return self.visit is not None

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the master table."""
        return {
            "clinical_session_id": (
                self.visit.d1_session_id or self.visit.b4_session_id
                if self.visit is not None
                else None
            ),
            "clinical_day": self.visit.clinical_day if self.visit is not None else None,
            "clinical_mri_gap_days": self.gap_days,
            "clinical_mri_abs_gap_days": self.abs_gap_days,
            "clinical_match_found": self.found,
            "clinical_match_valid": self.valid,
            "clinical_match_reason": self.reason,
            "clinical_match_ambiguous": self.ambiguous,
            "clinical_source": self.visit.source if self.visit is not None else None,
            "age_at_clinical_visit": (
                self.visit.age_at_clinical_visit if self.visit is not None else None
            ),
        }


@dataclass(frozen=True)
class CognitiveMatch:
    """Result of linking one MRI session to at most one C1 assessment.

    Kept separate from :class:`ClinicalMatch` on purpose: psychometric testing
    often happens on a different day from the diagnostic visit, so forcing them
    onto one day would silently distort both gaps.
    """

    session_id: str
    record: ClinicalRecord | None = field(default=None, repr=False)
    gap_days: int | None = None
    abs_gap_days: int | None = None
    valid: bool = False
    reason: str = "no_clinical_visit"

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the master table."""
        return {
            "cognitive_session_id": self.record.clinical_session_id if self.record else None,
            "cognitive_day": self.record.clinical_day if self.record else None,
            "cognitive_mri_gap_days": self.gap_days,
            "cognitive_mri_abs_gap_days": self.abs_gap_days,
            "cognitive_match_valid": self.valid,
            "cognitive_match_reason": self.reason,
        }


@dataclass(frozen=True)
class TrajectoryPoint:
    """One dated diagnosis in a subject's clinical history."""

    clinical_day: int
    cognitive_status: str
    ad_etiology: str
    classification_status: str


@dataclass(frozen=True)
class SubjectTrajectory:
    """A subject's ordered clinical history and its conversion event, if any.

    ``conversion_day`` is the day of the **first** visit showing the converted
    status. Everything here is derived strictly from ``clinical_day`` ordering.
    """

    subject_id: str
    points: tuple[TrajectoryPoint, ...] = ()
    baseline_diagnosis: str = "UNKNOWN"
    last_diagnosis: str = "UNKNOWN"
    trajectory: str = "UNKNOWN"
    #: Every progression in the history, ordered: ``((event, day), ...)``.
    #: A subject can convert more than once (CN->MCI, then MCI->AD).
    conversions: tuple[tuple[str, int], ...] = ()
    conversion_event: str | None = None
    conversion_day: int | None = None
    n_clinical_visits: int = 0
    followup_years: float | None = None

    def next_conversion_after(self, day: int) -> tuple[str, int] | None:
        """First progression occurring strictly after ``day``.

        This is what an MRI session needs: the conversion it could be used to
        predict, not one that had already happened when the scan was taken.
        """
        for event, conversion_day in self.conversions:
            if conversion_day > day:
                return event, conversion_day
        return None
