"""Readers that normalise the raw OASIS-3 clinical CSVs.

The source files are opened read-only and never rewritten. Normalisation is
limited to what is needed to join the tables reliably:

* ``OASISID``            -> ``subject_id``
* ``OASIS_session_label``-> ``clinical_session_id``
* ``days_to_visit``      -> ``clinical_day`` (int; D1 zero-pads it, B4 does not)
* ``age at visit``       -> ``age_at_visit`` (float)

Every other column is carried through verbatim in :attr:`ClinicalRecord.raw`,
with the OASIS missing-value conventions (``""`` and ``"."``) mapped to ``None``
so that downstream code has a single notion of "absent".
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..ids import SessionIdError, parse_session_id
from .models import SOURCE_B4, SOURCE_C1, SOURCE_D1, ClinicalRecord, MriSession

logger = logging.getLogger(__name__)

SUBJECT_COLUMN = "OASISID"
SESSION_LABEL_COLUMN = "OASIS_session_label"
DAYS_COLUMN = "days_to_visit"
AGE_COLUMN = "age at visit"

#: Strings OASIS uses for "no value recorded".
MISSING_TOKENS = frozenset({"", ".", "NA", "N/A", "NaN", "nan"})

#: Columns of the MRI demographic catalogue export.
CATALOGUE_LABEL_COLUMN = "Label"
CATALOGUE_SUBJECT_COLUMN = "Subject"
CATALOGUE_SEX_COLUMN = "M/F"
CATALOGUE_AGE_COLUMN = "Age"
CATALOGUE_SCANNER_COLUMN = "Scanner"


class ClinicalReaderError(ValueError):
    """Raised when a clinical source file cannot be read or is missing a key column."""


# ---------------------------------------------------------------------------
# scalar normalisation
# ---------------------------------------------------------------------------
def normalise_missing(value: Any) -> Any:
    """Map the OASIS missing-value tokens to ``None``; strip other strings."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped in MISSING_TOKENS else stripped
    return value


def parse_day(value: Any) -> int | None:
    """Parse ``days_to_visit`` into an int.

    Handles the zero-padded form D1 uses (``"0339"``) and the plain form B4 uses
    (``"339"``). Returns ``None`` for missing or unparsable values so the caller
    can raise a validation issue instead of guessing.
    """
    cleaned = normalise_missing(value)
    if cleaned is None:
        return None
    try:
        return int(str(cleaned).strip().lstrip("+"))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    """Parse a float, returning ``None`` for missing or unparsable values."""
    cleaned = normalise_missing(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def normalise_text(value: Any) -> str | None:
    """Collapse whitespace in a free-text clinical value, preserving its wording.

    Used for the B4 ``dx1``..``dx5`` labels. This is text hygiene only - it never
    maps a phrase onto a diagnostic category. That is the codebook's job, in
    :mod:`oasis_radiomics.clinical.classification`.
    """
    cleaned = normalise_missing(value)
    if cleaned is None:
        return None
    return " ".join(str(cleaned).split())


# ---------------------------------------------------------------------------
# generic clinical instrument reader
# ---------------------------------------------------------------------------
def _read_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a CSV into raw dictionaries, returning ``(fieldnames, rows)``."""
    path = Path(path)
    if not path.exists():
        raise ClinicalReaderError(f"Clinical file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not fieldnames:
        raise ClinicalReaderError(f"Clinical file has no header: {path}")
    return fieldnames, rows


def _require_columns(path: Path, fieldnames: Sequence[str], required: Iterable[str]) -> None:
    """Fail with an explicit message when a key column is absent."""
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ClinicalReaderError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Found: {', '.join(fieldnames[:10])}..."
        )


def read_clinical_instrument(path: Path, source: str) -> list[ClinicalRecord]:
    """Read any of the UDS instruments (D1, B4, C1) into typed records.

    Rows whose ``days_to_visit`` cannot be parsed are **kept out** of the
    returned records but logged; :mod:`oasis_radiomics.clinical.validation`
    reports them from the same file so nothing disappears silently.
    """
    path = Path(path)
    fieldnames, rows = _read_rows(path)
    _require_columns(path, fieldnames, (SUBJECT_COLUMN, DAYS_COLUMN))

    records: list[ClinicalRecord] = []
    skipped = 0
    for row in rows:
        subject_id = normalise_missing(row.get(SUBJECT_COLUMN))
        day = parse_day(row.get(DAYS_COLUMN))
        if not subject_id or day is None:
            skipped += 1
            continue

        raw = {key: normalise_missing(value) for key, value in row.items()}
        records.append(
            ClinicalRecord(
                subject_id=str(subject_id),
                clinical_session_id=str(
                    normalise_missing(row.get(SESSION_LABEL_COLUMN))
                    or f"{subject_id}_{source}_d{day:04d}"
                ),
                clinical_day=day,
                source=source,
                age_at_visit=parse_float(row.get(AGE_COLUMN)),
                raw=raw,
            )
        )

    if skipped:
        logger.warning(
            "%s: %d row(s) skipped (missing subject id or unparsable %s).",
            path.name,
            skipped,
            DAYS_COLUMN,
        )
    logger.info("%s: %d clinical record(s) read [%s].", path.name, len(records), source)
    return records


def read_d1(path: Path) -> list[ClinicalRecord]:
    """Read ``OASIS3_UDSd1_diagnoses.csv`` - the primary diagnostic source."""
    return read_clinical_instrument(path, SOURCE_D1)


def read_b4(path: Path) -> list[ClinicalRecord]:
    """Read ``OASIS3_UDSb4_cdr.csv`` - CDR, MMSE and supporting dx labels."""
    return read_clinical_instrument(path, SOURCE_B4)


def read_c1(path: Path) -> list[ClinicalRecord]:
    """Read ``OASIS3_UDSc1_cognitive_assessments.csv`` - optional psychometrics."""
    return read_clinical_instrument(path, SOURCE_C1)


# ---------------------------------------------------------------------------
# MRI demographic catalogue
# ---------------------------------------------------------------------------
def read_mri_catalogue(path: Path) -> list[MriSession]:
    """Read the OASIS MRI session catalogue export.

    Expects the columns of the official export: ``Label``, ``Subject``, ``M/F``,
    ``Age``, ``Scanner``. ``Label`` carries the canonical MRI session id
    (``OAS30001_MR_d0129``), from which subject and day are re-derived with the
    repository's existing parser rather than trusting the ``Subject`` column.

    Rows whose ``Label`` is not a parsable OASIS session id are skipped and
    logged; validation reports them separately.
    """
    path = Path(path)
    fieldnames, rows = _read_rows(path)
    _require_columns(path, fieldnames, (CATALOGUE_LABEL_COLUMN,))

    sessions: list[MriSession] = []
    skipped = 0
    for row in rows:
        label = normalise_missing(row.get(CATALOGUE_LABEL_COLUMN))
        if not label:
            skipped += 1
            continue
        try:
            key = parse_session_id(str(label))
        except SessionIdError:
            logger.warning("%s: unparsable MRI label %r; skipped.", path.name, label)
            skipped += 1
            continue

        sessions.append(
            MriSession(
                subject_id=key.subject_id,
                session_id=key.session_id,
                mri_day=key.days_from_reference,
                sex=normalise_missing(row.get(CATALOGUE_SEX_COLUMN)),
                age_at_mri=parse_float(row.get(CATALOGUE_AGE_COLUMN)),
                scanner=normalise_missing(row.get(CATALOGUE_SCANNER_COLUMN)),
                raw={key_: normalise_missing(value) for key_, value in row.items()},
            )
        )

    if skipped:
        logger.warning("%s: %d catalogue row(s) skipped.", path.name, skipped)

    sessions.sort(key=lambda item: (item.subject_id, item.mri_day, item.session_id))
    logger.info(
        "%s: %d MRI session(s) for %d subject(s).",
        path.name,
        len(sessions),
        len({item.subject_id for item in sessions}),
    )
    return sessions


def group_records_by_subject(
    records: Iterable[ClinicalRecord],
) -> dict[str, list[ClinicalRecord]]:
    """Group clinical records per subject, ordered by ``clinical_day``."""
    grouped: dict[str, list[ClinicalRecord]] = {}
    for record in records:
        grouped.setdefault(record.subject_id, []).append(record)
    for items in grouped.values():
        items.sort(key=lambda item: (item.clinical_day, item.clinical_session_id))
    return dict(sorted(grouped.items()))
