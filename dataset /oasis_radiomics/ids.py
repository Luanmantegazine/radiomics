"""Parsing and grouping of OASIS-3 subject / session identifiers.

OASIS-3 identifiers encode three things in one string::

    OAS30001_MR_d0129
    ^^^^^^^^ subject
             ^^ modality / pipeline tag
                ^^^^^ days from the subject's reference (entry) date

The same layout is used by the FreeSurfer identifiers handed to the official
downloader (``OAS30001_Freesurfer53_d0129``), so a single parser covers both.

Nothing here assumes the identifiers arrive in chronological order: temporal
ordering is always re-derived from ``days_from_reference``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

#: ``OAS30001_MR_d0129`` / ``OAS30001_Freesurfer53_d0129`` and friends.
SESSION_ID_RE = re.compile(
    r"^(?P<subject>OAS\d+)_(?P<tag>[A-Za-z0-9.+-]+)_d(?P<days>\d+)$"
)

#: Loose variant used to *find* an identifier inside a longer path component.
SESSION_ID_SEARCH_RE = re.compile(r"OAS\d+_[A-Za-z0-9.+-]+_d\d+")

#: Session identifiers use the ``MR`` tag; FreeSurfer ids use e.g. ``Freesurfer53``.
MR_SESSION_TAG = "MR"


class SessionIdError(ValueError):
    """Raised when an identifier does not follow the OASIS-3 convention."""


@dataclass(frozen=True, order=True)
class SessionKey:
    """A single imaging session, identified and placed on the time axis.

    Ordering is by ``(subject_id, days_from_reference, session_id)`` so that
    sorting a list of keys yields per-subject chronological order.
    """

    subject_id: str
    days_from_reference: int
    session_id: str

    @property
    def years_from_reference(self) -> float:
        """Days converted to years using the Julian year (365.25 days)."""
        return self.days_from_reference / 365.25

    def as_dict(self) -> dict[str, object]:
        """Flat mapping suitable for a dataframe row."""
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "days_from_reference": self.days_from_reference,
        }


def parse_session_id(session_id: str) -> SessionKey:
    """Parse a full OASIS-3 identifier into a :class:`SessionKey`.

    Parameters
    ----------
    session_id:
        For example ``"OAS30001_MR_d0129"``.

    Raises
    ------
    SessionIdError
        If the identifier does not match the OASIS-3 convention.
    """
    if not isinstance(session_id, str):
        raise SessionIdError(f"Session id must be a string, got {type(session_id).__name__}")

    match = SESSION_ID_RE.match(session_id.strip())
    if match is None:
        raise SessionIdError(
            f"Cannot parse OASIS session id {session_id!r}. "
            "Expected the form '<subject>_<tag>_d<days>', e.g. 'OAS30001_MR_d0129'."
        )

    return SessionKey(
        subject_id=match.group("subject"),
        days_from_reference=int(match.group("days")),
        session_id=session_id.strip(),
    )


def parse_subject_id(session_id: str) -> str:
    """Return the subject part of an OASIS-3 identifier.

    >>> parse_subject_id("OAS30001_MR_d0129")
    'OAS30001'
    """
    return parse_session_id(session_id).subject_id


def parse_session_time(session_id: str) -> int:
    """Return ``days_from_reference`` encoded in an OASIS-3 identifier.

    >>> parse_session_time("OAS30001_MR_d0757")
    757
    """
    return parse_session_id(session_id).days_from_reference


def find_session_id(text: str) -> str | None:
    """Extract the first OASIS identifier contained in ``text``.

    Returns ``None`` when the string carries no identifier, so callers can
    decide whether that is an error or simply an uninteresting path component.
    """
    match = SESSION_ID_SEARCH_RE.search(text)
    return match.group(0) if match else None


def freesurfer_id_to_session_id(freesurfer_id: str) -> str:
    """Convert a FreeSurfer id to the corresponding MR session id.

    >>> freesurfer_id_to_session_id("OAS30001_Freesurfer53_d0129")
    'OAS30001_MR_d0129'
    """
    key = parse_session_id(freesurfer_id)
    return f"{key.subject_id}_{MR_SESSION_TAG}_d{_days_token(freesurfer_id)}"


def _days_token(session_id: str) -> str:
    """Return the zero-padded day token exactly as written in ``session_id``."""
    match = SESSION_ID_RE.match(session_id.strip())
    if match is None:  # pragma: no cover - guarded by parse_session_id callers
        raise SessionIdError(f"Cannot parse OASIS session id {session_id!r}")
    return match.group("days")


def parse_many(session_ids: Iterable[str], strict: bool = True) -> list[SessionKey]:
    """Parse a collection of identifiers.

    Parameters
    ----------
    session_ids:
        Identifiers to parse.
    strict:
        When ``True`` (default) an unparseable identifier raises. When ``False``
        it is logged as a warning and skipped, which is what batch discovery
        wants.
    """
    keys: list[SessionKey] = []
    for session_id in session_ids:
        try:
            keys.append(parse_session_id(session_id))
        except SessionIdError:
            if strict:
                raise
            logger.warning("Skipping unparseable session id: %r", session_id)
    return keys


def group_by_subject(keys: Iterable[SessionKey]) -> dict[str, list[SessionKey]]:
    """Group sessions per subject, chronologically ordered within each subject.

    Duplicated ``(subject, days)`` pairs are kept (a subject can have more than
    one scan on the same day) but are reported, because they usually indicate a
    repeated or re-processed acquisition.
    """
    grouped: dict[str, list[SessionKey]] = {}
    for key in keys:
        grouped.setdefault(key.subject_id, []).append(key)

    for subject_id, sessions in grouped.items():
        sessions.sort()
        _warn_on_duplicate_timepoints(subject_id, sessions)

    return dict(sorted(grouped.items()))


def _warn_on_duplicate_timepoints(subject_id: str, sessions: Sequence[SessionKey]) -> None:
    """Log subjects that have several sessions sharing one ``days`` value."""
    seen: dict[int, str] = {}
    for session in sessions:
        previous = seen.get(session.days_from_reference)
        if previous is not None:
            logger.warning(
                "Subject %s has multiple sessions at day %d (%s, %s).",
                subject_id,
                session.days_from_reference,
                previous,
                session.session_id,
            )
        seen[session.days_from_reference] = session.session_id


def sort_sessions(keys: Iterable[SessionKey]) -> list[SessionKey]:
    """Return ``keys`` sorted by subject and then chronologically."""
    return sorted(keys)


def followup_summary(grouped: Mapping[str, Sequence[SessionKey]]) -> dict[str, dict[str, float]]:
    """Per-subject session count and follow-up span in days and years."""
    summary: dict[str, dict[str, float]] = {}
    for subject_id, sessions in grouped.items():
        if not sessions:
            continue
        days = [session.days_from_reference for session in sessions]
        span_days = max(days) - min(days)
        summary[subject_id] = {
            "n_sessions": float(len(sessions)),
            "followup_days": float(span_days),
            "followup_years": span_days / 365.25,
        }
    return summary
