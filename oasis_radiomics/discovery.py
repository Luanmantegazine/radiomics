"""Discovery of processed FreeSurfer sessions on disk.

Given the directory the OASIS-3 downloader wrote into, locate every session
that carries the pair the pipeline needs::

    <session>/mri/T1.mgz
    <session>/mri/aseg.mgz

No data is downloaded here; see :mod:`oasis_radiomics.download_oasis`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .ids import SessionIdError, SessionKey, find_session_id, group_by_subject, parse_session_id

logger = logging.getLogger(__name__)

T1_FILENAME = "T1.mgz"
ASEG_FILENAME = "aseg.mgz"


class DiscoveryError(RuntimeError):
    """Raised when the input directory contains no usable session."""


@dataclass(frozen=True)
class DiscoveredSession:
    """A FreeSurfer session that has both the image and the segmentation."""

    key: SessionKey
    t1_path: Path
    aseg_path: Path

    @property
    def subject_id(self) -> str:
        return self.key.subject_id

    @property
    def session_id(self) -> str:
        return self.key.session_id

    @property
    def days_from_reference(self) -> int:
        return self.key.days_from_reference


def infer_session_id(aseg_path: Path) -> str:
    """Derive the OASIS session id from the path of an ``aseg.mgz``.

    The FreeSurfer layout is ``<session>/mri/aseg.mgz``, but OASIS downloads
    sometimes nest that under an extra directory level, so every path component
    is searched from the deepest one outwards.

    Raises
    ------
    SessionIdError
        If no component of the path carries an OASIS identifier.
    """
    for part in reversed(aseg_path.resolve().parts):
        session_id = find_session_id(part)
        if session_id is not None:
            return session_id

    raise SessionIdError(
        f"No OASIS session identifier found in path {aseg_path}. "
        "Expected a component such as 'OAS30001_MR_d0129'."
    )


def discover_sessions(input_dir: Path, strict: bool = False) -> list[DiscoveredSession]:
    """Find every usable FreeSurfer session under ``input_dir``.

    Parameters
    ----------
    input_dir:
        Directory to search recursively (e.g. ``<out>/freesurfer``).
    strict:
        When ``True``, a session whose identifier cannot be parsed raises
        instead of being skipped.

    Returns
    -------
    list[DiscoveredSession]
        Sorted by subject and then chronologically.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise DiscoveryError(f"Input directory does not exist: {input_dir}")

    sessions: list[DiscoveredSession] = []
    seen: set[str] = set()

    for aseg_path in sorted(input_dir.rglob(ASEG_FILENAME)):
        t1_path = aseg_path.parent / T1_FILENAME
        if not t1_path.exists():
            logger.warning("Skipping %s: no %s next to it.", aseg_path, T1_FILENAME)
            continue

        try:
            session_id = infer_session_id(aseg_path)
            key = parse_session_id(session_id)
        except SessionIdError as exc:
            if strict:
                raise
            logger.warning("Skipping %s: %s", aseg_path, exc)
            continue

        if key.session_id in seen:
            logger.warning(
                "Duplicate session %s found at %s; keeping the first occurrence.",
                key.session_id,
                aseg_path,
            )
            continue

        seen.add(key.session_id)
        sessions.append(DiscoveredSession(key=key, t1_path=t1_path, aseg_path=aseg_path))

    sessions.sort(key=lambda session: session.key)
    logger.info(
        "Discovered %d FreeSurfer session(s) for %d subject(s) under %s",
        len(sessions),
        len({session.subject_id for session in sessions}),
        input_dir,
    )
    return sessions


def group_sessions_by_subject(
    sessions: Sequence[DiscoveredSession],
) -> dict[str, list[DiscoveredSession]]:
    """Group discovered sessions per subject, chronologically ordered."""
    by_id = {session.session_id: session for session in sessions}
    grouped_keys = group_by_subject(session.key for session in sessions)
    return {
        subject_id: [by_id[key.session_id] for key in keys]
        for subject_id, keys in grouped_keys.items()
    }
