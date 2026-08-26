"""Tests for FreeSurfer session discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_radiomics.discovery import (
    DiscoveryError,
    discover_sessions,
    group_sessions_by_subject,
    infer_session_id,
)
from oasis_radiomics.ids import SessionIdError


def _make_session(root: Path, session_id: str, with_t1: bool = True) -> Path:
    mri = root / session_id / "mri"
    mri.mkdir(parents=True)
    (mri / "aseg.mgz").write_bytes(b"")
    if with_t1:
        (mri / "T1.mgz").write_bytes(b"")
    return mri


def test_infer_session_id_from_freesurfer_layout() -> None:
    path = Path("/data/OAS30001_MR_d0129/mri/aseg.mgz")
    assert infer_session_id(path) == "OAS30001_MR_d0129"


def test_infer_session_id_through_extra_nesting() -> None:
    path = Path("/data/batch1/OAS30001_MR_d0757/some/deeper/mri/aseg.mgz")
    assert infer_session_id(path) == "OAS30001_MR_d0757"


def test_infer_session_id_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(SessionIdError):
        infer_session_id(tmp_path / "anonymous" / "mri" / "aseg.mgz")


def test_discover_sessions_sorts_chronologically(tmp_path: Path) -> None:
    for session_id in ("OAS30001_MR_d0757", "OAS30001_MR_d0129", "OAS30002_MR_d0010"):
        _make_session(tmp_path, session_id)

    sessions = discover_sessions(tmp_path)
    assert [session.session_id for session in sessions] == [
        "OAS30001_MR_d0129",
        "OAS30001_MR_d0757",
        "OAS30002_MR_d0010",
    ]


def test_discover_sessions_skips_missing_t1(tmp_path: Path) -> None:
    _make_session(tmp_path, "OAS30001_MR_d0129")
    _make_session(tmp_path, "OAS30001_MR_d0757", with_t1=False)
    assert [session.session_id for session in discover_sessions(tmp_path)] == [
        "OAS30001_MR_d0129"
    ]


def test_discover_sessions_skips_unparseable_directories(tmp_path: Path) -> None:
    _make_session(tmp_path, "OAS30001_MR_d0129")
    _make_session(tmp_path, "random_subject")
    assert len(discover_sessions(tmp_path)) == 1


def test_discover_sessions_strict_raises_on_unparseable(tmp_path: Path) -> None:
    _make_session(tmp_path, "random_subject")
    with pytest.raises(SessionIdError):
        discover_sessions(tmp_path, strict=True)


def test_discover_sessions_requires_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(DiscoveryError):
        discover_sessions(tmp_path / "absent")


def test_group_sessions_by_subject(tmp_path: Path) -> None:
    for session_id in ("OAS30001_MR_d0757", "OAS30001_MR_d0129", "OAS30002_MR_d0010"):
        _make_session(tmp_path, session_id)

    grouped = group_sessions_by_subject(discover_sessions(tmp_path))
    assert list(grouped) == ["OAS30001", "OAS30002"]
    assert [session.days_from_reference for session in grouped["OAS30001"]] == [129, 757]
