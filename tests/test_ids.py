"""Tests for OASIS-3 identifier parsing and grouping."""

from __future__ import annotations

import pytest

from oasis_radiomics.ids import (
    SessionIdError,
    SessionKey,
    find_session_id,
    followup_summary,
    freesurfer_id_to_session_id,
    group_by_subject,
    parse_many,
    parse_session_id,
    parse_session_time,
    parse_subject_id,
    sort_sessions,
)


@pytest.mark.parametrize(
    "session_id, subject, days",
    [
        ("OAS30001_MR_d0129", "OAS30001", 129),
        ("OAS30001_MR_d0757", "OAS30001", 757),
        ("OAS30001_Freesurfer53_d0129", "OAS30001", 129),
        ("OAS31172_MR_d0000", "OAS31172", 0),
        ("OAS30001_MR_d1234", "OAS30001", 1234),
    ],
)
def test_parse_session_id(session_id: str, subject: str, days: int) -> None:
    key = parse_session_id(session_id)
    assert key.subject_id == subject
    assert key.days_from_reference == days
    assert key.session_id == session_id


def test_parse_subject_id_and_time() -> None:
    assert parse_subject_id("OAS30001_MR_d0129") == "OAS30001"
    assert parse_session_time("OAS30001_MR_d0757") == 757


def test_parse_session_id_strips_whitespace() -> None:
    assert parse_session_id("  OAS30001_MR_d0129  ").session_id == "OAS30001_MR_d0129"


@pytest.mark.parametrize(
    "bad_id",
    ["", "OAS30001", "OAS30001_MR", "OAS30001_MR_0129", "subject_MR_d0129", "OAS30001-MR-d0129"],
)
def test_parse_session_id_rejects_malformed(bad_id: str) -> None:
    with pytest.raises(SessionIdError):
        parse_session_id(bad_id)


def test_parse_session_id_rejects_non_string() -> None:
    with pytest.raises(SessionIdError):
        parse_session_id(129)  # type: ignore[arg-type]


def test_days_are_not_inferred_from_order() -> None:
    """Ordering must come from the day count, not from the input order."""
    keys = parse_many(["OAS30001_MR_d0757", "OAS30001_MR_d0129", "OAS30001_MR_d1100"])
    assert [key.days_from_reference for key in sort_sessions(keys)] == [129, 757, 1100]


def test_parse_many_non_strict_skips_bad_ids(caplog) -> None:
    keys = parse_many(["OAS30001_MR_d0129", "garbage"], strict=False)
    assert [key.session_id for key in keys] == ["OAS30001_MR_d0129"]


def test_parse_many_strict_raises() -> None:
    with pytest.raises(SessionIdError):
        parse_many(["OAS30001_MR_d0129", "garbage"], strict=True)


def test_group_by_subject_orders_chronologically() -> None:
    keys = parse_many(
        [
            "OAS30002_MR_d0500",
            "OAS30001_MR_d1100",
            "OAS30001_MR_d0129",
            "OAS30002_MR_d0010",
            "OAS30001_MR_d0757",
        ]
    )
    grouped = group_by_subject(keys)

    assert list(grouped) == ["OAS30001", "OAS30002"]
    assert [key.days_from_reference for key in grouped["OAS30001"]] == [129, 757, 1100]
    assert [key.days_from_reference for key in grouped["OAS30002"]] == [10, 500]


def test_group_by_subject_warns_on_duplicate_days(caplog) -> None:
    keys = parse_many(["OAS30001_MR_d0129", "OAS30001_Freesurfer53_d0129"])
    with caplog.at_level("WARNING"):
        group_by_subject(keys)
    assert any("multiple sessions at day 129" in record.getMessage() for record in caplog.records)


def test_followup_summary() -> None:
    grouped = group_by_subject(parse_many(["OAS30001_MR_d0129", "OAS30001_MR_d0757"]))
    summary = followup_summary(grouped)["OAS30001"]
    assert summary["n_sessions"] == 2
    assert summary["followup_days"] == 628
    assert summary["followup_years"] == pytest.approx(628 / 365.25)


def test_session_key_years() -> None:
    assert SessionKey("OAS30001", 365, "OAS30001_MR_d0365").years_from_reference == pytest.approx(
        365 / 365.25
    )


def test_freesurfer_id_to_session_id() -> None:
    assert freesurfer_id_to_session_id("OAS30001_Freesurfer53_d0129") == "OAS30001_MR_d0129"


def test_find_session_id_in_path() -> None:
    assert find_session_id("/data/OAS30001_MR_d0129/mri/aseg.mgz") == "OAS30001_MR_d0129"
    assert find_session_id("/data/nothing/here") is None
