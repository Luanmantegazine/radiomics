"""Tests for clinical CSV normalisation and the classification codebook."""

from __future__ import annotations

from pathlib import Path

import pytest

from oasis_radiomics.clinical.classification import (
    ClassificationCodebook,
    CodebookError,
    classify_clinical_visit,
)
from oasis_radiomics.clinical.models import ClinicalRecord, ClinicalVisit
from oasis_radiomics.clinical.readers import (
    ClinicalReaderError,
    normalise_missing,
    normalise_text,
    parse_day,
    parse_float,
    read_clinical_instrument,
    read_mri_catalogue,
)

D1_CSV = """OASISID,OASIS_session_label,days_to_visit,age at visit,NORMCOG,PROBAD
OAS30001,OAS30001_UDSd1_d0000,0000,65.19,1,
OAS30001,OAS30001_UDSd1_d0339,0339,66.12,1,
OAS30002,OAS30002_UDSd1_d0010,0010,72.5,,1
OAS30003,OAS30003_UDSd1_dXXXX,notaday,70.0,1,
"""

CATALOGUE_CSV = """Label,Project,Date,Subject,M/F,Age,Type,Scanner,Scans
OAS30001_MR_d0129,OASIS3,,OAS30001,F,65,,3.0T,"T1w(2)"
OAS30001_MR_d0757,OASIS3,,OAS30001,F,67,,3.0T,"T1w(2)"
not_a_session,OASIS3,,OAS39999,M,80,,3.0T,"T1w(1)"
"""


@pytest.fixture
def d1_file(tmp_path: Path) -> Path:
    path = tmp_path / "d1.csv"
    path.write_text(D1_CSV, encoding="utf-8")
    return path


@pytest.fixture
def catalogue_file(tmp_path: Path) -> Path:
    path = tmp_path / "catalogue.csv"
    path.write_text(CATALOGUE_CSV, encoding="utf-8")
    return path


# --- scalar normalisation --------------------------------------------------
@pytest.mark.parametrize("token", ["", ".", "NA", "N/A", "NaN", "  "])
def test_missing_tokens_become_none(token: str) -> None:
    assert normalise_missing(token) is None


def test_normalise_missing_preserves_real_values() -> None:
    assert normalise_missing("  Cognitively normal  ") == "Cognitively normal"
    assert normalise_missing(0) == 0


def test_parse_day_handles_both_source_conventions() -> None:
    """D1 zero-pads days_to_visit; B4 does not."""
    assert parse_day("0339") == 339
    assert parse_day("339") == 339
    assert parse_day(339) == 339


def test_parse_day_accepts_negative_values_verbatim() -> None:
    """OASIS-3 contains negative days; readers keep them for validation to flag."""
    assert parse_day("-39520") == -39520


def test_parse_day_rejects_unparsable() -> None:
    assert parse_day("notaday") is None
    assert parse_day("") is None
    assert parse_day(None) is None


def test_parse_float() -> None:
    assert parse_float("65.19") == pytest.approx(65.19)
    assert parse_float(".") is None


def test_normalise_text_collapses_whitespace_without_interpreting() -> None:
    assert normalise_text("Cognitively   normal") == "Cognitively normal"
    assert normalise_text(".") is None


# --- instrument reading ----------------------------------------------------
def test_read_instrument_normalises_identifiers(d1_file: Path) -> None:
    records = read_clinical_instrument(d1_file, "d1")
    assert [record.subject_id for record in records] == ["OAS30001", "OAS30001", "OAS30002"]
    assert records[1].clinical_day == 339
    assert records[0].age_at_visit == pytest.approx(65.19)


def test_read_instrument_skips_unparsable_days(d1_file: Path) -> None:
    records = read_clinical_instrument(d1_file, "d1")
    assert all(record.subject_id != "OAS30003" for record in records)


def test_read_instrument_preserves_raw_values(d1_file: Path) -> None:
    records = read_clinical_instrument(d1_file, "d1")
    assert records[0].value("NORMCOG") == "1"
    assert records[0].value("PROBAD") is None
    assert records[2].value("PROBAD") == "1"


def test_read_instrument_requires_key_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ClinicalReaderError):
        read_clinical_instrument(path, "d1")


def test_read_instrument_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ClinicalReaderError):
        read_clinical_instrument(tmp_path / "absent.csv", "d1")


# --- MRI catalogue ---------------------------------------------------------
def test_read_catalogue_derives_ids_from_the_label(catalogue_file: Path) -> None:
    sessions = read_mri_catalogue(catalogue_file)
    assert [session.session_id for session in sessions] == [
        "OAS30001_MR_d0129",
        "OAS30001_MR_d0757",
    ]
    assert sessions[0].subject_id == "OAS30001"
    assert sessions[0].mri_day == 129


def test_read_catalogue_reads_demographics(catalogue_file: Path) -> None:
    session = read_mri_catalogue(catalogue_file)[0]
    assert session.sex == "F"
    assert session.age_at_mri == pytest.approx(65.0)
    assert session.scanner == "3.0T"


def test_read_catalogue_skips_unparsable_labels(catalogue_file: Path) -> None:
    assert len(read_mri_catalogue(catalogue_file)) == 2


# --- classification codebook ----------------------------------------------
def _visit_with(**d1_values) -> ClinicalVisit:
    record = ClinicalRecord("S", "S_UDSd1_d0000", 0, "d1", raw=dict(d1_values))
    return ClinicalVisit("S", 0, d1=record)


def test_repository_codebook_is_unfrozen_by_default() -> None:
    """The shipped codebook must not assert any D1 semantics."""
    codebook = ClassificationCodebook.load(None)
    assert codebook.frozen is False
    assert codebook.is_active is False


def test_unfrozen_codebook_derives_nothing() -> None:
    codebook = ClassificationCodebook.load(None)
    result = classify_clinical_visit(_visit_with(NORMCOG="1"), codebook)
    assert result.status == "unresolved_codebook"
    assert result.cognitive_status == "UNKNOWN"
    assert result.ad_etiology == "UNKNOWN"


def test_absent_visit_is_reported_as_no_clinical_data() -> None:
    result = classify_clinical_visit(None, ClassificationCodebook.load(None))
    assert result.status == "no_clinical_data"


def test_frozen_codebook_classifies() -> None:
    codebook = ClassificationCodebook.from_mapping(
        {
            "version": "test-v1",
            "codebook_frozen": True,
            "rules": [
                {
                    "id": "normcog",
                    "source": "d1",
                    "when": {"NORMCOG": "1"},
                    "cognitive_status": "CN",
                    "ad_etiology": "NON_AD",
                    "reference": "test fixture",
                }
            ],
        }
    )
    result = classify_clinical_visit(_visit_with(NORMCOG="1"), codebook)
    assert result.status == "classified"
    assert result.cognitive_status == "CN"
    assert result.codebook_version == "test-v1"


def test_frozen_codebook_reports_no_match_as_unresolved() -> None:
    codebook = ClassificationCodebook.from_mapping(
        {
            "version": "test-v1",
            "codebook_frozen": True,
            "rules": [
                {
                    "id": "normcog",
                    "source": "d1",
                    "when": {"NORMCOG": "1"},
                    "cognitive_status": "CN",
                    "reference": "test fixture",
                }
            ],
        }
    )
    assert classify_clinical_visit(_visit_with(NORMCOG="0"), codebook).status == (
        "unresolved_codebook"
    )


def test_conflicting_rules_are_reported_not_resolved() -> None:
    codebook = ClassificationCodebook.from_mapping(
        {
            "version": "test-v1",
            "codebook_frozen": True,
            "rules": [
                {"id": "a", "source": "d1", "when": {"X": "1"}, "cognitive_status": "CN", "reference": "t"},
                {"id": "b", "source": "d1", "when": {"Y": "1"}, "cognitive_status": "MCI", "reference": "t"},
            ],
        }
    )
    result = classify_clinical_visit(_visit_with(X="1", Y="1"), codebook)
    assert result.status == "conflicting"
    assert result.cognitive_status == "UNKNOWN"


def test_freezing_an_empty_codebook_is_rejected() -> None:
    with pytest.raises(CodebookError):
        ClassificationCodebook.from_mapping({"codebook_frozen": True})


def test_frozen_rules_must_cite_a_reference() -> None:
    with pytest.raises(CodebookError):
        ClassificationCodebook.from_mapping(
            {
                "codebook_frozen": True,
                "rules": [
                    {"id": "r", "source": "d1", "when": {"X": "1"}, "cognitive_status": "CN"}
                ],
            }
        )


def test_invalid_vocabulary_is_rejected() -> None:
    with pytest.raises(CodebookError):
        ClassificationCodebook.from_mapping(
            {"rules": [{"id": "r", "source": "d1", "when": {"X": "1"}, "cognitive_status": "PROBABLY_AD"}]}
        )
