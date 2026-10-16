"""Tests for the FreeSurfer morphometry inventory and extraction helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from oasis_radiomics.morphometry import (
    MEASURE_COLUMNS,
    RemoteLookupEntry,
    _read_official_freesurfer_index,
    build_and_write_outputs,
    build_download_manifest,
    build_preflight_rows,
    build_extraction_rows,
    build_inventory,
    build_session_substitutions,
    validate_primary_preflight_rows,
    summarize_remote_statuses,
    select_primary_sessions,
)
from scripts.extract_freesurfer_morphometry import build_parser as build_extract_parser


def _write_aseg(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# ColHeaders Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange",
                "1 17 100 3931.0 Left-Hippocampus 0 0 0 0 0",
                "2 53 110 4026.0 Right-Hippocampus 0 0 0 0 0",
                "3 18 120 194.0 Left-Amygdala 0 0 0 0 0",
                "4 54 130 205.0 Right-Amygdala 0 0 0 0 0",
                "5 6 140 33.0 Left-Inf-Lat-Vent 0 0 0 0 0",
                "6 7 150 34.0 Right-Inf-Lat-Vent 0 0 0 0 0",
                "7 4 160 900.0 Left-Lateral-Ventricle 0 0 0 0 0",
                "8 43 170 910.0 Right-Lateral-Ventricle 0 0 0 0 0",
                "# Measure EstimatedTotalIntraCranialVol, EstimatedTotalIntraCranialVol, 1500000.0, mm^3",
                "# Measure CortexVol, CortexVol, 900000.0, mm^3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_aparc(path: Path, left: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "# ColHeaders Index StructName NumVert SurfArea GrayVol ThickAvg MeanCurv GausCurv FoldInd CurvInd",
                f"1 entorhinal 10 20 30 {2.5 if left else 2.4} 0 0 0 0",
                f"2 parahippocampal 11 21 31 {2.6 if left else 2.3} 0 0 0 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _make_session(root: Path, session_id: str, complete: bool = True) -> Path:
    stats_dir = root / session_id / "stats"
    stats_dir.mkdir(parents=True)
    if complete:
        _write_aseg(stats_dir / "aseg.stats")
        _write_aparc(stats_dir / "lh.aparc.stats", True)
        _write_aparc(stats_dir / "rh.aparc.stats", False)
    else:
        _write_aseg(stats_dir / "aseg.stats")
    return stats_dir


def _write_input_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "subject_id,session_id,mri_day,supervised_label,training_eligible,clinical_match_valid,clinical_mri_abs_gap_days,scanner",
                "OAS30001,OAS30001_MR_d0129,129,CN,True,True,10,3.0T",
                "OAS30001,OAS30001_MR_d0757,757,CN,True,True,40,3.0T",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_primary_session_selection_prefers_smaller_gap() -> None:
    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "40",
        },
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0757",
            "mri_day": "757",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
        },
    ]
    assert select_primary_sessions(rows) == {"OAS30001": "OAS30001_MR_d0757"}


def test_inventory_and_extraction_from_synthetic_stats(tmp_path: Path) -> None:
    _make_session(tmp_path, "OAS30001_Freesurfer53_d0129", complete=False)
    _make_session(tmp_path, "OAS30001_Freesurfer53_d0757", complete=True)

    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
            "scanner": "3.0T",
        },
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0757",
            "mri_day": "757",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "40",
            "scanner": "3.0T",
        },
    ]
    inventory, directories, _ = build_inventory(rows, tmp_path)
    by_session = {row["session_id"]: row for row in inventory}

    assert by_session["OAS30001_MR_d0129"]["match_status"] == "exact_session"
    assert by_session["OAS30001_MR_d0129"]["local_stats_complete"] is False
    assert by_session["OAS30001_MR_d0757"]["local_stats_complete"] is True
    assert by_session["OAS30001_MR_d0129"]["selected_primary_session"] is True
    assert by_session["OAS30001_MR_d0757"]["selected_primary_session"] is False

    manifest = build_download_manifest(inventory)
    actions = {row["session_id"]: row["required_action"] for row in manifest}
    assert actions["OAS30001_MR_d0129"] == "remote_lookup_required"
    assert actions["OAS30001_MR_d0757"] == "backup_candidate"

    extraction_rows = build_extraction_rows(rows, inventory, directories)
    extracted = {row["session_id"]: row for row in extraction_rows}
    assert extracted["OAS30001_MR_d0757"]["fs_data_origin"] == "local_existing"
    assert extracted["OAS30001_MR_d0757"]["fs_left_hippocampus_mm3"] == 3931.0
    assert extracted["OAS30001_MR_d0757"]["fs_rh_parahippocampal_thickavg_mm"] == 2.3
    assert extracted["OAS30001_MR_d0129"]["fs_missing_fields"]
    assert extracted["OAS30001_MR_d0129"]["fs_qc_warning"]

    substitutions = build_session_substitutions(rows, inventory)
    assert substitutions == [
        {
            "subject_id": "OAS30001",
            "original_session_id": "OAS30001_MR_d0129",
            "replacement_session_id": "OAS30001_MR_d0757",
            "original_clinical_gap": "10",
            "replacement_clinical_gap": "40",
            "original_label": "CN",
            "replacement_label": "CN",
            "replacement_reason": "primary_session_missing_local_freesurfer",
        }
    ]


def test_default_manifest_scope_counts_real_dataset(repo_root: Path) -> None:
    input_csv = repo_root / "clinical_radiomics_CONSOLIDATED_v2 (1).csv"
    rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8-sig")))
    inventory, _, _ = build_inventory(rows, repo_root / "results")
    manifest = build_download_manifest(inventory)
    counts = {
        action: 0
        for action in (
            "remote_lookup_required",
            "backup_candidate",
            "not_required_mci",
            "not_required_ineligible",
            "not_required_unlabelled",
            "manual_review",
        )
    }
    for row in manifest:
        counts[row["required_action"]] = counts.get(row["required_action"], 0) + 1
    assert len(manifest) == 362
    assert counts == {
        "remote_lookup_required": 77,
        "backup_candidate": 62,
        "not_required_mci": 16,
        "not_required_ineligible": 24,
        "not_required_unlabelled": 183,
        "manual_review": 0,
    }
    assert sum(counts.values()) == 362
    assert sum(1 for row in manifest if row["selected_primary_session"]) == 77
    assert sum(1 for row in manifest if row["required_action"] == "backup_candidate") == 62
    assert sum(1 for row in manifest if row["remote_availability_status"] == "not_checked") == 362


def test_preflight_scope_primary_only_selects_principal_sessions(repo_root: Path) -> None:
    input_csv = repo_root / "clinical_radiomics_CONSOLIDATED_v2 (1).csv"
    rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8-sig")))
    inventory, _, _ = build_inventory(rows, repo_root / "results")
    manifest = build_download_manifest(inventory)
    preflight = build_preflight_rows(inventory, manifest, scope="primary")
    summary = validate_primary_preflight_rows(preflight)
    assert summary == {"total": 77, "CN": 51, "AD": 26, "subjects": 77, "sessions": 77}
    assert all(row["selected_primary_session"] for row in preflight)
    assert {row["planned_action"] for row in preflight} == {"remote_lookup_required"}
    assert all(row["resource_type_available"] == "not_checked" for row in preflight)
    assert summarize_remote_statuses(preflight) == {
        "remote_available": 0,
        "remote_not_available": 0,
        "remote_not_checked": 77,
        "remote_auth_required": 0,
        "remote_lookup_error": 0,
        "remote_multiple_candidates": 0,
    }


def test_official_spreadsheet_short_circuits_download(tmp_path: Path) -> None:
    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
        }
    ]
    inventory, _, _ = build_inventory(rows, tmp_path)
    measures = {
        ("OAS30001", "OAS30001_MR_d0129"): {column: 1.0 for column in MEASURE_COLUMNS}
    }
    manifest = build_download_manifest(inventory, measures, {})
    preflight = build_preflight_rows(inventory, manifest, measures, scope="primary")
    assert manifest[0]["required_action"] == "use_official_spreadsheet"
    assert manifest[0]["remote_availability_status"] == "not_checked"
    assert preflight[0]["planned_action"] == "use_official_spreadsheet"
    assert preflight[0]["resource_type_available"] == "official_spreadsheet"


def test_remote_available_marks_derivative_download(tmp_path: Path) -> None:
    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
        }
    ]
    inventory, _, _ = build_inventory(rows, tmp_path)
    index = {
        "OAS30001_MR_d0129": RemoteLookupEntry(
            session_id="OAS30001_MR_d0129",
            remote_identifier="OAS30001_Freesurfer53_d0129",
            remote_version="53",
            availability_status="available",
            candidate_columns=("freesurfer_id",),
            candidate_count=1,
        )
    }
    manifest = build_download_manifest(inventory, None, index)
    preflight = build_preflight_rows(inventory, manifest, None, scope="primary")
    assert manifest[0]["required_action"] == "download_freesurfer_derivative"
    assert manifest[0]["remote_availability_status"] == "available"
    assert preflight[0]["planned_action"] == "download_freesurfer_derivative"
    assert preflight[0]["resource_type_available"] == "freesurfer_derivative"


def test_official_index_parser_handles_nonstandard_headers(tmp_path: Path) -> None:
    index_csv = tmp_path / "oasis3_freesurfer_index.csv"
    index_csv.write_text(
        "\n".join(
            [
                "FreeSurfer Processing,Alternate Processing,FreeSurfer Release,Download URL",
                "OAS30001_Freesurfer53_d0129,,Freesurfer53,https://example.invalid/OAS30001_Freesurfer53_d0129",
                "OAS30001_Freesurfer53_d0757,OAS30001_Freesurfer53_d0999,Freesurfer53,https://example.invalid/OAS30001_Freesurfer53_d0757",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    index = _read_official_freesurfer_index(index_csv)
    assert index["OAS30001_MR_d0129"].availability_status == "available"
    assert index["OAS30001_MR_d0129"].remote_version == "53"
    assert index["OAS30001_MR_d0757"].availability_status == "multiple_remote_candidates"
    assert index["OAS30001_MR_d0757"].candidate_count == 2


def test_manifest_with_official_index_clears_not_checked(tmp_path: Path) -> None:
    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
        }
    ]
    inventory, _, _ = build_inventory(rows, tmp_path)
    index = {
        "OAS30001_MR_d0129": RemoteLookupEntry(
            session_id="OAS30001_MR_d0129",
            remote_identifier="OAS30001_Freesurfer53_d0129",
            remote_version="53",
            availability_status="available",
            candidate_columns=("FreeSurfer Processing",),
            candidate_count=1,
        )
    }
    manifest = build_download_manifest(inventory, None, index, remote_index_checked=True)
    preflight = build_preflight_rows(inventory, manifest, scope="primary")
    assert manifest[0]["remote_availability_status"] == "available"
    assert manifest[0]["required_action"] == "download_freesurfer_derivative"
    assert preflight[0]["planned_action"] == "download_freesurfer_derivative"
    assert preflight[0]["remote_availability_status"] == "available"


@pytest.mark.parametrize(
    "status, expected_action",
    [
        ("available", "download_freesurfer_derivative"),
        ("multiple_remote_candidates", "review_remote_candidates"),
        ("auth_required", "remote_auth_required"),
        ("not_available", "download_raw_t1"),
        ("not_checked", "remote_lookup_required"),
    ],
)
def test_remote_statuses_drive_manifest_actions(tmp_path: Path, status: str, expected_action: str) -> None:
    rows = [
        {
            "subject_id": "OAS30001",
            "session_id": "OAS30001_MR_d0129",
            "mri_day": "129",
            "supervised_label": "CN",
            "training_eligible": "True",
            "clinical_match_valid": "True",
            "clinical_mri_abs_gap_days": "10",
        }
    ]
    inventory, _, _ = build_inventory(rows, tmp_path)
    if status == "not_checked":
        manifest = build_download_manifest(inventory)
    else:
        index = {
            "OAS30001_MR_d0129": RemoteLookupEntry(
                session_id="OAS30001_MR_d0129",
                remote_identifier="OAS30001_Freesurfer53_d0129",
                remote_version="53",
                availability_status=status,
                candidate_columns=("freesurfer_id",),
                candidate_count=1 if status != "multiple_remote_candidates" else 2,
            )
        }
        manifest = build_download_manifest(inventory, None, index)
    assert manifest[0]["required_action"] == expected_action


def test_extract_parser_defaults_scope_to_primary() -> None:
    parser = build_extract_parser()
    args = parser.parse_args(
        [
            "--input-csv",
            "input.csv",
            "--subjects-dir",
            "subjects",
            "--output-csv",
            "out.csv",
            "--audit-dir",
            "audit",
        ]
    )
    assert args.scope == "primary"


def test_dry_run_does_not_call_downloader(tmp_path: Path, monkeypatch) -> None:
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv)
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    output_csv = tmp_path / "out.csv"
    audit_dir = tmp_path / "audit"
    download_dir = tmp_path / "download"

    def fail(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("downloader should not run during dry-run")

    monkeypatch.setattr("oasis_radiomics.morphometry.download_selected_sessions", fail)
    outputs = build_and_write_outputs(
        input_csv=input_csv,
        subjects_dir=subjects_dir,
        downloaded_subjects_dir=None,
        output_csv=output_csv,
        audit_dir=audit_dir,
        download_mode="freesurfer",
        scope="primary",
        dry_run=True,
        download_dir=download_dir,
    )
    assert outputs["preflight"].exists()
    assert not output_csv.exists()
    assert not any(download_dir.iterdir()) if download_dir.exists() else True
    with outputs["preflight"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["planned_action"] == "remote_lookup_required"
    assert {row["planned_action"] for row in rows} == {"remote_lookup_required"}
    assert sum(row["remote_availability_status"] == "not_checked" for row in rows) == 1


def test_no_credentials_in_logs(tmp_path: Path, caplog) -> None:
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv)
    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()
    with caplog.at_level("INFO"):
        build_and_write_outputs(
            input_csv=input_csv,
            subjects_dir=subjects_dir,
            downloaded_subjects_dir=None,
            output_csv=tmp_path / "out.csv",
            audit_dir=tmp_path / "audit",
            download_mode="none",
            scope="primary",
            dry_run=True,
            nitrc_user="secret-user",
        )
    assert "secret-user" not in caplog.text


def test_end_to_end_writeout_preserves_rows_and_writes_audits(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    _write_input_csv(input_csv)
    subjects_dir = tmp_path / "subjects"
    _make_session(subjects_dir, "OAS30001_Freesurfer53_d0129", complete=False)
    _make_session(subjects_dir, "OAS30001_Freesurfer53_d0757", complete=True)
    output_csv = tmp_path / "out.csv"
    audit_dir = tmp_path / "audit"

    outputs = build_and_write_outputs(
        input_csv=input_csv,
        subjects_dir=subjects_dir,
        downloaded_subjects_dir=None,
        output_csv=output_csv,
        audit_dir=audit_dir,
        scope="primary",
    )

    assert output_csv.exists()
    assert outputs["report"].exists()
    assert outputs["inventory"].exists()
    assert outputs["download_manifest"].exists()
    assert outputs["reprocessing_manifest"].exists()

    with output_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["session_id"] for row in rows] == ["OAS30001_MR_d0129", "OAS30001_MR_d0757"]
    assert rows[1]["fs_left_hippocampus_mm3"] == "3931.0"
    assert rows[1]["fs_data_origin"] == "local_existing"
    for column in MEASURE_COLUMNS:
        assert column in rows[0]
