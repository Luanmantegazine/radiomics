#!/usr/bin/env python3
"""Extract FreeSurfer morphometry and write the audit artefacts."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oasis_radiomics.morphometry import build_and_write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract FreeSurfer morphometry.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--downloaded-subjects-dir", type=Path, default=None)
    parser.add_argument("--download-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--download-mode", choices=["none", "freesurfer", "raw-t1"], default="none")
    parser.add_argument("--scope", choices=["primary", "all-eligible"], default="primary")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-all-eligible", action="store_true")
    parser.add_argument("--allow-raw-download", action="store_true")
    parser.add_argument("--session-map-csv", type=Path, default=None, help="Optional explicit directory/session mapping.")
    parser.add_argument("--official-spreadsheet-csv", type=Path, default=None, help="Legacy alias for --official-freesurfer-measures-csv.")
    parser.add_argument("--official-freesurfer-index-csv", type=Path, default=None, help="Official FreeSurfer index export.")
    parser.add_argument("--official-freesurfer-measures-csv", type=Path, default=None, help="Official FreeSurfer measures export.")
    parser.add_argument("--nitrc-user", type=str, default=None, help="NITRC-IR username; otherwise read from env or prompt.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = build_and_write_outputs(
        input_csv=args.input_csv,
        subjects_dir=args.subjects_dir,
        downloaded_subjects_dir=args.downloaded_subjects_dir,
        output_csv=args.output_csv,
        audit_dir=args.audit_dir,
        download_mode=args.download_mode,
        scope=args.scope,
        dry_run=args.dry_run,
        confirm_all_eligible=args.confirm_all_eligible,
        download_dir=args.download_dir,
        session_map_csv=args.session_map_csv,
        official_spreadsheet_csv=args.official_spreadsheet_csv,
        official_freesurfer_index_csv=args.official_freesurfer_index_csv,
        official_freesurfer_measures_csv=args.official_freesurfer_measures_csv,
        nitrc_user=args.nitrc_user,
        allow_raw_download=args.allow_raw_download,
        enforce_primary_preflight_validation=True,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    if args.dry_run:
        with outputs["preflight"].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with outputs["download_manifest"].open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        print(f"preflight_sessions={len(rows)}")
        print(f"preflight_patients={len({row['subject_id'] for row in rows})}")
        print(f"preflight_CN={sum(row['supervised_label'] == 'CN' for row in rows)}")
        print(f"preflight_AD={sum(row['supervised_label'] == 'AD' for row in rows)}")
        estimated_size = sum(
            int(float(row["estimated_size_bytes"]))
            for row in rows
            if row.get("estimated_size_bytes") not in (None, "", "nan")
        )
        print(f"preflight_estimated_size_bytes={estimated_size}")
        disk_path = args.download_dir if args.download_dir and args.download_dir.exists() else args.audit_dir
        free_space = shutil.disk_usage(disk_path).free
        print(f"free_space_bytes={free_space}")
        status_counts = {
            "remote_available": sum(row["remote_availability_status"] == "available" for row in rows),
            "remote_not_available": sum(row["remote_availability_status"] == "not_available" for row in rows),
            "remote_not_checked": sum(row["remote_availability_status"] == "not_checked" for row in rows),
            "remote_auth_required": sum(row["remote_availability_status"] == "auth_required" for row in rows),
            "remote_lookup_error": sum(row["remote_availability_status"] == "lookup_error" for row in rows),
            "remote_multiple_candidates": sum(row["remote_availability_status"] == "multiple_remote_candidates" for row in rows),
        }
        for key, value in status_counts.items():
            print(f"{key}={value}")
        print(f"sessions_with_multiple_candidates={sum(row['remote_availability_status'] == 'multiple_remote_candidates' for row in rows)}")
        print(f"sessions_served_by_official_spreadsheet={sum(row['planned_action'] == 'use_official_spreadsheet' for row in rows)}")
        print(f"sessions_needing_freesurfer_package={sum(row['planned_action'] == 'download_freesurfer_derivative' for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
