#!/usr/bin/env python3
"""Audit local FreeSurfer availability against the morphometry cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oasis_radiomics.morphometry import (
    INVENTORY_COLUMNS,
    DOWNLOAD_MANIFEST_COLUMNS,
    build_inventory,
    build_download_manifest,
    load_session_map,
    load_session_rows,
    select_primary_sessions,
    _write_csv,  # noqa: SLF001 - script wrapper
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit local FreeSurfer availability.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--subjects-dir", type=Path, required=True)
    parser.add_argument("--downloaded-subjects-dir", type=Path, default=None)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--session-map-csv", type=Path, default=None, help="Optional explicit directory/session mapping.")
    parser.add_argument("--official-freesurfer-index-csv", type=Path, default=None)
    parser.add_argument("--official-freesurfer-measures-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_session_rows(args.input_csv)
    session_map = load_session_map(args.session_map_csv)
    inventory, directories, _ = build_inventory(rows, args.subjects_dir, args.downloaded_subjects_dir, session_map)
    manifest = build_download_manifest(inventory)

    args.audit_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.audit_dir / "morphometry_inventory.csv"
    manifest_path = args.audit_dir / "morphometry_download_manifest.csv"
    _write_csv(inventory, inventory_path, INVENTORY_COLUMNS)
    _write_csv(manifest, manifest_path, DOWNLOAD_MANIFEST_COLUMNS)

    primary_sessions = select_primary_sessions(rows)
    print(f"Subjects: {len({row['subject_id'] for row in rows})}")
    print(f"Sessions: {len(rows)}")
    print(f"Primary CN/AD subjects: {len(primary_sessions)}")
    print(f"Local stats directories discovered: {len(directories)}")
    for row in inventory[:5]:
        print(
            f"{row['subject_id']} {row['session_id']} -> {row['local_subject_directory'] or 'not found'} "
            f"({row['match_status']})"
        )
    print(f"Inventory written to: {inventory_path}")
    print(f"Download manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
