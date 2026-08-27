#!/usr/bin/env python3
"""Build an OASIS-3 FreeSurfer download manifest without downloading data.

Input: a catalogue CSV exported/listed from OASIS/NITRC with a freesurfer_id column.
Output: downloader-compatible IDs plus a subject-level audit table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oasis_radiomics.acquisition import (
    eligible_subjects,
    read_catalogue,
    select_subjects,
    write_download_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the final OASIS-3 acquisition manifest.")
    parser.add_argument("--catalog", type=Path, required=True, help="CSV containing a freesurfer_id column.")
    parser.add_argument("--output", type=Path, default=Path("acquisition"), help="Output directory.")
    parser.add_argument("--min-sessions", type=int, default=2, help="Minimum FreeSurfer visits per subject.")
    parser.add_argument(
        "--target-subjects",
        type=int,
        default=None,
        help="Desired final participant count. Omit to select every eligible subject.",
    )
    parser.add_argument(
        "--oversample",
        type=float,
        default=1.20,
        help="Acquisition margin for expected file/QC attrition when target-subjects is set.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Reproducible subject sampling seed.")
    args = parser.parse_args()

    catalogue = read_catalogue(args.catalog)
    eligible = eligible_subjects(catalogue, min_sessions=args.min_sessions)
    selected = select_subjects(
        eligible,
        target_subjects=args.target_subjects,
        oversample=args.oversample,
        seed=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    ids_path, subjects_path = write_download_manifest(
        eligible,
        selected,
        args.output / "acquisition_freesurfer_ids.csv",
        args.output / "acquisition_subjects.csv",
    )

    n_sessions = sum(len(eligible[subject]) for subject in selected)
    print(f"Catalogue sessions: {len(catalogue)}")
    print(f"Longitudinally eligible subjects: {len(eligible)}")
    print(f"Selected subjects: {len(selected)}")
    print(f"Selected sessions: {n_sessions}")
    print(f"Downloader IDs: {ids_path}")
    print(f"Subject audit table: {subjects_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
