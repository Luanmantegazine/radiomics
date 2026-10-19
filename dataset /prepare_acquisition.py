#!/usr/bin/env python3
"""Build an OASIS-3 FreeSurfer download manifest without downloading data.

Accepted catalogue formats
--------------------------
1. FreeSurfer catalogue with a ``freesurfer_id`` column.
2. Official OASIS MRI session spreadsheet with columns such as
   ``Label,Subject,Scanner,Scans``. For this second format, 3.0T sessions that
   contain at least one T1w scan are converted into candidate
   ``Freesurfer53`` identifiers. Candidate means that the MRI session is
   eligible to be queried/downloaded; the actual FreeSurfer resource is still
   confirmed by the official NITRC downloader.

Output: downloader-compatible IDs plus a subject-level audit table.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from oasis_radiomics.acquisition import (
    AcquisitionError,
    CatalogueSession,
    eligible_subjects,
    read_catalogue,
    select_subjects,
    write_download_manifest,
)

MRI_LABEL_RE = re.compile(r"^(?P<subject>OAS\d+)_MR_d(?P<day>\d+)$")


def _read_input_catalogue(path: Path, freesurfer_version: str) -> tuple[list[CatalogueSession], str]:
    """Read either the native FreeSurfer-id CSV or the OASIS MRI spreadsheet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])

        if "freesurfer_id" in fieldnames:
            return read_catalogue(path), "freesurfer"

        required = {"Label", "Subject", "Scanner", "Scans"}
        if not required.issubset(fieldnames):
            raise AcquisitionError(
                f"{path} must contain either 'freesurfer_id' or the OASIS MRI "
                f"columns {sorted(required)}. Found: {sorted(fieldnames)}"
            )

        sessions: list[CatalogueSession] = []
        for row in reader:
            label = (row.get("Label") or "").strip()
            scanner = (row.get("Scanner") or "").strip()
            scans = (row.get("Scans") or "").strip()
            subject = (row.get("Subject") or "").strip()

            # Protocol v1.0 uses the OASIS 3T / FreeSurfer53 processing regime.
            if scanner != "3.0T" or "T1w(" not in scans:
                continue

            match = MRI_LABEL_RE.fullmatch(label)
            if not match:
                continue
            if subject and subject != match.group("subject"):
                raise AcquisitionError(
                    f"Subject/Label mismatch in {path}: Subject={subject!r}, Label={label!r}"
                )

            subject_id = match.group("subject")
            day_text = match.group("day")
            sessions.append(
                CatalogueSession(
                    freesurfer_id=f"{subject_id}_Freesurfer{freesurfer_version}_d{day_text}",
                    subject_id=subject_id,
                    days_from_reference=int(day_text),
                    freesurfer_version=str(freesurfer_version),
                )
            )

    unique = {session.freesurfer_id: session for session in sessions}
    parsed = sorted(
        unique.values(),
        key=lambda item: (item.subject_id, item.days_from_reference, item.freesurfer_id),
    )
    if not parsed:
        raise AcquisitionError(
            f"No 3.0T T1w MRI sessions could be converted from {path}."
        )
    return parsed, "mri_candidates"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the final OASIS-3 acquisition manifest.")
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help=(
            "CSV with either a freesurfer_id column or the official OASIS MRI "
            "columns Label, Subject, Scanner and Scans."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("acquisition"), help="Output directory.")
    parser.add_argument("--min-sessions", type=int, default=2, help="Minimum candidate FreeSurfer visits per subject.")
    parser.add_argument(
        "--freesurfer-version",
        default="53",
        help="Keep/infer this OASIS FreeSurfer processing version (default: 53).",
    )
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
        help="Acquisition margin for expected unavailable resources/QC attrition.",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Reproducible subject sampling seed.")
    args = parser.parse_args()

    all_catalogue, source_format = _read_input_catalogue(
        args.catalog, str(args.freesurfer_version)
    )
    catalogue = [
        session
        for session in all_catalogue
        if session.freesurfer_version == str(args.freesurfer_version)
    ]
    if not catalogue:
        parser.error(
            f"No sessions with Freesurfer{args.freesurfer_version} were found/inferred in {args.catalog}."
        )

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
    print(f"Catalogue format: {source_format}")
    print(f"Catalogue candidate sessions: {len(all_catalogue)}")
    print(f"Freesurfer{args.freesurfer_version} candidate sessions: {len(catalogue)}")
    print(f"Longitudinally eligible subjects: {len(eligible)}")
    print(f"Selected subjects: {len(selected)}")
    print(f"Selected candidate sessions: {n_sessions}")
    print(f"Downloader IDs: {ids_path}")
    print(f"Subject audit table: {subjects_path}")
    if source_format == "mri_candidates":
        print(
            "NOTE: IDs were inferred from 3.0T T1w MRI sessions. The official "
            "NITRC download step is the availability check for each FreeSurfer resource."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
