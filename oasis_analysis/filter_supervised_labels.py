#!/usr/bin/env python3
"""Prepare one consistently labeled CN or AD session per subject."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd


LABEL_COLUMN = "supervised_label"
SUBJECT_COLUMN = "subject_id"
GAP_COLUMN = "clinical_mri_gap_days"
LABELS_TO_KEEP = {"CN", "AD"}
SIMPLIFIED_COLUMNS = [
    SUBJECT_COLUMN,
    "session_id",
    "age_at_mri",
    "sex",
    LABEL_COLUMN,
]
SCRIPT_NAME = Path(__file__).stem
OUTPUT_DIR = Path("output") / SCRIPT_NAME
LOG_DIR = OUTPUT_DIR
LOGGER = logging.getLogger(__name__)


def default_output_path(input_path: Path) -> Path:
    """Return the default path for the retained rows."""
    return OUTPUT_DIR / f"{input_path.stem}_cn_ad.csv"


def discarded_output_path(input_path: Path) -> Path:
    """Return the default path for rows excluded by the filter."""
    return OUTPUT_DIR / f"{input_path.stem}_discarded.csv"


def simplified_output_path(output_path: Path) -> Path:
    """Return the simplified retained-output path."""
    return output_path.with_name(f"{output_path.stem}_simplified.csv")


def default_log_path(input_path: Path) -> Path:
    """Return the default path for the run log."""
    return LOG_DIR / f"{input_path.stem}_filter.log"


def configure_logging(log_path: Path) -> None:
    """Write informational messages to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def relative_path(path: Path) -> Path:
    """Return path relative to the current working directory for display."""
    return Path(os.path.relpath(path.resolve(), start=Path.cwd()))


def report(message: str) -> None:
    """Show a message on screen and write it to the log file."""
    print(message)
    LOGGER.info(message)


def filter_csv(
    input_path: Path,
    output_path: Path,
    simplified_path: Path,
    discarded_path: Path,
) -> dict[str, int]:
    """Filter the CSV and return summary statistics."""
    paths = {
        input_path.resolve(),
        output_path.resolve(),
        simplified_path.resolve(),
        discarded_path.resolve(),
    }
    if len(paths) != 4:
        raise ValueError("All input and output paths must differ.")

    data = pd.read_csv(input_path, low_memory=False)
    required_columns = set(SIMPLIFIED_COLUMNS) | {GAP_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Required column(s) not found: {missing}.")

    eligible_data = data[data[LABEL_COLUMN].isin(LABELS_TO_KEEP)].copy()

    labels_per_subject = eligible_data.groupby(SUBJECT_COLUMN)[LABEL_COLUMN].nunique()
    mixed_subjects = labels_per_subject[labels_per_subject > 1].index
    mixed_mask = eligible_data[SUBJECT_COLUMN].isin(mixed_subjects)
    mixed_rows = int(mixed_mask.sum())
    consistent_data = eligible_data[~mixed_mask].copy()

    numeric_gap = pd.to_numeric(consistent_data[GAP_COLUMN], errors="coerce")
    if numeric_gap.isna().any():
        invalid_rows = int(numeric_gap.isna().sum())
        raise ValueError(
            f"{GAP_COLUMN!r} contains {invalid_rows} missing or non-numeric value(s) "
            "among eligible rows."
        )

    consistent_data["_absolute_gap"] = numeric_gap.abs()
    kept_indices = consistent_data.groupby(SUBJECT_COLUMN, sort=False)[
        "_absolute_gap"
    ].idxmin()

    filtered_data = data.loc[kept_indices].sort_index()
    discarded_data = data.drop(index=kept_indices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simplified_path.parent.mkdir(parents=True, exist_ok=True)
    discarded_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_data.to_csv(output_path, index=False)
    filtered_data[SIMPLIFIED_COLUMNS].to_csv(simplified_path, index=False)
    discarded_data.to_csv(discarded_path, index=False)

    kept_counts = filtered_data[LABEL_COLUMN].value_counts()
    ad_kept = int(kept_counts.get("AD", 0))
    cn_kept = int(kept_counts.get("CN", 0))
    discarded = len(discarded_data)
    extra_sessions = len(consistent_data) - len(filtered_data)
    return {
        "initial_rows": len(data),
        "ad_kept": ad_kept,
        "cn_kept": cn_kept,
        "discarded": discarded,
        "mixed_subjects": len(mixed_subjects),
        "mixed_rows": mixed_rows,
        "extra_sessions": extra_sessions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep one CN or AD session per consistently labeled subject, selecting "
            "the session with the smallest absolute clinical/MRI gap."
        )
    )
    parser.add_argument("input_csv", type=Path, help="CSV file to filter")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Retained CSV "
            "(default: output/filter_supervised_labels/<input_name>_cn_ad.csv)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input_csv)
    simplified_path = simplified_output_path(output_path)
    discarded_path = discarded_output_path(args.input_csv)
    log_path = default_log_path(args.input_csv)
    configure_logging(log_path)
    report(f"Input: {relative_path(args.input_csv)}")

    try:
        stats = filter_csv(
            args.input_csv, output_path, simplified_path, discarded_path
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Filtering failed: %s", error)
        raise SystemExit(f"Error: {error}") from error

    total_kept = stats["ad_kept"] + stats["cn_kept"]
    report(f"Initial rows: {stats['initial_rows']}")
    report(f"Subjects removed for mixed AD/CN diagnoses: {stats['mixed_subjects']}")
    report(f"Rows removed for mixed AD/CN diagnoses: {stats['mixed_rows']}")
    report(f"Additional sessions discarded: {stats['extra_sessions']}")
    report(f"AD rows kept: {stats['ad_kept']}")
    report(f"CN rows kept: {stats['cn_kept']}")
    report(f"Total rows kept: {total_kept}")
    report(f"Total rows discarded: {stats['discarded']}")
    report(f"Retained output: {relative_path(output_path)}")
    report(f"Simplified retained output: {relative_path(simplified_path)}")
    report(f"Discarded output: {relative_path(discarded_path)}")
    report(f"Log: {relative_path(log_path)}")


if __name__ == "__main__":
    main()
