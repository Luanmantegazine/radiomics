#!/usr/bin/env python3
"""Validate that radiomics acquisition satisfies the frozen study protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from oasis_radiomics.acquisition import (
    read_long_feature_rows,
    validate_long_feature_rows,
    write_validation_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the final OASIS-3 radiomics acquisition table.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("results/radiomics_features_long.csv"),
        help="Long-format radiomics table.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/acquisition_validation.json"),
        help="Machine-readable validation report.",
    )
    args = parser.parse_args()

    summary = validate_long_feature_rows(read_long_feature_rows(args.features))
    write_validation_summary(summary, args.output)

    print(f"Protocol: {summary.protocol_version}")
    print(f"Subjects seen: {summary.subjects_seen}")
    print(f"Sessions seen: {summary.sessions_seen}")
    print(f"Valid sessions: {summary.valid_sessions}")
    print(f"Invalid sessions: {summary.invalid_sessions}")
    print(f"Expected: {summary.expected_rois_per_session} ROIs x {summary.expected_features_per_roi} features = {summary.expected_raw_features_per_session} raw features/session")
    print(f"Report: {args.output}")

    if summary.invalid_sessions:
        for issue in summary.issues[:20]:
            print(f"ERROR {issue.session_id}: {issue.code} ({issue.detail})")
        if len(summary.issues) > 20:
            print(f"... {len(summary.issues) - 20} additional issue(s) in the JSON report")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
