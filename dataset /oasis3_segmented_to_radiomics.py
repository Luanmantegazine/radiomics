#!/usr/bin/env python3
"""OASIS-3 smoke test - backwards-compatible entry point.

This script used to contain the whole proof of concept. The logic now lives in
the :mod:`oasis_radiomics` package and this file is a thin wrapper that keeps
the original command line working::

    python oasis3_segmented_to_radiomics.py --skip-download
    python oasis3_segmented_to_radiomics.py --nitrc-user <user> --max-cases 2
    python oasis3_segmented_to_radiomics.py --make-example-csv

What it does
------------
1. Reads a small CSV with OASIS-3 FreeSurfer IDs.
2. Optionally uses the official NrgXnat/oasis-scripts downloader to fetch only
   those FreeSurfer processed sessions from NITRC-IR.
3. Finds T1.mgz and aseg.mgz in the FreeSurfer outputs.
4. Builds left and right hippocampus masks from the aseg labels::

       17 = Left-Hippocampus
       53 = Right-Hippocampus

5. Converts image and masks to NIfTI.
6. Extracts Original-image radiomic features with PyRadiomics.
7. Saves ``<out>/radiomics_features.csv`` exactly where the original script did.

Difference from the original version
------------------------------------
The bilateral union ROI (17 + 53 in one mask) is **no longer extracted by
default**. It is two spatially disconnected objects, so its shape features are
not interpretable. Bilateral quantities are now derived tabularly - see
``python cli.py longitudinal``. Pass ``--legacy-bilateral`` to reproduce the old
three-ROI behaviour anyway.

For the longitudinal pipeline use the richer interface instead::

    python cli.py run --input oasis3_radiomics_smoketest/freesurfer --output results/

This is a proof-of-concept/smoke test, NOT a final radiomics protocol.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from oasis_radiomics.config import PipelineConfig
from oasis_radiomics.download_oasis import (
    create_example_ids_csv,
    download_freesurfer_sessions,
    read_freesurfer_ids,
    write_freesurfer_ids,
)
from oasis_radiomics.logging_setup import configure_logging
from oasis_radiomics.pipeline import LONG_LEADING_COLUMNS, run_extraction
from oasis_radiomics.tables import write_csv

logger = logging.getLogger("oasis3_segmented_to_radiomics")

LEGACY_FEATURES_CSV = "radiomics_features.csv"


def build_parser() -> argparse.ArgumentParser:
    """The original argument parser, plus ``--config`` and ``--legacy-bilateral``."""
    parser = argparse.ArgumentParser(
        description="Small OASIS-3 FreeSurfer -> hippocampus -> PyRadiomics smoke test."
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=Path("freesurfer_ids.csv"),
        help="CSV containing a freesurfer_id column.",
    )
    parser.add_argument(
        "--nitrc-user",
        help="NITRC-IR username. Required unless --skip-download is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("oasis3_radiomics_smoketest"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=2,
        help="Maximum number of FreeSurfer sessions to download/process (default: 2).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and process existing FreeSurfer files under <out>/freesurfer.",
    )
    parser.add_argument(
        "--make-example-csv",
        action="store_true",
        help="Create a 2-session example freesurfer_ids.csv and exit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to radiomics_config.yaml (defaults to the one next to this script).",
    )
    parser.add_argument(
        "--legacy-bilateral",
        action="store_true",
        help=(
            "Also extract the old bilateral union mask (17+53). Not recommended: "
            "shape features on a disconnected mask are not interpretable."
        ),
    )
    return parser


def _config_for(args: argparse.Namespace) -> PipelineConfig:
    """Load the configuration, honouring ``--legacy-bilateral``."""
    config = PipelineConfig.load(args.config)
    if not args.legacy_bilateral:
        return config

    logger.warning(
        "--legacy-bilateral: extracting the union ROI (17+53). Its shape features "
        "describe two disconnected objects and must not be interpreted."
    )
    bilateral = dataclasses.replace(config.bilateral, extract_union_mask=True)
    return dataclasses.replace(config, bilateral=bilateral)


def main(argv: list[str] | None = None) -> int:
    """Run the original smoke test on top of the refactored pipeline."""
    args = build_parser().parse_args(argv)
    configure_logging(logging.INFO)

    if args.make_example_csv:
        create_example_ids_csv(args.ids)
        logger.info("Review the IDs in NITRC-IR before downloading if needed.")
        return 0

    if args.max_cases < 1:
        raise ValueError("--max-cases must be >= 1")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    download_dir = out_dir / "freesurfer"

    ids = read_freesurfer_ids(args.ids, args.max_cases)
    write_freesurfer_ids(ids, out_dir / "selected_freesurfer_ids.csv")
    logger.info("Selected FreeSurfer IDs: %s", ", ".join(ids))

    if not args.skip_download:
        if not args.nitrc_user:
            raise ValueError("--nitrc-user is required unless --skip-download is used.")
        download_freesurfer_sessions(
            ids_csv=out_dir / "selected_freesurfer_ids.csv",
            download_dir=download_dir,
            nitrc_user=args.nitrc_user,
            repo_dir=out_dir / "_oasis_scripts",
        )

    result = run_extraction(
        input_dir=download_dir,
        output_dir=out_dir,
        config=_config_for(args),
        max_sessions=args.max_cases,
        prepared_dir=out_dir / "prepared_nifti",
        write_outputs=False,
    )

    output_csv = out_dir / LEGACY_FEATURES_CSV
    write_csv(result.long_rows, output_csv, LONG_LEADING_COLUMNS)

    logger.info("DONE")
    logger.info("Radiomic feature table: %s", output_csv)
    logger.info("Prepared NIfTI files:   %s", out_dir / "prepared_nifti")
    logger.info("Downloaded FreeSurfer:  %s", download_dir)
    logger.info(
        "Reminder: bin width, normalization, resampling, ROI definitions and "
        "feature robustness must be defined rigorously before the real study."
    )
    logger.info("For the longitudinal dataset run: python cli.py run --input %s", download_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
