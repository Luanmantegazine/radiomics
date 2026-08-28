"""Command line interface for the OASIS-3 longitudinal radiomics pipeline.

Subcommands
-----------
``extract``       FreeSurfer sessions on disk -> ``radiomics_features_long.csv``
``longitudinal``  long feature table -> wide / deltas / slopes
``qc``            quality control only, no radiomics extraction
``run``           ``extract`` + ``longitudinal`` + ``run_metadata.json``
``download``      fetch sessions with the official NITRC-IR downloader (explicit only)
``clinical-link`` MRI catalogue + D1/B4/C1 -> clinical imaging master + validation
``clinical-radiomics`` clinical master + radiomics tables -> analysis datasets

Examples
--------
::

    python cli.py extract --input oasis3_radiomics_smoketest/freesurfer --output results/
    python cli.py longitudinal --features results/radiomics_features_long.csv --output results/
    python cli.py run --input oasis3_radiomics_smoketest/freesurfer --output results/

    python cli.py clinical-link \
        --mri-catalog oasis3_mri_catalog.csv \
        --d1 diagnostic/OASIS3_UDSd1_diagnoses.csv \
        --b4 diagnostic/OASIS3_UDSb4_cdr.csv \
        --c1 diagnostic/OASIS3_UDSc1_cognitive_assessments.csv \
        --clinical-window-days 180 --output clinical_results/

    python cli.py clinical-radiomics \
        --clinical clinical_results/clinical_imaging_master.csv \
        --radiomics results/radiomics_features_wide.csv \
        --deltas results/radiomics_longitudinal_deltas.csv \
        --slopes results/radiomics_longitudinal_slopes.csv \
        --output dataset/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .clinical import DEFAULT_CLINICAL_WINDOW_DAYS
from .clinical.classification import ClassificationCodebook, CodebookError
from .clinical.readers import ClinicalReaderError
from .clinical_dataset import (
    ClinicalDatasetError,
    build_clinical_linkage,
    build_clinical_radiomics,
    write_linkage_outputs,
)
from .config import ConfigError, PipelineConfig
from .discovery import DiscoveryError, discover_sessions
from .download_oasis import (
    DownloadError,
    create_example_ids_csv,
    download_freesurfer_sessions,
    read_freesurfer_ids,
    write_freesurfer_ids,
)
from .logging_setup import configure_logging
from .masks import build_roi_masks, load_volume
from .pipeline import (
    QC_CSV,
    QC_LEADING_COLUMNS,
    PipelineError,
    run_extraction,
    run_longitudinal_from_csv,
    run_pipeline,
)
from .quality_control import check_session, flag_outliers
from .tables import write_csv

logger = logging.getLogger("oasis_radiomics.cli")

DEFAULT_INPUT = Path("oasis3_radiomics_smoketest/freesurfer")
DEFAULT_OUTPUT = Path("results")
DEFAULT_FEATURES = DEFAULT_OUTPUT / "radiomics_features_long.csv"
DEFAULT_CLINICAL_OUTPUT = Path("clinical_results")
DEFAULT_DATASET_OUTPUT = Path("dataset")


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "OASIS-3 FreeSurfer -> hippocampal radiomics -> longitudinal dataset. "
            "Operates on data already present on disk; downloads are explicit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Extract radiomic features from FreeSurfer sessions on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(extract)
    extract.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory holding FreeSurfer sessions.")
    extract.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the result tables.")
    extract.add_argument("--max-sessions", type=int, default=None, help="Process at most N sessions.")
    extract.set_defaults(handler=_handle_extract)

    longitudinal = subparsers.add_parser(
        "longitudinal",
        help="Derive wide, delta and slope tables from a long feature table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(longitudinal)
    longitudinal.add_argument("--features", type=Path, default=DEFAULT_FEATURES, help="Long-format feature CSV.")
    longitudinal.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the result tables.")
    longitudinal.set_defaults(handler=_handle_longitudinal)

    quality = subparsers.add_parser(
        "qc",
        help="Run quality control only; no radiomics extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(quality)
    quality.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory holding FreeSurfer sessions.")
    quality.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for quality_control.csv.")
    quality.set_defaults(handler=_handle_qc)

    run = subparsers.add_parser(
        "run",
        help="Full pipeline: extraction + longitudinal derivation + metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(run)
    run.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Directory holding FreeSurfer sessions.")
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for the result tables.")
    run.add_argument("--max-sessions", type=int, default=None, help="Process at most N sessions.")
    run.set_defaults(handler=_handle_run)

    download = subparsers.add_parser(
        "download",
        help="Download FreeSurfer sessions with the official NITRC-IR scripts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(download)
    download.add_argument("--ids", type=Path, default=Path("freesurfer_ids.csv"), help="CSV with a freesurfer_id column.")
    download.add_argument("--output", type=Path, default=Path("oasis3_radiomics_smoketest"), help="Download root.")
    download.add_argument("--nitrc-user", help="NITRC-IR username (the password is prompted for by the official script).")
    download.add_argument("--max-cases", type=int, default=None, help="Download at most N sessions.")
    download.add_argument("--make-example-csv", action="store_true", help="Write a two-session example id CSV and exit.")
    download.set_defaults(handler=_handle_download)

    clinical_link = subparsers.add_parser(
        "clinical-link",
        help="Link MRI sessions to OASIS-3 clinical visits (D1/B4, optional C1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(clinical_link)
    clinical_link.add_argument("--mri-catalog", type=Path, required=True, help="OASIS MRI session catalogue export (Label/Subject/M-F/Age/Scanner).")
    clinical_link.add_argument("--d1", type=Path, required=True, help="OASIS3_UDSd1_diagnoses.csv (primary diagnostic source).")
    clinical_link.add_argument("--b4", type=Path, required=True, help="OASIS3_UDSb4_cdr.csv (CDR / MMSE / dx labels).")
    clinical_link.add_argument("--c1", type=Path, default=None, help="OASIS3_UDSc1_cognitive_assessments.csv (optional psychometrics).")
    clinical_link.add_argument("--clinical-window-days", type=int, default=DEFAULT_CLINICAL_WINDOW_DAYS, help="Half-width of the MRI<->clinical matching window, in days.")
    clinical_link.add_argument("--cognitive-window-days", type=int, default=None, help="Separate window for C1; defaults to --clinical-window-days.")
    clinical_link.add_argument("--codebook", type=Path, default=None, help="Path to clinical_classification.yaml.")
    clinical_link.add_argument("--output", type=Path, default=DEFAULT_CLINICAL_OUTPUT, help="Directory for the linkage outputs.")
    clinical_link.set_defaults(handler=_handle_clinical_link)

    clinical_radiomics = subparsers.add_parser(
        "clinical-radiomics",
        help="Join the clinical imaging master with the radiomics tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_common_arguments(clinical_radiomics)
    clinical_radiomics.add_argument("--clinical", type=Path, required=True, help="clinical_imaging_master.csv from 'clinical-link'.")
    clinical_radiomics.add_argument("--radiomics", type=Path, required=True, help="radiomics_features_wide.csv (read-only).")
    clinical_radiomics.add_argument("--deltas", type=Path, default=None, help="radiomics_longitudinal_deltas.csv (read-only).")
    clinical_radiomics.add_argument("--slopes", type=Path, default=None, help="radiomics_longitudinal_slopes.csv (read-only).")
    clinical_radiomics.add_argument("--clinical-visits", type=Path, default=None, help="clinical_visits.csv; auto-detected next to --clinical when omitted.")
    clinical_radiomics.add_argument("--output", type=Path, default=DEFAULT_DATASET_OUTPUT, help="Directory for the analysis datasets.")
    clinical_radiomics.set_defaults(handler=_handle_clinical_radiomics)

    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Options accepted both before and after the subcommand name."""
    parser.add_argument("--config", type=Path, default=None, help="Path to radiomics_config.yaml.")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--log-file", type=Path, default=None, help="Also write logs to this file.")


def _resolve_common(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Apply defaults for options that may appear on either side of the command."""
    args.log_level = args.log_level or "INFO"
    configure_logging(getattr(logging, args.log_level), args.log_file)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------
def _handle_extract(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``extract``: run the per-session radiomics extraction."""
    result = run_extraction(
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        max_sessions=args.max_sessions,
    )
    logger.info(
        "Extracted %d ROI row(s) from %d session(s).", len(result.long_rows), result.n_sessions
    )
    return 0


def _handle_longitudinal(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``longitudinal``: derive wide/delta/slope tables from a long CSV."""
    result = run_longitudinal_from_csv(args.features, args.output, config)
    logger.info(
        "Derived %d wide row(s), %d delta row(s), %d slope row(s).",
        len(result.wide_rows),
        len(result.delta_rows),
        len(result.slope_rows),
    )
    return 0


def _handle_qc(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``qc``: build masks and run quality control without extracting features."""
    sessions = discover_sessions(args.input)
    if not sessions:
        raise DiscoveryError(f"No FreeSurfer sessions found under {args.input}")

    rows = []
    for session in sessions:
        t1 = load_volume(session.t1_path)
        aseg = load_volume(session.aseg_path)
        masks = build_roi_masks(aseg, config.extraction_roi_labels)
        qc = check_session(
            session=session,
            t1=t1,
            aseg=aseg,
            masks=masks,
            config=config.quality_control,
            left_roi=config.bilateral.left_roi,
            right_roi=config.bilateral.right_roi,
        )
        rows.append(qc.as_row())

    rows = flag_outliers(rows, config.quality_control.outliers)
    write_csv(rows, Path(args.output) / QC_CSV, QC_LEADING_COLUMNS)
    return 0


def _handle_run(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``run``: the full pipeline."""
    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        config=config,
        max_sessions=args.max_sessions,
    )
    return 0


def _handle_download(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``download``: fetch sessions with the official OASIS scripts."""
    if args.make_example_csv:
        create_example_ids_csv(args.ids)
        return 0

    if not args.nitrc_user:
        raise DownloadError("--nitrc-user is required to download OASIS-3 data.")

    output_dir = Path(args.output)
    ids = read_freesurfer_ids(args.ids, args.max_cases)
    selected_csv = write_freesurfer_ids(ids, output_dir / "selected_freesurfer_ids.csv")

    logger.info("Selected FreeSurfer id(s): %s", ", ".join(ids))
    download_freesurfer_sessions(
        ids_csv=selected_csv,
        download_dir=output_dir / "freesurfer",
        nitrc_user=args.nitrc_user,
        repo_dir=output_dir / "_oasis_scripts",
    )
    return 0


def _handle_clinical_link(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``clinical-link``: build the clinical imaging master table."""
    codebook = ClassificationCodebook.load(args.codebook)
    result = build_clinical_linkage(
        mri_catalog=args.mri_catalog,
        d1_path=args.d1,
        b4_path=args.b4,
        c1_path=args.c1,
        window_days=args.clinical_window_days,
        cognitive_window_days=args.cognitive_window_days,
        codebook=codebook,
    )
    outputs = write_linkage_outputs(result, args.output)
    for name, path in outputs.items():
        logger.info("  %-24s -> %s", name, path)
    logger.info(
        "Linked %d/%d MRI session(s) within +/-%d days.",
        result.summary["valid_matches"],
        result.summary["mri_sessions"],
        args.clinical_window_days,
    )
    if not codebook.is_active:
        logger.warning(
            "Diagnostic classes were NOT derived: the classification codebook is "
            "unfrozen. Raw D1/B4 variables are preserved in the outputs."
        )
    return 0


def _handle_clinical_radiomics(args: argparse.Namespace, config: PipelineConfig) -> int:
    """``clinical-radiomics``: join the clinical master with the radiomics tables."""
    outputs = build_clinical_radiomics(
        clinical_master=args.clinical,
        radiomics_wide=args.radiomics,
        output_dir=args.output,
        deltas=args.deltas,
        slopes=args.slopes,
        clinical_visits=args.clinical_visits,
    )
    for name, path in outputs.items():
        logger.info("  %-30s -> %s", name, path)
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging and dispatch to the handler."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _resolve_common(args, parser)

    try:
        config = PipelineConfig.load(args.config)
        return int(args.handler(args, config))
    except (
        ConfigError,
        DiscoveryError,
        DownloadError,
        PipelineError,
        ClinicalDatasetError,
        ClinicalReaderError,
        CodebookError,
    ) as exc:
        logger.error("%s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
