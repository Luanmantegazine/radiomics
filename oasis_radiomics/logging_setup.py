"""Logging configuration shared by the CLI and the legacy entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(levelname)-7s %(name)-28s %(message)s"
LOG_FORMAT_VERBOSE = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"

#: PyRadiomics is extremely chatty at INFO level; keep it at WARNING by default.
PYRADIOMICS_LOG_LEVEL = logging.WARNING


def configure_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Install a stream handler (and optionally a file handler) on the root logger.

    Existing handlers are replaced so repeated calls (tests, notebooks) do not
    duplicate every log line.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        LOG_FORMAT_VERBOSE if level <= logging.DEBUG else LOG_FORMAT
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT_VERBOSE))
        root.addHandler(file_handler)

    root.setLevel(level)
    quiet_pyradiomics(level)


def quiet_pyradiomics(level: int = logging.INFO) -> None:
    """Tame PyRadiomics' logger so it neither floods nor duplicates the output.

    PyRadiomics installs its own ``StreamHandler`` on import *and* leaves
    propagation on, so every warning it emits is printed twice. Its handlers are
    removed here and its level is raised to WARNING unless the caller asked for
    DEBUG, in which case its per-feature messages are genuinely useful.
    """
    try:
        import radiomics
    except ImportError:  # pragma: no cover - environment problem
        return

    for handler in list(radiomics.logger.handlers):
        radiomics.logger.removeHandler(handler)
    radiomics.logger.propagate = True
    radiomics.logger.setLevel(level if level <= logging.DEBUG else PYRADIOMICS_LOG_LEVEL)
