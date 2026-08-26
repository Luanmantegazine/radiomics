"""Run provenance: versions, configuration and dataset counts.

Every run writes a ``run_metadata.json`` next to its CSVs. Without it a feature
table is not reproducible: the same ROI can yield very different texture values
under a different bin width, a different normalisation, or a different
PyRadiomics build.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import __version__
from .config import PipelineConfig

logger = logging.getLogger(__name__)

RUN_METADATA_FILENAME = "run_metadata.json"


def _package_version(module_name: str) -> str:
    """Version of an installed package, or ``'not installed'``."""
    try:
        module = __import__(module_name)
    except ImportError:
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def collect_environment() -> dict[str, str]:
    """Versions of everything that can change a radiomic feature value."""
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "oasis_radiomics": __version__,
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "nibabel": _package_version("nibabel"),
        "scipy": _package_version("scipy"),
        "SimpleITK": _package_version("SimpleITK"),
        "pyradiomics": _package_version("radiomics"),
    }


def build_run_metadata(
    config: PipelineConfig,
    input_dir: Path,
    output_dir: Path,
    counts: Mapping[str, Any],
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Assemble the run metadata document."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": collect_environment(),
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "config": config.to_dict(),
        "counts": dict(counts),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "notes": [
            "Left and right hippocampi are extracted separately; bilateral "
            "quantities are derived tabularly, never from a disconnected union mask.",
            "NumPy 2.x must not be used with PyRadiomics 3.0.1 in this environment "
            "(observed segmentation fault); see requirements.txt.",
        ],
    }


def write_run_metadata(metadata: Mapping[str, Any], output_dir: Path) -> Path:
    """Serialise the run metadata to ``<output_dir>/run_metadata.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / RUN_METADATA_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=False, default=str)
        handle.write("\n")
    logger.info("Wrote run metadata: %s", path)
    return path
