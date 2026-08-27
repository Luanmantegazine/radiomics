"""Thin, configuration-driven wrapper around PyRadiomics.

Everything methodological (bin width, normalisation, feature classes, image
types) comes from :class:`~oasis_radiomics.config.PipelineConfig`; this module
only translates that configuration into a
``radiomics.featureextractor.RadiomicsFeatureExtractor`` and turns its output
into plain Python dictionaries.

Environment note
----------------
PyRadiomics 3.0.1 must be paired with NumPy 1.26.x. NumPy 2.x has been observed
to crash the extractor with a segmentation fault; see ``requirements.txt``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .config import PipelineConfig

logger = logging.getLogger(__name__)

DIAGNOSTICS_PREFIX = "diagnostics_"


class ExtractionError(RuntimeError):
    """Raised when PyRadiomics fails on a given image/mask pair."""


def build_extractor(config: PipelineConfig):
    """Instantiate a PyRadiomics extractor from ``config``.

    Only the feature classes and image types listed in the configuration are
    enabled; PyRadiomics' own defaults are switched off first so a run never
    silently picks up features nobody asked for.
    """
    try:
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ExtractionError(
            "PyRadiomics is not installed. Install the pinned environment with "
            "'pip install -r requirements.txt'."
        ) from exc

    settings = dict(config.radiomics)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllFeatures()
    for feature_class in config.feature_classes:
        extractor.enableFeatureClassByName(feature_class)

    extractor.disableAllImageTypes()
    for image_type, kwargs in config.image_types.items():
        extractor.enableImageTypeByName(image_type, customArgs=dict(kwargs) or None)

    logger.info(
        "PyRadiomics extractor ready (image types: %s; feature classes: %s)",
        ", ".join(config.image_types),
        ", ".join(config.feature_classes),
    )
    logger.debug("PyRadiomics settings: %s", settings)
    return extractor


def extract_roi_features(
    extractor,
    image_path: Path,
    mask_path: Path,
    keep_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run PyRadiomics on one image/mask pair and return plain Python values.

    Parameters
    ----------
    extractor:
        The object returned by :func:`build_extractor`.
    image_path, mask_path:
        NIfTI files sharing the same geometry.
    keep_diagnostics:
        Keep PyRadiomics' ``diagnostics_*`` entries. They are dropped by default
        because they are provenance strings, not features.

    Raises
    ------
    ExtractionError
        Wrapping any failure reported by PyRadiomics, with the offending paths.
    """
    try:
        result = extractor.execute(str(image_path), str(mask_path))
    except Exception as exc:
        raise ExtractionError(
            f"PyRadiomics failed on image={image_path} mask={mask_path}: {exc}"
        ) from exc

    features: dict[str, Any] = {}
    for key, value in result.items():
        if not keep_diagnostics and key.startswith(DIAGNOSTICS_PREFIX):
            continue
        features[key] = _to_python_scalar(value)

    if not features:
        raise ExtractionError(f"PyRadiomics returned no features for mask {mask_path}.")

    return features


def _to_python_scalar(value: Any) -> Any:
    """Convert NumPy scalars to built-in types; leave everything else alone."""
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def feature_names(row: Mapping[str, Any]) -> list[str]:
    """Radiomic feature names present in a row, in their original order."""
    return [key for key in row if not key.startswith(DIAGNOSTICS_PREFIX)]


def pyradiomics_version() -> str:
    """Installed PyRadiomics version, or ``'not installed'``."""
    try:
        import radiomics
    except ImportError:  # pragma: no cover - environment problem
        return "not installed"
    return str(getattr(radiomics, "__version__", "unknown"))
