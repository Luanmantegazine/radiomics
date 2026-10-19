"""Configuration-driven, protocol-locked wrapper around PyRadiomics.

Acquisition runs use a frozen 107-feature Original-image signature per ROI.
The explicit names live in :mod:`oasis_radiomics.protocol`, so upgrading a
library or changing defaults cannot silently change the dataset schema.

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
from .protocol import (
    EXPECTED_FEATURES_BY_CLASS,
    EXPECTED_FEATURE_COUNT,
    validate_extracted_features,
)

logger = logging.getLogger(__name__)

DIAGNOSTICS_PREFIX = "diagnostics_"


class ExtractionError(RuntimeError):
    """Raised when PyRadiomics fails or its output violates the frozen protocol."""


def build_extractor(config: PipelineConfig):
    """Instantiate the protocol-locked PyRadiomics extractor.

    The final acquisition protocol intentionally accepts only the ``Original``
    image type and the seven feature classes frozen in ``protocol.py``.  Every
    individual feature is enabled by name rather than relying on PyRadiomics
    class defaults.
    """
    try:
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ExtractionError(
            "PyRadiomics is not installed. Install the pinned environment with "
            "'pip install -r requirements.txt'."
        ) from exc

    expected_classes = tuple(EXPECTED_FEATURES_BY_CLASS)
    if tuple(config.feature_classes) != expected_classes:
        raise ExtractionError(
            "The acquisition protocol requires feature classes "
            f"{expected_classes}, got {tuple(config.feature_classes)}."
        )
    if set(config.image_types) != {"Original"}:
        raise ExtractionError(
            "The acquisition protocol is frozen to the Original image type only; "
            f"configured image types are {sorted(config.image_types)}."
        )

    settings = dict(config.radiomics)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(
        **{name: list(features) for name, features in EXPECTED_FEATURES_BY_CLASS.items()}
    )

    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")

    logger.info(
        "PyRadiomics extractor ready: Original image, %d frozen features/ROI (%s)",
        EXPECTED_FEATURE_COUNT,
        ", ".join(f"{name}={len(features)}" for name, features in EXPECTED_FEATURES_BY_CLASS.items()),
    )
    logger.debug("PyRadiomics settings: %s", settings)
    return extractor


def extract_roi_features(
    extractor,
    image_path: Path,
    mask_path: Path,
    keep_diagnostics: bool = False,
) -> dict[str, Any]:
    """Run PyRadiomics and require the frozen 107-feature output signature."""
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

    if not keep_diagnostics:
        try:
            validate_extracted_features(features)
        except RuntimeError as exc:
            raise ExtractionError(str(exc)) from exc

    logger.debug("Validated %d radiomic features for %s", EXPECTED_FEATURE_COUNT, mask_path)
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
