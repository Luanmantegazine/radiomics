"""Radiomic feature extraction with PyRadiomics under a versioned configuration.

Every PyRadiomics setting that affects feature values is spelled out in the
configuration file (no reliance on library defaults), and the configuration is
identified by ``config_id`` plus a content hash recorded with every result.

IBSI
----
PyRadiomics follows the IBSI feature definitions with documented deviations,
which this module does not correct:

* fixed-bin-width discretisation uses bin edges at multiples of ``binWidth``
  (IBSI anchors them at the ROI minimum / resegmentation bound);
* resampling aligns the grids by the origin voxel, not the image centre;
* first-order kurtosis is not excess kurtosis (it is 3 for a normal
  distribution);
* features PyRadiomics marks as deprecated are not extracted.

Hypotheses and limitations
--------------------------
* T1-weighted MRI has no absolute intensity scale. With ``normalize: true``
  intensities are z-scored over the **whole image** (PyRadiomics behaviour),
  which makes them depend on field of view and skull-stripping. Upstream
  preprocessing (bias correction, registration, skull-stripping) is *not*
  performed here; it is described by the ``preprocessing`` field and must be
  identical across sites.
* The feature schema must be identical for every case: a case that yields a
  different set of feature names, or a non-finite value, fails.
* Feature values depend on the PyRadiomics/NumPy/SimpleITK versions, which are
  recorded in the provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from client.imaging.validation import ValidationSettings

PIPELINE_VERSION = "1"
DIAGNOSTICS_PREFIX = "diagnostics_"
FEATURE_CLASSES = ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
IMAGE_TYPES = ("Original", "LoG", "Wavelet")
Interpolator = Literal[
    "sitkNearestNeighbor", "sitkLinear", "sitkBSpline", "sitkGaussian", "sitkLanczosWindowedSinc"
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RadiomicsSettings(_Model):
    """PyRadiomics settings, under their PyRadiomics names. All are required."""

    binWidth: float | None = Field(gt=0)
    binCount: int | None = Field(ge=2)
    normalize: bool
    normalizeScale: float = Field(gt=0)
    removeOutliers: float | None = Field(gt=0)
    resampledPixelSpacing: tuple[float, float, float] | None
    interpolator: Interpolator
    padDistance: int = Field(ge=0)
    preCrop: bool
    resegmentRange: tuple[float, float] | None
    resegmentMode: Literal["absolute", "relative", "sigma"]
    force2D: bool
    force2Ddimension: int = Field(ge=0, le=2)
    distances: tuple[int, ...] = Field(min_length=1)
    symmetricalGLCM: bool
    weightingNorm: Literal["manhattan", "euclidean", "infinity", "no_weighting"] | None
    gldm_a: int = Field(ge=0)
    voxelArrayShift: float
    minimumROIDimensions: int = Field(ge=1, le=3)
    minimumROISize: int | None = Field(ge=1)

    @model_validator(mode="after")
    def _one_discretisation(self) -> RadiomicsSettings:
        if (self.binWidth is None) == (self.binCount is None):
            raise ValueError("set exactly one of binWidth / binCount (the other must be null)")
        return self

    def to_pyradiomics(self) -> dict[str, Any]:
        settings = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.model_dump().items()
        }
        # PyRadiomics treats the presence of both keys as a conflict.
        settings.pop("binCount" if self.binCount is None else "binWidth")
        return settings


class ExtractionConfig(_Model):
    """Versioned extraction protocol."""

    schema_version: Literal[1]
    config_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    description: str = Field(max_length=2000)
    modality: Literal["T1w"]
    preprocessing: str = Field(min_length=1, max_length=2000)
    """Upstream preprocessing the inputs are assumed to have received."""
    validation: ValidationSettings
    settings: RadiomicsSettings
    image_types: dict[Literal["Original", "LoG", "Wavelet"], dict[str, Any]] = Field(min_length=1)
    """Image type -> custom arguments (e.g. ``LoG: {sigma: [1.0, 2.0]}``)."""
    feature_classes: tuple[
        Literal["shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"], ...
    ] = Field(min_length=1)
    expected_features_per_roi: int = Field(ge=1)
    """Guard against silent schema drift (library upgrade, changed defaults)."""

    @model_validator(mode="after")
    def _checks(self) -> ExtractionConfig:
        if len(set(self.feature_classes)) != len(self.feature_classes):
            raise ValueError("feature_classes must be unique")
        if "LoG" in self.image_types and not self.image_types["LoG"].get("sigma"):
            raise ValueError("LoG requires an explicit 'sigma' list")
        return self

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_extraction_config(path: str | Path) -> ExtractionConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return ExtractionConfig.model_validate(yaml.safe_load(handle))


class ExtractionError(RuntimeError):
    """Extraction failed for one case. ``code`` is a short machine-readable tag."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    features: dict[str, float]
    diagnostics: dict[str, Any]


def library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("pyradiomics", "numpy", "SimpleITK", "PyWavelets"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    try:
        import radiomics

        # The installed distribution reports a placeholder version when built
        # from git; the module version carries the actual tag distance + commit.
        versions["pyradiomics"] = str(radiomics.__version__)
    except ImportError:
        pass
    return versions


class PyRadiomicsExtractor:
    """`client.interfaces.FeatureExtractor` backed by PyRadiomics."""

    def __init__(self, config: ExtractionConfig) -> None:
        try:
            from radiomics import featureextractor
        except ImportError as exc:  # pragma: no cover - environment problem
            raise ExtractionError(
                "pyradiomics_missing", "install with: uv sync --extra imaging"
            ) from exc
        # Emitted once per case and purely informational (redundant feature pair).
        logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)
        self.config = config
        self.config_id = config.config_id
        extractor = featureextractor.RadiomicsFeatureExtractor(
            **config.settings.to_pyradiomics(),
            geometryTolerance=config.validation.geometry_tolerance,
            correctMask=False,  # geometry mismatches are validation errors, never resampled
            additionalInfo=True,
        )
        extractor.disableAllImageTypes()
        for image_type, arguments in config.image_types.items():
            extractor.enableImageTypeByName(image_type, customArgs=dict(arguments))
        extractor.disableAllFeatures()
        for feature_class in config.feature_classes:
            extractor.enableFeatureClassByName(feature_class)
        self._extractor = extractor

    def extract(self, image: Path, mask: Path, label: int = 1) -> ExtractionResult:
        """Extract the configured features for one ROI."""
        try:
            raw = self._extractor.execute(str(image), str(mask), label=label)
        except Exception as exc:  # PyRadiomics raises plain ValueError/RuntimeError
            raise ExtractionError("pyradiomics_error", str(exc)) from exc
        features: dict[str, float] = {}
        diagnostics: dict[str, Any] = {}
        for key, value in raw.items():
            if key.startswith(DIAGNOSTICS_PREFIX):
                diagnostics[key.removeprefix(DIAGNOSTICS_PREFIX)] = value
                continue
            number = float(value)
            if not math.isfinite(number):
                raise ExtractionError("non_finite_feature", key)
            features[key] = number
        if len(features) != self.config.expected_features_per_roi:
            raise ExtractionError(
                "unexpected_feature_count",
                f"{len(features)} != {self.config.expected_features_per_roi}",
            )
        return ExtractionResult(features=features, diagnostics=diagnostics)

    def provenance(self) -> dict[str, Any]:
        """Everything needed to reproduce (or refuse to compare) the features."""
        return {
            "pipeline_version": PIPELINE_VERSION,
            "config_id": self.config.config_id,
            "config_hash": self.config.content_hash,
            "config": self.config.model_dump(mode="json"),
            "libraries": library_versions(),
        }
