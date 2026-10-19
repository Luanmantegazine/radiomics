"""Typed configuration for the OASIS-3 radiomics pipeline.

The whole pipeline is driven by :class:`PipelineConfig`, normally loaded from
``radiomics_config.yaml``. Keeping every methodological knob in one declarative
place is what makes a run reproducible: :mod:`oasis_radiomics.metadata` dumps
the resolved configuration next to the results.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "radiomics_config.yaml"

#: FreeSurfer ``aseg`` labels used when no ROI section is supplied.
DEFAULT_ROI_LABELS: dict[str, tuple[int, ...]] = {
    "left_hippocampus": (17,),
    "right_hippocampus": (53,),
}

DEFAULT_FEATURE_CLASSES: tuple[str, ...] = (
    "firstorder",
    "shape",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)

DEFAULT_ADDITIVE_FEATURES: tuple[str, ...] = (
    "mask_voxels",
    "mask_volume_mm3",
    "original_shape_MeshVolume",
    "original_shape_VoxelVolume",
    "original_shape_SurfaceArea",
)

ASYMMETRY_MODES = ("positive_only", "always")
OUTLIER_METHODS = ("mad", "iqr", "none")


class ConfigError(ValueError):
    """Raised when a configuration file is structurally invalid."""


@dataclass(frozen=True)
class BilateralConfig:
    """How left/right ROIs are combined into derived tabular columns."""

    extract_union_mask: bool = False
    left_roi: str = "left_hippocampus"
    right_roi: str = "right_hippocampus"
    additive_features: tuple[str, ...] = DEFAULT_ADDITIVE_FEATURES
    asymmetry_mode: str = "positive_only"

    def __post_init__(self) -> None:
        if self.asymmetry_mode not in ASYMMETRY_MODES:
            raise ConfigError(
                f"bilateral.asymmetry_mode must be one of {ASYMMETRY_MODES}, "
                f"got {self.asymmetry_mode!r}"
            )


@dataclass(frozen=True)
class OutlierConfig:
    """Cohort-relative outlier flagging. Never removes anything."""

    method: str = "mad"
    threshold: float = 3.5
    iqr_multiplier: float = 1.5
    min_samples: int = 8
    columns: tuple[str, ...] = (
        "left_volume_mm3",
        "right_volume_mm3",
        "total_volume_mm3",
        "volume_asymmetry",
    )

    def __post_init__(self) -> None:
        if self.method not in OUTLIER_METHODS:
            raise ConfigError(
                f"quality_control.outliers.method must be one of {OUTLIER_METHODS}, "
                f"got {self.method!r}"
            )
        if self.min_samples < 1:
            raise ConfigError("quality_control.outliers.min_samples must be >= 1")


@dataclass(frozen=True)
class QualityControlConfig:
    """Per-session sanity bounds used to raise QC warnings."""

    min_hippocampus_volume_mm3: float = 1000.0
    max_hippocampus_volume_mm3: float = 7000.0
    min_hippocampus_voxels: int = 500
    max_absolute_volume_asymmetry: float = 0.20
    outliers: OutlierConfig = field(default_factory=OutlierConfig)


@dataclass(frozen=True)
class LongitudinalConfig:
    """Delta/slope derivation settings."""

    delta_comparisons: tuple[str, ...] = ("consecutive", "baseline")
    min_sessions_for_slope: int = 2
    days_per_year: float = 365.25

    def __post_init__(self) -> None:
        allowed = {"consecutive", "baseline"}
        unknown = set(self.delta_comparisons) - allowed
        if unknown:
            raise ConfigError(
                f"longitudinal.delta_comparisons contains unknown entries: {sorted(unknown)}"
            )
        if self.min_sessions_for_slope < 2:
            raise ConfigError("longitudinal.min_sessions_for_slope must be >= 2")
        if self.days_per_year <= 0:
            raise ConfigError("longitudinal.days_per_year must be > 0")


@dataclass(frozen=True)
class PipelineConfig:
    """Fully resolved pipeline configuration."""

    radiomics: Mapping[str, Any] = field(
        default_factory=lambda: {
            "binWidth": 25,
            "normalize": True,
            "normalizeScale": 100,
            "correctMask": True,
            "label": 1,
            "enableCExtensions": False,
        }
    )
    image_types: Mapping[str, Any] = field(default_factory=lambda: {"Original": {}})
    feature_classes: tuple[str, ...] = DEFAULT_FEATURE_CLASSES
    roi_labels: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(DEFAULT_ROI_LABELS)
    )
    bilateral: BilateralConfig = field(default_factory=BilateralConfig)
    quality_control: QualityControlConfig = field(default_factory=QualityControlConfig)
    longitudinal: LongitudinalConfig = field(default_factory=LongitudinalConfig)
    source_path: Path | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source_path: Path | None = None
    ) -> "PipelineConfig":
        """Build a configuration from an already-parsed mapping.

        Unknown top-level keys are reported as warnings rather than errors so a
        config file can carry documentation sections without breaking runs.
        """
        if not isinstance(raw, Mapping):
            raise ConfigError(f"Configuration root must be a mapping, got {type(raw).__name__}")

        known = {
            "radiomics",
            "image_types",
            "features",
            "rois",
            "bilateral",
            "quality_control",
            "longitudinal",
        }
        for key in raw:
            if key not in known:
                logger.warning("Ignoring unknown configuration section: %r", key)

        defaults = cls()

        radiomics_settings = _as_mapping(raw.get("radiomics"), "radiomics", defaults.radiomics)
        image_types = _parse_image_types(raw.get("image_types"), defaults.image_types)
        feature_classes = _parse_feature_classes(raw.get("features"), defaults.feature_classes)
        roi_labels = _parse_rois(raw.get("rois"), defaults.roi_labels)

        bilateral = _build_dataclass(
            BilateralConfig,
            _as_mapping(raw.get("bilateral"), "bilateral", {}),
            tuple_fields=("additive_features",),
        )
        qc_raw = dict(_as_mapping(raw.get("quality_control"), "quality_control", {}))
        outliers = _build_dataclass(
            OutlierConfig,
            _as_mapping(qc_raw.pop("outliers", None), "quality_control.outliers", {}),
            tuple_fields=("columns",),
        )
        quality_control = _build_dataclass(
            QualityControlConfig, qc_raw, tuple_fields=(), extra={"outliers": outliers}
        )
        longitudinal = _build_dataclass(
            LongitudinalConfig,
            _as_mapping(raw.get("longitudinal"), "longitudinal", {}),
            tuple_fields=("delta_comparisons",),
        )

        _validate_bilateral_rois(bilateral, roi_labels)

        return cls(
            radiomics=dict(radiomics_settings),
            image_types=image_types,
            feature_classes=feature_classes,
            roi_labels=roi_labels,
            bilateral=bilateral,
            quality_control=quality_control,
            longitudinal=longitudinal,
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        logger.info("Loaded configuration from %s", path)
        return cls.from_mapping(raw, source_path=path.resolve())

    @classmethod
    def load(cls, path: Path | str | None) -> "PipelineConfig":
        """Load ``path``; fall back to defaults when ``path`` is ``None``.

        When no path is given, ``radiomics_config.yaml`` next to the repository
        root is used if it exists.
        """
        if path is not None:
            return cls.from_yaml(path)

        default_path = Path(__file__).resolve().parent.parent / DEFAULT_CONFIG_FILENAME
        if default_path.exists():
            return cls.from_yaml(default_path)

        logger.warning("No configuration file found; using built-in defaults.")
        return cls()

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable snapshot, used for ``run_metadata.json``."""
        payload = asdict(self)
        payload["roi_labels"] = {name: list(labels) for name, labels in self.roi_labels.items()}
        payload["feature_classes"] = list(self.feature_classes)
        payload["bilateral"]["additive_features"] = list(self.bilateral.additive_features)
        payload["quality_control"]["outliers"]["columns"] = list(
            self.quality_control.outliers.columns
        )
        payload["longitudinal"]["delta_comparisons"] = list(
            self.longitudinal.delta_comparisons
        )
        payload["source_path"] = str(self.source_path) if self.source_path else None
        return payload

    # -- convenience --------------------------------------------------------
    @property
    def extraction_roi_labels(self) -> dict[str, tuple[int, ...]]:
        """ROIs handed to PyRadiomics, including the optional union mask.

        The bilateral union is only added when explicitly enabled; its shape
        features are not interpretable because the mask has two disconnected
        components.
        """
        rois = dict(self.roi_labels)
        if self.bilateral.extract_union_mask:
            left = self.roi_labels.get(self.bilateral.left_roi, ())
            right = self.roi_labels.get(self.bilateral.right_roi, ())
            rois["bilateral_hippocampus"] = tuple(sorted(set(left) | set(right)))
        return rois


# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------
def _as_mapping(value: Any, name: str, default: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``value`` as a mapping, or ``default`` when it is ``None``."""
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise ConfigError(f"Configuration section {name!r} must be a mapping.")
    return value


def _parse_image_types(value: Any, default: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the ``image_types`` section to PyRadiomics' ``{name: kwargs}``.

    ``true`` is accepted as shorthand for "enabled with default kwargs"; ``false``
    disables the image type entirely.
    """
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ConfigError("Configuration section 'image_types' must be a mapping.")

    parsed: dict[str, Any] = {}
    for name, settings in value.items():
        if settings is False:
            continue
        if settings is True or settings is None:
            parsed[str(name)] = {}
        elif isinstance(settings, Mapping):
            parsed[str(name)] = dict(settings)
        else:
            raise ConfigError(
                f"image_types.{name} must be a boolean or a mapping, got {type(settings).__name__}"
            )

    if not parsed:
        raise ConfigError("At least one image type must be enabled.")
    return parsed


def _parse_feature_classes(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    """Normalise the ``features`` section to an ordered tuple of class names."""
    if value is None:
        return tuple(default)
    if isinstance(value, Mapping):
        enabled = tuple(str(name) for name, flag in value.items() if flag)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        enabled = tuple(str(name) for name in value)
    else:
        raise ConfigError("Configuration section 'features' must be a mapping or a list.")

    if not enabled:
        raise ConfigError("At least one PyRadiomics feature class must be enabled.")
    return enabled


def _parse_rois(value: Any, default: Mapping[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """Normalise the ``rois`` section to ``{roi_name: (label, ...)}``."""
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ConfigError("Configuration section 'rois' must be a mapping.")

    parsed: dict[str, tuple[int, ...]] = {}
    for name, labels in value.items():
        if isinstance(labels, int):
            labels = [labels]
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ConfigError(f"rois.{name} must be an int or a list of ints.")
        try:
            parsed[str(name)] = tuple(int(label) for label in labels)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"rois.{name} must contain integer aseg labels.") from exc
        if not parsed[str(name)]:
            raise ConfigError(f"rois.{name} must list at least one aseg label.")

    if not parsed:
        raise ConfigError("At least one ROI must be configured.")
    return parsed


def _build_dataclass(
    cls: type,
    raw: Mapping[str, Any],
    tuple_fields: Sequence[str],
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Instantiate a config dataclass from ``raw``, ignoring unknown keys."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            logger.warning("Ignoring unknown configuration key: %s.%s", cls.__name__, key)
            continue
        kwargs[key] = tuple(value) if key in tuple_fields and value is not None else value
    if extra:
        kwargs.update(extra)
    return cls(**kwargs)


def _validate_bilateral_rois(
    bilateral: BilateralConfig, roi_labels: Mapping[str, tuple[int, ...]]
) -> None:
    """Ensure the configured left/right ROI names actually exist."""
    for side in ("left_roi", "right_roi"):
        name = getattr(bilateral, side)
        if name not in roi_labels:
            raise ConfigError(
                f"bilateral.{side} refers to ROI {name!r}, which is not defined "
                f"in the 'rois' section (available: {sorted(roi_labels)})."
            )
