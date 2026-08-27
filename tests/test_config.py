"""Tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from oasis_radiomics.config import ConfigError, PipelineConfig
from oasis_radiomics.protocol import ALZHEIMER_ROIS


def test_repository_config_loads(config: PipelineConfig) -> None:
    assert config.radiomics["binWidth"] == 25
    assert config.radiomics["enableCExtensions"] is False
    assert config.feature_classes == (
        "firstorder",
        "shape",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    )
    assert config.roi_labels == ALZHEIMER_ROIS


def test_wavelet_and_log_are_disabled_by_default(config: PipelineConfig) -> None:
    """They multiply the feature count and require a separate protocol version."""
    assert set(config.image_types) == {"Original"}


def test_bilateral_union_is_off_by_default(config: PipelineConfig) -> None:
    assert config.bilateral.extract_union_mask is False
    assert "bilateral_hippocampus" not in config.extraction_roi_labels


def test_extraction_rois_include_union_when_enabled() -> None:
    config = PipelineConfig.from_mapping({"bilateral": {"extract_union_mask": True}})
    assert config.extraction_roi_labels["bilateral_hippocampus"] == (17, 53)


def test_image_types_boolean_shorthand() -> None:
    config = PipelineConfig.from_mapping({"image_types": {"Original": True, "Wavelet": False}})
    assert config.image_types == {"Original": {}}


def test_image_types_with_arguments() -> None:
    config = PipelineConfig.from_mapping({"image_types": {"LoG": {"sigma": [1.0, 3.0]}}})
    assert config.image_types["LoG"] == {"sigma": [1.0, 3.0]}


def test_features_as_list_is_accepted() -> None:
    assert PipelineConfig.from_mapping({"features": ["shape"]}).feature_classes == ("shape",)


def test_single_int_roi_label_is_accepted() -> None:
    config = PipelineConfig.from_mapping(
        {"rois": {"left_hippocampus": 17, "right_hippocampus": 53}}
    )
    assert config.roi_labels["left_hippocampus"] == (17,)


def test_unknown_sections_are_ignored_with_a_warning(caplog) -> None:
    with caplog.at_level("WARNING"):
        PipelineConfig.from_mapping({"nonsense": 1})
    assert any("nonsense" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
    "raw",
    [
        {"image_types": {}},
        {"features": {}},
        {"rois": {}},
        {"rois": {"left_hippocampus": []}},
        {"rois": {"left_hippocampus": "seventeen"}},
        {"bilateral": {"asymmetry_mode": "nonsense"}},
        {"bilateral": {"left_roi": "not_an_roi"}},
        {"quality_control": {"outliers": {"method": "nonsense"}}},
        {"longitudinal": {"delta_comparisons": ["nonsense"]}},
        {"longitudinal": {"min_sessions_for_slope": 1}},
        {"longitudinal": {"days_per_year": 0}},
    ],
)
def test_invalid_configurations_are_rejected(raw: dict) -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_mapping(raw)


def test_missing_file_raises() -> None:
    with pytest.raises(ConfigError):
        PipelineConfig.from_yaml("does/not/exist.yaml")


def test_to_dict_is_json_serialisable(config: PipelineConfig) -> None:
    import json

    payload = json.loads(json.dumps(config.to_dict()))
    assert payload["roi_labels"]["left_hippocampus"] == [17]
    assert payload["roi_labels"]["right_precuneus"] == [2025]
    assert payload["feature_classes"][0] == "firstorder"
