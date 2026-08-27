"""Tests for the frozen OASIS-3 Alzheimer acquisition protocol."""

from oasis_radiomics.protocol import (
    ALZHEIMER_ROIS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_FEATURES_BY_CLASS,
    EXPECTED_RAW_FEATURES_PER_SESSION,
    EXPECTED_ROI_COUNT,
    REQUIRED_SEGMENTATION_FILENAME,
    expected_feature_keys,
    validate_protocol_definition,
)


def test_protocol_arithmetic_is_frozen() -> None:
    validate_protocol_definition()
    assert EXPECTED_ROI_COUNT == 16
    assert len(ALZHEIMER_ROIS) == 16
    assert EXPECTED_FEATURE_COUNT == 107
    assert len(expected_feature_keys()) == 107
    assert EXPECTED_RAW_FEATURES_PER_SESSION == 1712


def test_feature_family_counts() -> None:
    assert {name: len(features) for name, features in EXPECTED_FEATURES_BY_CLASS.items()} == {
        "firstorder": 18,
        "shape": 14,
        "glcm": 24,
        "glrlm": 16,
        "glszm": 16,
        "gldm": 14,
        "ngtdm": 5,
    }


def test_final_protocol_requires_aparc_aseg() -> None:
    assert REQUIRED_SEGMENTATION_FILENAME == "aparc+aseg.mgz"


def test_roi_labels_are_unique() -> None:
    labels = [label for values in ALZHEIMER_ROIS.values() for label in values]
    assert len(labels) == len(set(labels)) == 16
