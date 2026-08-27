"""Frozen acquisition protocol for the OASIS-3 Alzheimer radiomics study.

This module contains the decisions that must not drift between acquisition runs:

* the FreeSurfer segmentation product required by the study;
* the 16 Alzheimer-related ROIs (8 bilateral anatomical regions);
* the exact 107 Original-image PyRadiomics features extracted per ROI.

The original project text estimated 104 features/ROI (21 GLCM features).  The
validated PyRadiomics 3.0.1 environment exposes 24 non-deprecated GLCM features,
which yields 107 features/ROI.  We do not silently delete three valid features
merely to reproduce the old arithmetic.  This protocol therefore records the
methodological amendment explicitly and freezes the actual extractor output.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROTOCOL_VERSION = "oasis3-ad-radiomics-v1.0"
REQUIRED_SEGMENTATION_FILENAME = "aparc+aseg.mgz"
EXPECTED_FEATURE_COUNT = 107
EXPECTED_ROI_COUNT = 16
EXPECTED_RAW_FEATURES_PER_SESSION = EXPECTED_FEATURE_COUNT * EXPECTED_ROI_COUNT

# FreeSurfer aparc+aseg labels.  Cortical labels follow the Desikan-Killiany
# convention (1000-series left hemisphere, 2000-series right hemisphere).
# Subcortical hippocampus/amygdala labels come from aseg.
ALZHEIMER_ROIS: dict[str, tuple[int, ...]] = {
    "left_hippocampus": (17,),
    "right_hippocampus": (53,),
    "left_amygdala": (18,),
    "right_amygdala": (54,),
    "left_entorhinal": (1006,),
    "right_entorhinal": (2006,),
    "left_fusiform": (1007,),
    "right_fusiform": (2007,),
    "left_inferior_temporal": (1009,),
    "right_inferior_temporal": (2009,),
    "left_middle_temporal": (1015,),
    "right_middle_temporal": (2015,),
    "left_parahippocampal": (1016,),
    "right_parahippocampal": (2016,),
    "left_precuneus": (1025,),
    "right_precuneus": (2025,),
}

# Exact non-deprecated features observed and validated with PyRadiomics 3.0.1.
# Feature names are passed directly to RadiomicsFeatureExtractor.enableFeaturesByName.
EXPECTED_FEATURES_BY_CLASS: dict[str, tuple[str, ...]] = {
    "firstorder": (
        "10Percentile",
        "90Percentile",
        "Energy",
        "Entropy",
        "InterquartileRange",
        "Kurtosis",
        "Maximum",
        "MeanAbsoluteDeviation",
        "Mean",
        "Median",
        "Minimum",
        "Range",
        "RobustMeanAbsoluteDeviation",
        "RootMeanSquared",
        "Skewness",
        "TotalEnergy",
        "Uniformity",
        "Variance",
    ),
    "shape": (
        "Elongation",
        "Flatness",
        "LeastAxisLength",
        "MajorAxisLength",
        "Maximum2DDiameterColumn",
        "Maximum2DDiameterRow",
        "Maximum2DDiameterSlice",
        "Maximum3DDiameter",
        "MeshVolume",
        "MinorAxisLength",
        "Sphericity",
        "SurfaceArea",
        "SurfaceVolumeRatio",
        "VoxelVolume",
    ),
    "glcm": (
        "Autocorrelation",
        "ClusterProminence",
        "ClusterShade",
        "ClusterTendency",
        "Contrast",
        "Correlation",
        "DifferenceAverage",
        "DifferenceEntropy",
        "DifferenceVariance",
        "Id",
        "Idm",
        "Idmn",
        "Idn",
        "Imc1",
        "Imc2",
        "InverseVariance",
        "JointAverage",
        "JointEnergy",
        "JointEntropy",
        "MCC",
        "MaximumProbability",
        "SumAverage",
        "SumEntropy",
        "SumSquares",
    ),
    "glrlm": (
        "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized",
        "GrayLevelVariance",
        "HighGrayLevelRunEmphasis",
        "LongRunEmphasis",
        "LongRunHighGrayLevelEmphasis",
        "LongRunLowGrayLevelEmphasis",
        "LowGrayLevelRunEmphasis",
        "RunEntropy",
        "RunLengthNonUniformity",
        "RunLengthNonUniformityNormalized",
        "RunPercentage",
        "RunVariance",
        "ShortRunEmphasis",
        "ShortRunHighGrayLevelEmphasis",
        "ShortRunLowGrayLevelEmphasis",
    ),
    "glszm": (
        "GrayLevelNonUniformity",
        "GrayLevelNonUniformityNormalized",
        "GrayLevelVariance",
        "HighGrayLevelZoneEmphasis",
        "LargeAreaEmphasis",
        "LargeAreaHighGrayLevelEmphasis",
        "LargeAreaLowGrayLevelEmphasis",
        "LowGrayLevelZoneEmphasis",
        "SizeZoneNonUniformity",
        "SizeZoneNonUniformityNormalized",
        "SmallAreaEmphasis",
        "SmallAreaHighGrayLevelEmphasis",
        "SmallAreaLowGrayLevelEmphasis",
        "ZoneEntropy",
        "ZonePercentage",
        "ZoneVariance",
    ),
    "gldm": (
        "DependenceEntropy",
        "DependenceNonUniformity",
        "DependenceNonUniformityNormalized",
        "DependenceVariance",
        "GrayLevelNonUniformity",
        "GrayLevelVariance",
        "HighGrayLevelEmphasis",
        "LargeDependenceEmphasis",
        "LargeDependenceHighGrayLevelEmphasis",
        "LargeDependenceLowGrayLevelEmphasis",
        "LowGrayLevelEmphasis",
        "SmallDependenceEmphasis",
        "SmallDependenceHighGrayLevelEmphasis",
        "SmallDependenceLowGrayLevelEmphasis",
    ),
    "ngtdm": (
        "Busyness",
        "Coarseness",
        "Complexity",
        "Contrast",
        "Strength",
    ),
}


def expected_feature_keys() -> tuple[str, ...]:
    """Return the exact PyRadiomics keys expected for one Original-image ROI."""
    return tuple(
        f"original_{feature_class}_{feature_name}"
        for feature_class, names in EXPECTED_FEATURES_BY_CLASS.items()
        for feature_name in names
    )


def validate_protocol_definition() -> None:
    """Fail fast if a future edit changes the frozen study arithmetic."""
    actual_features = sum(len(names) for names in EXPECTED_FEATURES_BY_CLASS.values())
    if actual_features != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"Protocol defines {actual_features} features, expected {EXPECTED_FEATURE_COUNT}."
        )
    if len(ALZHEIMER_ROIS) != EXPECTED_ROI_COUNT:
        raise RuntimeError(
            f"Protocol defines {len(ALZHEIMER_ROIS)} ROIs, expected {EXPECTED_ROI_COUNT}."
        )
    labels = [label for labels in ALZHEIMER_ROIS.values() for label in labels]
    if len(labels) != len(set(labels)):
        raise RuntimeError("Protocol contains duplicated FreeSurfer ROI labels.")


def validate_extracted_features(features: Mapping[str, Any]) -> None:
    """Require exactly the frozen 107-feature signature for one ROI."""
    expected = set(expected_feature_keys())
    actual = {key for key in features if not key.startswith("diagnostics_")}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise RuntimeError(
            "PyRadiomics output does not match the frozen acquisition protocol: "
            + "; ".join(details)
        )


validate_protocol_definition()
