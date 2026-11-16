"""PyRadiomics integration: configuration, known-answer checks, provenance."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from client.imaging.extraction import (
    ExtractionConfig,
    ExtractionError,
    PyRadiomicsExtractor,
    load_extraction_config,
)
from tests.federated import phantoms


@pytest.fixture
def pair(tmp_path):
    return (
        phantoms.write(phantoms.texture(1), tmp_path / "t1.nii.gz"),
        phantoms.write(phantoms.labels(), tmp_path / "mask.nii.gz"),
    )


def test_shipped_configuration_is_explicit_and_versioned():
    config = load_extraction_config(phantoms.DEFAULT_CONFIG)
    assert config.config_id == "t1-original-v1"
    assert config.content_hash == load_extraction_config(phantoms.DEFAULT_CONFIG).content_hash
    assert phantoms.config(binWidth=10).content_hash != config.content_hash
    raw = config.model_dump(mode="json")
    # No PyRadiomics setting may be left to a library default.
    del raw["settings"]["interpolator"]
    with pytest.raises(ValidationError):
        ExtractionConfig.model_validate(raw)
    raw = config.model_dump(mode="json")
    raw["settings"]["someNewSetting"] = 1
    with pytest.raises(ValidationError):
        ExtractionConfig.model_validate(raw)
    with pytest.raises(ValidationError, match="exactly one"):
        phantoms.config(binCount=32)


def test_extracts_the_declared_feature_schema(pair):
    result = PyRadiomicsExtractor(phantoms.config()).extract(*pair, label=1)
    names = list(result.features)
    assert len(names) == 107
    classes = {name.split("_")[1] for name in names}
    assert classes == {"shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"}
    assert all(name.startswith("original_") for name in names)
    assert all(np.isfinite(v) for v in result.features.values())
    assert result.diagnostics["Mask-original_VoxelNum"] == 640


def test_known_answers_for_first_order_and_shape(tmp_path):
    """Values that can be derived by hand, on an anisotropic grid."""
    spacing = (1.0, 1.0, 2.0)
    array = phantoms.texture(3)
    mask = phantoms.labels()
    image_path = phantoms.write(array, tmp_path / "i.nii.gz", spacing=spacing)
    mask_path = phantoms.write(mask, tmp_path / "m.nii.gz", spacing=spacing)
    extractor = PyRadiomicsExtractor(phantoms.config(normalize=False))
    features = extractor.extract(image_path, mask_path, label=1).features
    roi = array[mask == 1].astype(float)
    assert features["original_firstorder_Mean"] == pytest.approx(roi.mean(), rel=1e-6)
    assert features["original_firstorder_Minimum"] == pytest.approx(roi.min())
    assert features["original_firstorder_Maximum"] == pytest.approx(roi.max())
    assert features["original_firstorder_Variance"] == pytest.approx(roi.var(), rel=1e-6)
    assert features["original_shape_VoxelVolume"] == pytest.approx(roi.size * 2.0)
    # The box is 8 x 10 x 8 voxels = 8 x 10 x 16 mm; the surface mesh lies between
    # the voxel-centre extent and the voxel-edge extent.
    centres, edges = np.sqrt(7**2 + 9**2 + 14**2), np.sqrt(8**2 + 10**2 + 16**2)
    assert centres < features["original_shape_Maximum3DDiameter"] < edges
    assert 0.9 * roi.size * 2.0 < features["original_shape_MeshVolume"] < roi.size * 2.0
    # The mask label selects the ROI: label 2 is the sphere, not the box.
    sphere = extractor.extract(image_path, mask_path, label=2).features
    assert sphere["original_shape_VoxelVolume"] == pytest.approx((mask == 2).sum() * 2.0)
    assert sphere["original_shape_Sphericity"] > features["original_shape_Sphericity"]


def test_known_answer_for_discretisation(tmp_path):
    """Two equally frequent intensities, two bins apart -> entropy of exactly 1 bit."""
    array = np.full(phantoms.SHAPE, 10, dtype=np.float32)
    array[:, :, ::2] = 60
    image_path = phantoms.write(array, tmp_path / "i.nii.gz")
    mask_path = phantoms.write(phantoms.labels(), tmp_path / "m.nii.gz")
    features = (
        PyRadiomicsExtractor(phantoms.config(normalize=False, binWidth=25))
        .extract(image_path, mask_path, label=1)
        .features
    )
    assert features["original_firstorder_Entropy"] == pytest.approx(1.0)
    assert features["original_firstorder_Uniformity"] == pytest.approx(0.5)


def test_extraction_is_deterministic(pair):
    first = PyRadiomicsExtractor(phantoms.config()).extract(*pair, label=2).features
    second = PyRadiomicsExtractor(phantoms.config()).extract(*pair, label=2).features
    assert first == second


def test_settings_reach_pyradiomics(pair):
    coarse = PyRadiomicsExtractor(phantoms.config(binWidth=50)).extract(*pair).features
    fine = PyRadiomicsExtractor(phantoms.config(binWidth=5)).extract(*pair).features
    assert coarse["original_firstorder_Entropy"] < fine["original_firstorder_Entropy"]
    assert coarse["original_shape_MeshVolume"] == fine["original_shape_MeshVolume"]


def test_filtered_image_types_are_configurable(pair):
    raw = phantoms.config().model_dump(mode="json")
    raw["image_types"] = {"Original": {}, "LoG": {"sigma": [1.0]}}
    raw["feature_classes"] = ["firstorder"]
    raw["expected_features_per_roi"] = 36
    features = PyRadiomicsExtractor(ExtractionConfig.model_validate(raw)).extract(*pair).features
    assert sum(name.startswith("log-sigma-1-0-mm-3D_") for name in features) == 18
    raw["image_types"] = {"LoG": {}}
    with pytest.raises(ValidationError, match="sigma"):
        ExtractionConfig.model_validate(raw)


def test_schema_drift_is_detected(pair):
    raw = phantoms.config().model_dump(mode="json")
    raw["expected_features_per_roi"] = 100
    with pytest.raises(ExtractionError, match="unexpected_feature_count"):
        PyRadiomicsExtractor(ExtractionConfig.model_validate(raw)).extract(*pair)


def test_geometry_mismatch_is_never_silently_resampled(tmp_path, pair):
    shifted = phantoms.write(phantoms.labels(), tmp_path / "s.nii.gz", origin=(0.0, 3.0, 0.0))
    with pytest.raises(ExtractionError, match="pyradiomics_error"):
        PyRadiomicsExtractor(phantoms.config()).extract(pair[0], shifted)


def test_provenance_records_versions_and_configuration():
    provenance = PyRadiomicsExtractor(phantoms.config()).provenance()
    assert provenance["config_id"] == "t1-original-v1"
    assert provenance["config_hash"] == phantoms.config().content_hash
    assert provenance["config"]["settings"]["binWidth"] == 25
    assert provenance["config"]["preprocessing"]
    assert set(provenance["libraries"]) == {
        "python", "pyradiomics", "numpy", "SimpleITK", "PyWavelets",
    }
    assert provenance["libraries"]["pyradiomics"].startswith("3.1.0")
    assert provenance["libraries"]["numpy"].startswith("1.26.")
