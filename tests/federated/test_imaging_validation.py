"""Manifest parsing and image/mask validation (Stage 2)."""

from __future__ import annotations

import numpy as np
import pytest

from client.imaging.manifest import ManifestError, load_manifest
from client.imaging.validation import IssueCode, ValidationSettings, validate_pair
from tests.federated import phantoms

SETTINGS = ValidationSettings(
    geometry_tolerance=1e-4, min_roi_voxels=64, max_spacing_anisotropy=3.0
)


def _codes(report, severity=None):
    return {i.code for i in report.issues if severity is None or i.severity.value == severity}


@pytest.fixture
def pair(tmp_path):
    return (
        phantoms.write(phantoms.texture(1), tmp_path / "t1.nii.gz"),
        phantoms.write(phantoms.labels(), tmp_path / "mask.nii.gz"),
    )


def test_valid_pair(pair):
    report = validate_pair(*pair, 1, SETTINGS)
    assert report.ok and not report.issues
    assert report.roi_voxels == 8 * 10 * 8
    assert report.roi_components == 1
    assert report.image_geometry.size == phantoms.SHAPE
    assert report.image_geometry.orientation == "LPS"


@pytest.mark.parametrize(
    ("mask_kwargs", "code"),
    [
        ({"spacing": (1.0, 1.0, 1.2)}, IssueCode.SPACING_MISMATCH),
        ({"origin": (0.0, 5.0, 0.0)}, IssueCode.ORIGIN_MISMATCH),
        ({"direction": (-1, 0, 0, 0, 1, 0, 0, 0, 1)}, IssueCode.DIRECTION_MISMATCH),
    ],
)
def test_geometry_mismatch_is_an_error(tmp_path, pair, mask_kwargs, code):
    mask = phantoms.write(phantoms.labels(), tmp_path / "other.nii.gz", **mask_kwargs)
    report = validate_pair(pair[0], mask, 1, SETTINGS)
    assert not report.ok and code in _codes(report, "error")


def test_size_mismatch(tmp_path, pair):
    mask = phantoms.write(phantoms.labels((28, 30, 30)), tmp_path / "small.nii.gz")
    assert IssueCode.SIZE_MISMATCH in _codes(validate_pair(pair[0], mask, 1, SETTINGS), "error")


def test_sub_tolerance_differences_are_accepted(tmp_path, pair):
    mask = phantoms.write(phantoms.labels(), tmp_path / "m.nii.gz", origin=(0.0, 0.00001, 0.0))
    assert validate_pair(pair[0], mask, 1, SETTINGS).ok


def test_non_finite_image_values(tmp_path, pair):
    array = phantoms.texture(1)
    array[3, 3, 3] = np.nan
    image = phantoms.write(array, tmp_path / "nan.nii.gz")
    assert IssueCode.NON_FINITE_VALUES in _codes(validate_pair(image, pair[1], 1, SETTINGS))


def test_mask_problems(tmp_path, pair):
    image = pair[0]
    assert IssueCode.LABEL_ABSENT in _codes(validate_pair(image, pair[1], 7, SETTINGS), "error")

    fractional = phantoms.labels().astype(np.float32) * 0.5
    mask = phantoms.write(fractional, tmp_path / "frac.nii.gz")
    assert IssueCode.MASK_NOT_INTEGER in _codes(validate_pair(image, mask, 1, SETTINGS), "error")

    tiny = np.zeros(phantoms.SHAPE, dtype=np.uint8)
    tiny[10:12, 10:12, 10:12] = 1
    mask = phantoms.write(tiny, tmp_path / "tiny.nii.gz")
    assert IssueCode.ROI_TOO_SMALL in _codes(validate_pair(image, mask, 1, SETTINGS), "error")


def test_constant_roi_intensity(tmp_path, pair):
    image = phantoms.write(np.full(phantoms.SHAPE, 50, dtype=np.float32), tmp_path / "c.nii.gz")
    report = validate_pair(image, pair[1], 1, SETTINGS)
    assert IssueCode.ROI_CONSTANT_INTENSITY in _codes(report, "error")


def test_warnings_do_not_block(tmp_path):
    split = np.zeros(phantoms.SHAPE, dtype=np.uint8)
    split[0:6, 4:10, 4:10] = 1  # touches the border
    split[16:22, 4:10, 4:10] = 1  # second component
    spacing = (1.0, 1.0, 4.0)
    image = phantoms.write(phantoms.texture(2), tmp_path / "i.nii.gz", spacing=spacing)
    mask = phantoms.write(split, tmp_path / "m.nii.gz", spacing=spacing)
    settings = SETTINGS.model_copy(update={"expected_orientation": "RAS"})
    report = validate_pair(image, mask, 1, settings)
    assert report.ok
    assert _codes(report, "warning") == {
        IssueCode.ROI_MULTIPLE_COMPONENTS, IssueCode.ROI_TOUCHES_BORDER,
        IssueCode.ANISOTROPIC_SPACING, IssueCode.UNEXPECTED_ORIENTATION,
    }


def test_unusable_files(tmp_path, pair):
    assert IssueCode.FILE_NOT_FOUND in _codes(
        validate_pair(tmp_path / "missing.nii.gz", pair[1], 1, SETTINGS)
    )
    garbage = tmp_path / "garbage.nii.gz"
    garbage.write_bytes(b"not a nifti file")
    assert IssueCode.UNREADABLE in _codes(validate_pair(garbage, pair[1], 1, SETTINGS))
    other = tmp_path / "scan.mha"
    other.write_bytes(b"x")
    assert IssueCode.UNSUPPORTED_FORMAT in _codes(validate_pair(other, pair[1], 1, SETTINGS))

    import SimpleITK as sitk

    four_d = tmp_path / "bold.nii.gz"
    sitk.WriteImage(sitk.Image([8, 8, 8, 3], sitk.sitkFloat32), str(four_d))
    assert IssueCode.NOT_3D_SCALAR in _codes(validate_pair(four_d, pair[1], 1, SETTINGS))


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
ROW = {"participant_id": "P1", "roi": "hippo_l", "image_path": "i.nii.gz",
       "mask_path": "m.nii.gz", "mask_label": 17, "time_months": 24.5, "event": 1}


def test_manifest_association_and_followup(tmp_path):
    path = phantoms.write_manifest(
        tmp_path / "site" / "manifest.csv",
        [
            ROW,
            {**ROW, "roi": "hippo_r", "mask_label": 53},
            {**ROW, "participant_id": "P2", "image_path": "j.nii.gz", "event": 0},
            {**ROW, "participant_id": "P3", "image_path": "k.nii.gz",
             "time_months": "", "event": ""},
        ],
    )
    manifest = load_manifest(path)
    assert manifest.participants == ("P1", "P2", "P3")
    assert manifest.rois == ("hippo_l", "hippo_r")
    first = manifest.cases[0]
    assert (first.participant_id, first.roi, first.mask_label) == ("P1", "hippo_l", 17)
    assert first.image_path == (tmp_path / "site" / "i.nii.gz").resolve()
    # P3 has no follow-up: reported as unknown, not as a censored non-converter.
    assert set(manifest.followup) == {"P1", "P2"}
    assert manifest.followup["P2"].event is False
    summary = manifest.followup_summary()
    assert (summary.n_with_followup, summary.n_without_followup) == (2, 1)
    assert (summary.n_events, summary.n_censored) == (1, 1)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([ROW, ROW], "duplicate"),
        ([ROW, {**ROW, "roi": "other", "image_path": "z.nii.gz"}], "more than one image"),
        ([ROW, {**ROW, "roi": "other", "time_months": 30}], "inconsistent follow-up"),
        ([{**ROW, "event": ""}], "both set or both blank"),
        ([{**ROW, "time_months": -3}], "positive finite"),
        ([{**ROW, "event": "maybe"}], "event must be 0 or 1"),
        ([{**ROW, "roi": "Left Hippocampus"}], "roi must match"),
        ([{**ROW, "mask_label": 0}], "mask_label"),
        ([{**ROW, "diagnosis": "MCI"}], "unknown column"),
        ([{k: v for k, v in ROW.items() if k != "mask_path"}], "missing column"),
    ],
)
def test_manifest_errors(tmp_path, rows, message):
    path = phantoms.write_manifest(tmp_path / "manifest.csv", rows)
    with pytest.raises(ManifestError, match=message):
        load_manifest(path)


def test_itk_would_hide_non_finite_values(tmp_path):
    """Documents why NaN/Inf are checked on the raw file, not through SimpleITK."""
    import SimpleITK as sitk

    array = phantoms.texture(1)
    array[3, 3, 3], array[4, 4, 4] = np.nan, np.inf
    path = phantoms.write(array, tmp_path / "nan.nii.gz")
    as_read_by_itk = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert np.isfinite(as_read_by_itk).all()  # silently replaced by zeros


def test_qform_sform_disagreement_is_flagged(tmp_path, pair):
    import nibabel as nib

    image = nib.load(pair[0])
    shifted = image.affine.copy()
    shifted[0, 3] += 7.0
    image.header.set_qform(image.affine, code=1)
    image.header.set_sform(shifted, code=1)
    path = tmp_path / "conflict.nii.gz"
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), None, image.header), path)
    report = validate_pair(path, path, 1, SETTINGS)
    assert IssueCode.QFORM_SFORM_DISAGREE in _codes(report, "warning")
