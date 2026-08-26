"""Tests for aseg-derived hippocampal masks."""

from __future__ import annotations

import numpy as np
import pytest

from oasis_radiomics.masks import (
    LEFT_HIPPOCAMPUS_LABEL,
    RIGHT_HIPPOCAMPUS_LABEL,
    MaskError,
    bounding_box,
    build_roi_masks,
    create_label_mask,
    create_left_hippocampus_mask,
    create_right_hippocampus_mask,
    describe_mask,
)


@pytest.fixture
def aseg() -> np.ndarray:
    """A tiny segmentation with 3 left, 2 right and 4 unrelated labelled voxels."""
    volume = np.zeros((8, 8, 8), dtype=np.int16)
    volume[1, 1, 1] = LEFT_HIPPOCAMPUS_LABEL
    volume[1, 2, 1] = LEFT_HIPPOCAMPUS_LABEL
    volume[2, 1, 3] = LEFT_HIPPOCAMPUS_LABEL
    volume[5, 1, 1] = RIGHT_HIPPOCAMPUS_LABEL
    volume[6, 2, 2] = RIGHT_HIPPOCAMPUS_LABEL
    volume[4, 4, 4] = 2      # Left-Cerebral-White-Matter
    volume[4, 4, 5] = 41     # Right-Cerebral-White-Matter
    volume[0, 0, 0] = 18     # Left-Amygdala: adjacent label, must not leak in
    volume[7, 7, 7] = 54     # Right-Amygdala
    return volume


def test_create_left_hippocampus_mask(aseg: np.ndarray) -> None:
    mask = create_left_hippocampus_mask(aseg)
    assert mask.dtype == np.uint8
    assert mask.sum() == 3
    assert mask[1, 1, 1] == 1
    assert mask[5, 1, 1] == 0  # right hippocampus excluded
    assert mask[0, 0, 0] == 0  # amygdala excluded


def test_create_right_hippocampus_mask(aseg: np.ndarray) -> None:
    mask = create_right_hippocampus_mask(aseg)
    assert mask.sum() == 2
    assert mask[5, 1, 1] == 1
    assert mask[1, 1, 1] == 0


def test_left_and_right_masks_are_disjoint(aseg: np.ndarray) -> None:
    left = create_left_hippocampus_mask(aseg)
    right = create_right_hippocampus_mask(aseg)
    assert np.count_nonzero(left & right) == 0


def test_masks_are_binary(aseg: np.ndarray) -> None:
    mask = create_left_hippocampus_mask(aseg)
    assert set(np.unique(mask)).issubset({0, 1})


def test_float_segmentation_is_rounded() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.float32)
    volume[1, 1, 1] = 17.0000001
    volume[2, 2, 2] = 16.9999999
    assert create_left_hippocampus_mask(volume).sum() == 2


def test_empty_mask_when_label_absent() -> None:
    mask = create_left_hippocampus_mask(np.zeros((4, 4, 4), dtype=np.int16))
    assert mask.sum() == 0


def test_create_label_mask_requires_labels() -> None:
    with pytest.raises(MaskError):
        create_label_mask(np.zeros((2, 2, 2)), [])


def test_create_label_mask_with_multiple_labels(aseg: np.ndarray) -> None:
    union = create_label_mask(aseg, [LEFT_HIPPOCAMPUS_LABEL, RIGHT_HIPPOCAMPUS_LABEL])
    assert union.sum() == 5


def test_bounding_box(aseg: np.ndarray) -> None:
    assert bounding_box(create_left_hippocampus_mask(aseg)) == (1, 2, 1, 2, 1, 3)


def test_bounding_box_of_empty_mask_is_none() -> None:
    assert bounding_box(np.zeros((4, 4, 4), dtype=np.uint8)) is None


def test_describe_mask_computes_volume(aseg: np.ndarray) -> None:
    mask = create_left_hippocampus_mask(aseg)
    roi = describe_mask("left_hippocampus", [17], mask, voxel_volume_mm3=0.5)
    assert roi.n_voxels == 3
    assert roi.volume_mm3 == pytest.approx(1.5)
    assert not roi.is_empty


def test_build_roi_masks_keeps_empty_rois(aseg: np.ndarray) -> None:
    """Empty ROIs must survive so quality control can report them."""

    class FakeVolume:
        data = aseg
        voxel_volume_mm3 = 1.0
        path = "in-memory"

    masks = build_roi_masks(
        FakeVolume(), {"left_hippocampus": (17,), "right_hippocampus": (53,), "absent": (999,)}
    )
    assert set(masks) == {"left_hippocampus", "right_hippocampus", "absent"}
    assert masks["absent"].is_empty
