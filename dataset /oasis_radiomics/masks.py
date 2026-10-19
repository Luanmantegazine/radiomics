"""Volume loading and construction of hippocampal ROI masks from FreeSurfer.

The masks come straight from the ``aseg.mgz`` produced by the FreeSurfer run
shipped with OASIS-3; no re-segmentation is performed.

Label convention (FreeSurferColorLUT)::

    17 = Left-Hippocampus
    53 = Right-Hippocampus

Left and right are kept as **separate** masks. A union of the two is two
spatially disconnected objects, and PyRadiomics' shape features (sphericity,
surface-to-volume ratio, axis lengths, ...) are meaningless on such a mask.
Bilateral quantities are therefore derived tabularly instead, in
:mod:`oasis_radiomics.longitudinal`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

LEFT_HIPPOCAMPUS_LABEL = 17
RIGHT_HIPPOCAMPUS_LABEL = 53

#: Value written into the binary mask; must match ``radiomics.label`` in the config.
MASK_FOREGROUND = 1

IMAGE_FILENAME = "T1.nii.gz"


class MaskError(RuntimeError):
    """Raised when an ROI mask cannot be built for a session."""


@dataclass(frozen=True)
class LoadedVolume:
    """A FreeSurfer volume loaded into memory together with its geometry."""

    data: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    path: Path

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def voxel_volume_mm3(self) -> float:
        """Volume of a single voxel in mm^3."""
        return float(np.prod(self.spacing))


@dataclass(frozen=True)
class RoiMask:
    """A binary ROI mask plus the descriptors QC and reporting need."""

    name: str
    labels: tuple[int, ...]
    data: np.ndarray
    n_voxels: int
    volume_mm3: float
    bounding_box: tuple[int, int, int, int, int, int] | None

    @property
    def is_empty(self) -> bool:
        return self.n_voxels == 0


def load_volume(path: Path) -> LoadedVolume:
    """Load a ``.mgz``/``.nii.gz`` volume with nibabel.

    Raises
    ------
    MaskError
        If the file is missing or cannot be read.
    """
    import nibabel as nib

    path = Path(path)
    if not path.exists():
        raise MaskError(f"Volume not found: {path}")

    try:
        image = nib.load(str(path))
    except Exception as exc:  # nibabel raises a variety of errors
        raise MaskError(f"Could not read volume {path}: {exc}") from exc

    data = np.asarray(image.dataobj)
    zooms = image.header.get_zooms()[:3]
    spacing = tuple(float(zoom) for zoom in zooms)
    return LoadedVolume(
        data=data, affine=np.asarray(image.affine), spacing=spacing, path=path
    )


def create_label_mask(aseg_data: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Binary mask of every voxel whose ``aseg`` label is in ``labels``.

    The segmentation is rounded to the nearest integer first: some FreeSurfer
    volumes are stored as float even though the labels are categorical.
    """
    if not labels:
        raise MaskError("At least one aseg label is required to build a mask.")

    segmentation = np.rint(np.asarray(aseg_data)).astype(np.int32)
    mask = np.isin(segmentation, list(labels))
    return (mask * MASK_FOREGROUND).astype(np.uint8)


def create_left_hippocampus_mask(aseg_data: np.ndarray) -> np.ndarray:
    """Binary mask of the left hippocampus (``aseg`` label 17)."""
    return create_label_mask(aseg_data, [LEFT_HIPPOCAMPUS_LABEL])


def create_right_hippocampus_mask(aseg_data: np.ndarray) -> np.ndarray:
    """Binary mask of the right hippocampus (``aseg`` label 53)."""
    return create_label_mask(aseg_data, [RIGHT_HIPPOCAMPUS_LABEL])


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """Inclusive ``(x0, x1, y0, y1, z0, z1)`` bounding box, or ``None`` if empty."""
    indices = np.nonzero(mask)
    if indices[0].size == 0:
        return None

    box: list[int] = []
    for axis_indices in indices:
        box.extend((int(axis_indices.min()), int(axis_indices.max())))
    return tuple(box)  # type: ignore[return-value]


def describe_mask(
    name: str, labels: Sequence[int], mask: np.ndarray, voxel_volume_mm3: float
) -> RoiMask:
    """Wrap a raw binary mask into a :class:`RoiMask` with its descriptors."""
    n_voxels = int(mask.sum())
    return RoiMask(
        name=name,
        labels=tuple(labels),
        data=mask,
        n_voxels=n_voxels,
        volume_mm3=n_voxels * voxel_volume_mm3,
        bounding_box=bounding_box(mask),
    )


def build_roi_masks(
    aseg: LoadedVolume, roi_labels: Mapping[str, Sequence[int]]
) -> dict[str, RoiMask]:
    """Build every configured ROI mask for one session.

    Empty masks are returned as well (rather than dropped) so that quality
    control can report them explicitly instead of them vanishing silently.
    """
    masks: dict[str, RoiMask] = {}
    for roi_name, labels in roi_labels.items():
        mask = create_label_mask(aseg.data, labels)
        roi = describe_mask(roi_name, labels, mask, aseg.voxel_volume_mm3)
        if roi.is_empty:
            logger.warning("ROI %s is empty in %s (labels=%s).", roi_name, aseg.path, list(labels))
        else:
            logger.info(
                "ROI %s: %d voxels (%.1f mm^3)", roi_name, roi.n_voxels, roi.volume_mm3
            )
        masks[roi_name] = roi
    return masks


def write_session_nifti(
    session_id: str,
    t1: LoadedVolume,
    masks: Mapping[str, RoiMask],
    prepared_dir: Path,
) -> tuple[Path, dict[str, Path]]:
    """Write the T1 and every non-empty mask as NIfTI for PyRadiomics.

    All masks are saved with the **image affine** so PyRadiomics sees image and
    mask in exactly the same geometry.

    Returns
    -------
    tuple
        ``(image_path, {roi_name: mask_path})``. Empty masks are not written.
    """
    import nibabel as nib

    session_dir = Path(prepared_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    image_path = session_dir / IMAGE_FILENAME
    t1_data = np.asarray(t1.data, dtype=np.float32)
    nib.save(nib.Nifti1Image(t1_data, t1.affine), str(image_path))

    mask_paths: dict[str, Path] = {}
    for roi_name, roi in masks.items():
        if roi.is_empty:
            logger.warning("Not writing %s for %s: mask is empty.", roi_name, session_id)
            continue
        mask_path = session_dir / f"{roi_name}_mask.nii.gz"
        nib.save(nib.Nifti1Image(roi.data, t1.affine), str(mask_path))
        mask_paths[roi_name] = mask_path

    logger.debug("Wrote %s and %d mask(s) to %s", image_path.name, len(mask_paths), session_dir)
    return image_path, mask_paths
