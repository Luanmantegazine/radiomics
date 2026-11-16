"""Validation of an image/mask pair before feature extraction.

Checks dimensionality, grid geometry (size, spacing, origin, direction),
orientation, invalid values and the ROI itself. Errors block extraction;
warnings are recorded and the case proceeds.

Hypotheses and limitations
--------------------------
* Geometry is the one interpreted by ITK's NIfTI reader (via SimpleITK), which
  is what PyRadiomics uses. A file whose qform and sform disagree is flagged
  with a warning because other tools may place it differently.
* ITK's NIfTI reader silently replaces NaN/Inf voxels with 0, so PyRadiomics
  would treat them as real zero intensities. Non-finite values are therefore
  detected on the raw file (nibabel) and are an error.
* A geometry mismatch is an error: the mask is **not** resampled onto the image
  grid. Resampling a segmentation is a protocol decision (interpolator, label
  handling) that must not happen silently.
* These checks cannot tell whether the mask delineates the intended anatomy or
  whether image and mask belong to the same person; that relies on the manifest.
* Thresholds (minimum ROI size, anisotropy) are configuration, not validated
  constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class IssueCode(StrEnum):
    FILE_NOT_FOUND = "file_not_found"
    UNSUPPORTED_FORMAT = "unsupported_format"
    UNREADABLE = "unreadable"
    NOT_3D_SCALAR = "not_3d_scalar"
    SIZE_MISMATCH = "size_mismatch"
    SPACING_MISMATCH = "spacing_mismatch"
    ORIGIN_MISMATCH = "origin_mismatch"
    DIRECTION_MISMATCH = "direction_mismatch"
    NON_FINITE_VALUES = "non_finite_values"
    MASK_NOT_INTEGER = "mask_not_integer"
    LABEL_ABSENT = "label_absent"
    ROI_TOO_SMALL = "roi_too_small"
    ROI_CONSTANT_INTENSITY = "roi_constant_intensity"
    UNEXPECTED_ORIENTATION = "unexpected_orientation"
    QFORM_SFORM_DISAGREE = "qform_sform_disagree"
    ANISOTROPIC_SPACING = "anisotropic_spacing"
    ROI_MULTIPLE_COMPONENTS = "roi_multiple_components"
    ROI_TOUCHES_BORDER = "roi_touches_border"


@dataclass(frozen=True, slots=True)
class Issue:
    code: IssueCode
    severity: Severity
    detail: str = ""


class ValidationSettings(BaseModel):
    """Thresholds of the validation step (part of the versioned configuration)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    geometry_tolerance: float = Field(gt=0)
    """Maximum absolute difference (mm for spacing/origin, unitless for direction
    cosines) for image and mask grids to be considered identical. Also passed to
    PyRadiomics so both steps agree."""
    min_roi_voxels: int = Field(ge=1)
    max_spacing_anisotropy: float = Field(ge=1)
    """Warn when max(spacing) / min(spacing) exceeds this ratio."""
    expected_orientation: str | None = Field(default=None, pattern=r"^[LRAPIS]{3}$")
    """Anatomical orientation code (ITK convention, e.g. 'LPS'); warn if different."""


@dataclass(frozen=True, slots=True)
class Geometry:
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]
    orientation: str


@dataclass(slots=True)
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    image_geometry: Geometry | None = None
    mask_geometry: Geometry | None = None
    roi_voxels: int | None = None
    roi_components: int | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def add(self, code: IssueCode, severity: Severity, detail: str = "") -> None:
        self.issues.append(Issue(code, severity, detail))


def read_nifti(path: Path) -> sitk.Image:
    """Read a NIfTI volume (``.nii`` / ``.nii.gz``)."""
    return sitk.ReadImage(str(path), imageIO="NiftiImageIO")


def geometry_of(image: sitk.Image) -> Geometry:
    orientation = ""
    if image.GetDimension() == 3:
        orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            image.GetDirection()
        )
    return Geometry(
        size=tuple(image.GetSize()),
        spacing=tuple(image.GetSpacing()),
        origin=tuple(image.GetOrigin()),
        direction=tuple(image.GetDirection()),
        orientation=orientation,
    )


def _load(path: Path, role: str, report: ValidationReport) -> sitk.Image | None:
    if not path.is_file():
        report.add(IssueCode.FILE_NOT_FOUND, Severity.ERROR, role)
        return None
    if not path.name.endswith((".nii", ".nii.gz")):
        report.add(IssueCode.UNSUPPORTED_FORMAT, Severity.ERROR, f"{role}: expected NIfTI")
        return None
    try:
        image = read_nifti(path)
    except RuntimeError:
        report.add(IssueCode.UNREADABLE, Severity.ERROR, role)
        return None
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        report.add(
            IssueCode.NOT_3D_SCALAR,
            Severity.ERROR,
            f"{role}: dimension={image.GetDimension()}, "
            f"components={image.GetNumberOfComponentsPerPixel()}",
        )
        return None
    return image


def _load_checked(
    path: Path, role: str, tolerance: float, report: ValidationReport
) -> sitk.Image | None:
    image = _load(path, role, report)
    if image is not None:
        _inspect_raw(path, role, tolerance, report)
    return image


def _inspect_raw(path: Path, role: str, tolerance: float, report: ValidationReport) -> None:
    """Checks on the file as stored, bypassing ITK (see module docstring)."""
    raw = nib.load(path)
    data = np.asanyarray(raw.dataobj)
    if np.issubdtype(data.dtype, np.floating) and not np.isfinite(data).all():
        report.add(IssueCode.NON_FINITE_VALUES, Severity.ERROR, role)
    header = raw.header
    if int(header["qform_code"]) > 0 and int(header["sform_code"]) > 0:
        difference = np.abs(header.get_qform() - header.get_sform()).max()
        if difference > max(tolerance, 1e-3):  # qform is stored with limited precision
            report.add(IssueCode.QFORM_SFORM_DISAGREE, Severity.WARNING, role)


def _max_abs_diff(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def validate_pair(
    image_path: Path, mask_path: Path, label: int, settings: ValidationSettings
) -> ValidationReport:
    """Validate one image/mask pair for the ROI identified by ``label``."""
    report = ValidationReport()
    image = _load_checked(image_path, "image", settings.geometry_tolerance, report)
    mask = _load_checked(mask_path, "mask", settings.geometry_tolerance, report)
    if image is not None:
        report.image_geometry = geometry_of(image)
    if mask is not None:
        report.mask_geometry = geometry_of(mask)
    if image is None or mask is None:
        return report
    img_geo, mask_geo = report.image_geometry, report.mask_geometry
    tol = settings.geometry_tolerance

    # -- grid geometry ------------------------------------------------------ #
    if img_geo.size != mask_geo.size:
        report.add(IssueCode.SIZE_MISMATCH, Severity.ERROR, f"{img_geo.size} vs {mask_geo.size}")
    if _max_abs_diff(img_geo.spacing, mask_geo.spacing) > tol:
        report.add(IssueCode.SPACING_MISMATCH, Severity.ERROR,
                   f"{img_geo.spacing} vs {mask_geo.spacing}")
    if _max_abs_diff(img_geo.origin, mask_geo.origin) > tol:
        report.add(IssueCode.ORIGIN_MISMATCH, Severity.ERROR,
                   f"{img_geo.origin} vs {mask_geo.origin}")
    if _max_abs_diff(img_geo.direction, mask_geo.direction) > tol:
        report.add(IssueCode.DIRECTION_MISMATCH, Severity.ERROR,
                   f"{img_geo.orientation} vs {mask_geo.orientation}")
    if settings.expected_orientation and img_geo.orientation != settings.expected_orientation:
        report.add(IssueCode.UNEXPECTED_ORIENTATION, Severity.WARNING,
                   f"{img_geo.orientation}, expected {settings.expected_orientation}")
    if max(img_geo.spacing) / min(img_geo.spacing) > settings.max_spacing_anisotropy:
        report.add(IssueCode.ANISOTROPIC_SPACING, Severity.WARNING, f"{img_geo.spacing}")

    # -- values ------------------------------------------------------------- #
    image_array = sitk.GetArrayViewFromImage(image)
    mask_array = sitk.GetArrayViewFromImage(mask)
    if any(i.code is IssueCode.NON_FINITE_VALUES and i.detail == "mask" for i in report.issues):
        return report
    if not np.array_equal(mask_array, np.rint(mask_array)):
        report.add(IssueCode.MASK_NOT_INTEGER, Severity.ERROR)
        return report

    # -- region of interest -------------------------------------------------- #
    roi = np.asarray(mask_array) == label
    report.roi_voxels = int(roi.sum())
    if report.roi_voxels == 0:
        report.add(IssueCode.LABEL_ABSENT, Severity.ERROR, f"label {label}")
        return report
    if report.roi_voxels < settings.min_roi_voxels:
        report.add(IssueCode.ROI_TOO_SMALL, Severity.ERROR,
                   f"{report.roi_voxels} < {settings.min_roi_voxels} voxels")
    components = sitk.ConnectedComponent(sitk.GetImageFromArray(roi.astype(np.uint8)))
    report.roi_components = int(sitk.GetArrayViewFromImage(components).max())
    if report.roi_components > 1:
        report.add(IssueCode.ROI_MULTIPLE_COMPONENTS, Severity.WARNING,
                   f"{report.roi_components} components")
    if any(
        roi.take(index, axis=axis).any() for axis in range(3) for index in (0, -1)
    ):
        report.add(IssueCode.ROI_TOUCHES_BORDER, Severity.WARNING)
    if roi.shape == image_array.shape and report.ok:
        values = np.asarray(image_array)[roi]
        if float(values.max()) == float(values.min()):
            report.add(IssueCode.ROI_CONSTANT_INTENSITY, Severity.ERROR)
    return report
