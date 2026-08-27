"""Shared fixtures and path setup for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: FreeSurfer sessions of the local pilot, if they were downloaded.
SMOKETEST_FREESURFER_DIR = REPO_ROOT / "oasis3_radiomics_smoketest" / "freesurfer"
PILOT_SESSIONS = ("OAS30001_MR_d0129", "OAS30001_MR_d0757")


def local_data_available() -> bool:
    """Whether both pilot sessions are present with T1 and aseg."""
    return all(
        (SMOKETEST_FREESURFER_DIR / session / "mri" / name).exists()
        for session in PILOT_SESSIONS
        for name in ("T1.mgz", "aseg.mgz")
    )


requires_local_data = pytest.mark.skipif(
    not local_data_available(),
    reason=(
        "OASIS-3 data is not distributed with this repository. Download the "
        "pilot sessions first: python cli.py download --nitrc-user <user>"
    ),
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config():
    """The repository's default pipeline configuration."""
    from oasis_radiomics.config import PipelineConfig

    return PipelineConfig.load(None)


@pytest.fixture
def long_rows() -> list[dict]:
    """A synthetic long table: one subject, three visits, two ROIs."""
    rows = []
    for session_index, (days, left, right) in enumerate(
        [(129, 4000.0, 4200.0), (757, 3900.0, 4100.0), (1400, 3800.0, 4000.0)]
    ):
        session_id = f"OAS39999_MR_d{days:04d}"
        for roi, volume in (("left_hippocampus", left), ("right_hippocampus", right)):
            rows.append(
                {
                    "subject_id": "OAS39999",
                    "session_id": session_id,
                    "days_from_reference": days,
                    "roi": roi,
                    "mask_voxels": volume,
                    "mask_volume_mm3": volume,
                    "image_path": f"/tmp/{session_id}/T1.nii.gz",
                    "mask_path": f"/tmp/{session_id}/{roi}_mask.nii.gz",
                    "original_shape_MeshVolume": volume - 10.0,
                    "original_firstorder_Entropy": 2.5 - 0.1 * session_index,
                    "original_firstorder_Skewness": -0.1 - 0.01 * session_index,
                }
            )
    return rows
