"""Synthetic NIfTI phantoms for testing the imaging pipeline.

Random textures with geometric masks: they exercise I/O, validation and
extraction mechanics only and have no anatomical or scientific meaning.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from client.imaging.extraction import ExtractionConfig, load_extraction_config

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "experiments/configs/extraction/t1_original_v1.yaml"
SHAPE = (28, 30, 32)  # (x, y, z)


def config(**settings) -> ExtractionConfig:
    """Shipped extraction config with some PyRadiomics settings overridden."""
    base = load_extraction_config(DEFAULT_CONFIG).model_dump(mode="json")
    base["settings"].update(settings)
    return ExtractionConfig.model_validate(base)


def write(array_xyz: np.ndarray, path: Path, *, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
          direction=None) -> Path:
    image = sitk.GetImageFromArray(np.transpose(array_xyz, (2, 1, 0)))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))
    return path


def texture(seed: int, shape=SHAPE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 60, shape[0])[:, None, None]
    return (100 + ramp + rng.normal(0, 25, shape)).astype(np.float32)


def labels(shape=SHAPE, variant: int = 0) -> np.ndarray:
    """Label 1: a box; label 2: a sphere. ``variant`` changes their sizes."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4 : 12 + variant % 4, 5 : 15 - variant % 3, 6 : 14 + variant % 5] = 1
    x, y, z = np.indices(shape)
    mask[(x - 19) ** 2 + (y - 20) ** 2 + (z - 20) ** 2 <= (6 - variant % 3) ** 2] = 2
    return mask


def write_site(root: Path, n_participants: int, *, seed: int, prefix: str = "P",
               rois: dict[str, int] | None = None, followup: bool = True,
               vary_masks: bool = False) -> Path:
    """Write images, masks and a manifest for one simulated site."""
    rois = rois or {"box": 1, "sphere": 2}
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_participants):
        participant = f"{prefix}{seed:02d}{index:03d}"
        write(texture(seed * 1000 + index), root / "images" / f"{participant}.nii.gz")
        write(labels(variant=index if vary_masks else 0),
              root / "masks" / f"{participant}.nii.gz")
        time_months = f"{rng.uniform(6, 48):.2f}"
        for roi, label in rois.items():
            row = {
                "participant_id": participant, "roi": roi,
                "image_path": f"images/{participant}.nii.gz",
                "mask_path": f"masks/{participant}.nii.gz", "mask_label": label,
            }
            if followup:
                row["time_months"] = time_months
                row["event"] = int(index % 3 == 0)
            rows.append(row)
    return write_manifest(root / "manifest.csv", rows)


def write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def survival_cohort(n: int = 300, p: int = 12, seed: int = 0):
    """Synthetic time-to-event data with a planted structure (method tests only).

    f0, f1 drive the hazard; f2 is a near-copy of f0; the rest is noise.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    x[:, 2] = x[:, 0] + rng.normal(scale=0.1, size=n)
    hazard = 0.03 * np.exp(0.9 * x[:, 0] + 0.7 * x[:, 1])
    event_time = rng.exponential(1 / hazard)
    censor_time = rng.uniform(5, 60, size=n)
    time = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    return tuple(f"f{i}" for i in range(p)), x, time, event
