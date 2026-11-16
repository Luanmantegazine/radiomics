"""Reliability study: produce repeated observations and estimate ICCs.

A study is a list of *families*; each family compares the baseline extraction
with one or more alternative conditions of the same kind:

``mask_morphology``
    The ROI is eroded or dilated by a number of voxels - a controlled proxy
    for segmentation variability.
``settings_override``
    The extraction runs with modified PyRadiomics settings (e.g. another bin
    width) - robustness to processing choices.
``alternative_manifest``
    Another manifest for the same participants, typically pointing to
    alternative segmentations (second rater, another algorithm).

Hypotheses and limitations
--------------------------
* Perturbation robustness is **not** test-retest reliability: it does not
  capture scanner, positioning or biological variability. Only repeated scans
  can. Each family answers a different question, so results are reported per
  family and never pooled into a single ICC.
* Erosion/dilation is isotropic in voxels and systematically changes volume;
  real segmentation disagreement is neither. Shape features will look
  unreliable under ``icc2`` by construction.
* Conditions are chosen by the analyst (fixed, not sampled), so the ICC
  describes agreement across *these* conditions only.
* Participants for whom any condition fails are dropped from that family
  (complete cases), which can bias towards easier cases.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
import SimpleITK as sitk
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from client.imaging.extraction import ExtractionConfig
from client.imaging.manifest import Manifest, load_manifest
from client.imaging.pipeline import FEATURES_FILE, run_extraction
from client.imaging.validation import read_nifti
from client.reliability.icc import IccForm, icc
from client.reliability.repeated import assemble, load_feature_table
from shared.logging import get_logger

RELIABILITY_FILE = "reliability.csv"
SUMMARY_FILE = "reliability_summary.csv"
_NAME = r"^[a-z0-9_]{1,48}$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MaskMorphology(_Model):
    kind: Literal["mask_morphology"]
    name: str = Field(pattern=_NAME)
    operation: Literal["erode", "dilate"]
    radius_voxels: int = Field(ge=1, le=5)


class SettingsOverride(_Model):
    kind: Literal["settings_override"]
    name: str = Field(pattern=_NAME)
    settings: dict[str, Any] = Field(min_length=1)


class AlternativeManifest(_Model):
    kind: Literal["alternative_manifest"]
    name: str = Field(pattern=_NAME)
    manifest_path: str


Condition = Annotated[
    MaskMorphology | SettingsOverride | AlternativeManifest, Field(discriminator="kind")
]


class Family(_Model):
    name: str = Field(pattern=_NAME)
    conditions: tuple[Condition, ...] = Field(min_length=1)


class ReliabilityStudyConfig(_Model):
    schema_version: Literal[1]
    study_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]{1,64}$")
    icc_form: IccForm
    confidence_level: float = Field(gt=0, lt=1)
    min_subjects: int = Field(ge=2)
    """Refuse to estimate an ICC from fewer complete participants."""
    families: tuple[Family, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_names(self) -> ReliabilityStudyConfig:
        names = [f.name for f in self.families]
        if len(set(names)) != len(names) or "baseline" in names:
            raise ValueError("family names must be unique and different from 'baseline'")
        for family in self.families:
            inner = [c.name for c in family.conditions]
            if len(set(inner)) != len(inner) or "baseline" in inner:
                raise ValueError(f"condition names in family {family.name!r} must be unique")
        return self


def load_study_config(path: str | Path) -> ReliabilityStudyConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return ReliabilityStudyConfig.model_validate(yaml.safe_load(handle))


def _write_morphology_manifest(
    manifest: Manifest, condition: MaskMorphology, directory: Path
) -> Path:
    """Write eroded/dilated binary masks and a manifest pointing to them."""
    masks = directory / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    operate = sitk.BinaryErode if condition.operation == "erode" else sitk.BinaryDilate
    rows = []
    for index, case in enumerate(manifest.cases):
        target = masks / f"{index:06d}.nii.gz"
        try:
            binary = sitk.Cast(read_nifti(case.mask_path) == case.mask_label, sitk.sitkUInt8)
            sitk.WriteImage(operate(binary, [condition.radius_voxels] * 3), str(target))
        except RuntimeError:
            target = case.mask_path.parent / "__unreadable__.nii.gz"  # fails validation later
        rows.append(
            {
                "participant_id": case.participant_id,
                "roi": case.roi,
                "image_path": str(case.image_path),
                "mask_path": str(target),
                "mask_label": 1,
            }
        )
    path = directory / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _run_condition(
    manifest: Manifest, extraction: ExtractionConfig, condition: Condition, directory: Path
) -> pd.DataFrame:
    config, source = extraction, manifest
    if isinstance(condition, MaskMorphology):
        source = load_manifest(_write_morphology_manifest(manifest, condition, directory))
    elif isinstance(condition, AlternativeManifest):
        source = load_manifest(condition.manifest_path)
    else:
        raw = extraction.model_dump(mode="json")
        unknown = set(condition.settings) - set(raw["settings"])
        if unknown:
            raise ValueError(f"unknown PyRadiomics settings in override: {sorted(unknown)}")
        raw["settings"].update(condition.settings)
        raw["config_id"] = f"{extraction.config_id}.{condition.name}"[:64]
        config = ExtractionConfig.model_validate(raw)
    run_extraction(source, config, directory, overwrite=True)
    return load_feature_table(directory / FEATURES_FILE)


def run_reliability_study(
    manifest_path: str | Path,
    extraction: ExtractionConfig,
    study: ReliabilityStudyConfig,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Run every condition, estimate ICCs per family and write local reports.

    Returns the per-feature summary (also written to ``reliability_summary.csv``).
    """
    log = get_logger("client.reliability")
    output_dir = Path(output_dir)
    manifest = load_manifest(manifest_path)
    run_extraction(manifest, extraction, output_dir / "baseline", overwrite=True)
    baseline = load_feature_table(output_dir / "baseline" / FEATURES_FILE)

    frames = []
    for family in study.families:
        tables = {"baseline": baseline}
        for condition in family.conditions:
            tables[condition.name] = _run_condition(
                manifest, extraction, condition, output_dir / family.name / condition.name
            )
        repeated = assemble(tables)
        result = icc(
            repeated.values,
            study.icc_form,
            confidence_level=study.confidence_level,
            min_subjects=study.min_subjects,
        )
        frames.append(
            pd.DataFrame(
                {
                    "feature": repeated.feature_names,
                    "family": family.name,
                    "icc_form": study.icc_form.value,
                    "icc": result.estimate,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "n_subjects": result.n_subjects,
                    "n_conditions": result.n_conditions,
                }
            )
        )
        log.info(
            "family_estimated",
            family=family.name,
            n_subjects=result.n_subjects,
            n_conditions=result.n_conditions,
        )
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(output_dir / RELIABILITY_FILE, index=False, lineterminator="\n")
    order = {name: i for i, name in enumerate(baseline.columns)}
    summary = (
        detail.groupby("feature")
        .agg(
            min_icc=("icc", lambda s: s.min(skipna=False)),
            min_ci_low=("ci_low", lambda s: s.min(skipna=False)),
            n_families=("family", "nunique"),
        )
        .reset_index()
        .sort_values("feature", key=lambda s: s.map(order))
    )
    summary.insert(1, "study_id", study.study_id)
    summary.insert(2, "icc_form", study.icc_form.value)
    summary.to_csv(output_dir / SUMMARY_FILE, index=False, lineterminator="\n")
    return summary


def load_reliability(path: str | Path, column: str = "min_icc") -> dict[str, float]:
    """Feature -> reliability value from a summary file (NaN = undefined)."""
    table = pd.read_csv(path)
    if column not in table.columns:
        raise ValueError(f"{path} has no column {column!r}")
    return dict(zip(table["feature"], table[column].astype(float), strict=True))


__all__ = [
    "ReliabilityStudyConfig", "load_reliability", "load_study_config", "run_reliability_study",
]
