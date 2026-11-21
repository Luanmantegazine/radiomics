"""Experiment configuration templates and validation (no scientific logic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from shared.config import ExperimentConfig, config_hash


def phases(config: ExperimentConfig) -> list[str]:
    out = []
    if config.rounds:
        out.append("infrastructure")
    for name in ("harmonization", "feature_selection", "training", "evaluation"):
        if getattr(config, name) is not None:
            out.append(name)
    return out


def validate(raw: Any) -> dict:
    """Structured validation result for a candidate configuration."""
    try:
        config = ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {"location": ".".join(str(p) for p in error["loc"]) or "(root)",
                 "message": error["msg"]}
                for error in exc.errors()
            ],
        }
    return {
        "valid": True,
        "config_hash": config_hash(config),
        "phases": phases(config),
        "expected_clients": list(config.expected_clients),
        "data_origin": config.data_origin.value,
    }


def list_templates(directory: str | Path) -> list[dict]:
    """Top-level experiment configurations shipped with the code."""
    out = []
    for path in sorted(Path(directory).glob("*.yaml")):
        try:
            config = ExperimentConfig.model_validate(yaml.safe_load(path.read_text("utf-8")))
        except (ValidationError, yaml.YAMLError):
            continue  # not an experiment configuration (or broken): not offered
        out.append({
            "id": path.stem,
            "name": config.name,
            "description": config.description,
            "data_origin": config.data_origin.value,
            "phases": phases(config),
            "config": config.model_dump(mode="json"),
        })
    return out
