from __future__ import annotations

from pathlib import Path

import pytest

from shared.config import ExperimentConfig, load_experiment_config
from shared.logging import configure_logging

CONFIG_PATH = Path(__file__).resolve().parents[2] / "experiments/configs/stage1_simulation.yaml"


@pytest.fixture(autouse=True, scope="session")
def _quiet_logs() -> None:
    configure_logging("ERROR")


@pytest.fixture
def config() -> ExperimentConfig:
    return load_experiment_config(CONFIG_PATH)


def with_updates(config: ExperimentConfig, **changes) -> ExperimentConfig:
    """Copy of a config with top-level fields replaced (re-validated)."""
    return ExperimentConfig.model_validate({**config.model_dump(mode="json"), **changes})
