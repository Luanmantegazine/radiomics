"""Versioned experiment configuration (YAML -> validated Pydantic models)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from shared.protocol import (
    PROTOCOL_VERSION,
    ClientId,
    DataOrigin,
    FeatureName,
    FederatedPhase,
    PrivacyPolicy,
    TaskKind,
    Token,
)
from shared.selection import FederatedSelectionConfig

CONFIG_SCHEMA_VERSION = 1


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RoundSpec(ConfigModel):
    """One federated round: which local task runs and how results are aggregated."""

    phase: FederatedPhase
    task: TaskKind
    aggregator: Token
    params: dict[Token, bool | int | float | str | None] = Field(default_factory=dict)


class DerivedFeatureSpec(ConfigModel):
    """A synthetic feature equal to another one plus independent noise."""

    source: FeatureName
    noise_sd: float = Field(ge=0)


class SyntheticDataSourceSpec(ConfigModel):
    """Synthetic local dataset. For infrastructure tests only, never for science."""

    kind: Literal["synthetic"] = "synthetic"
    n_participants: int = Field(ge=1)
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    log_hazard_effects: dict[FeatureName, float] = Field(default_factory=dict)
    """Planted effects on the conversion hazard (empty = no relation at all)."""
    derived_features: dict[FeatureName, DerivedFeatureSpec] = Field(default_factory=dict)
    """Correlated copies (to exercise redundancy handling)."""
    measurement_noise_sd: dict[FeatureName, float] | None = None
    """When set, the site simulates a second, noisy observation of every feature
    (noise 0 for features not listed) and estimates its reliability from it."""

    @model_validator(mode="after")
    def _known_features(self) -> SyntheticDataSourceSpec:
        listed = (
            set(self.log_hazard_effects)
            | set(self.measurement_noise_sd or {})
            | set(self.derived_features)
            | {d.source for d in self.derived_features.values()}
        )
        if not listed <= set(self.feature_names):
            raise ValueError("effects/noise refer to features the source does not have")
        if any(sd < 0 for sd in (self.measurement_noise_sd or {}).values()):
            raise ValueError("measurement_noise_sd must be non-negative")
        if set(self.derived_features) & {d.source for d in self.derived_features.values()}:
            raise ValueError("a derived feature cannot be the source of another")
        return self


class FeatureTableSourceSpec(ConfigModel):
    """Feature table produced by the site's local radiomic pipeline (Stage 2)."""

    kind: Literal["feature_table"] = "feature_table"
    features_path: str
    manifest_path: str
    reliability_path: str | None = None
    """``reliability_summary.csv`` of the site's reliability study, if any."""
    data_origin: DataOrigin
    """Declared provenance of the underlying images (phantoms are 'synthetic')."""


DataSourceSpec = Annotated[
    SyntheticDataSourceSpec | FeatureTableSourceSpec, Field(discriminator="kind")
]


class SimulatedClientSpec(ConfigModel):
    client_id: ClientId
    data_source: DataSourceSpec


class SimulationSpec(ConfigModel):
    """Single-machine simulation settings.

    In a distributed deployment this section does not exist: each institution
    keeps its own local data configuration, which the coordinator never sees.
    """

    clients: tuple[SimulatedClientSpec, ...] = Field(min_length=1)


class ExperimentConfig(ConfigModel):
    schema_version: Literal[1] = CONFIG_SCHEMA_VERSION
    name: Token
    description: str = Field(default="", max_length=2000)
    seed: int = Field(ge=0)
    protocol_version: Token = PROTOCOL_VERSION
    data_origin: DataOrigin
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    expected_clients: tuple[ClientId, ...] = Field(min_length=2)
    """Allowlist of pseudonymised sites permitted to register."""
    rounds: tuple[RoundSpec, ...] = ()
    """Fixed rounds (infrastructure phase), run before any feature selection."""
    feature_selection: FederatedSelectionConfig | None = None
    simulation: SimulationSpec | None = None

    @model_validator(mode="after")
    def _consistency(self) -> ExperimentConfig:
        if not self.rounds and self.feature_selection is None:
            raise ValueError("configure at least one round or a feature_selection")
        if len(set(self.expected_clients)) != len(self.expected_clients):
            raise ValueError("expected_clients must be unique")
        if len(self.expected_clients) < self.privacy.min_clients_per_aggregate:
            raise ValueError("fewer expected clients than privacy.min_clients_per_aggregate")
        if self.simulation is not None:
            simulated = [c.client_id for c in self.simulation.clients]
            if len(set(simulated)) != len(simulated):
                raise ValueError("simulated client ids must be unique")
            origins = {
                getattr(c.data_source, "data_origin", DataOrigin.SYNTHETIC)
                for c in self.simulation.clients
            }
            if origins != {self.data_origin}:
                raise ValueError("every simulated data source must match data_origin")
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment configuration file."""
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentConfig.model_validate(yaml.safe_load(handle))


def config_hash(config: ExperimentConfig) -> str:
    """Stable short hash of a configuration, used to version every result."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
