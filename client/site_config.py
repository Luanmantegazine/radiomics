"""Local configuration of a participating site (never sent anywhere).

The site decides here which experiment it takes part in, where its data is
and how to reach the coordinator. Example (site.yaml)::

    client_id: client-01
    experiment_id: exp-0123456789ab
    coordinator_url: https://coordinator.example.org
    token_file: /run/secrets/fedrad_site_token
    data_source:
      kind: feature_table
      features_path: /data/radiomics/features.csv
      manifest_path: /data/radiomics/manifest.csv
      reliability_path: /data/radiomics/reliability/reliability_summary.csv
      data_origin: real
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.config import DataSourceSpec
from shared.protocol import ClientId, Identifier


class SiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    client_id: ClientId
    experiment_id: Identifier
    """The one experiment this site agrees to serve; other experiments are refused."""
    coordinator_url: str = Field(pattern=r"^https?://")
    token_file: str
    """File holding the site's token (kept out of the configuration itself)."""
    ca_bundle: str | None = None
    """CA certificate(s) to verify the coordinator's TLS certificate."""
    data_source: DataSourceSpec
    seed: int = Field(default=0, ge=0)
    """Seed of a synthetic data source (demos only)."""
    poll_wait_s: float = Field(default=20.0, gt=0, le=60)
    idle_timeout_s: float = Field(default=3600.0, gt=0)
    """Give up when the coordinator sends nothing for this long."""


def load_site_config(path: str | Path) -> SiteConfig:
    return SiteConfig.model_validate(yaml.safe_load(Path(path).read_text("utf-8")))
