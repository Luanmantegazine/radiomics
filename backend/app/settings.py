"""Backend settings, read from environment variables (prefix ``FEDRAD_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEDRAD_", env_file=".env", extra="ignore")

    #: PostgreSQL in deployment; SQLite is supported for local simulation/tests.
    database_url: str = "sqlite:///./var/fedradiomics.db"
    #: MLflow tracking URI, or "none" to disable tracking.
    mlflow_tracking_uri: str = "sqlite:///./var/mlflow.db"
    #: Artifact root used when the backend creates MLflow experiments itself
    #: (ignored when a tracking server manages artifacts). Empty = MLflow default.
    mlflow_artifact_root: str = "./var/mlartifacts"
    #: "simulation" spawns the single-machine simulation; "distributed" spawns the
    #: coordinator and waits for remote sites (see experiments.run_distributed);
    #: "none" disables runs.
    launcher: str = "simulation"
    #: Distributed launcher: token hashes of the sites, and where sites connect.
    sites_file: str = ""
    coordinator_port: int = 8700
    registration_timeout_s: float = 600.0
    round_timeout_s: float = 900.0
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]
    #: When set, every endpoint except /health requires ``Authorization: Bearer <token>``.
    api_token: str = ""
    #: Directory of experiment configuration templates offered to the dashboard.
    templates_dir: str = "experiments/configs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
