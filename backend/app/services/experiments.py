"""Experiment lifecycle (create, read, state transitions)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import Experiment, utcnow
from shared.config import ExperimentConfig, config_hash


class ExperimentNotFound(LookupError):
    pass


class InvalidTransition(RuntimeError):
    pass


def create_experiment(session: Session, config: ExperimentConfig) -> Experiment:
    experiment = Experiment(
        id=f"exp-{uuid.uuid4().hex[:12]}",
        name=config.name,
        status="created",
        data_origin=config.data_origin.value,
        protocol_version=config.protocol_version,
        seed=config.seed,
        config=config.model_dump(mode="json"),
        config_hash=config_hash(config),
    )
    session.add(experiment)
    session.commit()
    return experiment


def get_experiment(session: Session, experiment_id: str) -> Experiment:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise ExperimentNotFound(experiment_id)
    return experiment


def list_experiments(session: Session) -> list[Experiment]:
    return list(session.scalars(select(Experiment).order_by(Experiment.created_at.desc())))


def mark_queued(session: Session, experiment_id: str) -> Experiment:
    """Atomically move created -> queued so an experiment runs at most once."""
    experiment = get_experiment(session, experiment_id)
    if experiment.status != "created":
        raise InvalidTransition(f"experiment is '{experiment.status}', expected 'created'")
    experiment.status = "queued"
    session.commit()
    return experiment


def mark_failed(session: Session, experiment_id: str, error: str) -> None:
    experiment = get_experiment(session, experiment_id)
    if experiment.status not in {"completed", "failed"}:
        experiment.status = "failed"
        experiment.error = error[-4000:]
        experiment.finished_at = utcnow()
        session.commit()
