"""ORM models.

By design the central database has **no** table or column for medical images,
masks or individual clinical data. It stores experiment configurations,
pseudonymised client identifiers, run state, global aggregates and costs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="created")
    data_origin: Mapped[str] = mapped_column(String(16))
    protocol_version: Mapped[str] = mapped_column(String(16))
    seed: Mapped[int] = mapped_column(Integer)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    rounds: Mapped[list[FederatedRound]] = relationship(
        back_populates="experiment", order_by="FederatedRound.round_number"
    )


class ClientRecord(Base):
    """A pseudonymised site registered in an experiment."""

    __tablename__ = "clients"
    __table_args__ = (UniqueConstraint("experiment_id", "client_id"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    client_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    reason_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    data_origin: Mapped[str] = mapped_column(String(16))
    software: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class FederatedRound(Base):
    __tablename__ = "federated_rounds"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    round_number: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(32))
    task: Mapped[str] = mapped_column(String(32))
    aggregator: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    aggregate: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    """Global aggregate only; individual client payloads are never stored."""
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    experiment: Mapped[Experiment] = relationship(back_populates="rounds")
    participations: Mapped[list[RoundParticipation]] = relationship(back_populates="round")


class RoundParticipation(Base):
    """Per-client round metadata: status and cohort size, no statistics."""

    __tablename__ = "round_participations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    round_id: Mapped[str] = mapped_column(ForeignKey("federated_rounds.id"), index=True)
    client_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    reason_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    n_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)

    round: Mapped[FederatedRound] = relationship(back_populates="participations")


class CostRecord(Base):
    """One cost measurement (long format)."""

    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    round_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    scope: Mapped[str] = mapped_column(String(16))  # "client" | "coordinator" | "experiment"
    client_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metric: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(16))


class FeatureSelectionRecord(Base):
    """Released result of a federated selection run (global statistics only)."""

    __tablename__ = "feature_selections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id"), index=True, unique=True
    )
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class SelectedFeatureRecord(Base):
    """One selected feature of one strategy (queryable form of the result)."""

    __tablename__ = "selected_features"
    __table_args__ = (UniqueConstraint("experiment_id", "strategy", "rank"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    strategy: Mapped[str] = mapped_column(String(64))
    rank: Mapped[int] = mapped_column(Integer)
    feature: Mapped[str] = mapped_column(String(160))
    ranking_score: Mapped[float] = mapped_column(Float)
    reliability: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_p: Mapped[float | None] = mapped_column(Float, nullable=True)
    agreement: Mapped[float | None] = mapped_column(Float, nullable=True)


class HarmonizationRecord(Base):
    """Harmonization summary: site effects before/after, clinical site differences."""

    __tablename__ = "harmonizations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id"), index=True, unique=True
    )
    result: Mapped[dict[str, Any]] = mapped_column(JSON)


class EvaluationRecord(Base):
    """Pooled (and, if released, per-site) metrics of one model/variant/scope."""

    __tablename__ = "evaluations"
    __table_args__ = (UniqueConstraint("experiment_id", "model_name", "variant", "scope"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    variant: Mapped[str] = mapped_column(String(16))
    scope: Mapped[str] = mapped_column(String(96))
    n_clients: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)


class ModelRecord(Base):
    """A trained (or failed) model of an experiment.

    ``parameters`` is the released model: global coefficients, standardisation,
    pooled baseline and fit diagnostics - no per-site quantity.
    """

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("experiment_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("experiments.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    features_source: Mapped[str] = mapped_column(String(96))
    feature_names: Mapped[list[str]] = mapped_column(JSON, default=list)
    parameters: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    mlflow_model_uri: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
