"""Read-side summaries for the dashboard and reports (bookkeeping only)."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models import (
    ClientRecord,
    CostRecord,
    EvaluationRecord,
    Experiment,
    FeatureSelectionRecord,
    FederatedRound,
    HarmonizationRecord,
    ModelRecord,
)
from backend.app.services.templates import phases
from shared.config import ExperimentConfig


def _exists(session: Session, model, experiment_id: str) -> bool:
    return session.scalar(
        select(func.count()).select_from(model).where(model.experiment_id == experiment_id)
    ) > 0


def experiment_summary(session: Session, experiment: Experiment) -> dict:
    clients = session.scalars(
        select(ClientRecord).where(ClientRecord.experiment_id == experiment.id)
    ).all()
    rounds = experiment.rounds
    by_phase: dict[str, int] = defaultdict(int)
    for r in rounds:
        by_phase[r.phase] += 1
    totals = {
        c.metric: c.value
        for c in session.scalars(select(CostRecord).where(
            CostRecord.experiment_id == experiment.id, CostRecord.scope == "experiment"))
    }
    try:
        planned = phases(ExperimentConfig.model_validate(experiment.config))
    except ValueError:  # pragma: no cover - stored configs were validated on creation
        planned = []
    return {
        "experiment": experiment,
        "registered_clients": sum(c.status == "registered" for c in clients),
        "rejected_clients": sum(c.status != "registered" for c in clients),
        "rounds": {"total": len(rounds), "failed": sum(r.status != "completed" for r in rounds),
                   "by_phase": dict(by_phase)},
        "planned_phases": planned,
        "available_results": {
            "harmonization": _exists(session, HarmonizationRecord, experiment.id),
            "feature_selection": _exists(session, FeatureSelectionRecord, experiment.id),
            "models": _exists(session, ModelRecord, experiment.id),
            "evaluation": _exists(session, EvaluationRecord, experiment.id),
        },
        "total_bytes": totals.get("total_bytes"),
        "duration_s": totals.get("total_time"),
    }


def cost_summary(session: Session, experiment_id: str) -> dict:
    records = session.scalars(
        select(CostRecord).where(CostRecord.experiment_id == experiment_id)
    ).all()
    rounds = {
        r.id: r for r in session.scalars(
            select(FederatedRound).where(FederatedRound.experiment_id == experiment_id))
    }
    totals = {r.metric: r.value for r in records if r.round_id is None}
    phase = defaultdict(lambda: defaultdict(float))
    per_round = defaultdict(lambda: defaultdict(float))
    client = defaultdict(lambda: defaultdict(float))
    client_rounds = defaultdict(set)
    memory: dict[str, float] = {}
    for r in records:
        if r.round_id is None:
            continue
        round_ = rounds.get(r.round_id)
        if r.scope == "coordinator" and round_ is not None:
            phase[round_.phase][r.metric] += r.value
            per_round[r.round_id][r.metric] += r.value
        elif r.scope == "client" and r.client_id:
            client_rounds[r.client_id].add(r.round_id)
            if r.metric == "peak_memory":
                memory[r.client_id] = max(memory.get(r.client_id, 0.0), r.value)
            else:
                client[r.client_id][r.metric] += r.value
    phase_rounds = defaultdict(int)
    for round_ in rounds.values():
        phase_rounds[round_.phase] += 1
    return {
        "experiment_id": experiment_id,
        "totals": totals,
        "by_phase": [
            {"phase": name, "rounds": phase_rounds[name],
             "bytes_from_clients": values["bytes_from_clients"],
             "bytes_to_clients": values["bytes_to_clients"],
             "communication_s": values["communication_time"],
             "aggregation_s": values["aggregation_time"],
             "round_time_s": values["round_time"]}
            for name, values in phase.items()
        ],
        "by_client": [
            {"client_id": cid, "rounds": len(client_rounds[cid]),
             "bytes_sent": values["bytes_sent"], "bytes_received": values["bytes_received"],
             "local_execution_s": values["local_execution_time"],
             "peak_memory_bytes": memory.get(cid)}
            for cid, values in sorted(client.items())
        ],
        "by_round": [
            {"round_number": r.round_number, "phase": r.phase, "task": r.task,
             "status": r.status,
             "bytes_from_clients": per_round[r.id]["bytes_from_clients"],
             "bytes_to_clients": per_round[r.id]["bytes_to_clients"],
             "round_time_s": per_round[r.id]["round_time"]}
            for r in sorted(rounds.values(), key=lambda r: r.round_number)
        ],
    }
