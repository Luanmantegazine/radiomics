from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import CostRecord
from backend.app.schemas import CostOut, ResultsOut, RoundResult
from backend.app.services import experiments as service

router = APIRouter(prefix="/results", tags=["results"])


def _experiment_or_404(session: Session, experiment_id: str):
    try:
        return service.get_experiment(session, experiment_id)
    except service.ExperimentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found") from None


@router.get("/{experiment_id}", response_model=ResultsOut)
def get_results(experiment_id: str, session: Session = Depends(get_session)):
    """Global aggregates per round (never per-client statistics)."""
    experiment = _experiment_or_404(session, experiment_id)
    return ResultsOut(
        experiment_id=experiment.id,
        status=experiment.status,
        data_origin=experiment.data_origin,
        config_hash=experiment.config_hash,
        rounds=[
            RoundResult(
                round_id=r.id, round_number=r.round_number, phase=r.phase, task=r.task,
                aggregator=r.aggregator, status=r.status, aggregate=r.aggregate,
            )
            for r in experiment.rounds
        ],
    )


@router.get("/{experiment_id}/costs", response_model=list[CostOut])
def get_costs(experiment_id: str, session: Session = Depends(get_session)):
    """Computational and communication costs recorded for an experiment."""
    _experiment_or_404(session, experiment_id)
    return list(
        session.scalars(
            select(CostRecord).where(CostRecord.experiment_id == experiment_id).order_by(
                CostRecord.id
            )
        )
    )
