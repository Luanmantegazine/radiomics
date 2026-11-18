from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import (
    CostRecord,
    EvaluationRecord,
    FeatureSelectionRecord,
    HarmonizationRecord,
    SelectedFeatureRecord,
)
from backend.app.schemas import CostOut, ResultsOut, RoundResult, SelectedFeatureOut
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


@router.get("/{experiment_id}/feature-selection")
def get_feature_selection(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    """Released federated selection result: strategies, selections, stability."""
    experiment = _experiment_or_404(session, experiment_id)
    record = session.scalar(
        select(FeatureSelectionRecord).where(FeatureSelectionRecord.experiment_id == experiment_id)
    )
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no feature selection for this experiment")
    return {"experiment_id": experiment_id, "data_origin": experiment.data_origin, **record.result}


@router.get("/{experiment_id}/selected-features", response_model=list[SelectedFeatureOut])
def get_selected_features(
    experiment_id: str, strategy: str | None = None, session: Session = Depends(get_session)
):
    _experiment_or_404(session, experiment_id)
    query = select(SelectedFeatureRecord).where(
        SelectedFeatureRecord.experiment_id == experiment_id
    )
    if strategy:
        query = query.where(SelectedFeatureRecord.strategy == strategy)
    return list(
        session.scalars(query.order_by(SelectedFeatureRecord.strategy, SelectedFeatureRecord.rank))
    )


@router.get("/{experiment_id}/harmonization")
def get_harmonization(experiment_id: str, session: Session = Depends(get_session)) -> dict:
    """Site effects before/after harmonization and clinical differences between sites."""
    experiment = _experiment_or_404(session, experiment_id)
    record = session.scalar(
        select(HarmonizationRecord).where(HarmonizationRecord.experiment_id == experiment_id)
    )
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no harmonization for this experiment")
    return {"experiment_id": experiment_id, "data_origin": experiment.data_origin,
            **record.result}


@router.get("/{experiment_id}/evaluation")
def get_evaluation(
    experiment_id: str,
    model_name: str | None = None,
    scope: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Pooled (site-stratified) metrics per model, variant and scope."""
    experiment = _experiment_or_404(session, experiment_id)
    query = select(EvaluationRecord).where(EvaluationRecord.experiment_id == experiment_id)
    if model_name:
        query = query.where(EvaluationRecord.model_name == model_name)
    if scope:
        query = query.where(EvaluationRecord.scope == scope)
    rows = session.scalars(query.order_by(EvaluationRecord.id)).all()
    return {
        "experiment_id": experiment_id,
        "data_origin": experiment.data_origin,
        "evaluations": [
            {"model_name": r.model_name, "variant": r.variant, "scope": r.scope,
             "n_clients": r.n_clients, **r.metrics}
            for r in rows
        ],
    }


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
