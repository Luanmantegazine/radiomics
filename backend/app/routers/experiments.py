from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.deps import get_launcher, get_session
from backend.app.schemas import ExperimentDetail, ExperimentOut
from backend.app.services import experiments as service
from backend.app.services.launchers import ExperimentLauncher
from shared.config import ExperimentConfig

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentDetail, status_code=status.HTTP_201_CREATED)
def create_experiment(config: ExperimentConfig, session: Session = Depends(get_session)):
    """Register a new experiment from a validated configuration."""
    return service.create_experiment(session, config)


@router.get("", response_model=list[ExperimentOut])
def list_experiments(session: Session = Depends(get_session)):
    return service.list_experiments(session)


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: str, session: Session = Depends(get_session)):
    try:
        return service.get_experiment(session, experiment_id)
    except service.ExperimentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found") from None


@router.post(
    "/{experiment_id}/run", response_model=ExperimentOut, status_code=status.HTTP_202_ACCEPTED
)
def run_experiment(
    experiment_id: str,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    launcher: ExperimentLauncher | None = Depends(get_launcher),
):
    """Trigger the federated run asynchronously; poll the experiment for status."""
    if launcher is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no launcher configured")
    try:
        experiment = service.mark_queued(session, experiment_id)
    except service.ExperimentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found") from None
    except service.InvalidTransition as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    background.add_task(launcher.launch, experiment_id)
    return experiment
