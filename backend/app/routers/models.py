from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import ModelRecord
from backend.app.schemas import ModelDetail, ModelOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models(experiment_id: str | None = None, session: Session = Depends(get_session)):
    """Trained models (status, features). Parameters: ``/models/{id}``."""
    query = select(ModelRecord).order_by(ModelRecord.created_at.desc())
    if experiment_id:
        query = query.where(ModelRecord.experiment_id == experiment_id)
    return list(session.scalars(query))


@router.get("/{model_id}", response_model=ModelDetail)
def get_model(model_id: int, session: Session = Depends(get_session)):
    """Released model: global coefficients, standardisation, baseline, diagnostics."""
    record = session.get(ModelRecord, model_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model not found")
    return record
