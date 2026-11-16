from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import ModelRecord
from backend.app.schemas import ModelOut

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models(experiment_id: str | None = None, session: Session = Depends(get_session)):
    """Registered model versions. Empty until prognostic modelling (Stage 5)."""
    query = select(ModelRecord).order_by(ModelRecord.created_at.desc())
    if experiment_id:
        query = query.where(ModelRecord.experiment_id == experiment_id)
    return list(session.scalars(query))
