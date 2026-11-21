from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import ClientRecord
from backend.app.schemas import ClientOut

router = APIRouter(prefix="/clients", tags=["clients"])


@router.get("", response_model=list[ClientOut])
def list_clients(experiment_id: str | None = None, session: Session = Depends(get_session)):
    """Pseudonymised sites and their registration status."""
    query = select(ClientRecord).order_by(ClientRecord.experiment_id, ClientRecord.client_id)
    if experiment_id:
        query = query.where(ClientRecord.experiment_id == experiment_id)
    return list(session.scalars(query))
