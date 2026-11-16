from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.models import FederatedRound
from backend.app.schemas import RoundDetail, RoundOut

router = APIRouter(prefix="/federated-rounds", tags=["federated-rounds"])


@router.get("", response_model=list[RoundOut])
def list_rounds(experiment_id: str | None = None, session: Session = Depends(get_session)):
    query = select(FederatedRound).order_by(
        FederatedRound.experiment_id, FederatedRound.round_number
    )
    if experiment_id:
        query = query.where(FederatedRound.experiment_id == experiment_id)
    return list(session.scalars(query))


@router.get("/{round_id}", response_model=RoundDetail)
def get_round(round_id: str, session: Session = Depends(get_session)):
    round_ = session.get(FederatedRound, round_id)
    if round_ is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "round not found")
    return round_
