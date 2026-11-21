from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from backend.app.deps import get_session
from backend.app.services import experiments as service
from backend.app.services.reports import TABLES, bundle, csv_table, html_report

router = APIRouter(prefix="/reports", tags=["reports"])


def _experiment(session: Session, experiment_id: str):
    try:
        return service.get_experiment(session, experiment_id)
    except service.ExperimentNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found") from None


def _attachment(name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


@router.get("/{experiment_id}")
def export_report(
    experiment_id: str,
    format: Literal["json", "html"] = Query(default="json"),
    download: bool = Query(default=False),
    session: Session = Depends(get_session),
):
    """Full report of an experiment: JSON bundle or self-contained HTML page."""
    experiment = _experiment(session, experiment_id)
    if format == "html":
        headers = _attachment(f"{experiment_id}.html") if download else None
        return HTMLResponse(html_report(session, experiment), headers=headers)
    headers = _attachment(f"{experiment_id}.json") if download else None
    return JSONResponse(bundle(session, experiment), headers=headers)


@router.get("/{experiment_id}/{table}.csv", response_class=PlainTextResponse)
def export_table(experiment_id: str, table: str, session: Session = Depends(get_session)):
    """One result table as CSV: evaluation, selected-features, models, costs, rounds."""
    if table not in TABLES:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown table; available: {TABLES}")
    experiment = _experiment(session, experiment_id)
    return PlainTextResponse(
        csv_table(session, experiment, table), media_type="text/csv",
        headers=_attachment(f"{experiment_id}-{table}.csv"),
    )
