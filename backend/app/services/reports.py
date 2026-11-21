"""Experiment reports: JSON bundle, CSV tables and a self-contained HTML page.

Formatting only: every number comes from what the coordinator released and the
recorder stored. Nothing here computes a scientific result.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    ClientRecord,
    EvaluationRecord,
    Experiment,
    FeatureSelectionRecord,
    HarmonizationRecord,
    ModelRecord,
    SelectedFeatureRecord,
)
from backend.app.schemas import ExperimentDetail
from backend.app.services.summaries import cost_summary, experiment_summary
from shared.protocol import PROTOCOL_VERSION

TABLES = ("evaluation", "selected-features", "models", "costs", "rounds")
SYNTHETIC_NOTICE = (
    "SYNTHETIC DATA: these results validate the software only and have no scientific meaning."
)
_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).resolve().parents[1] / "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _version() -> str:
    try:
        return metadata.version("fedradiomics")
    except metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def bundle(session: Session, experiment: Experiment) -> dict:
    """Everything stored about an experiment, in one JSON-serialisable document."""
    eid = experiment.id

    def rows(model):
        return session.scalars(select(model).where(model.experiment_id == eid)).all()

    selection = session.scalar(
        select(FeatureSelectionRecord).where(FeatureSelectionRecord.experiment_id == eid))
    harmonization = session.scalar(
        select(HarmonizationRecord).where(HarmonizationRecord.experiment_id == eid))
    summary = experiment_summary(session, experiment)
    summary["experiment"] = ExperimentDetail.model_validate(experiment).model_dump(mode="json")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "software": {"fedradiomics": _version(), "protocol_version": PROTOCOL_VERSION},
        "notice": SYNTHETIC_NOTICE if experiment.data_origin == "synthetic" else None,
        "summary": summary,
        "clients": [
            {"client_id": c.client_id, "status": c.status, "reason_code": c.reason_code,
             "data_origin": c.data_origin, "software": c.software}
            for c in rows(ClientRecord)
        ],
        "rounds": [
            {"round_number": r.round_number, "phase": r.phase, "task": r.task,
             "aggregator": r.aggregator, "status": r.status, "failure_reason": r.failure_reason}
            for r in experiment.rounds
        ],
        "harmonization": harmonization.result if harmonization else None,
        "feature_selection": selection.result if selection else None,
        "models": [
            {"name": m.name, "kind": m.kind, "status": m.status,
             "failure_reason": m.failure_reason, "features_source": m.features_source,
             "feature_names": m.feature_names, "parameters": m.parameters}
            for m in rows(ModelRecord)
        ],
        "evaluation": [
            {"model_name": e.model_name, "variant": e.variant, "scope": e.scope,
             "n_clients": e.n_clients, **e.metrics}
            for e in rows(EvaluationRecord)
        ],
        "costs": cost_summary(session, eid),
    }


def _table_rows(session: Session, experiment: Experiment, table: str) -> list[dict]:
    eid = experiment.id
    if table == "evaluation":
        out = []
        for e in session.scalars(select(EvaluationRecord).where(
                EvaluationRecord.experiment_id == eid).order_by(EvaluationRecord.id)):
            groups = [("pooled", e.metrics["pooled"])] + [
                (s["client_id"], s["horizons"]) for s in e.metrics.get("sites", [])
            ]
            for level, horizons in groups:
                for h in horizons:
                    out.append({
                        "model": e.model_name, "variant": e.variant, "scope": e.scope,
                        "level": level, "horizon_months": h["horizon_months"],
                        "available": h["available"], "n": h["n"], "n_events": h["n_events"],
                        **{k: h.get(k) for k in ("auc", "c_index", "brier", "scaled_brier",
                                                  "o_e", "calibration_error",
                                                  "observed_incidence")},
                    })
        return out
    if table == "selected-features":
        return [
            {k: getattr(r, k) for k in ("strategy", "rank", "feature", "ranking_score",
                                        "reliability", "relevance_z", "relevance_p",
                                        "agreement")}
            for r in session.scalars(select(SelectedFeatureRecord).where(
                SelectedFeatureRecord.experiment_id == eid).order_by(
                SelectedFeatureRecord.strategy, SelectedFeatureRecord.rank))
        ]
    if table == "models":
        out = []
        for m in session.scalars(select(ModelRecord).where(ModelRecord.experiment_id == eid)):
            p = m.parameters
            if not p:
                out.append({"model": m.name, "kind": m.kind, "status": m.status})
                continue
            fits = ([(None, p["coefficients"], p["standard_errors"], None)]
                    if p["kind"] == "cox" else
                    [(h["horizon_months"], h["coefficients"], h["standard_errors"],
                      h["intercept"]) for h in p["horizons"]])
            for horizon, coefficients, errors, intercept in fits:
                for j, feature in enumerate(p["feature_names"]):
                    out.append({
                        "model": m.name, "kind": m.kind, "status": m.status,
                        "horizon_months": horizon, "intercept": intercept, "feature": feature,
                        "coefficient": coefficients[j], "standard_error": errors[j],
                        "center": p["center"][j], "scale": p["scale"][j],
                    })
        return out
    if table == "costs":
        return cost_summary(session, eid)["by_round"]
    if table == "rounds":
        return [
            {"round_number": r.round_number, "phase": r.phase, "task": r.task,
             "aggregator": r.aggregator, "status": r.status, "failure_reason": r.failure_reason}
            for r in experiment.rounds
        ]
    raise KeyError(table)


def csv_table(session: Session, experiment: Experiment, table: str) -> str:
    rows = _table_rows(session, experiment, table)
    buffer = io.StringIO()
    if experiment.data_origin == "synthetic":
        buffer.write(f"# {SYNTHETIC_NOTICE}\n")
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return buffer.getvalue()


def html_report(session: Session, experiment: Experiment) -> str:
    data = bundle(session, experiment)
    return _ENV.get_template("report.html.j2").render(
        data=data,
        evaluation_rows=_table_rows(session, experiment, "evaluation"),
        model_rows=_table_rows(session, experiment, "models"),
    )
