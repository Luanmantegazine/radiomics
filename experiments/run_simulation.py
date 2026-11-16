"""CLI: run a simulated federated experiment and record it.

    python -m experiments.run_simulation --config experiments/configs/stage1_simulation.yaml
    python -m experiments.run_simulation --experiment-id exp-0123456789ab

Results are written to the database (``FEDRAD_DATABASE_URL``) and MLflow
(``FEDRAD_MLFLOW_TRACKING_URI``). Synthetic data only: infrastructure check.
"""

from __future__ import annotations

import argparse
import json
import sys

from backend.app.db import init_db, make_engine
from backend.app.services import experiments as experiment_service
from backend.app.services.recording import DatabaseRunRecorder
from backend.app.settings import get_settings
from backend.app.tracking import make_tracker
from experiments.simulation import run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from shared.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="YAML experiment configuration to register and run")
    source.add_argument("--experiment-id", help="run an experiment already registered in the DB")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level)
    sessions = init_db(make_engine(settings.database_url))

    with sessions() as session:
        if args.config:
            experiment = experiment_service.create_experiment(
                session, load_experiment_config(args.config)
            )
        else:
            experiment = experiment_service.get_experiment(session, args.experiment_id)
        experiment_id = experiment.id
        config = ExperimentConfig.model_validate(experiment.config)

    recorder = DatabaseRunRecorder(
        experiment_id,
        sessions,
        make_tracker(settings.mlflow_tracking_uri, settings.mlflow_artifact_root),
    )
    try:
        outcome = run_simulation(experiment_id, config, recorder=recorder)
    except Exception as exc:
        with sessions() as session:
            experiment_service.mark_failed(session, experiment_id, f"{type(exc).__name__}: {exc}")
        raise
    summary = {
        "experiment_id": outcome.experiment_id,
        "completed": outcome.completed,
        "data_origin": config.data_origin.value,
        "registered_clients": outcome.registered_clients,
        "rounds": [
            {"round": r.round_number, "task": r.task.value, "completed": r.completed,
             "aggregate": r.aggregate.model_dump(mode="json") if r.aggregate else None}
            for r in outcome.rounds
        ],
        "total_bytes": outcome.total_bytes,
        "duration_s": round(outcome.duration_s, 4),
    }
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if outcome.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
