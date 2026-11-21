"""Coordinator process of a distributed run (sites connect over HTTPS).

    python -m experiments.run_distributed --experiment-id exp-0123456789ab \
        --sites infrastructure/distributed/sites.yaml --port 8700 \
        --ssl-certfile cert.pem --ssl-keyfile key.pem

Waits for the expected sites to register (``--registration-timeout``), runs the
configured phases exactly as the simulation does, records the results in the
database and MLflow, then tells the sites the session is over.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from backend.app.db import init_db, make_engine
from backend.app.services import experiments as experiment_service
from backend.app.services.recording import DatabaseRunRecorder
from backend.app.settings import get_settings
from backend.app.tracking import make_tracker
from coordinator.engine import FederatedCoordinator
from coordinator.network.api import create_site_api
from coordinator.network.hub import SiteHub
from coordinator.network.server import BackgroundServer
from coordinator.network.tokens import load_sites
from coordinator.ports import RunRecorder
from shared.config import ExperimentConfig, load_experiment_config
from shared.logging import configure_logging, get_logger


def run_distributed(
    experiment_id: str,
    config: ExperimentConfig,
    token_hashes: dict[str, str],
    *,
    host: str = "0.0.0.0",
    port: int = 8700,
    registration_timeout_s: float = 600.0,
    round_timeout_s: float = 900.0,
    recorder: RunRecorder | None = None,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    capture: bool = False,
    on_listening=None,
):
    """Serve the site endpoints and run the experiment. Returns (outcome, hub)."""
    log = get_logger("coordinator.distributed").bind(experiment_id=experiment_id)
    expected = set(config.expected_clients)
    unknown = expected - set(token_hashes)
    if unknown:
        raise ValueError(f"no token configured for expected sites: {sorted(unknown)}")
    hub = SiteHub(experiment_id, expected, round_timeout_s=round_timeout_s, capture=capture)
    server = BackgroundServer(
        create_site_api(hub, {c: h for c, h in token_hashes.items() if c in expected}),
        host, port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile,
    ).start()
    if on_listening is not None:
        on_listening(server.port)
    log.info("waiting_for_sites", port=server.port, expected=sorted(expected))
    outcome = None
    try:
        registered = hub.wait_for_registrations(registration_timeout_s)
        log.info("sites_registered", registered=list(registered),
                 missing=sorted(expected - set(registered)))
        outcome = FederatedCoordinator(experiment_id, config, hub, recorder).run()
    finally:
        hub.end_session(bool(outcome and outcome.completed))
        deadline = time.monotonic() + 30
        while not hub.drained() and time.monotonic() < deadline:
            time.sleep(0.1)  # let the sites fetch the end of session
        server.stop()
    return outcome, hub


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--experiment-id")
    parser.add_argument("--sites", required=True, help="sites.yaml with token hashes")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8700)
    parser.add_argument("--registration-timeout", type=float, default=600.0)
    parser.add_argument("--round-timeout", type=float, default=900.0)
    parser.add_argument("--ssl-certfile")
    parser.add_argument("--ssl-keyfile")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(args.log_level or settings.log_level)
    sessions = init_db(make_engine(settings.database_url))
    with sessions() as session:
        if args.config:
            experiment = experiment_service.create_experiment(
                session, load_experiment_config(args.config))
        else:
            experiment = experiment_service.get_experiment(session, args.experiment_id)
        experiment_id = experiment.id
        config = ExperimentConfig.model_validate(experiment.config)
    if config.simulation is not None:
        print("this configuration describes simulated sites; use run_simulation",
              file=sys.stderr)
        return 2
    recorder = DatabaseRunRecorder(
        experiment_id, sessions,
        make_tracker(settings.mlflow_tracking_uri, settings.mlflow_artifact_root),
    )
    try:
        outcome, _ = run_distributed(
            experiment_id, config, load_sites(args.sites), host=args.host, port=args.port,
            registration_timeout_s=args.registration_timeout,
            round_timeout_s=args.round_timeout, recorder=recorder,
            ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile,
        )
    except Exception as exc:
        with sessions() as session:
            experiment_service.mark_failed(session, experiment_id, f"{type(exc).__name__}: {exc}")
        raise
    json.dump({"experiment_id": experiment_id, "completed": outcome.completed,
               "registered_clients": outcome.registered_clients,
               "rounds": len(outcome.rounds), "total_bytes": outcome.total_bytes},
              sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if outcome.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
