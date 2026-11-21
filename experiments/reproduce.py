"""Check that an experiment reproduces.

    python -m experiments.reproduce --config experiments/configs/stage5_federated_models.yaml
    python -m experiments.reproduce --experiment-id exp-0123456789ab

``--config`` runs the simulation ``--runs`` times and requires identical
digests. ``--experiment-id`` re-runs a recorded (simulated) experiment and
compares with the digest stored when it ran; on a mismatch the differences in
code provenance (commit, lockfile, library versions) are listed, since they
are the usual explanation. A manifest can be written with ``--output``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coordinator.provenance import code_provenance, outcome_digest
from experiments.simulation import run_simulation
from shared.config import ExperimentConfig, config_hash, load_experiment_config
from shared.logging import configure_logging


def _differences(recorded: dict | None, current: dict) -> dict:
    if not recorded:
        return {"recorded": "no provenance stored"}
    out = {}
    for key in ("git_commit", "uv_lock_sha256", "python"):
        if recorded.get(key) != current.get(key):
            out[key] = {"recorded": recorded.get(key), "current": current.get(key)}
    for name, version in current.get("libraries", {}).items():
        before = recorded.get("libraries", {}).get(name)
        if before != version:
            out[f"libraries.{name}"] = {"recorded": before, "current": version}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.reproduce")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--experiment-id")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", help="write the reproducibility manifest (JSON)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    provenance = code_provenance()

    if args.config:
        config = load_experiment_config(args.config)
        experiment_id = f"repro-{config_hash(config)}"
        digests = [outcome_digest(run_simulation(experiment_id, config))
                   for _ in range(max(args.runs, 2))]
        manifest = {"source": args.config, "config_hash": config_hash(config),
                    "digests": digests, "reproducible": len(set(digests)) == 1,
                    "code": provenance}
    else:
        from backend.app.db import init_db, make_engine
        from backend.app.services import experiments as service
        from backend.app.settings import get_settings

        sessions = init_db(make_engine(get_settings().database_url))
        with sessions() as session:
            experiment = service.get_experiment(session, args.experiment_id)
            config = ExperimentConfig.model_validate(experiment.config)
            recorded, recorded_code = experiment.outcome_digest, experiment.code_version
        if recorded is None:
            print("this experiment has no recorded digest (not finished, or older than "
                  "Stage 8)", file=sys.stderr)
            return 2
        if config.simulation is None:
            print("only simulated experiments can be re-run here; a distributed run needs "
                  "the sites", file=sys.stderr)
            return 2
        digest = outcome_digest(run_simulation(args.experiment_id, config))
        manifest = {"source": args.experiment_id, "config_hash": config_hash(config),
                    "recorded_digest": recorded, "digest": digest,
                    "reproducible": digest == recorded, "code": provenance,
                    "code_differences": _differences(recorded_code, provenance)}
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if manifest["reproducible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
