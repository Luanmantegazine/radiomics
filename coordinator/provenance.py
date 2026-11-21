"""Reproducibility: a digest of released results and the provenance of the code.

``outcome_digest`` hashes everything an experiment *released* (round
aggregates, harmonization, selection, models, evaluation, which sites took
part) in canonical JSON, and nothing that measures the run itself (times,
bytes, memory). Two runs of the same configuration with the same code and
library versions must produce the same digest, whether simulated or
distributed.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from coordinator.outcomes import ExperimentOutcome

REPO = Path(__file__).resolve().parents[1]
LIBRARIES = ("fedradiomics", "numpy", "scipy", "scikit-learn", "scikit-survival", "pandas",
             "SimpleITK", "pyradiomics", "pydantic")


def released_results(outcome: ExperimentOutcome) -> dict:
    """The reproducible part of an outcome (no timing or traffic measurements)."""
    return {
        "completed": outcome.completed,
        "registered_clients": list(outcome.registered_clients),
        "rounds": [
            {"round_number": r.round_number, "phase": r.phase.value, "task": r.task.value,
             "aggregator": r.aggregator, "completed": r.completed,
             "failure_reason": r.failure_reason,
             "aggregate": r.aggregate.model_dump(mode="json") if r.aggregate else None,
             "clients": [{"client_id": c.client_id, "status": c.status.value,
                          "reason_code": c.reason_code.value if c.reason_code else None,
                          "n_samples": c.n_samples} for c in r.clients]}
            for r in outcome.rounds
        ],
        **{name: (getattr(outcome, name).model_dump(mode="json")
                  if getattr(outcome, name) is not None else None)
           for name in ("harmonization", "feature_selection", "training", "evaluation")},
    }


def outcome_digest(outcome: ExperimentOutcome) -> str:
    canonical = json.dumps(released_results(outcome), sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                check=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def code_provenance() -> dict:
    """Commit, lockfile hash and library versions of the running code."""
    lock = REPO / "uv.lock"
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    # pyradiomics built from git reports a placeholder version; the exact commit
    # is pinned in uv.lock, whose hash is recorded below.
    commit = _git("rev-parse", "HEAD") or os.environ.get("FEDRAD_GIT_COMMIT")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
        "python": platform.python_version(),
        "libraries": versions,
    }
