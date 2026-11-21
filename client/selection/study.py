"""Local selection study: run the configured selectors and compare them.

Outputs (local files):

``selection.json``
    Per selector: ranked selection, exclusion counts, bootstrap stability;
    pairwise Jaccard overlap between selectors; cohort counts.
``selected_features.csv``
    One row per (selector, rank, feature) with its score and bootstrap
    selection frequency.

This compares *which features* each method picks and how stable the picks are.
It says nothing about predictive performance, which is evaluated in Stage 6.

Bootstrap limitations: resampling participants with replacement creates tied
duplicates and understates the instability of small cohorts; resamples with
too few events are skipped and counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from client.data import LocalDataset
from client.selection.config import SelectionStudyConfig
from client.selection.selectors import (
    SelectionError,
    SelectionInputs,
    SelectionResult,
    run_selector,
)
from evaluation.stability import jaccard, nogueira_stability, selection_frequency
from shared.logging import get_logger
from shared.seeding import derive_seed

SELECTION_FILE = "selection.json"
SELECTED_FILE = "selected_features.csv"


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    n_resamples: int
    n_failed: int
    stability: float
    frequency: dict[str, float]


@dataclass(frozen=True, slots=True)
class StudyResult:
    selections: dict[str, SelectionResult]
    overlap: dict[str, float]
    bootstrap: dict[str, BootstrapSummary]
    n_participants: int
    n_events: int


def _inputs(dataset: LocalDataset, reliability, rows=None) -> SelectionInputs:
    rows = slice(None) if rows is None else rows
    return SelectionInputs(
        feature_names=dataset.feature_names,
        features=dataset.features[rows],
        time=dataset.time_months[rows],
        event=dataset.event[rows],
        reliability=reliability,
    )


def run_selection_study(
    dataset: LocalDataset,
    config: SelectionStudyConfig,
    reliability: dict[str, dict[str, float]] | None = None,
) -> StudyResult:
    """Run every selector on the dataset and, if configured, on bootstrap resamples."""
    log = get_logger("client.selection")
    inputs = _inputs(dataset, reliability)
    selections = {
        spec.name: run_selector(spec, inputs, min_events=config.min_events)
        for spec in config.selectors
    }
    overlap = {
        f"{a}|{b}": jaccard(selections[a].selected, selections[b].selected)
        for a, b in combinations(selections, 2)
    }
    bootstrap: dict[str, BootstrapSummary] = {}
    if config.bootstrap is not None:
        n = dataset.n_participants
        index = {name: i for i, name in enumerate(dataset.feature_names)}
        for spec in config.selectors:
            rows, failed = [], 0
            for b in range(config.bootstrap.n_resamples):
                rng = np.random.default_rng(derive_seed(config.bootstrap.seed, "bootstrap", b))
                sample = rng.integers(0, n, size=n)
                try:
                    picked = run_selector(
                        spec, _inputs(dataset, reliability, sample), min_events=config.min_events
                    ).selected
                except SelectionError:
                    failed += 1
                    continue
                row = np.zeros(len(index), dtype=bool)
                row[[index[name] for name in picked]] = True
                rows.append(row)
            matrix = np.array(rows, dtype=bool).reshape(len(rows), len(index))
            frequency = selection_frequency(matrix) if len(rows) else np.zeros(len(index))
            bootstrap[spec.name] = BootstrapSummary(
                n_resamples=config.bootstrap.n_resamples,
                n_failed=failed,
                stability=nogueira_stability(matrix) if len(rows) >= 2 else float("nan"),
                frequency={name: float(frequency[i]) for name, i in index.items()},
            )
    log.info("selection_study_finished", study_id=config.study_id, selectors=len(selections))
    return StudyResult(
        selections=selections,
        overlap=overlap,
        bootstrap=bootstrap,
        n_participants=dataset.n_participants,
        n_events=int(dataset.event.sum()),
    )


def write_study(result: StudyResult, config: SelectionStudyConfig, output_dir: str | Path) -> Path:
    """Write the local report files and return the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, selectors = [], {}
    for name, selection in result.selections.items():
        boot = result.bootstrap.get(name)
        reasons: dict[str, int] = {}
        for reason in selection.excluded.values():
            reasons[reason] = reasons.get(reason, 0) + 1
        selectors[name] = {
            "method": selection.method,
            "selected": list(selection.selected),
            "excluded_counts": dict(sorted(reasons.items())),
            "bootstrap": None
            if boot is None
            else {
                "n_resamples": boot.n_resamples,
                "n_failed": boot.n_failed,
                "nogueira_stability": None if np.isnan(boot.stability) else boot.stability,
            },
        }
        for rank, feature in enumerate(selection.selected, start=1):
            rows.append(
                {
                    "selector": name,
                    "rank": rank,
                    "feature": feature,
                    "score": selection.scores[feature],
                    "bootstrap_frequency": boot.frequency[feature] if boot else None,
                }
            )
    report = {
        "study_id": config.study_id,
        "config": config.model_dump(mode="json"),
        "n_participants": result.n_participants,
        "n_events": result.n_events,
        "n_censored": result.n_participants - result.n_events,
        "selectors": selectors,
        "jaccard_overlap": result.overlap,
    }
    (output_dir / SELECTION_FILE).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        rows, columns=["selector", "rank", "feature", "score", "bootstrap_frequency"]
    ).to_csv(output_dir / SELECTED_FILE, index=False, lineterminator="\n")
    return output_dir
