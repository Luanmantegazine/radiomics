"""Local training at one site: fit configured models, write models and predictions.

Outputs (local files; predictions are individual-level and stay at the site):

``models/<name>.json``  released format (Cox, horizon logistic)
``predictions.csv``     participant_id, partition, model, p_<horizon> per horizon
``training.json``       configuration, feature list, partition sizes, status
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from client.data import LocalDataset
from client.models.cox import fit_cox
from client.models.forest import RandomSurvivalForestModel
from client.models.logistic import fit_horizon_logistic
from client.models.prediction import predict_conversion
from client.splits import partition
from shared.newton import NewtonSettings
from shared.protocol import PartitionSpec
from shared.training import (
    CoxSpec,
    DataSplitSpec,
    HorizonLogisticSpec,
    ModelSpec,
    RandomSurvivalForestSpec,
)


class LocalTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    horizons_months: tuple[float, ...] = Field(min_length=1)
    data_split: DataSplitSpec | None
    """Null trains on every participant (then nothing is held out for evaluation)."""
    min_events: int = Field(ge=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)


def load_local_config(path: str | Path) -> LocalTrainingConfig:
    with Path(path).open(encoding="utf-8") as handle:
        return LocalTrainingConfig.model_validate(yaml.safe_load(handle))


def _settings(spec) -> NewtonSettings:
    o = spec.optimizer
    return NewtonSettings(o.ridge, o.max_iterations, o.tolerance)


def _columns(dataset: LocalDataset, names: tuple[str, ...]) -> np.ndarray:
    index = {name: i for i, name in enumerate(dataset.feature_names)}
    missing = [n for n in names if n not in index]
    if missing:
        raise ValueError(f"features not in the table: {missing}")
    return dataset.features[:, [index[n] for n in names]]


def train_locally(
    dataset: LocalDataset,
    feature_names: tuple[str, ...],
    config: LocalTrainingConfig,
    output_dir: str | Path,
) -> dict:
    output_dir = Path(output_dir)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    split = config.data_split
    spec = None if split is None else PartitionSpec(
        name="train", test_fraction=split.test_fraction, seed=split.seed
    )
    train = partition(dataset, spec)
    horizons = config.horizons_months
    x_train = _columns(train, feature_names)
    status: dict[str, str] = {}
    predictions = []
    for model_spec in config.models:
        try:
            if isinstance(model_spec, CoxSpec):
                model = fit_cox(feature_names, x_train, train.time_months, train.event,
                                _settings(model_spec), horizons, min_events=config.min_events)
            elif isinstance(model_spec, HorizonLogisticSpec):
                model = fit_horizon_logistic(
                    feature_names, x_train, train.time_months, train.event, horizons,
                    model_spec.censoring, _settings(model_spec), min_events=config.min_events,
                )
            else:
                assert isinstance(model_spec, RandomSurvivalForestSpec)
                model = RandomSurvivalForestModel(model_spec, horizons).fit(
                    x_train, train.time_months, train.event
                )
        except ValueError as exc:
            status[model_spec.name] = f"failed: {exc}"
            continue
        status[model_spec.name] = "trained"
        if isinstance(model, RandomSurvivalForestModel):
            probabilities = model.predict_conversion(_columns(dataset, feature_names))
        else:
            (output_dir / "models" / f"{model_spec.name}.json").write_text(
                model.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            _, probabilities = predict_conversion(model, dataset.feature_names, dataset.features)
        frame = pd.DataFrame(probabilities, columns=[f"p_{h:g}" for h in horizons])
        frame.insert(0, "model", model_spec.name)
        frame.insert(0, "partition", _partition_labels(dataset, split))
        frame.insert(0, "participant_id", dataset.participant_ids)
        predictions.append(frame)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output_dir / "predictions.csv", index=False, lineterminator="\n"
        )
    report = {
        "config": config.model_dump(mode="json"),
        "feature_names": list(feature_names),
        "n_train": train.n_participants,
        "n_events_train": int(train.event.sum()),
        "n_total": dataset.n_participants,
        "models": status,
    }
    (output_dir / "training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _partition_labels(dataset: LocalDataset, split: DataSplitSpec | None) -> list[str]:
    if split is None:
        return ["train"] * dataset.n_participants
    from client.splits import test_mask

    mask = test_mask(dataset, split.test_fraction, split.seed)
    return ["test" if m else "train" for m in mask]
