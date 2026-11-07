"""Checks for the paired ROC-AUC comparison of out-of-fold predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from oasis_analysis.compare_models import compare, delong_test


def predictions(probability: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [f"OAS{index:05d}" for index in range(labels.size)],
        "session_id": [f"OAS{index:05d}_MR_d0000" for index in range(labels.size)],
        "supervised_label": np.where(labels == 1, "AD", "CN"),
        "fold": folds,
        "probability_ad": probability,
    })


@pytest.fixture
def scores() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1], 60)
    strong = labels * 1.5 + rng.normal(size=labels.size)
    weak = labels * 0.4 + rng.normal(size=labels.size)
    return labels, strong, weak


def test_delong_matches_auc_and_detects_a_clear_difference(scores) -> None:
    labels, strong, weak = scores
    result = delong_test(labels, strong, weak)

    assert result["auc_a"] == pytest.approx(roc_auc_score(labels, strong))
    assert result["auc_b"] == pytest.approx(roc_auc_score(labels, weak))
    assert result["auc_difference"] > 0
    assert result["ci95_low"] > 0
    assert result["p_value"] < 0.05


def test_identical_models_have_no_difference(scores) -> None:
    labels, strong, _ = scores
    result = delong_test(labels, strong, strong)
    assert result["auc_difference"] == 0
    assert result["p_value"] == 1.0


def test_compare_aligns_rows_and_writes_report(tmp_path: Path, scores) -> None:
    labels, strong, weak = scores
    folds = np.arange(labels.size) % 5 + 1
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    predictions(strong, labels, folds).to_csv(first, index=False)
    predictions(weak, labels, folds).iloc[::-1].to_csv(second, index=False)

    result = compare(first, second, tmp_path / "out", name_a="morph", name_b="demo",
                     resamples=200)

    saved = json.loads((tmp_path / "out" / "morph_vs_demo.json").read_text())
    assert saved["delong"] == pytest.approx(result["delong"])
    assert result["same_folds"] is True
    assert result["participants"] == labels.size
    assert result["delong"]["auc_a"] == pytest.approx(roc_auc_score(labels, strong))
    assert result["paired_bootstrap"]["ci95_low"] > 0


def test_compare_rejects_different_participants(tmp_path: Path, scores) -> None:
    labels, strong, weak = scores
    folds = np.ones(labels.size, dtype=int)
    predictions(strong, labels, folds).to_csv(tmp_path / "a.csv", index=False)
    predictions(weak, labels, folds).iloc[1:].to_csv(tmp_path / "b.csv", index=False)

    with pytest.raises(ValueError, match="same participants"):
        compare(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "out", resamples=200)
