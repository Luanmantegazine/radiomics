#!/usr/bin/env python3
"""Paired ROC-AUC comparison of two models' out-of-fold predictions.

Both inputs are ``oof_predictions.csv`` files produced on the same participants
and folds (for example, the morphological model and the demographic baseline).
The script reports DeLong's paired test and a stratified paired bootstrap of the
AUC difference (model A minus model B).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output/compare_models"
KEY_COLUMNS = ["subject_id", "session_id"]
REQUIRED_COLUMNS = [*KEY_COLUMNS, "supervised_label", "fold", "probability_ad"]
LABEL_TO_INT = {"CN": 0, "AD": 1}


def load_predictions(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {', '.join(missing)}.")
    data = data.loc[:, REQUIRED_COLUMNS]
    if data[KEY_COLUMNS].duplicated().any():
        raise ValueError(f"{path}: duplicated subject/session rows.")
    if not np.isfinite(data["probability_ad"]).all():
        raise ValueError(f"{path}: probability_ad must be finite.")
    return data


def paired_predictions(path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray, bool]:
    """Return labels and both probability vectors aligned by participant."""
    a, b = load_predictions(path_a), load_predictions(path_b)
    merged = a.merge(b, on=KEY_COLUMNS, how="outer", suffixes=("_a", "_b"),
                     indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Both prediction files must cover the same participants.")
    if (merged["supervised_label_a"] != merged["supervised_label_b"]).any():
        raise ValueError("Labels differ between the two prediction files.")
    labels = set(merged["supervised_label_a"])
    if labels != set(LABEL_TO_INT):
        raise ValueError(f"Expected CN and AD labels only; found {sorted(labels)}.")
    y = merged["supervised_label_a"].map(LABEL_TO_INT).to_numpy(dtype=int)
    same_folds = bool((merged["fold_a"] == merged["fold_b"]).all())
    return (y, merged["probability_ad_a"].to_numpy(dtype=float),
            merged["probability_ad_b"].to_numpy(dtype=float), same_folds)


def _placements(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives, negatives = score[y == 1], score[y == 0]
    kernel = ((positives[:, None] > negatives[None, :])
              + 0.5 * (positives[:, None] == negatives[None, :]))
    return kernel.mean(axis=1), kernel.mean(axis=0)


def delong_test(y: np.ndarray, score_a: np.ndarray,
                score_b: np.ndarray) -> dict[str, float]:
    """DeLong, DeLong and Clarke-Pearson (1988) test for correlated AUCs."""
    positives_a, negatives_a = _placements(y, score_a)
    positives_b, negatives_b = _placements(y, score_b)
    covariance = (np.cov(np.vstack([positives_a, positives_b])) / positives_a.size
                  + np.cov(np.vstack([negatives_a, negatives_b])) / negatives_a.size)
    auc_a, auc_b = positives_a.mean(), positives_b.mean()
    difference = auc_a - auc_b
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    standard_error = float(np.sqrt(max(variance, 0.0)))
    if standard_error == 0:
        z, p_value = 0.0, 1.0
    else:
        z = difference / standard_error
        p_value = float(2 * norm.sf(abs(z)))
    return {
        "auc_a": float(auc_a), "auc_b": float(auc_b),
        "auc_difference": float(difference),
        "standard_error": standard_error,
        "ci95_low": float(difference - 1.959964 * standard_error),
        "ci95_high": float(difference + 1.959964 * standard_error),
        "z": float(z), "p_value": p_value,
        "auc_a_standard_error": float(np.sqrt(covariance[0, 0])),
        "auc_b_standard_error": float(np.sqrt(covariance[1, 1])),
    }


def paired_bootstrap(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                     *, resamples: int, random_state: int) -> dict[str, float]:
    """Stratified participant bootstrap of the paired AUC difference."""
    if resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100.")
    rng = np.random.default_rng(random_state)
    by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    differences = np.empty(resamples)
    for index in range(resamples):
        sample = np.concatenate([rng.choice(rows, rows.size, replace=True)
                                 for rows in by_class])
        differences[index] = (roc_auc_score(y[sample], score_a[sample])
                              - roc_auc_score(y[sample], score_b[sample]))
    low, high = np.quantile(differences, [0.025, 0.975])
    return {"ci95_low": float(low), "ci95_high": float(high),
            "fraction_difference_le_zero": float(np.mean(differences <= 0)),
            "resamples": resamples, "random_state": random_state}


def compare(path_a: Path, path_b: Path, output_dir: Path, *, name_a: str = "A",
            name_b: str = "B", resamples: int = 5000,
            random_state: int = 42) -> dict[str, object]:
    y, score_a, score_b, same_folds = paired_predictions(path_a, path_b)
    result = {
        "model_a": {"name": name_a, "predictions": str(path_a.resolve())},
        "model_b": {"name": name_b, "predictions": str(path_b.resolve())},
        "participants": int(y.size), "cn_participants": int((y == 0).sum()),
        "ad_participants": int((y == 1).sum()), "same_folds": same_folds,
        "prediction_correlation": float(np.corrcoef(score_a, score_b)[0, 1]),
        "delong": delong_test(y, score_a, score_b),
        "paired_bootstrap": paired_bootstrap(
            y, score_a, score_b, resamples=resamples, random_state=random_state),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name_a}_vs_{name_b}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="oof_predictions.csv of model A.")
    parser.add_argument("--b", type=Path, required=True, help="oof_predictions.csv of model B.")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    try:
        result = compare(args.a, args.b, args.output_dir, name_a=args.name_a,
                         name_b=args.name_b, resamples=args.bootstrap_resamples,
                         random_state=args.random_state)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    delong = result["delong"]
    if not result["same_folds"]:
        print("Warning: the two models used different fold assignments.")
    print(f"AUC {args.name_a}={delong['auc_a']:.3f}, {args.name_b}={delong['auc_b']:.3f}")
    print(f"Difference {delong['auc_difference']:+.3f} "
          f"(DeLong 95% CI {delong['ci95_low']:.3f} to {delong['ci95_high']:.3f}; "
          f"p={delong['p_value']:.2g})")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
