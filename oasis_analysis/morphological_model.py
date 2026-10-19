#!/usr/bin/env python3
"""Nested, subject-grouped CN/AD evaluation of morphometric measurements.

Read the filtered table with original volumes and eTIV. Prepared out-of-fold
feature CSVs are for audit only: residualization must be refit inside every
inner and outer training split during model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    # Keep serialized pipelines importable when this file is launched directly.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.modules["oasis_analysis.morphological_model"] = sys.modules[__name__]

os.environ.setdefault("MPLCONFIGDIR", "/tmp/oasis-morphological-model-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from oasis_analysis.prepare_morphological_features import (
    DEFAULT_ASYMMETRY_PAIRS,
    DEFAULT_SOURCE_COLUMNS,
    EXTENDED_ASYMMETRY_PAIRS,
    EXTENDED_SOURCE_COLUMNS,
    adjusted_name,
    make_validation_folds,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output/morphological_model"
CLASSIFIERS = ("elasticnet", "svm", "rf", "xgboost")
LABEL_TO_INT = {"CN": 0, "AD": 1}


class ETIVAdjustedFeatures(TransformerMixin, BaseEstimator):
    """Adjust volumes using CN rows passed to fit; copy thickness unchanged."""

    def __init__(self, *, extended_data: bool = False, asymmetry: bool = False,
                 proportional: bool = False):
        self.extended_data = extended_data
        self.asymmetry = asymmetry
        self.proportional = proportional

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> ETIVAdjustedFeatures:
        mapping = EXTENDED_SOURCE_COLUMNS if self.extended_data else DEFAULT_SOURCE_COLUMNS
        pairs = (
            (EXTENDED_ASYMMETRY_PAIRS if self.extended_data else DEFAULT_ASYMMETRY_PAIRS)
            if self.asymmetry else {}
        )
        self.mapping_ = mapping
        self.volume_targets_ = [
            target for target in mapping if target != "etiv_mm3" and target.endswith("_mm3")
        ]
        self.thickness_targets_ = [
            target for target in mapping if target.endswith("_thickness_mm")
        ]
        self.asymmetry_pairs_ = pairs
        self.feature_names_ = [
            *(adjusted_name(target, self.proportional) for target in self.volume_targets_),
            *self.thickness_targets_, *pairs,
        ]
        self.cn_fits_ = {}
        if not self.proportional:
            if y is None:
                raise ValueError("Residual adjustment requires training labels.")
            labels = np.asarray(y)
            controls = labels == LABEL_TO_INT["CN"]
            if controls.sum() < 2:
                raise ValueError("Residual adjustment needs at least two CN training rows.")
            etiv = X.loc[:, mapping["etiv_mm3"]].to_numpy(dtype=float)[controls]
            centered = etiv - etiv.mean()
            denominator = float(np.dot(centered, centered))
            if denominator <= 0:
                raise ValueError("CN training eTIV has zero variance.")
            for target in self.volume_targets_:
                volumes = X.loc[:, mapping[target]].to_numpy(dtype=float)[controls]
                mean_volume = float(volumes.mean())
                slope = float(np.dot(centered, volumes - mean_volume) / denominator)
                self.cn_fits_[target] = {
                    "intercept": mean_volume - slope * float(etiv.mean()),
                    "slope": slope,
                    "cn_training_rows": int(controls.sum()),
                }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        etiv = X.loc[:, self.mapping_["etiv_mm3"]].to_numpy(dtype=float)
        features = []
        for target in self.volume_targets_:
            volume = X.loc[:, self.mapping_[target]].to_numpy(dtype=float)
            if self.proportional:
                adjusted = volume / etiv
            else:
                fit = self.cn_fits_[target]
                adjusted = volume - (fit["intercept"] + fit["slope"] * etiv)
            features.append(adjusted)
        for target in self.thickness_targets_:
            features.append(X.loc[:, self.mapping_[target]].to_numpy(dtype=float))
        for left_target, right_target in self.asymmetry_pairs_.values():
            left = X.loc[:, self.mapping_[left_target]].to_numpy(dtype=float)
            right = X.loc[:, self.mapping_[right_target]].to_numpy(dtype=float)
            features.append((left - right) / (left + right))
        return np.column_stack(features)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


ETIVAdjustedFeatures.__module__ = "oasis_analysis.morphological_model"


def load_cohort(path: Path, *, extended_data: bool, outer_folds: int) -> pd.DataFrame:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), None)
    if header is None:
        raise ValueError("Input CSV has no header.")
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate input columns: {', '.join(duplicates)}")
    mapping = EXTENDED_SOURCE_COLUMNS if extended_data else DEFAULT_SOURCE_COLUMNS
    needed = ["subject_id", "session_id", "supervised_label", *mapping.values()]
    missing = sorted(set(needed) - set(header))
    if missing:
        raise ValueError(f"Required input columns are absent: {', '.join(missing)}")
    data = pd.read_csv(path, usecols=needed, low_memory=False)
    if data.empty or data[needed].isna().any().any():
        raise ValueError("Input cohort is empty or has missing required values.")
    if data["subject_id"].duplicated().any() or data["session_id"].duplicated().any():
        raise ValueError("Expected exactly one unique session per subject.")
    if set(data["supervised_label"]) != set(LABEL_TO_INT):
        raise ValueError("Input must contain CN and AD labels only.")
    for source in mapping.values():
        numeric = pd.to_numeric(data[source], errors="raise")
        if not np.isfinite(numeric).all() or (numeric <= 0).any():
            raise ValueError(f"{source} must contain finite positive measurements.")
        data[source] = numeric.astype(float)
    counts = data["supervised_label"].value_counts()
    if outer_folds < 2 or int(counts.min()) < outer_folds:
        raise ValueError("Each class needs at least one participant per outer fold.")
    return data.reset_index(drop=True)


def classifier_and_grid(name: str, random_state: int):
    if name == "elasticnet":
        estimator = LogisticRegression(
            l1_ratio=0.5, solver="saga", class_weight="balanced",
            max_iter=5000, tol=1e-3, random_state=random_state,
        )
        grid = {"classifier__C": [0.1, 1.0, 10.0],
                "classifier__l1_ratio": [0.1, 0.5, 0.9]}
    elif name == "svm":
        estimator = SVC(probability=True, class_weight="balanced", random_state=random_state)
        grid = {"classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"]}
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=1, random_state=random_state
        )
        grid = {"classifier__max_depth": [None, 4],
                "classifier__min_samples_leaf": [1, 3]}
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError(
                "XGBoost is optional; install xgboost from oasis_analysis/requirements.txt "
                "before using --classifier xgboost."
            ) from error
        estimator = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            n_estimators=150, n_jobs=1, random_state=random_state,
        )
        grid = {"classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8, 1.0]}
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return estimator, grid


def checked_splits(splitter: StratifiedGroupKFold, X: pd.DataFrame,
                   y: np.ndarray, groups: np.ndarray, *, name: str):
    splits = list(splitter.split(X, y, groups))
    for fold, (train, validation) in enumerate(splits, start=1):
        if set(groups[train]) & set(groups[validation]):
            raise RuntimeError(f"Subject leakage in {name} fold {fold}.")
        if len(set(y[train])) != 2 or len(set(y[validation])) != 2:
            raise ValueError(f"{name} fold {fold} lacks CN or AD in train/validation.")
    return splits


def classification_metrics(y: np.ndarray, probability: np.ndarray,
                           threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
        "threshold": threshold,
    }


def bootstrap_auc_ci(y: np.ndarray, probability: np.ndarray,
                     *, resamples: int, random_state: int) -> tuple[float, float]:
    if resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100.")
    rng = np.random.default_rng(random_state)
    by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    scores = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = np.concatenate([rng.choice(indices, len(indices), replace=True)
                                 for indices in by_class])
        scores[index] = roc_auc_score(y[sample], probability[sample])
    low, high = np.quantile(scores, [0.025, 0.975])
    return float(low), float(high)


def calibration_table(y: np.ndarray, probability: np.ndarray,
                      *, bins: int = 10) -> tuple[pd.DataFrame, float]:
    rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probability >= lower) & (probability < upper if index < bins - 1
                                          else probability <= upper)
        if not mask.any():
            continue
        rows.append({"bin": index + 1, "lower": lower, "upper": upper,
                     "participants": int(mask.sum()),
                     "mean_probability_ad": float(probability[mask].mean()),
                     "observed_ad_fraction": float(y[mask].mean())})
    table = pd.DataFrame(rows)
    ece = float(sum(row["participants"] * abs(row["mean_probability_ad"] -
                                             row["observed_ad_fraction"])
                    for row in rows) / len(y))
    return table, ece


def save_calibration_plot(table: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Perfect calibration")
    axis.plot(table["mean_probability_ad"], table["observed_ad_fraction"],
              marker="o", label="Out-of-fold")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted AD probability",
             ylabel="Observed AD fraction", title="Morphometric model calibration")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_model(input_path: Path, output_dir: Path, *, classifier: str = "elasticnet",
              extended_data: bool = False, asymmetry: bool = False,
              proportional: bool = False, outer_folds: int = 5,
              inner_folds: int = 3, random_state: int = 42,
              threshold: float = 0.5, bootstrap_resamples: int = 2000,
              fold_assignments: Path | None = None) -> dict[str, float | int]:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if inner_folds < 2:
        raise ValueError("inner-folds must be at least 2.")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100.")
    data = load_cohort(input_path, extended_data=extended_data, outer_folds=outer_folds)
    mapping = EXTENDED_SOURCE_COLUMNS if extended_data else DEFAULT_SOURCE_COLUMNS
    X = data.loc[:, list(mapping.values())]
    y = data["supervised_label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = data["subject_id"].to_numpy()
    estimator, grid = classifier_and_grid(classifier, random_state)
    pipeline = Pipeline([
        ("morphology", ETIVAdjustedFeatures(
            extended_data=extended_data, asymmetry=asymmetry, proportional=proportional)),
        ("zscore", StandardScaler()),
        ("classifier", estimator),
    ])
    if fold_assignments is None:
        outer_splitter = StratifiedGroupKFold(
            n_splits=outer_folds, shuffle=True, random_state=random_state)
        outer_splits = checked_splits(outer_splitter, X, y, groups, name="outer")
    else:
        assignment_rows = data.loc[:, ["subject_id", "session_id", "supervised_label"]].to_dict("records")
        validation_folds = make_validation_folds(
            assignment_rows, outer_folds, random_state, fold_assignments)
        outer_splits = []
        for validation in validation_folds:
            validation_array = np.asarray(validation, dtype=int)
            held_out = set(validation)
            train_array = np.asarray(
                [index for index in range(len(data)) if index not in held_out], dtype=int)
            outer_splits.append((train_array, validation_array))

    output_dir.mkdir(parents=True, exist_ok=True)
    probability = np.full(len(data), np.nan)
    fold_number = np.full(len(data), -1, dtype=int)
    fold_metrics = []
    fold_details = []
    for fold, (train, validation) in enumerate(outer_splits, start=1):
        if len(set(y[train])) != 2 or len(set(y[validation])) != 2:
            raise ValueError(f"Outer fold {fold} lacks CN or AD.")
        if min(Counter(y[train]).values()) < inner_folds:
            raise ValueError(f"Outer fold {fold} has too few examples for {inner_folds} inner folds.")
        inner_splitter = StratifiedGroupKFold(
            n_splits=inner_folds, shuffle=True, random_state=random_state + fold)
        inner_splits = checked_splits(
            inner_splitter, X.iloc[train], y[train], groups[train], name=f"inner/outer-{fold}")
        search = GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=inner_splits,
            n_jobs=1, refit=True, error_score="raise", return_train_score=False,
        )
        search.fit(X.iloc[train], y[train])
        predicted = search.predict_proba(X.iloc[validation])[:, 1]
        probability[validation] = predicted
        fold_number[validation] = fold
        fitted = search.best_estimator_
        dump(fitted, output_dir / f"fold_{fold:02d}_model.joblib")
        pd.DataFrame(search.cv_results_).loc[:,
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].to_csv(output_dir / f"fold_{fold:02d}_inner_search.csv", index=False)
        fold_table = pd.DataFrame({
            "subject_id": groups[validation],
            "session_id": data.iloc[validation]["session_id"].to_numpy(),
            "supervised_label": data.iloc[validation]["supervised_label"].to_numpy(),
            "fold": fold, "probability_ad": predicted,
        })
        fold_table.to_csv(output_dir / f"fold_{fold:02d}_validation.csv", index=False)
        fold_metric = classification_metrics(y[validation], predicted, threshold)
        fold_metrics.append({"fold": fold, "train_cn": int((y[train] == 0).sum()),
                             "train_ad": int((y[train] == 1).sum()),
                             "validation_cn": int((y[validation] == 0).sum()),
                             "validation_ad": int((y[validation] == 1).sum()),
                             **fold_metric})
        transformer = fitted.named_steps["morphology"]
        scaler = fitted.named_steps["zscore"]
        fold_details.append({
            "fold": fold,
            "inner_best_auc": float(search.best_score_),
            "best_params": search.best_params_,
            "cn_etiv_fits": transformer.cn_fits_,
            "feature_names": transformer.feature_names_,
            "zscore_mean": scaler.mean_.tolist(),
            "zscore_scale": scaler.scale_.tolist(),
            "train_subject_ids": groups[train].tolist(),
            "validation_subject_ids": groups[validation].tolist(),
        })
        print(f"Fold {fold}/{outer_folds}: inner AUC={search.best_score_:.3f}, "
              f"outer AUC={fold_metric['roc_auc']:.3f}", flush=True)

    if not np.isfinite(probability).all() or (fold_number < 1).any():
        raise RuntimeError("Every participant must receive one outer validation prediction.")
    metrics = classification_metrics(y, probability, threshold)
    low, high = bootstrap_auc_ci(
        y, probability, resamples=bootstrap_resamples, random_state=random_state)
    metrics.update({"roc_auc_ci95_low": low, "roc_auc_ci95_high": high,
                    "participants": len(data), "cn_participants": int((y == 0).sum()),
                    "ad_participants": int((y == 1).sum())})
    calibration, ece = calibration_table(y, probability)
    metrics["expected_calibration_error"] = ece
    oof = data.loc[:, ["subject_id", "session_id", "supervised_label"]].copy()
    oof["fold"] = fold_number
    oof["probability_ad"] = probability
    oof["predicted_label"] = np.where(probability >= threshold, "AD", "CN")
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
    calibration.to_csv(output_dir / "calibration.csv", index=False)
    save_calibration_plot(calibration, output_dir / "calibration.png")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    software = {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__}
    if classifier == "xgboost":
        import xgboost
        software["xgboost"] = xgboost.__version__
    configuration = {
        "input": str(input_path.resolve()), "classifier": classifier,
        "extended_data": extended_data, "asymmetry": asymmetry,
        "volume_adjustment": "proportional" if proportional else "residual_cn_training_only",
        "outer_cv": "StratifiedGroupKFold" if fold_assignments is None else "provided_assignments",
        "fold_assignments": str(fold_assignments.resolve()) if fold_assignments else None,
        "inner_cv": "StratifiedGroupKFold", "group": "subject_id",
        "outer_folds": outer_folds, "inner_folds": inner_folds,
        "random_state": random_state, "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_method": "Stratified subject bootstrap of pooled outer-fold predictions; percentile 95% CI",
        "threshold": threshold, "zscore": "Fit within each inner and outer training split",
        "feature_source_columns": mapping,
        "fold_details": fold_details,
        "software": software,
    }
    (output_dir / "run_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Filtered CN/AD CSV with original ROI measurements and eTIV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--classifier", choices=CLASSIFIERS, default="elasticnet")
    parser.add_argument("--extended-data", action="store_true")
    parser.add_argument("--asymmetry", action="store_true")
    parser.add_argument("--proportional", action="store_true")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--fold-assignments", type=Path)
    args = parser.parse_args()
    try:
        metrics = run_model(
            args.input, args.output_dir, classifier=args.classifier,
            extended_data=args.extended_data, asymmetry=args.asymmetry,
            proportional=args.proportional, outer_folds=args.outer_folds,
            inner_folds=args.inner_folds, random_state=args.random_state,
            threshold=args.threshold, bootstrap_resamples=args.bootstrap_resamples,
            fold_assignments=args.fold_assignments,
        )
    except (OSError, ValueError, RuntimeError, csv.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Nested OOF AUC={metrics['roc_auc']:.3f} "
          f"(95% bootstrap CI {metrics['roc_auc_ci95_low']:.3f}-"
          f"{metrics['roc_auc_ci95_high']:.3f})")
    print(f"Results: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
