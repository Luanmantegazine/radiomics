#!/usr/bin/env python3
"""Evaluate age-and-sex classifiers with leakage-safe OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import joblib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/demographic-baseline-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ANALYSIS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ANALYSIS_DIR
    / "output/filter_supervised_labels/radiomics_sessions_cn_ad_simplified.csv"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_DIR / "output/demographic_baseline"
LOG_DIR = ANALYSIS_DIR / "logs"
PREDICTORS = ["age_at_mri", "sex"]
REQUIRED_COLUMNS = ["subject_id", "session_id", *PREDICTORS, "supervised_label"]
LABEL_TO_INT = {"CN": 0, "AD": 1}
SEX_CATEGORIES = ["F", "M"]
CLASSIFIERS = ("logistic", "elasticnet", "svm", "rf", "xgboost")
# Shared with morphological_model.py so both pipelines tune the same grid.
ELASTICNET_C_GRID = (0.003, 0.01, 0.03, 0.1, 1.0)
LOGGER = logging.getLogger(__name__)


def default_log_path(input_path: Path) -> Path:
    """Return the default path for the baseline run log."""
    return LOG_DIR / f"{input_path.stem}_demographic_baseline.log"


def configure_logging(log_path: Path) -> None:
    """Write informational messages to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )


def relative_path(path: Path) -> Path:
    """Return path relative to the current working directory for display."""
    return Path(os.path.relpath(path.resolve(), start=Path.cwd()))


def report(message: str) -> None:
    """Show a message on screen and write it to the log file."""
    print(message)
    LOGGER.info(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the reproducible age-and-sex CN/AD baseline."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="Grouped inner folds used to tune non-default classifiers (default: 3).",
    )
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        default="logistic",
        help="Classifier to evaluate (default: logistic).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability cutoff for classifying AD (default: 0.5).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_cohort(path: Path, n_splits: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = data.loc[:, REQUIRED_COLUMNS].copy()
    if data.empty:
        raise ValueError("The input cohort is empty.")
    if data[REQUIRED_COLUMNS].isna().any().any():
        counts = data[REQUIRED_COLUMNS].isna().sum()
        details = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
        raise ValueError(f"Missing values are not allowed: {details}")
    if data["subject_id"].duplicated().any():
        duplicate_count = int(data["subject_id"].duplicated(keep=False).sum())
        raise ValueError(
            "Expected one session per participant, but found "
            f"{duplicate_count} rows with duplicated subject_id values."
        )
    if data["session_id"].duplicated().any():
        raise ValueError("session_id values must be unique.")

    labels = set(data["supervised_label"].astype(str))
    if labels != set(LABEL_TO_INT):
        raise ValueError(f"Expected labels CN and AD only; found {sorted(labels)}.")
    sexes = set(data["sex"].astype(str))
    if not sexes.issubset(SEX_CATEGORIES):
        raise ValueError(f"Expected sex values F or M; found {sorted(sexes)}.")

    data["age_at_mri"] = pd.to_numeric(data["age_at_mri"], errors="raise")
    if not np.isfinite(data["age_at_mri"]).all():
        raise ValueError("age_at_mri must contain finite numeric values.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    smallest_class = int(data["supervised_label"].value_counts().min())
    if n_splits > smallest_class:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest class size ({smallest_class})."
        )
    return data


def classifier_and_grid(
    name: str, random_state: int
) -> tuple[object, dict[str, list[object]]]:
    """Return a classifier and the same tuning grid as the morphology pipeline."""
    if name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        )
        grid: dict[str, list[object]] = {}
    elif name == "elasticnet":
        estimator = LogisticRegression(
            l1_ratio=0.5,
            solver="saga",
            class_weight="balanced",
            max_iter=5000,
            tol=1e-3,
            random_state=random_state,
        )
        grid = {
            "classifier__C": list(ELASTICNET_C_GRID),
            "classifier__l1_ratio": [0.5],
        }
    elif name == "svm":
        estimator = SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
        grid = {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"],
        }
    elif name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=1,
            random_state=random_state,
        )
        grid = {
            "classifier__max_depth": [None, 4],
            "classifier__min_samples_leaf": [1, 3],
        }
    elif name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ValueError(
                "XGBoost is optional; install xgboost from "
                "oasis_analysis/requirements.txt before using "
                "--classifier xgboost."
            ) from error
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=150,
            n_jobs=1,
            random_state=random_state,
        )
        grid = {
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.03, 0.1],
            "classifier__subsample": [0.8, 1.0],
        }
    else:
        raise ValueError(f"Unknown classifier: {name}")
    return estimator, grid


def build_pipeline(random_state: int, classifier: str = "logistic") -> Pipeline:
    """Create preprocessing and classifier as one leakage-safe estimator."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("age", StandardScaler(), ["age_at_mri"]),
            (
                "sex",
                OneHotEncoder(
                    categories=[SEX_CATEGORIES],
                    drop="first",
                    handle_unknown="error",
                ),
                ["sex"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator, _ = classifier_and_grid(classifier, random_state)
    return Pipeline(
        steps=[("preprocessing", preprocessing), ("classifier", estimator)]
    )


def checked_inner_splits(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
    outer_fold: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build grouped inner splits and validate class coverage and leakage."""
    if n_splits < 2:
        raise ValueError("inner_folds must be at least 2.")
    if n_splits > int(np.bincount(y).min()):
        raise ValueError(
            f"Outer fold {outer_fold} has too few examples for "
            f"inner_folds={n_splits}."
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state + outer_fold,
    )
    splits = list(splitter.split(x, y, groups))
    for inner_fold, (train, validation) in enumerate(splits, start=1):
        if set(groups[train]).intersection(groups[validation]):
            raise RuntimeError(
                f"Participant leakage detected in outer fold {outer_fold}, "
                f"inner fold {inner_fold}."
            )
        if len(set(y[train])) != 2 or len(set(y[validation])) != 2:
            raise ValueError(
                f"Outer fold {outer_fold}, inner fold {inner_fold} lacks CN or AD."
            )
    return splits


def calculate_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict[str, float | int]:
    prediction = (probability >= decision_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "threshold": decision_threshold,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_performance_curves(
    y_true: np.ndarray,
    probability: np.ndarray,
    metrics: dict[str, float | int],
    output_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    prevalence = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="0.55", label="Chance")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(recall, precision, linewidth=2, label=f"AP = {metrics['average_precision']:.3f}")
    axes[1].axhline(prevalence, linestyle="--", color="0.55", label=f"Prevalence = {prevalence:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Demographic baseline: pooled out-of-fold performance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    metrics: dict[str, float | int],
    output_path: Path,
    decision_threshold: float = 0.5,
) -> None:
    matrix = np.array(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ]
    )
    fig, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=14)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["CN", "AD"],
        yticklabels=["CN", "AD"],
        xlabel=f"Predicted class (threshold = {decision_threshold:g})",
        ylabel="True class",
        title="OOF confusion matrix",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def build_split_table(
    data: pd.DataFrame,
    row_indices: np.ndarray,
    y: np.ndarray,
    fold: int,
    split: str,
    probabilities: np.ndarray | None = None,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Return an auditable copy of one fold's training or validation rows."""
    table = data.iloc[row_indices].copy()
    table.insert(0, "cohort_row_index", row_indices)
    table.insert(1, "source_csv_row", row_indices + 2)
    table["fold"] = fold
    table["split"] = split
    table["true_label_int"] = y[row_indices]
    if probabilities is not None:
        table["probability_ad"] = probabilities
        table["predicted_label"] = np.where(
            probabilities >= decision_threshold, "AD", "CN"
        )
    return table


def run_baseline(
    input_path: Path,
    output_dir: Path,
    n_splits: int,
    random_state: int,
    decision_threshold: float = 0.5,
    classifier: str = "logistic",
    inner_folds: int = 3,
) -> dict[str, Any]:
    if not np.isfinite(decision_threshold) or not 0.0 <= decision_threshold <= 1.0:
        raise ValueError("decision_threshold must be a finite number between 0 and 1 inclusive.")
    data = load_and_validate_cohort(input_path, n_splits)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = data[PREDICTORS]
    y = data["supervised_label"].map(LABEL_TO_INT).to_numpy()
    groups = data["subject_id"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    pipeline = build_pipeline(random_state, classifier)
    _, parameter_grid = classifier_and_grid(classifier, random_state)

    for pattern in (
        "fold_*_train.csv",
        "fold_*_validation.csv",
        "fold_*_pipeline.joblib",
        "fold_*_coefficients.csv",
        "fold_*_feature_importances.csv",
        "fold_*_inner_search.csv",
    ):
        for old_split_file in output_dir.glob(pattern):
            old_split_file.unlink()
    for legacy_file in (
        output_dir / "demographic_logistic_pipeline.joblib",
        output_dir / "coefficients.csv",
    ):
        if legacy_file.exists():
            legacy_file.unlink()

    probabilities = np.full(len(data), np.nan, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)
    fold_rows: list[dict[str, float | int]] = []
    split_audit: list[dict[str, float | int]] = []
    split_manifest: list[dict[str, str | int]] = []
    fold_search_details: list[dict[str, Any]] = []

    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        train_subjects = set(groups[train_index])
        validation_subjects = set(groups[validation_index])
        overlap = train_subjects.intersection(validation_subjects)
        if overlap:
            raise RuntimeError(f"Participant leakage detected in fold {fold}: {sorted(overlap)}")

        fold_template = clone(pipeline)
        if classifier == "xgboost":
            # Match class_weight="balanced" used by the other classifiers.
            negatives, positives = np.bincount(y[train_index], minlength=2)
            fold_template.set_params(classifier__scale_pos_weight=negatives / positives)
        if parameter_grid:
            inner_splits = checked_inner_splits(
                x.iloc[train_index],
                y[train_index],
                groups[train_index],
                inner_folds,
                random_state,
                fold,
            )
            search = GridSearchCV(
                fold_template,
                parameter_grid,
                scoring="roc_auc",
                cv=inner_splits,
                n_jobs=1,
                refit=True,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(x.iloc[train_index], y[train_index])
            fold_pipeline = search.best_estimator_
            search_table = pd.DataFrame(search.cv_results_).loc[
                :, ["params", "mean_test_score", "std_test_score", "rank_test_score"]
            ]
            search_table.to_csv(
                output_dir / f"fold_{fold:02d}_inner_search.csv", index=False
            )
            fold_search_details.append(
                {
                    "fold": fold,
                    "inner_best_auc": float(search.best_score_),
                    "best_params": search.best_params_,
                }
            )
        else:
            fold_pipeline = fold_template
            fold_pipeline.fit(x.iloc[train_index], y[train_index])
            fold_search_details.append(
                {"fold": fold, "inner_best_auc": None, "best_params": {}}
            )
        fold_probability = fold_pipeline.predict_proba(x.iloc[validation_index])[:, 1]
        age_scaler = fold_pipeline.named_steps["preprocessing"].named_transformers_["age"]
        joblib.dump(fold_pipeline, output_dir / f"fold_{fold:02d}_pipeline.joblib")
        feature_names = fold_pipeline.named_steps["preprocessing"].get_feature_names_out()
        fitted_classifier = fold_pipeline.named_steps["classifier"]
        if hasattr(fitted_classifier, "coef_"):
            coefficients = fitted_classifier.coef_[0]
            intercept = getattr(fitted_classifier, "intercept_", np.array([0.0]))[0]
            coefficient_table = pd.DataFrame(
                {
                    "feature": ["intercept", *feature_names],
                    "coefficient": [intercept, *coefficients],
                }
            )
            coefficient_table.to_csv(
                output_dir / f"fold_{fold:02d}_coefficients.csv",
                index=False,
                float_format="%.10f",
            )
        elif hasattr(fitted_classifier, "feature_importances_"):
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": fitted_classifier.feature_importances_,
                }
            ).to_csv(
                output_dir / f"fold_{fold:02d}_feature_importances.csv",
                index=False,
                float_format="%.10f",
            )
        probabilities[validation_index] = fold_probability
        fold_ids[validation_index] = fold

        split_tables = {
            "train": build_split_table(
                data, train_index, y, fold=fold, split="train"
            ),
            "validation": build_split_table(
                data,
                validation_index,
                y,
                fold=fold,
                split="validation",
                probabilities=fold_probability,
                decision_threshold=decision_threshold,
            ),
        }
        for split_name, split_table in split_tables.items():
            split_path = output_dir / f"fold_{fold:02d}_{split_name}.csv"
            split_table.to_csv(split_path, index=False, float_format="%.10f")
            split_manifest.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "file": split_path.name,
                    "rows": len(split_table),
                    "cn": int((split_table["supervised_label"] == "CN").sum()),
                    "ad": int((split_table["supervised_label"] == "AD").sum()),
                    "sha256": sha256(split_path),
                }
            )

        fold_metrics = calculate_metrics(
            y[validation_index], fold_probability, decision_threshold
        )
        fold_metrics["fold"] = fold
        fold_rows.append(fold_metrics)
        split_audit.append(
            {
                "fold": fold,
                "train_participants": len(train_subjects),
                "validation_participants": len(validation_subjects),
                "participant_overlap": len(overlap),
                "train_cn": int(np.sum(y[train_index] == 0)),
                "train_ad": int(np.sum(y[train_index] == 1)),
                "validation_cn": int(np.sum(y[validation_index] == 0)),
                "validation_ad": int(np.sum(y[validation_index] == 1)),
                "training_ad_proportion": float(np.mean(y[train_index])),
                "validation_ad_proportion": float(np.mean(y[validation_index])),
                "input_ad_proportion": float(np.mean(y)),
                "training_age_mean": float(x.iloc[train_index]["age_at_mri"].mean()),
                "scaler_age_mean": float(age_scaler.mean_[0]),
                "scaler_age_scale": float(age_scaler.scale_[0]),
            }
        )

    if np.isnan(probabilities).any() or np.any(fold_ids < 1):
        raise RuntimeError("Not every participant received exactly one OOF prediction.")
    validation_ad_counts = [int(row["validation_ad"]) for row in split_audit]
    validation_cn_counts = [int(row["validation_cn"]) for row in split_audit]
    if max(validation_ad_counts) - min(validation_ad_counts) > 1:
        raise RuntimeError("AD participants are not distributed evenly across folds.")
    if max(validation_cn_counts) - min(validation_cn_counts) > 1:
        raise RuntimeError("CN participants are not distributed evenly across folds.")

    overall_metrics = calculate_metrics(y, probabilities, decision_threshold)
    overall_metrics.update(
        {
            "participants": len(data),
            "cn_participants": int(np.sum(y == 0)),
            "ad_participants": int(np.sum(y == 1)),
            "n_splits": n_splits,
            "random_state": random_state,
            "classifier": classifier,
        }
    )

    oof = data.copy()
    oof["fold"] = fold_ids
    oof["true_label_int"] = y
    oof["probability_ad"] = probabilities
    oof["predicted_label"] = np.where(
        probabilities >= decision_threshold, "AD", "CN"
    )
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, float_format="%.10f")

    assignments = data.copy()
    assignments["validation_fold"] = fold_ids
    assignments["training_folds"] = [
        ",".join(str(fold) for fold in range(1, n_splits + 1) if fold != validation_fold)
        for validation_fold in fold_ids
    ]
    assignments.to_csv(output_dir / "fold_assignments.csv", index=False)

    pd.DataFrame(fold_rows).sort_values("fold").to_csv(
        output_dir / "fold_metrics.csv", index=False, float_format="%.10f"
    )
    pd.DataFrame(split_audit).sort_values("fold").to_csv(
        output_dir / "split_audit.csv", index=False, float_format="%.10f"
    )
    pd.DataFrame(split_manifest).sort_values(["fold", "split"]).to_csv(
        output_dir / "split_manifest.csv", index=False
    )

    cohort_summary = (
        data.groupby(["supervised_label", "sex"], observed=True)
        .agg(participants=("subject_id", "size"), mean_age=("age_at_mri", "mean"), sd_age=("age_at_mri", "std"))
        .reset_index()
    )
    cohort_summary.to_csv(output_dir / "cohort_summary.csv", index=False, float_format="%.4f")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(json_ready(overall_metrics), stream, indent=2, sort_keys=True)
        stream.write("\n")

    save_performance_curves(y, probabilities, overall_metrics, output_dir / "performance_curves.png")
    save_confusion_matrix(
        overall_metrics,
        output_dir / "confusion_matrix.png",
        decision_threshold,
    )

    config = {
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "output_dir": str(output_dir),
        "predictors": PREDICTORS,
        "label_mapping": LABEL_TO_INT,
        "sex_categories": SEX_CATEGORIES,
        "sex_reference_category": "F",
        "cross_validation": {
            "method": "StratifiedGroupKFold",
            "n_splits": n_splits,
            "shuffle": True,
            "group": "subject_id",
            "random_state": random_state,
            "split_files": "fold_XX_train.csv and fold_XX_validation.csv",
            "split_manifest": "split_manifest.csv",
            "assignment_file": "fold_assignments.csv",
            "training_rule": "For each iteration, fit on four folds and call predict_proba only on the held-out fifth fold",
            "class_balance_rule": "Distribute each class as evenly as integer counts permit across five validation folds",
        },
        "pipeline": {
            "age": "StandardScaler fitted separately in each training fold",
            "sex": "OneHotEncoder(categories=[['F', 'M']], drop='first')",
            "classifier": classifier,
            "parameter_grid": parameter_grid,
        },
        "inner_cross_validation": {
            "method": "StratifiedGroupKFold" if parameter_grid else None,
            "n_splits": inner_folds if parameter_grid else None,
            "scoring": "roc_auc" if parameter_grid else None,
            "fold_search_details": fold_search_details,
        },
        "decision_threshold": decision_threshold,
        "saved_models": "fold_XX_pipeline.joblib; each model is fitted only on that fold's training CSV",
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
    }
    if classifier == "xgboost":
        import xgboost

        config["software"]["xgboost"] = xgboost.__version__
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")

    return overall_metrics


def main() -> None:
    args = parse_args()
    log_path = default_log_path(args.input)
    configure_logging(log_path)
    report(f"Input: {relative_path(args.input)}")

    try:
        metrics = run_baseline(
            input_path=args.input,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            decision_threshold=args.decision_threshold,
            classifier=args.classifier,
            inner_folds=args.inner_folds,
        )
    except (OSError, ValueError, RuntimeError, pd.errors.ParserError) as error:
        LOGGER.exception("Demographic baseline failed: %s", error)
        raise SystemExit(f"Error: {error}") from error

    report(
        f"Participants: {metrics['participants']} "
        f"(CN={metrics['cn_participants']}, AD={metrics['ad_participants']})"
    )
    report(f"Classifier: {args.classifier}")
    report(f"OOF ROC-AUC: {metrics['roc_auc']:.4f}")
    report(f"OOF balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    report(f"Results: {relative_path(args.output_dir)}")
    report(f"Log: {relative_path(log_path)}")


if __name__ == "__main__":
    main()
