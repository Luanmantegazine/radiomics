"""Checks for demographic classifier parity and nested validation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from oasis_analysis.demographic_baseline import (
    CLASSIFIERS,
    classifier_and_grid,
    run_baseline,
)


def cohort() -> pd.DataFrame:
    rows = []
    for index in range(24):
        label = "CN" if index < 12 else "AD"
        rows.append(
            {
                "subject_id": f"OAS{index:05d}",
                "session_id": f"OAS{index:05d}_MR_d0000",
                "age_at_mri": 55 + index + (index % 3),
                "sex": "F" if index % 2 == 0 else "M",
                "supervised_label": label,
            }
        )
    return pd.DataFrame(rows)


class DemographicBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.input_path = Path(self.temporary.name) / "cohort.csv"
        self.output_dir = Path(self.temporary.name) / "output"
        cohort().to_csv(self.input_path, index=False)

    def test_classifier_options_match_morphological_pipeline(self) -> None:
        self.assertEqual(
            CLASSIFIERS,
            ("logistic", "elasticnet", "svm", "rf", "xgboost"),
        )

        elasticnet, elasticnet_grid = classifier_and_grid("elasticnet", 42)
        svm, svm_grid = classifier_and_grid("svm", 42)
        forest, forest_grid = classifier_and_grid("rf", 42)
        xgboost, xgboost_grid = classifier_and_grid("xgboost", 42)

        self.assertIsInstance(elasticnet, LogisticRegression)
        self.assertEqual(elasticnet.solver, "saga")
        self.assertIsInstance(svm, SVC)
        self.assertIsInstance(forest, RandomForestClassifier)
        self.assertEqual(xgboost.__class__.__name__, "XGBClassifier")
        self.assertEqual(
            elasticnet_grid,
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__l1_ratio": [0.1, 0.5, 0.9],
            },
        )
        self.assertEqual(
            svm_grid,
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["linear", "rbf"],
            },
        )
        self.assertEqual(
            forest_grid,
            {
                "classifier__max_depth": [None, 4],
                "classifier__min_samples_leaf": [1, 3],
            },
        )
        self.assertEqual(
            xgboost_grid,
            {
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8, 1.0],
            },
        )

    def test_tuned_classifier_uses_inner_search_in_each_outer_fold(self) -> None:
        estimator = LogisticRegression(
            solver="liblinear", class_weight="balanced", random_state=42
        )
        small_grid = {"classifier__C": [0.1, 1.0]}
        with patch(
            "oasis_analysis.demographic_baseline.classifier_and_grid",
            return_value=(estimator, small_grid),
        ):
            metrics = run_baseline(
                self.input_path,
                self.output_dir,
                n_splits=3,
                inner_folds=2,
                random_state=42,
                classifier="elasticnet",
            )

        oof = pd.read_csv(self.output_dir / "oof_predictions.csv")
        config = json.loads((self.output_dir / "run_config.json").read_text())
        self.assertEqual(len(oof), len(cohort()))
        self.assertTrue(np.isfinite(oof["probability_ad"]).all())
        self.assertEqual(len(list(self.output_dir.glob("fold_*_inner_search.csv"))), 3)
        self.assertEqual(len(list(self.output_dir.glob("fold_*_pipeline.joblib"))), 3)
        self.assertEqual(config["pipeline"]["classifier"], "elasticnet")
        self.assertEqual(config["inner_cross_validation"]["n_splits"], 2)
        self.assertEqual(len(config["inner_cross_validation"]["fold_search_details"]), 3)
        self.assertEqual(metrics["classifier"], "elasticnet")
        self.assertIn("recall", metrics)
        self.assertNotIn("recall_sensitivity", metrics)

        saved_metrics = json.loads((self.output_dir / "metrics.json").read_text())
        self.assertIn("recall", saved_metrics)
        self.assertNotIn("recall_sensitivity", saved_metrics)


if __name__ == "__main__":
    unittest.main()
