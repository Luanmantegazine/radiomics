"""Focused checks for nested morphometric modeling and leakage boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import load
from sklearn.linear_model import LogisticRegression

from oasis_analysis.morphological_model import (
    ETIVAdjustedFeatures,
    classifier_and_grid,
    run_model,
)
from oasis_analysis.prepare_morphological_features import (
    DEFAULT_SOURCE_COLUMNS,
    EXTENDED_SOURCE_COLUMNS,
)


def cohort() -> pd.DataFrame:
    rows = []
    for index in range(12):
        etiv = 1_100_000 + 60_000 * index
        row = {
            "subject_id": f"OAS{index:05d}",
            "session_id": f"OAS{index:05d}_MR_d0000",
            "supervised_label": "CN" if index < 6 else "AD",
        }
        for target, source in DEFAULT_SOURCE_COLUMNS.items():
            if target == "etiv_mm3":
                row[source] = etiv
            else:
                row[source] = 0.004 * etiv + (index % 3) * 120 - (index >= 6) * 300
        rows.append(row)
    return pd.DataFrame(rows)


class MorphologicalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.input = Path(self.temporary.name) / "input.csv"
        self.output = Path(self.temporary.name) / "results"

    def test_residual_fit_uses_training_cn_only(self) -> None:
        data = cohort()
        X = data.loc[:, list(DEFAULT_SOURCE_COLUMNS.values())]
        y = data["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()
        first = ETIVAdjustedFeatures().fit(X, y)
        changed = X.copy()
        changed.loc[y == 1, "mask_volume_mm3_left"] += 100_000
        second = ETIVAdjustedFeatures().fit(changed, y)

        self.assertEqual(first.cn_fits_, second.cn_fits_)
        self.assertEqual(first.cn_fits_["left_hippocampus_mm3"]["cn_training_rows"], 6)
        transformed = first.transform(X)
        self.assertEqual(transformed.shape, (12, 8))
        self.assertAlmostEqual(transformed[y == 0, 0].mean(), 0.0, places=9)

    def test_proportional_and_asymmetry_features(self) -> None:
        data = cohort()
        X = data.loc[:, list(DEFAULT_SOURCE_COLUMNS.values())].copy()
        X.loc[0, "mask_volume_mm3_left"] = 3000
        X.loc[0, "mask_volume_mm3_right"] = 2000
        transformer = ETIVAdjustedFeatures(proportional=True, asymmetry=True).fit(X)
        transformed = transformer.transform(X)

        self.assertEqual(transformed.shape, (12, 12))
        self.assertAlmostEqual(transformed[0, 0], 3000 / X.iloc[0]["EstimatedTotalIntraCranialVol"])
        self.assertAlmostEqual(transformed[0, 8], 0.2)
        self.assertIn("hippocampus_asymmetry", transformer.get_feature_names_out())

    def test_extended_mode_preserves_cortical_thickness(self) -> None:
        original = cohort()
        X = pd.DataFrame(index=original.index)
        for target, source in EXTENDED_SOURCE_COLUMNS.items():
            if target == "etiv_mm3":
                X[source] = original["EstimatedTotalIntraCranialVol"]
            elif target.endswith("_thickness_mm"):
                X[source] = 2.0 + original.index * 0.1
            else:
                X[source] = original["mask_volume_mm3_left"]
        y = original["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()

        transformer = ETIVAdjustedFeatures(extended_data=True, asymmetry=True).fit(X, y)
        transformed = transformer.transform(X)
        self.assertEqual(transformed.shape, (12, 15))
        self.assertAlmostEqual(transformed[0, 6], 2.0)
        self.assertIn("inf_lat_vent_asymmetry", transformer.get_feature_names_out())

    def test_all_secondary_classifiers_are_available_as_options(self) -> None:
        for name in ("elasticnet", "svm", "rf"):
            estimator, grid = classifier_and_grid(name, 42)
            self.assertIsNotNone(estimator)
            self.assertTrue(grid)

    def test_nested_run_exports_oof_metrics_and_fold_models(self) -> None:
        cohort().to_csv(self.input, index=False)
        small_grid = (LogisticRegression(l1_ratio=0.5, solver="saga", max_iter=2000,
                                         random_state=42), {"classifier__C": [1.0]})
        with patch("oasis_analysis.morphological_model.classifier_and_grid",
                   return_value=small_grid):
            metrics = run_model(
                self.input, self.output, outer_folds=3, inner_folds=2,
                bootstrap_resamples=100,
            )

        oof = pd.read_csv(self.output / "oof_predictions.csv")
        config = json.loads((self.output / "run_config.json").read_text())
        self.assertEqual(len(oof), 12)
        self.assertEqual(oof["subject_id"].nunique(), 12)
        self.assertEqual(set(oof["fold"]), {1, 2, 3})
        self.assertTrue(np.isfinite(oof["probability_ad"]).all())
        self.assertEqual(len(list(self.output.glob("fold_*_model.joblib"))), 3)
        saved = load(self.output / "fold_01_model.joblib")
        self.assertEqual(saved.named_steps["morphology"].__class__.__module__,
                         "oasis_analysis.morphological_model")
        self.assertEqual(config["volume_adjustment"], "residual_cn_training_only")
        self.assertEqual(len(config["fold_details"]), 3)
        self.assertIn("roc_auc_ci95_low", metrics)
        self.assertIn("brier_score", metrics)
        self.assertTrue((self.output / "calibration.png").exists())


if __name__ == "__main__":
    unittest.main()
