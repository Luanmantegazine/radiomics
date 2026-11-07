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
    ELASTICNET_C_GRID,
    ETIVAdjustedFeatures,
    classifier_and_grid,
    load_cohort,
    run_model,
)
from oasis_analysis.prepare_morphological_features import (
    DEFAULT_SOURCE_COLUMNS,
    EXTENDED_SOURCE_COLUMNS,
)


def cohort(size: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(size):
        etiv = 1_100_000 + 60_000 * index
        row = {
            "subject_id": f"OAS{index:05d}",
            "session_id": f"OAS{index:05d}_MR_d0000",
            "supervised_label": "CN" if index < size // 2 else "AD",
            "age_at_mri": 60.0 + 2.5 * index + (index % 4),
            "sex": "F" if index % 2 == 0 else "M",
        }
        for target, source in DEFAULT_SOURCE_COLUMNS.items():
            if target == "etiv_mm3":
                row[source] = etiv
            else:
                row[source] = 0.004 * etiv + (index % 3) * 120 - (index >= size // 2) * 300
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

    def test_age_sex_adjustment_is_fit_on_training_cn_only(self) -> None:
        data = cohort()
        columns = [*DEFAULT_SOURCE_COLUMNS.values(), "age_at_mri", "sex"]
        X = data.loc[:, columns]
        y = data["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()
        first = ETIVAdjustedFeatures(adjust_age_sex=True).fit(X, y)
        changed = X.copy()
        changed.loc[y == 1, "mask_volume_mm3_left"] += 100_000
        changed.loc[y == 1, "age_at_mri"] += 20
        second = ETIVAdjustedFeatures(adjust_age_sex=True).fit(changed, y)

        self.assertEqual(first.cn_fits_, second.cn_fits_)
        fit = first.cn_fits_["left_hippocampus_mm3"]
        self.assertEqual(set(fit["coefficients"]), {"etiv_mm3", "age_at_mri", "male"})
        self.assertEqual(fit["cn_training_rows"], 6)
        transformed = first.transform(X)
        self.assertEqual(transformed.shape, (12, 8))
        controls = transformed[y == 0, 0]
        self.assertAlmostEqual(controls.mean(), 0.0, places=6)
        self.assertAlmostEqual(
            float(np.dot(controls, X.loc[y == 0, "age_at_mri"] - X.loc[y == 0, "age_at_mri"].mean())),
            0.0, places=3)
        self.assertEqual(first.get_feature_names_out()[0],
                         "left_hippocampus_age_sex_etiv_residual_mm3")

    def test_age_sex_adjustment_of_thickness_and_ratios(self) -> None:
        original = cohort()
        X = original.loc[:, ["age_at_mri", "sex"]].copy()
        for target, source in EXTENDED_SOURCE_COLUMNS.items():
            if target == "etiv_mm3":
                X[source] = original["EstimatedTotalIntraCranialVol"]
            elif target.endswith("_thickness_mm"):
                X[source] = 3.2 - 0.01 * original["age_at_mri"] + (original.index % 3) * 0.05
            else:
                X[source] = original["mask_volume_mm3_left"]
        y = original["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()

        transformer = ETIVAdjustedFeatures(
            extended_data=True, adjust_age_sex=True, proportional=True).fit(X, y)
        names = list(transformer.get_feature_names_out())
        thickness = transformer.cn_fits_["left_entorhinal_thickness_mm"]
        self.assertEqual(set(thickness["coefficients"]), {"age_at_mri", "male"})
        self.assertEqual(set(transformer.cn_fits_["left_hippocampus_mm3"]["coefficients"]),
                         {"age_at_mri", "male"})
        self.assertIn("left_hippocampus_etiv_ratio_age_sex_residual", names)
        self.assertIn("left_entorhinal_thickness_mm_age_sex_residual", names)
        transformed = transformer.transform(X)
        self.assertAlmostEqual(transformed[y == 0, names.index(
            "left_entorhinal_thickness_mm_age_sex_residual")].mean(), 0.0, places=9)

    def test_age_sex_adjustment_rejects_single_sex_controls(self) -> None:
        data = cohort()
        data["sex"] = "F"
        X = data.loc[:, [*DEFAULT_SOURCE_COLUMNS.values(), "age_at_mri", "sex"]]
        y = data["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()
        with self.assertRaisesRegex(ValueError, "collinear or constant"):
            ETIVAdjustedFeatures(adjust_age_sex=True).fit(X, y)

    def test_demographic_features_are_appended(self) -> None:
        data = cohort()
        X = data.loc[:, [*DEFAULT_SOURCE_COLUMNS.values(), "age_at_mri", "sex"]]
        y = data["supervised_label"].map({"CN": 0, "AD": 1}).to_numpy()
        transformer = ETIVAdjustedFeatures(include_demographics=True).fit(X, y)
        transformed = transformer.transform(X)
        self.assertEqual(transformed.shape, (12, 10))
        self.assertEqual(list(transformer.get_feature_names_out()[-2:]), ["age_at_mri", "male"])
        np.testing.assert_allclose(transformed[:, -2], X["age_at_mri"])
        np.testing.assert_allclose(transformed[:, -1], (X["sex"] == "M").astype(float))

    def test_demographic_columns_are_validated_only_when_requested(self) -> None:
        data = cohort().drop(columns=["age_at_mri"])
        data.to_csv(self.input, index=False)
        self.assertEqual(len(load_cohort(self.input, extended_data=False, outer_folds=3)), 12)
        with self.assertRaisesRegex(ValueError, "age_at_mri"):
            load_cohort(self.input, extended_data=False, outer_folds=3, demographics=True)
        data = cohort()
        data.loc[0, "sex"] = "X"
        data.to_csv(self.input, index=False)
        with self.assertRaisesRegex(ValueError, "sex values"):
            load_cohort(self.input, extended_data=False, outer_folds=3, demographics=True)

    def test_elasticnet_grid_extends_below_previous_boundary(self) -> None:
        _, grid = classifier_and_grid("elasticnet", 42)
        self.assertEqual(grid["classifier__C"], list(ELASTICNET_C_GRID))
        self.assertLess(min(grid["classifier__C"]), 0.1)
        self.assertEqual(grid["classifier__l1_ratio"], [0.5])

    def test_run_with_age_sex_options_records_configuration(self) -> None:
        cohort(36).to_csv(self.input, index=False)
        self.output.mkdir()
        stale = self.output / "fold_09_model.joblib"
        stale.write_text("old")
        small_grid = (LogisticRegression(l1_ratio=0.5, solver="saga", max_iter=2000,
                                         random_state=42), {"classifier__C": [1.0]})
        with patch("oasis_analysis.morphological_model.classifier_and_grid",
                   return_value=small_grid):
            metrics = run_model(
                self.input, self.output, outer_folds=3, inner_folds=2,
                bootstrap_resamples=100, adjust_age_sex=True, include_demographics=True,
            )

        config = json.loads((self.output / "run_config.json").read_text())
        self.assertFalse(stale.exists())
        self.assertEqual(config["volume_adjustment"],
                         "residual_on_etiv_age_sex_cn_training_only")
        self.assertEqual(config["demographic_features"], ["age_at_mri", "male"])
        self.assertEqual(config["fold_details"][0]["feature_names"][-2:],
                         ["age_at_mri", "male"])
        self.assertIn("fold_roc_auc_mean", metrics)
        self.assertIn("fold_roc_auc_sd", metrics)

    def test_xgboost_is_weighted_by_training_class_balance(self) -> None:
        data = cohort()
        data = pd.concat([data, data.iloc[:4].assign(
            subject_id=lambda frame: frame["subject_id"] + "b",
            session_id=lambda frame: frame["session_id"] + "b")], ignore_index=True)
        data.to_csv(self.input, index=False)
        estimator, _ = classifier_and_grid("xgboost", 42)
        with patch("oasis_analysis.morphological_model.classifier_and_grid",
                   return_value=(estimator, {"classifier__max_depth": [2]})):
            run_model(self.input, self.output, classifier="xgboost", outer_folds=3,
                      inner_folds=2, bootstrap_resamples=100)

        config = json.loads((self.output / "run_config.json").read_text())
        for detail in config["fold_details"]:
            train = data[data["subject_id"].isin(detail["train_subject_ids"])]
            counts = train["supervised_label"].value_counts()
            self.assertAlmostEqual(detail["scale_pos_weight"], counts["CN"] / counts["AD"])


if __name__ == "__main__":
    unittest.main()
