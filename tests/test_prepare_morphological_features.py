"""Check leakage-safe morphometric adjustment and export."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from oasis_analysis.prepare_morphological_features import (
    DEFAULT_SOURCE_COLUMNS,
    EXTENDED_SOURCE_COLUMNS,
    fit_cn_residuals,
    prepare,
)


def write_input(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def cohort(*, extended: bool = False) -> list[dict[str, str]]:
    source_columns = EXTENDED_SOURCE_COLUMNS if extended else DEFAULT_SOURCE_COLUMNS
    rows = []
    for index in range(10):
        etiv = 1_000_000 + index * 100_000
        row = {
            "subject_id": f"OAS{index:05d}",
            "session_id": f"OAS{index:05d}_MR_d0000",
            "supervised_label": "CN" if index < 5 else "AD",
            "MMSE": "30",
        }
        for target, source in source_columns.items():
            if target == "etiv_mm3":
                row[source] = str(etiv)
            elif target.endswith("_thickness_mm"):
                row[source] = str(2 + index / 10)
            else:
                row[source] = str(0.004 * etiv + index * index * 10 + (index >= 5) * 500)
        rows.append(row)
    return rows


class PrepareMorphologicalFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "input.csv"
        self.output = Path(self.temporary.name) / "morphology.csv"

    def read_output(self) -> list[dict[str, str]]:
        with self.output.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_residual_fit_ignores_ad_and_held_out_cn(self) -> None:
        rows = [
            {"supervised_label": "CN", "etiv_mm3": "1", "left_hippocampus_mm3": "2"},
            {"supervised_label": "CN", "etiv_mm3": "2", "left_hippocampus_mm3": "4"},
            {"supervised_label": "AD", "etiv_mm3": "3", "left_hippocampus_mm3": "1000"},
            {"supervised_label": "CN", "etiv_mm3": "4", "left_hippocampus_mm3": "1000"},
        ]

        fits = fit_cn_residuals(rows, [0, 1, 2], ["left_hippocampus_mm3"])
        self.assertAlmostEqual(fits["left_hippocampus_mm3"]["slope"], 2.0)
        self.assertAlmostEqual(fits["left_hippocampus_mm3"]["intercept"], 0.0)
        self.assertEqual(fits["left_hippocampus_mm3"]["cn_training_rows"], 2)

    def test_residual_is_default_and_fitted_on_training_cn_only(self) -> None:
        rows = cohort()
        write_input(self.source, rows)

        manifest = prepare(self.source, self.output)
        output = self.read_output()
        self.assertEqual(len(output), len(rows))
        self.assertEqual(manifest["adjustment_method"], "residual")
        self.assertIn("left_hippocampus_residual_mm3", output[0])
        self.assertNotIn("left_hippocampus_mm3", output[0])
        self.assertNotIn("MMSE", output[0])
        self.assertNotIn("hippocampus_asymmetry", output[0])
        self.assertEqual({int(row["fold"]) for row in output}, set(range(1, 6)))
        for files in manifest["fold_files"]:
            with Path(files["train"]).open(newline="", encoding="utf-8") as stream:
                train = list(csv.DictReader(stream))
            with Path(files["validation"]).open(newline="", encoding="utf-8") as stream:
                validation = list(csv.DictReader(stream))
            fits = json.loads(Path(files["cn_residual_fits"]).read_text())
            self.assertEqual(len(train) + len(validation), len(rows))
            self.assertFalse({row["subject_id"] for row in train} &
                             {row["subject_id"] for row in validation})
            self.assertEqual(
                fits["left_hippocampus_mm3"]["cn_training_rows"],
                sum(row["supervised_label"] == "CN" for row in train),
            )
            for transformed in validation:
                original = next(row for row in rows if row["subject_id"] == transformed["subject_id"])
                fit = fits["left_hippocampus_mm3"]
                expected = float(original["mask_volume_mm3_left"]) - (
                    fit["intercept"] + fit["slope"] * float(original["EstimatedTotalIntraCranialVol"])
                )
                self.assertAlmostEqual(float(transformed["left_hippocampus_residual_mm3"]), expected)
                oof = next(row for row in output if row["subject_id"] == transformed["subject_id"])
                self.assertEqual(oof, transformed)

    def test_proportional_flag_divides_each_volume_by_etiv(self) -> None:
        rows = cohort()
        write_input(self.source, rows)

        manifest = prepare(self.source, self.output, proportional=True)
        output = self.read_output()
        self.assertEqual(manifest["adjustment_method"], "proportional")
        self.assertIsNone(manifest["fold_files"][0]["cn_residual_fits"])
        self.assertIn("left_hippocampus_etiv_ratio", output[0])
        for original, transformed in zip(rows, output):
            self.assertAlmostEqual(
                float(transformed["left_hippocampus_etiv_ratio"]),
                float(original["mask_volume_mm3_left"])
                / float(original["EstimatedTotalIntraCranialVol"]),
            )

    def test_asymmetry_flag_is_independent_of_adjustment_method(self) -> None:
        write_input(self.source, cohort())
        for proportional in (False, True):
            manifest = prepare(self.source, self.output, asymmetry=True, proportional=proportional)
            output = self.read_output()
            self.assertIn("hippocampus_asymmetry", output[0])
            self.assertAlmostEqual(float(output[0]["hippocampus_asymmetry"]), 0.0)
            self.assertIn("hippocampus_asymmetry", manifest["predictors"])

    def test_extended_thickness_is_unchanged_and_volumes_are_adjusted(self) -> None:
        rows = cohort(extended=True)
        write_input(self.source, rows)

        manifest = prepare(self.source, self.output, extended_data=True, asymmetry=True)
        output = self.read_output()
        self.assertEqual(manifest["mode"], "extended_data")
        self.assertIn("left_inf_lat_vent_residual_mm3", output[0])
        self.assertEqual(output[0]["left_entorhinal_thickness_mm"],
                         rows[0]["lh.aparc.entorhinal.ThickAvg"])
        self.assertIn("entorhinal_asymmetry", output[0])

    def test_reuses_explicit_fold_assignments(self) -> None:
        rows = cohort()
        write_input(self.source, rows)
        assignments = Path(self.temporary.name) / "fold_assignments.csv"
        with assignments.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "validation_fold"])
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({"subject_id": row["subject_id"], "session_id": row["session_id"],
                                 "validation_fold": index % 5 + 1})

        manifest = prepare(self.source, self.output, fold_assignments=assignments)
        output = self.read_output()
        self.assertEqual(manifest["cross_validation"]["method"], "provided_assignments")
        self.assertEqual([int(row["fold"]) for row in output], [i % 5 + 1 for i in range(10)])

    def test_missing_etiv_stops_before_writing(self) -> None:
        rows = cohort()
        for row in rows:
            del row["EstimatedTotalIntraCranialVol"]
        write_input(self.source, rows)
        with self.assertRaisesRegex(ValueError, "EstimatedTotalIntraCranialVol"):
            prepare(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_extended_mode_requires_freesurfer_stats(self) -> None:
        write_input(self.source, cohort())
        with self.assertRaisesRegex(ValueError, "aseg.Left-Inf-Lat-Vent.Volume_mm3"):
            prepare(self.source, self.output, extended_data=True)
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_subjects(self) -> None:
        rows = cohort()
        rows[1]["subject_id"] = rows[0]["subject_id"]
        write_input(self.source, rows)
        with self.assertRaisesRegex(ValueError, "duplicated subject_id"):
            prepare(self.source, self.output)


if __name__ == "__main__":
    unittest.main()
