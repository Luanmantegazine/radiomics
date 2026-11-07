"""The synthetic eTIV generator must stay isolated from real input data."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from statistics import median

from oasis_analysis.create_synthetic_etiv_input import (
    ETIV_COLUMN,
    REGION_TO_ICV_RATIO,
    SOURCE_COLUMNS,
    create_test_input,
)


class SyntheticEtivInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name) / "source.csv"
        self.output = Path(self.temporary.name) / "test.csv"

    def test_creates_separate_test_csv_with_recorded_formula(self) -> None:
        source_volumes = {
            column: 1000 * (index + 1)
            for index, column in enumerate(SOURCE_COLUMNS)
        }
        with self.source.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["subject_id", *SOURCE_COLUMNS])
            writer.writeheader()
            writer.writerow({"subject_id": "S1", **source_volumes})

        self.assertEqual(create_test_input(self.source, self.output), 1)
        with self.output.open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))

        estimates = [
            (source_volumes[f"mask_volume_mm3_left_{region}"]
             + source_volumes[f"mask_volume_mm3_right_{region}"]) / ratio
            for region, ratio in REGION_TO_ICV_RATIO.items()
        ]
        self.assertAlmostEqual(float(row[ETIV_COLUMN]), median(estimates), places=6)
        self.assertNotIn(ETIV_COLUMN, self.source.read_text(encoding="utf-8"))
        metadata = json.loads(self.output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["source_columns"], list(SOURCE_COLUMNS))
        self.assertIn("median", metadata["formula"])

    def test_refuses_to_replace_a_real_etiv_column(self) -> None:
        self.source.write_text(
            f"{','.join(SOURCE_COLUMNS)},{ETIV_COLUMN}\n"
            f"{','.join('1' for _ in SOURCE_COLUMNS)},1500000\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "already contains"):
            create_test_input(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_requires_all_four_bilateral_regions(self) -> None:
        self.source.write_text("subject_id\nS1\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "mask_volume_mm3_left_fusiform"):
            create_test_input(self.source, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
