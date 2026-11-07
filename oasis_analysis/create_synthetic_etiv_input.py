#!/usr/bin/env python3
"""Create a TEST-ONLY CSV with synthetic eTIV from four non-model regions.

Reference: Voevodskaya et al. (2014), Tables 1 and 4. In 406 cognitively
normal 75-year-olds, cortical regional volumes were averaged across hemispheres.
The bilateral-to-ICV ratios below double the sex-weighted unilateral means and
divide by the sex-weighted mean ICV. They are only for a pipeline smoke test.
https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2014.00264/full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from statistics import median
from pathlib import Path


ETIV_COLUMN = "EstimatedTotalIntraCranialVol"
REFERENCE_URL = "https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2014.00264/full"
REFERENCE_FEMALE_COUNT = 193
REFERENCE_MALE_COUNT = 213
REFERENCE_ICV_MM3 = (
    REFERENCE_FEMALE_COUNT * 1440320 + REFERENCE_MALE_COUNT * 1638419
) / (REFERENCE_FEMALE_COUNT + REFERENCE_MALE_COUNT)
# Table 4 reports the mean of left/right cortical volumes, not their sum.
REFERENCE_UNILATERAL_MM3 = {
    "fusiform": (8077, 8842),
    "inferior_temporal": (9039, 10024),
    "middle_temporal": (9508, 10275),
    "precuneus": (7859, 8643),
}
REGION_TO_ICV_RATIO = {
    region: 2
    * (REFERENCE_FEMALE_COUNT * female + REFERENCE_MALE_COUNT * male)
    / (REFERENCE_FEMALE_COUNT + REFERENCE_MALE_COUNT)
    / REFERENCE_ICV_MM3
    for region, (female, male) in REFERENCE_UNILATERAL_MM3.items()
}
SOURCE_COLUMNS = tuple(
    f"mask_volume_mm3_{side}_{region}"
    for region in REGION_TO_ICV_RATIO
    for side in ("left", "right")
)


def create_test_input(input_path: Path, output_path: Path) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ; preserve the original CSV.")
    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        missing = [column for column in SOURCE_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}.")
        if ETIV_COLUMN in reader.fieldnames:
            raise ValueError(f"Input already contains {ETIV_COLUMN}; refusing to replace it.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=[*reader.fieldnames, ETIV_COLUMN])
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise ValueError(f"Row {line_number} has more fields than the header.")
                volumes = {}
                for column in SOURCE_COLUMNS:
                    try:
                        value = float(row[column])
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"Row {line_number}: invalid {column}.") from error
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError(f"Row {line_number}: {column} must be positive.")
                    volumes[column] = value
                estimates = [
                    (
                        volumes[f"mask_volume_mm3_left_{region}"]
                        + volumes[f"mask_volume_mm3_right_{region}"]
                    )
                    / ratio
                    for region, ratio in REGION_TO_ICV_RATIO.items()
                ]
                row[ETIV_COLUMN] = f"{median(estimates):.6f}"
                writer.writerow(row)
                count += 1

    metadata = {
        "test_only": True,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": count,
        "column": ETIV_COLUMN,
        "formula": "median((left_region + right_region) / reference_bilateral_region_to_icv_ratio for each of four regions)",
        "regions": list(REGION_TO_ICV_RATIO),
        "source_columns": list(SOURCE_COLUMNS),
        "reference_bilateral_region_to_icv_ratios": REGION_TO_ICV_RATIO,
        "reference_cohort": "PIVUS cognitively normal adults age 75, N=406; Tables 1 and 4",
        "reference": REFERENCE_URL,
        "warning": "Synthetic eTIV depends on cortical volumes and must not be used to assess model performance or to normalize volumes in scientific analyses.",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        count = create_test_input(args.input, args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Created TEST-ONLY CSV with synthetic eTIV for {count} rows: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
