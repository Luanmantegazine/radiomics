#!/usr/bin/env python3
"""Export leakage-safe, eTIV-adjusted features for the CN/AD morphology model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import fmean


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output/prepare_morphological_features/morphological_features.csv"

# Default: the eight ROI mask volumes already available in the consolidated CSV.
DEFAULT_SOURCE_COLUMNS = {
    "etiv_mm3": "EstimatedTotalIntraCranialVol",
    "left_hippocampus_mm3": "mask_volume_mm3_left",
    "right_hippocampus_mm3": "mask_volume_mm3_right",
    "left_amygdala_mm3": "mask_volume_mm3_left_amygdala",
    "right_amygdala_mm3": "mask_volume_mm3_right_amygdala",
    "left_entorhinal_mm3": "mask_volume_mm3_left_entorhinal",
    "right_entorhinal_mm3": "mask_volume_mm3_right_entorhinal",
    "left_parahippocampal_mm3": "mask_volume_mm3_left_parahippocampal",
    "right_parahippocampal_mm3": "mask_volume_mm3_right_parahippocampal",
}

# Opt-in: the stricter flattened FreeSurfer stats schema. If a stats merger
# uses different headers, adapt the mapping there or here.
EXTENDED_SOURCE_COLUMNS = {
    "etiv_mm3": "EstimatedTotalIntraCranialVol",
    "left_hippocampus_mm3": "aseg.Left-Hippocampus.Volume_mm3",
    "right_hippocampus_mm3": "aseg.Right-Hippocampus.Volume_mm3",
    "left_amygdala_mm3": "aseg.Left-Amygdala.Volume_mm3",
    "right_amygdala_mm3": "aseg.Right-Amygdala.Volume_mm3",
    "left_inf_lat_vent_mm3": "aseg.Left-Inf-Lat-Vent.Volume_mm3",
    "right_inf_lat_vent_mm3": "aseg.Right-Inf-Lat-Vent.Volume_mm3",
    "left_entorhinal_thickness_mm": "lh.aparc.entorhinal.ThickAvg",
    "right_entorhinal_thickness_mm": "rh.aparc.entorhinal.ThickAvg",
    "left_parahippocampal_thickness_mm": "lh.aparc.parahippocampal.ThickAvg",
    "right_parahippocampal_thickness_mm": "rh.aparc.parahippocampal.ThickAvg",
}
# Headers used by the consolidated FreeSurfer tier export (morpho_tier_consolidado.csv).
EXTENDED_SOURCE_ALIASES = {
    "EstimatedTotalIntraCranialVol": "fs_etiv_mm3",
    "aseg.Left-Hippocampus.Volume_mm3": "fs_left_hippocampus_mm3",
    "aseg.Right-Hippocampus.Volume_mm3": "fs_right_hippocampus_mm3",
    "aseg.Left-Amygdala.Volume_mm3": "fs_left_amygdala_mm3",
    "aseg.Right-Amygdala.Volume_mm3": "fs_right_amygdala_mm3",
    "aseg.Left-Inf-Lat-Vent.Volume_mm3": "fs_left_inf_lat_vent_mm3",
    "aseg.Right-Inf-Lat-Vent.Volume_mm3": "fs_right_inf_lat_vent_mm3",
    "lh.aparc.entorhinal.ThickAvg": "fs_lh_entorhinal_thickavg_mm",
    "rh.aparc.entorhinal.ThickAvg": "fs_rh_entorhinal_thickavg_mm",
    "lh.aparc.parahippocampal.ThickAvg": "fs_lh_parahippocampal_thickavg_mm",
    "rh.aparc.parahippocampal.ThickAvg": "fs_rh_parahippocampal_thickavg_mm",
}
DEFAULT_ASYMMETRY_PAIRS = {
    "hippocampus_asymmetry": ("left_hippocampus_mm3", "right_hippocampus_mm3"),
    "amygdala_asymmetry": ("left_amygdala_mm3", "right_amygdala_mm3"),
    "entorhinal_asymmetry": ("left_entorhinal_mm3", "right_entorhinal_mm3"),
    "parahippocampal_asymmetry": (
        "left_parahippocampal_mm3", "right_parahippocampal_mm3"
    ),
}
EXTENDED_ASYMMETRY_PAIRS = {
    "hippocampus_asymmetry": ("left_hippocampus_mm3", "right_hippocampus_mm3"),
    "amygdala_asymmetry": ("left_amygdala_mm3", "right_amygdala_mm3"),
    "inf_lat_vent_asymmetry": ("left_inf_lat_vent_mm3", "right_inf_lat_vent_mm3"),
    "entorhinal_asymmetry": (
        "left_entorhinal_thickness_mm", "right_entorhinal_thickness_mm"
    ),
    "parahippocampal_asymmetry": (
        "left_parahippocampal_thickness_mm", "right_parahippocampal_thickness_mm"
    ),
}
IDENTIFIER_COLUMNS = ("subject_id", "session_id")
LABEL_COLUMN = "supervised_label"


def adjusted_name(column: str, proportional: bool) -> str:
    stem = column.removesuffix("_mm3")
    return f"{stem}_etiv_ratio" if proportional else f"{stem}_residual_mm3"


def fit_cn_residuals(
    rows: list[dict[str, str]], train_indices: list[int], volume_columns: list[str]
) -> dict[str, dict[str, float | int]]:
    """Fit volume = intercept + slope * eTIV using only training CN rows."""
    cn_rows = [rows[index] for index in train_indices if rows[index][LABEL_COLUMN] == "CN"]
    if len(cn_rows) < 2:
        raise ValueError("Residual adjustment needs at least two CN training rows per fold.")
    etiv = [float(row["etiv_mm3"]) for row in cn_rows]
    mean_etiv = fmean(etiv)
    centered_etiv = [value - mean_etiv for value in etiv]
    denominator = sum(value * value for value in centered_etiv)
    if denominator <= 0:
        raise ValueError("CN training eTIV has zero variance; cannot fit residual adjustment.")
    coefficients = {}
    for column in volume_columns:
        volumes = [float(row[column]) for row in cn_rows]
        mean_volume = fmean(volumes)
        slope = sum(
            x * (volume - mean_volume)
            for x, volume in zip(centered_etiv, volumes)
        ) / denominator
        coefficients[column] = {
            "intercept": mean_volume - slope * mean_etiv,
            "slope": slope,
            "cn_training_rows": len(cn_rows),
        }
    return coefficients


def adjusted_row(
    row: dict[str, str],
    volume_columns: list[str],
    coefficients: dict[str, dict[str, float | int]],
    *,
    proportional: bool,
    fold: int,
    thickness_columns: list[str],
    asymmetry_columns: list[str],
) -> dict[str, str | int]:
    etiv = float(row["etiv_mm3"])
    result: dict[str, str | int] = {
        **{column: row[column] for column in (*IDENTIFIER_COLUMNS, LABEL_COLUMN)},
        "fold": fold,
        "etiv_mm3": row["etiv_mm3"],
    }
    for column in volume_columns:
        volume = float(row[column])
        if proportional:
            adjusted = volume / etiv
        else:
            fit = coefficients[column]
            adjusted = volume - (fit["intercept"] + fit["slope"] * etiv)
        result[adjusted_name(column, proportional)] = repr(adjusted)
    result.update({column: row[column] for column in (*thickness_columns, *asymmetry_columns)})
    return result


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_validation_folds(
    rows: list[dict[str, str]], n_splits: int, random_state: int,
    fold_assignments: Path | None,
) -> list[list[int]]:
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    if fold_assignments is not None:
        with fold_assignments.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"subject_id", "session_id", "validation_fold"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Fold assignments missing: {', '.join(sorted(missing))}.")
            assignments = {}
            for assignment in reader:
                key = (assignment["subject_id"], assignment["session_id"])
                if key in assignments:
                    raise ValueError(f"Duplicate fold assignment: {key}.")
                try:
                    fold = int(assignment["validation_fold"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid validation_fold for {key}.") from error
                if not 1 <= fold <= n_splits:
                    raise ValueError(f"validation_fold out of range for {key}.")
                assignments[key] = fold - 1
        expected = {(row["subject_id"], row["session_id"]) for row in rows}
        if set(assignments) != expected:
            raise ValueError("Fold assignments must match all input subject/session pairs exactly.")
        for index, row in enumerate(rows):
            folds[assignments[(row["subject_id"], row["session_id"])]].append(index)
    else:
        rng = random.Random(random_state)
        for label in ("CN", "AD"):
            indices = [index for index, row in enumerate(rows) if row[LABEL_COLUMN] == label]
            rng.shuffle(indices)
            for position, index in enumerate(indices):
                folds[position % n_splits].append(index)
    for fold, indices in enumerate(folds, start=1):
        labels = Counter(rows[index][LABEL_COLUMN] for index in indices)
        if not labels["CN"] or not labels["AD"]:
            raise ValueError(f"Validation fold {fold} must contain CN and AD rows.")
        indices.sort()
    return folds


def prepare(
    input_path: Path,
    output_path: Path,
    *,
    extended_data: bool = False,
    asymmetry: bool = False,
    proportional: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
    fold_assignments: Path | None = None,
) -> dict[str, object]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    source_columns = (
        EXTENDED_SOURCE_COLUMNS if extended_data else DEFAULT_SOURCE_COLUMNS
    )
    asymmetry_pairs = (
        (EXTENDED_ASYMMETRY_PAIRS if extended_data else DEFAULT_ASYMMETRY_PAIRS)
        if asymmetry
        else {}
    )
    required_columns = (*IDENTIFIER_COLUMNS, LABEL_COLUMN, *source_columns.values())

    with input_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header.")
        duplicates = sorted(name for name, count in Counter(reader.fieldnames).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate input columns: {', '.join(duplicates)}")
        missing = sorted(set(required_columns) - set(reader.fieldnames))
        if missing:
            if extended_data:
                detail = "Merge the FreeSurfer stats by session_id first."
            else:
                detail = "Add eTIV from FreeSurfer stats by session_id."
            raise ValueError(
                f"Required input columns are absent: {', '.join(missing)}. {detail}"
            )

        rows = []
        sessions = set()
        subjects = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Row {line_number} has more fields than the header.")
            for column in (*IDENTIFIER_COLUMNS, LABEL_COLUMN):
                if not row[column] or not row[column].strip():
                    raise ValueError(f"Row {line_number}: {column} is empty.")
            session = row["session_id"]
            if session in sessions:
                raise ValueError(f"Row {line_number}: duplicated session_id {session!r}.")
            sessions.add(session)
            subject = row["subject_id"]
            if subject in subjects:
                raise ValueError(
                    f"Row {line_number}: duplicated subject_id {subject!r}. "
                    "Run filter_supervised_labels before this script."
                )
            subjects.add(subject)
            if row[LABEL_COLUMN] not in {"CN", "AD"}:
                raise ValueError(
                    f"Row {line_number}: unexpected supervised_label "
                    f"{row[LABEL_COLUMN]!r}. Run filter_supervised_labels first."
                )

            selected = {column: row[column] for column in (*IDENTIFIER_COLUMNS, LABEL_COLUMN)}
            for target, source in source_columns.items():
                value = row[source]
                if value is None or not value.strip():
                    raise ValueError(f"Row {line_number}: {source} is empty.")
                try:
                    number = float(value)
                except ValueError as error:
                    raise ValueError(f"Row {line_number}: {source} is not numeric: {value!r}.") from error
                if not math.isfinite(number) or number <= 0:
                    raise ValueError(f"Row {line_number}: {source} must be finite and positive.")
                selected[target] = value
            for target, (left, right) in asymmetry_pairs.items():
                left_value = float(selected[left])
                right_value = float(selected[right])
                selected[target] = str(
                    (left_value - right_value) / (left_value + right_value)
                )
            rows.append(selected)

    if not rows:
        raise ValueError("Input CSV has no data rows.")
    counts = Counter(row[LABEL_COLUMN] for row in rows)
    if n_splits < 2 or min(counts.get("CN", 0), counts.get("AD", 0)) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} participants in each of CN and AD."
        )

    volume_columns = [
        column for column in source_columns
        if column != "etiv_mm3" and column.endswith("_mm3")
    ]
    thickness_columns = [column for column in source_columns if column.endswith("_thickness_mm")]
    output_columns = (
        *IDENTIFIER_COLUMNS, LABEL_COLUMN, "fold", "etiv_mm3",
        *(adjusted_name(column, proportional) for column in volume_columns),
        *thickness_columns, *asymmetry_pairs,
    )
    validation_folds = make_validation_folds(
        rows, n_splits, random_state, fold_assignments
    )
    fold_directory = output_path.parent / f"{output_path.stem}_folds"
    fold_outputs = []
    oof_rows: list[dict[str, str | int] | None] = [None] * len(rows)
    for fold, validation_indices in enumerate(validation_folds, start=1):
        held_out = set(validation_indices)
        train_indices = [index for index in range(len(rows)) if index not in held_out]
        coefficients = (
            {} if proportional else fit_cn_residuals(rows, train_indices, volume_columns)
        )
        train_rows = [
            adjusted_row(
                rows[index], volume_columns, coefficients, proportional=proportional,
                fold=fold, thickness_columns=thickness_columns,
                asymmetry_columns=list(asymmetry_pairs),
            )
            for index in train_indices
        ]
        validation_rows = [
            adjusted_row(
                rows[index], volume_columns, coefficients, proportional=proportional,
                fold=fold, thickness_columns=thickness_columns,
                asymmetry_columns=list(asymmetry_pairs),
            )
            for index in validation_indices
        ]
        for index, transformed in zip(validation_indices, validation_rows):
            if oof_rows[index] is not None:
                raise RuntimeError(f"Row {index} assigned to multiple validation folds.")
            oof_rows[index] = transformed
        fold_outputs.append((fold, train_rows, validation_rows, coefficients))
    if any(row is None for row in oof_rows):
        raise RuntimeError("Some rows were not assigned to a validation fold.")

    write_csv(output_path, output_columns, oof_rows)
    fold_files = []
    for fold, train_rows, validation_rows, coefficients in fold_outputs:
        train_path = fold_directory / f"fold_{fold:02d}_train.csv"
        validation_path = fold_directory / f"fold_{fold:02d}_validation.csv"
        write_csv(train_path, output_columns, train_rows)
        write_csv(validation_path, output_columns, validation_rows)
        coefficients_path = None
        if not proportional:
            coefficients_path = fold_directory / f"fold_{fold:02d}_cn_residual_fits.json"
            coefficients_path.write_text(json.dumps(coefficients, indent=2) + "\n", encoding="utf-8")
        fold_files.append({
            "fold": fold,
            "train": str(train_path.resolve()),
            "validation": str(validation_path.resolve()),
            "cn_residual_fits": str(coefficients_path.resolve()) if coefficients_path else None,
        })

    manifest = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": len(rows),
        "mode": "extended_data" if extended_data else "mask_volumes",
        "adjustment_method": "proportional" if proportional else "residual",
        "cross_validation": {
            "method": "provided_assignments" if fold_assignments else "stratified_subject_round_robin",
            "n_splits": n_splits,
            "shuffle": fold_assignments is None,
            "random_state": random_state,
            "group": "subject_id",
            "fold_assignments": str(fold_assignments.resolve()) if fold_assignments else None,
            "main_csv": "Each row is transformed as a held-out validation row, using that fold's CN training fit when residual adjustment is selected.",
        },
        "source_columns": source_columns,
        "adjusted_volume_columns": {
            adjusted_name(column, proportional): {
                "source": source_columns[column],
                "etiv_source": source_columns["etiv_mm3"],
                "method": "proportional" if proportional else "cn_training_fold_residual",
            }
            for column in volume_columns
        },
        "derived_columns": {
            target: {
                "left": left,
                "right": right,
                "formula": "(left - right) / (left + right)",
            }
            for target, (left, right) in asymmetry_pairs.items()
        },
        "volume_adjustment": (
            "volume / eTIV; no fitted parameters"
            if proportional else
            "For each fold and volume, fit volume = intercept + slope * eTIV on CN training rows only; output volume - predicted volume for both train and validation."
        ),
        "cortical_thickness": "Copied unchanged; never adjusted by eTIV.",
        "predictors": [
            *(adjusted_name(column, proportional) for column in volume_columns),
            *thickness_columns,
            *asymmetry_pairs,
        ],
        "fold_files": fold_files,
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extended-data",
        action="store_true",
        help="Require FreeSurfer aseg volumes, inferior lateral ventricles and cortical thickness instead of the eight mask volumes.",
    )
    parser.add_argument(
        "--asymmetry",
        action="store_true",
        help="Add (left - right) / (left + right) for each bilateral structure.",
    )
    parser.add_argument(
        "--proportional",
        action="store_true",
        help="Use volume / eTIV instead of the default CN-training-fold residual adjustment.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--fold-assignments", type=Path,
        help="Reuse a fold_assignments.csv containing subject_id, session_id and validation_fold.",
    )
    args = parser.parse_args()
    try:
        manifest = prepare(
            args.input,
            args.output,
            extended_data=args.extended_data,
            asymmetry=args.asymmetry,
            proportional=args.proportional,
            n_splits=args.n_splits,
            random_state=args.random_state,
            fold_assignments=args.fold_assignments,
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    print(
        f"Selected {manifest['rows']} rows and "
        f"{len(manifest['predictors']) + len(IDENTIFIER_COLUMNS) + 3} columns: "
        f"{args.output}"
    )
    print(f"Column mapping: {args.output.with_suffix('.manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
