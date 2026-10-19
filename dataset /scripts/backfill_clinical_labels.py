#!/usr/bin/env python3
"""Backfill clinical pairing and supervised labels into the consolidated CSV.

Context
-------
``clinical_radiomics_CONSOLIDATED_v2`` joins the radiomic/morphometric features
of 362 MRI sessions with clinical columns, but 183 of them carry no
``supervised_label``. This script recovers the missing pairing and labels by
running the **existing** production path - ``build_clinical_linkage`` followed by
``build_supervised_datasets`` - over the full MRI catalogue and the raw OASIS-3
clinical forms, then copying the result into the rows that lack it.

It adds no rule of its own. Policy, window and eligibility all come from
``supervised_labels.yaml`` through ``LabelPolicy``; nothing here relaxes the
clinical window or decides a diagnosis.

Guarantees
----------
* The input CSV is never modified; a new file is written.
* Only **empty** cells are filled. The single exception is
  ``clinical_master_found``, a join flag, which is moved ``False`` -> ``True``
  once the session has been found in the full-catalogue clinical master.
* Rows that already carry a ``supervised_label`` are byte-identical (checked).
* Radiomic and morphometric columns are never touched (checked).
* A session whose nearest clinical visit is outside the window is **not**
  labelled (category B); its probable label goes to a side file for review.
* A session on which the policy cannot decide stays unlabelled (category D).
* Console output carries counts and session identifiers only.

Usage
-----
::

    python scripts/backfill_clinical_labels.py \\
        --consolidated "clinical_radiomics_CONSOLIDATED_v2 (1).csv" \\
        --mri-catalog luanmantegazine_8_26_2026_23_7_46.csv \\
        --d1 diagnostic/OASIS3_UDSd1_diagnoses.csv \\
        --b4 diagnostic/OASIS3_UDSb4_cdr.csv \\
        --c1 diagnostic/OASIS3_UDSc1_cognitive_assessments.csv \\
        --crosscheck-labels acquisition_labels/supervised_radiomics_sessions.csv \\
        --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oasis_radiomics.clinical.labels import (  # noqa: E402
    EXCLUSION_OUTSIDE_WINDOW,
    STATUS_LABELLED,
    LabelPolicy,
)
from oasis_radiomics.clinical.readers import read_b4, read_c1, read_d1  # noqa: E402
from oasis_radiomics.clinical_dataset import (  # noqa: E402
    build_clinical_linkage,
    write_linkage_outputs,
)
from oasis_radiomics.supervised_dataset import (  # noqa: E402
    DEFAULT_CLINICAL_WINDOW_DAYS,
    build_supervised_datasets,
)

logger = logging.getLogger("backfill_clinical_labels")

# --- taxonomy --------------------------------------------------------------
CAT_A = "A"  # subject has no row in any clinical file
CAT_B = "B"  # nearest clinical visit is outside the window
CAT_C = "C"  # valid visit exists; pairing was absent from v2; policy decides
CAT_D = "D"  # valid pairing; the existing policy returns no label
CAT_E = "E"  # anything else, described in the detail column
CATEGORIES = (CAT_A, CAT_B, CAT_C, CAT_D, CAT_E)

NOTE_COLUMN = "label_backfill_note"
NOTE_PREFIX = "backfill_v1:"
MASTER_FLAG = "clinical_master_found"

LABEL_COLUMN = "supervised_label"
CLASS_ORDER = ("CN", "MCI", "AD", "IMPAIRED_NOT_MCI", "OTHER_DEMENTIA")

#: Pairing columns written for categories A and B (no label, no raw diagnosis).
PAIRING_COLUMNS = (
    "clinical_session_id",
    "clinical_day",
    "clinical_mri_gap_days",
    "clinical_mri_abs_gap_days",
    "clinical_match_found",
    "clinical_match_valid",
    "clinical_match_reason",
    "clinical_match_ambiguous",
    "clinical_source",
    "age_at_clinical_visit",
)
#: Demographics taken from the MRI catalogue, filled only where empty.
DEMOGRAPHIC_COLUMNS = ("mri_day", "age_at_mri", "scanner", "sex")
#: Eligibility columns written for category B.
B_ELIGIBILITY_COLUMNS = ("training_eligible", "training_exclusion_reason")

DEMOGRAPHIC_SOURCES = {
    "mri_day": "MRI catalogue: day suffix of 'Label' (_dNNNN); the 'Date' column is empty",
    "age_at_mri": "MRI catalogue: 'Age'",
    "scanner": "MRI catalogue: 'Scanner'",
    "sex": "MRI catalogue: 'M/F' (already complete in v2; used only as a consistency check)",
}

#: Prefixes of columns that must never change (radiomic / morphometric / merge).
PROTECTED_PREFIXES = ("original_", "diagnostics_", "fs_")
PROTECTED_COLUMNS = frozenset(
    {
        "diagnosis_label",
        "primary_diagnosis",
        "source_files",
        "merge_conflict_count",
        "merge_conflicts",
        "days_from_reference",
    }
)


class BackfillError(RuntimeError):
    """Raised when an invariant of the backfill is violated."""


# ---------------------------------------------------------------------------
# text-level CSV IO (cell text is preserved exactly)
# ---------------------------------------------------------------------------
def read_text_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV keeping every cell as its original text."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [dict(zip(header, record)) for record in reader]
    return header, rows


def serialise_table(header: Sequence[str], rows: Iterable[Mapping[str, str]]) -> bytes:
    """Serialise exactly as pandas does: ``\\n`` terminators, minimal quoting."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow([row.get(column, "") for column in header])
    return buffer.getvalue().encode("utf-8")


def assert_lossless_roundtrip(path: Path) -> None:
    """Refuse to continue if rewriting the source would alter any byte."""
    header, rows = read_text_table(path)
    if serialise_table(header, rows) != Path(path).read_bytes():
        raise BackfillError(
            f"{path} does not survive a csv round-trip byte-for-byte; rewriting it "
            "could silently change unlabelled-to-labelled-neutral rows."
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# style mirroring
# ---------------------------------------------------------------------------
_FLOAT0 = re.compile(r"-?\d+\.0")
_INT = re.compile(r"-?\d+")


def column_style(existing_cells: Iterable[str]) -> str:
    """``float0`` when every non-empty cell is an integral float like ``129.0``.

    pandas writes an integer column as ``129.0`` whenever the column contained
    blanks, which is the case for the day/gap columns of v2. New cells must follow
    the same convention or the column would mix ``129`` and ``129.0``.
    """
    cells = [cell for cell in existing_cells if cell != ""]
    if cells and all(_FLOAT0.fullmatch(cell) for cell in cells):
        return "float0"
    return "raw"


def mirror_style(text: str, style: str | None) -> str:
    """Render ``text`` in the numeric style of its destination column."""
    if style == "float0" and _INT.fullmatch(text):
        return f"{text}.0"
    return text


# ---------------------------------------------------------------------------
# classification of the unlabelled sessions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Triage:
    """Category assigned to one unlabelled session."""

    session_id: str
    subject_id: str
    category: str
    detail: str


def classify_unlabelled(
    v2_row: Mapping[str, str],
    fresh_row: Mapping[str, str] | None,
    subject_in_clinical_files: bool,
    window_days: int,
) -> tuple[str, str]:
    """Assign exactly one category to an unlabelled session.

    ``fresh_row`` is the session's row from the full-catalogue production run.
    The test order is A, B, then the two valid-pairing outcomes:

    * the policy **does not decide** (``label_status`` != ``labelled``) -> D,
      whether or not v2 had a pairing;
    * the policy decides and v2 lacked the pairing -> C;
    * the policy decides and v2 already had the pairing -> E (the label step was
      simply never run on this session).
    """
    if fresh_row is None:
        return CAT_E, "session absent from the MRI catalogue run"

    found = fresh_row.get("clinical_match_found") == "True"
    valid = fresh_row.get("clinical_match_valid") == "True"

    if not found:
        if not subject_in_clinical_files:
            return CAT_A, "subject has no row in D1, B4 or C1"
        return CAT_E, (
            "subject appears in the clinical files but has no D1/B4 visit "
            f"(reason={fresh_row.get('clinical_match_reason')})"
        )

    if not valid:
        gap = fresh_row.get("clinical_mri_abs_gap_days", "")
        return CAT_B, f"nearest clinical visit is {gap} days from the MRI (window {window_days})"

    if fresh_row.get("label_status") != STATUS_LABELLED:
        return CAT_D, (
            f"rule {fresh_row.get('label_rule_id')} returns {fresh_row.get('label_status')}: "
            f"{fresh_row.get('label_reason')}"
        )

    gap = fresh_row.get("clinical_mri_abs_gap_days", "")
    if v2_row.get("clinical_match_found", "") == "":
        return CAT_C, (
            "pairing absent from v2 (its clinical master did not contain the session); "
            f"visit found in the full-catalogue run at {gap} days"
        )
    return CAT_E, (
        "valid pairing already present in v2 but the label step never ran; "
        f"the policy decides (visit at {gap} days)"
    )


# ---------------------------------------------------------------------------
# cell planning
# ---------------------------------------------------------------------------
def allowed_columns(category: str, fresh_columns: Sequence[str]) -> list[str]:
    """Columns a category may fill, in the order of the fresh table."""
    if category == CAT_A:
        wanted = set(PAIRING_COLUMNS) | set(DEMOGRAPHIC_COLUMNS)
    elif category == CAT_B:
        wanted = set(PAIRING_COLUMNS) | set(DEMOGRAPHIC_COLUMNS) | set(B_ELIGIBILITY_COLUMNS)
    else:  # C, D, E: every label/pairing column the production run produces
        wanted = set(fresh_columns)
        if category == CAT_D:
            wanted.discard(LABEL_COLUMN)  # the policy did not decide: no label
    return [column for column in fresh_columns if column in wanted]


@dataclass
class CellPlan:
    """What would change in one row."""

    updates: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    conflict_values: list[tuple[str, str, str]] = field(default_factory=list)  # (column, kept, fresh)


def plan_cells(
    category: str,
    v2_row: Mapping[str, str],
    fresh_row: Mapping[str, str],
    fresh_columns: Sequence[str],
    styles: Mapping[str, str],
) -> CellPlan:
    """Decide which cells of ``v2_row`` to fill.

    Only empty cells are filled. A non-empty cell that disagrees with the fresh
    value is **kept** and listed in ``conflicts`` so the report can show it.
    """
    plan = CellPlan()
    for column in allowed_columns(category, fresh_columns):
        if column not in v2_row:
            continue
        new = mirror_style(fresh_row.get(column, ""), styles.get(column))
        if new == "":
            continue
        old = v2_row[column]
        if old == "":
            plan.updates[column] = new
        elif old != new:
            plan.conflicts.append(column)
            plan.conflict_values.append((column, old, new))

    # Join flag: the session is now known to be in the full-catalogue master.
    if MASTER_FLAG in v2_row and v2_row[MASTER_FLAG] != "True":
        plan.updates[MASTER_FLAG] = "True"
    return plan


# ---------------------------------------------------------------------------
# counts (computed from tables, never by hand)
# ---------------------------------------------------------------------------
def compute_counts(rows: Sequence[Mapping[str, str]]) -> dict:
    """Class distribution by session and by subject, all labelled and eligible."""
    labelled = [row for row in rows if row.get(LABEL_COLUMN, "") != ""]
    eligible = [row for row in labelled if row.get("training_eligible") == "True"]

    def tally(selection: Sequence[Mapping[str, str]]) -> tuple[Counter, Counter]:
        sessions = Counter(row[LABEL_COLUMN] for row in selection)
        subjects: dict[str, set[str]] = defaultdict(set)
        for row in selection:
            subjects[row[LABEL_COLUMN]].add(row["subject_id"])
        return sessions, Counter({label: len(ids) for label, ids in subjects.items()})

    def multi_class(selection: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
        per_subject: dict[str, set[str]] = defaultdict(set)
        for row in selection:
            per_subject[row["subject_id"]].add(row[LABEL_COLUMN])
        return {s: tuple(sorted(c)) for s, c in per_subject.items() if len(c) > 1}

    all_sessions, all_subjects = tally(labelled)
    el_sessions, el_subjects = tally(eligible)
    return {
        "rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "labelled_rows": len(labelled),
        "unlabelled_rows": len(rows) - len(labelled),
        "eligible_rows": len(eligible),
        "eligible_subjects": len({row["subject_id"] for row in eligible}),
        "labelled_sessions": dict(all_sessions),
        "labelled_subjects": dict(all_subjects),
        "eligible_sessions": dict(el_sessions),
        "eligible_subjects_by_class": dict(el_subjects),
        "multi_class_eligible": multi_class(eligible),
        "multi_class_labelled": multi_class(labelled),
    }


# ---------------------------------------------------------------------------
# production run (reuse, no reimplementation)
# ---------------------------------------------------------------------------
@dataclass
class FreshRun:
    """Output of the production path over the full catalogue."""

    columns: list[str]
    rows: dict[str, dict[str, str]]
    sessions_csv: Path
    policy: LabelPolicy
    window_days: int
    subjects_with_clinical_rows: set[str]


def run_production_path(args: argparse.Namespace, work_dir: Path) -> FreshRun:
    """Run ``clinical-link`` then ``supervised-labels`` exactly as the CLI does."""
    policy = LabelPolicy.load(args.label_policy)
    window = int(args.clinical_window_days)

    linkage = build_clinical_linkage(
        mri_catalog=args.mri_catalog,
        d1_path=args.d1,
        b4_path=args.b4,
        c1_path=args.c1,
        window_days=window,
    )
    outputs = write_linkage_outputs(linkage, work_dir / "clinical")
    _, sup_outputs = build_supervised_datasets(
        clinical_radiomics=outputs["clinical_imaging_master"],
        clinical_visits=outputs["clinical_visits"],
        output_dir=work_dir / "supervised",
        policy=policy,
        window_days=window,
    )
    sessions_csv = sup_outputs["supervised_radiomics_sessions"]
    columns, rows = read_text_table(sessions_csv)

    subjects: set[str] = set()
    for reader, path in ((read_d1, args.d1), (read_b4, args.b4), (read_c1, args.c1)):
        if path:
            subjects.update(record.subject_id for record in reader(Path(path)))

    return FreshRun(
        columns=columns,
        rows={row["session_id"]: row for row in rows},
        sessions_csv=sessions_csv,
        policy=policy,
        window_days=window,
        subjects_with_clinical_rows=subjects,
    )


# ---------------------------------------------------------------------------
# backfill
# ---------------------------------------------------------------------------
@dataclass
class BackfillResult:
    """Everything the backfill computed (nothing is written yet)."""

    columns: list[str]
    rows: list[dict[str, str]]
    triage: list[Triage]
    changed: dict[str, str]  # session_id -> category
    updated_cells: Counter
    conflicts_kept: Counter
    conflict_values: dict[str, list[tuple[str, str]]]
    master_flag_overwrites: int
    reproduction_mismatch_cells: int
    reproduction_mismatch_columns: Counter
    candidates: list[dict[str, str]]
    unmatched_catalogue: list[str]


def styles_from(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> dict[str, str]:
    """Numeric style of every column, inferred from its existing cells."""
    return {column: column_style(row.get(column, "") for row in rows) for column in columns}


def apply_backfill(
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    fresh: FreshRun,
) -> BackfillResult:
    """Compute the v3 table without touching ``rows``."""
    styles = styles_from(rows, [c for c in fresh.columns if c in header])
    new_rows = [dict(row) for row in rows]

    # How faithfully does the production path reproduce the rows already labelled?
    # This also validates the style mirroring: any residue is reported.
    repro_cells = 0
    repro_columns: Counter = Counter()
    for row in new_rows:
        if row.get(LABEL_COLUMN, "") == "":
            continue
        fresh_row = fresh.rows.get(row["session_id"])
        if fresh_row is None:
            repro_cells += 1
            repro_columns["<session absent from fresh run>"] += 1
            continue
        for column in fresh.columns:
            if column not in row:
                continue
            if mirror_style(fresh_row.get(column, ""), styles.get(column)) != row[column]:
                repro_cells += 1
                repro_columns[column] += 1

    triage: list[Triage] = []
    changed: dict[str, str] = {}
    updated: Counter = Counter()
    conflicts: Counter = Counter()
    conflict_values: dict[str, list[tuple[str, str]]] = defaultdict(list)
    master_overwrites = 0
    candidates: list[dict[str, str]] = []
    unmatched: list[str] = []

    for row in new_rows:
        if row.get(LABEL_COLUMN, "") != "":
            continue
        session_id = row["session_id"]
        fresh_row = fresh.rows.get(session_id)
        if fresh_row is None:
            unmatched.append(session_id)
        category, detail = classify_unlabelled(
            row,
            fresh_row,
            row["subject_id"] in fresh.subjects_with_clinical_rows,
            fresh.window_days,
        )
        triage.append(Triage(session_id, row["subject_id"], category, detail))
        if fresh_row is None:
            continue

        plan = plan_cells(category, row, fresh_row, fresh.columns, styles)
        if MASTER_FLAG in plan.updates and row.get(MASTER_FLAG, "") == "False":
            master_overwrites += 1
        for column, value in plan.updates.items():
            row[column] = value
            updated[column] += 1
        for column in plan.conflicts:
            conflicts[column] += 1
        for column, kept, other in plan.conflict_values:
            conflict_values[column].append((kept, other))
        if plan.updates:
            changed[session_id] = category

        if category == CAT_B:
            candidates.append(
                {
                    "session_id": session_id,
                    "subject_id": row["subject_id"],
                    "mri_day": fresh_row.get("mri_day", ""),
                    "nearest_clinical_day": fresh_row.get("clinical_day", ""),
                    "signed_gap_days": fresh_row.get("clinical_mri_gap_days", ""),
                    "abs_gap_days": fresh_row.get("clinical_mri_abs_gap_days", ""),
                    "window_days": str(fresh.window_days),
                    "nearest_dx1": fresh_row.get("dx1", ""),
                    "probable_label": fresh_row.get(LABEL_COLUMN, ""),
                    "label_rule_id": fresh_row.get("label_rule_id", ""),
                    "note": "NOT applied: outside the clinical window",
                }
            )

    output_header = list(header) + [NOTE_COLUMN]
    for row in new_rows:
        category = changed.get(row["session_id"])
        row[NOTE_COLUMN] = f"{NOTE_PREFIX}{category}" if category else ""

    return BackfillResult(
        columns=output_header,
        rows=new_rows,
        triage=triage,
        changed=changed,
        updated_cells=updated,
        conflicts_kept=conflicts,
        conflict_values=dict(conflict_values),
        master_flag_overwrites=master_overwrites,
        reproduction_mismatch_cells=repro_cells,
        reproduction_mismatch_columns=repro_columns,
        candidates=candidates,
        unmatched_catalogue=unmatched,
    )


def describe_conflicts(pairs: Sequence[tuple[str, str]]) -> str:
    """Aggregate description of kept-vs-fresh values for one column.

    Numeric columns get the median and maximum absolute difference; others get
    the counted ``kept -> fresh`` pairs. Never lists individual rows.
    """
    try:
        diffs = sorted(abs(float(kept) - float(other)) for kept, other in pairs)
    except ValueError:
        counted = Counter(f"{kept!r} -> {other!r}" for kept, other in pairs)
        return "; ".join(f"{k} x{n}" for k, n in counted.most_common(6))
    median = diffs[len(diffs) // 2] if len(diffs) % 2 else (diffs[len(diffs) // 2 - 1] + diffs[len(diffs) // 2]) / 2
    return f"absolute difference: median {median:.2f}, max {diffs[-1]:.2f}"


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def is_protected(column: str) -> bool:
    return column.startswith(PROTECTED_PREFIXES) or column in PROTECTED_COLUMNS or (
        column.startswith("mask_") and column != "mask_voxels"
    )


def verify_backfill(
    v2_header: Sequence[str],
    v2_rows: Sequence[Mapping[str, str]],
    v3_header: Sequence[str],
    v3_rows: Sequence[Mapping[str, str]],
    window_days: int,
) -> dict[str, int]:
    """Check every acceptance invariant on the two tables. Raises on violation.

    Works on the tables as *re-read from disk*, so it validates what was written.
    """
    if len(v3_rows) != len(v2_rows):
        raise BackfillError(f"row count changed: {len(v2_rows)} -> {len(v3_rows)}")
    if list(v3_header) != list(v2_header) + [NOTE_COLUMN]:
        raise BackfillError("v3 columns are not v2 columns plus the backfill note")

    regression_rows = 0
    changed_cells = 0
    for old, new in zip(v2_rows, v3_rows):
        if old["session_id"] != new["session_id"]:
            raise BackfillError("row order changed")
        labelled = old.get(LABEL_COLUMN, "") != ""
        note = new[NOTE_COLUMN]

        if labelled:
            regression_rows += 1
            if note != "":
                raise BackfillError(f"labelled row {old['session_id']} carries a note")
            diffs = [c for c in v2_header if old[c] != new[c]]
            if diffs:
                raise BackfillError(
                    f"labelled row {old['session_id']} changed in {len(diffs)} column(s)"
                )
            continue

        for column in v2_header:
            if old[column] == new[column]:
                continue
            changed_cells += 1
            if is_protected(column):
                raise BackfillError(f"protected column {column} changed in {old['session_id']}")
            if old[column] != "" and not (column == MASTER_FLAG and new[column] == "True"):
                raise BackfillError(
                    f"non-empty cell {column} overwritten in {old['session_id']}"
                )
        if any(old[c] != new[c] for c in v2_header) != (note != ""):
            raise BackfillError(f"note/change mismatch for {old['session_id']}")

        if new[LABEL_COLUMN] != "" and note != "":
            gap = new.get("clinical_mri_abs_gap_days", "")
            if new.get("clinical_match_valid") != "True" or gap == "" or float(gap) > window_days:
                raise BackfillError(f"{old['session_id']} labelled outside the clinical window")

    return {"regression_rows_identical": regression_rows, "changed_cells": changed_cells}


# ---------------------------------------------------------------------------
# cross-check against the repository's own full-catalogue output
# ---------------------------------------------------------------------------
def crosscheck_labels(fresh: FreshRun, reference: Path) -> dict:
    """Compare the fresh run with an earlier output of the same pipeline."""
    same_bytes = sha256(fresh.sessions_csv) == sha256(reference)
    _, ref_rows = read_text_table(reference)
    differing = 0
    for row in ref_rows:
        other = fresh.rows.get(row["session_id"])
        if other is None or any(other.get(c) != v for c, v in row.items() if c in other):
            differing += 1
    return {
        "reference": str(reference),
        "byte_identical": same_bytes,
        "reference_rows": len(ref_rows),
        "rows_with_any_difference": differing,
    }


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_simple(path: Path, header: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    Path(path).write_bytes(serialise_table(header, rows))


def _fmt_counter(counter: Mapping[str, int], order: Sequence[str] = CLASS_ORDER) -> list[tuple[str, int]]:
    keys = [k for k in order if k in counter] + sorted(k for k in counter if k not in order)
    return [(k, counter[k]) for k in keys]


def build_report(
    *,
    args: argparse.Namespace,
    fresh: FreshRun,
    result: BackfillResult,
    verification: Mapping[str, int],
    v2_counts: Mapping,
    v3_counts: Mapping,
    crosscheck: Mapping | None,
    observed_max_gap: float | None,
    v2_outside_window_labelled: int,
    primary_diagnosis_table: Mapping[str, Mapping[str, int]],
) -> str:
    """Markdown report. Counts only; no clinical rows."""
    by_category = Counter(t.category for t in result.triage)
    lines: list[str] = []
    add = lines.append

    add("# Backfill report - clinical pairing and supervised labels\n")
    add(f"Policy `{fresh.policy.version}` (primary source {fresh.policy.primary_source}), "
        f"clinical window **{fresh.window_days} days**, training labels "
        f"{list(fresh.policy.training_labels)}.\n")

    add("## 1. Policy applied (extracted from the code, not re-implemented)\n")
    add("- **Window.** `training_eligibility()` applies the temporal gate first: a session is "
        "ineligible when `clinical_match_valid` is false or `|gap| > window`.")
    add("- **Then** conflicting, missing and unmapped diagnoses are excluded.")
    add("- **Classes.** `CN`, `MCI`, `AD` are training labels. `IMPAIRED_NOT_MCI`, `OTHER_DEMENTIA` "
        "and `DEMENTIA_UNKNOWN_ETIOLOGY` receive a label but are never eligible "
        "(`impaired_not_mci`, `other_dementia`, `dementia_unknown_etiology`).")
    add("- **MCI** comes from the D1 indicators MCIAMEM, MCIAPLUS, MCINON1, MCINON2; "
        "B4 `dx1` is only an independent audit (`b4_label`, `b4_agreement`).")
    add(f"- **Largest |gap| among the eligible rows of v2:** "
        f"{observed_max_gap if observed_max_gap is not None else 'n/a'} days "
        f"(window {fresh.window_days}).")
    add(f"- **Existing behaviour kept untouched:** the policy also labels sessions outside the "
        f"window (marking them ineligible); v2 contains **{v2_outside_window_labelled}** such rows. "
        "They are not changed. For the backfill the instruction is stricter: category B is "
        "**not** labelled.\n")

    add("## 2. Triage of the unlabelled sessions\n")
    add("| category | meaning | sessions |")
    add("|---|---|---:|")
    meanings = {
        CAT_A: "subject absent from every clinical file",
        CAT_B: "nearest visit outside the window - left unlabelled",
        CAT_C: "valid visit; pairing absent from v2; policy labels it",
        CAT_D: "valid pairing; the policy returns no label - left unlabelled",
        CAT_E: "other (see `detalhe` in the triage file)",
    }
    for cat in CATEGORIES:
        add(f"| {cat} | {meanings[cat]} | {by_category.get(cat, 0)} |")
    add(f"| **total** | | **{sum(by_category.values())}** |\n")
    add("Category **E** here means a valid pairing was already present in v2 but the label step "
        "never ran; the policy decides, so it was applied (note `backfill_v1:E`).\n")

    add("## 3. What changed\n")
    add(f"- Rows changed: **{len(result.changed)}** of {v3_counts['rows']} "
        f"(`{NOTE_COLUMN}` is set only on these).")
    add(f"- Rows labelled by the backfill: **{sum(1 for r in result.rows if r[NOTE_COLUMN] and r[LABEL_COLUMN])}**.")
    add(f"- `{MASTER_FLAG}` set to True on rows now found in the full-catalogue master "
        f"(**{result.master_flag_overwrites}** overwrote an explicit `False`).")
    add("\nCells filled, by column (top 12):\n")
    add("| column | rows filled |")
    add("|---|---:|")
    for column, count in result.updated_cells.most_common(12):
        add(f"| `{column}` | {count} |")
    add("\nSources of the demographic fields (filled only where empty):\n")
    for column, source in DEMOGRAPHIC_SOURCES.items():
        add(f"- `{column}`: {source}")
    if result.conflicts_kept:
        add("\n### Non-empty cells that disagree with the fresh value (kept, not overwritten)\n")
        add(f"{sum(result.conflicts_kept.values())} cells in {len(result.conflicts_kept)} columns. "
            "The rule was to fill empty cells only, so v2 was kept in every case:\n")
        add("| column | cells | what differs |")
        add("|---|---:|---|")
        for column, count in result.conflicts_kept.most_common():
            add(f"| `{column}` | {count} | {describe_conflicts(result.conflict_values.get(column, []))} |")
        add("\nReading: `scanner` is formatting only (`3T` vs `3.0T`). `ad_etiology` is a stale "
            "`UNKNOWN` left by the earlier clinical merge, where the policy gives `AD`/`NON_AD`; "
            "the row's `ad_etiology_role` and `supervised_label` already reflect the policy. "
            "`age_at_mri` is not formatting: in these rows v2 holds fractional ages that track "
            "the age at a *clinical visit*, whereas the catalogue gives integer MRI ages, and the "
            "other rows of the file use the catalogue. Overwriting these cells is a decision for "
            "the data owner; it needs a deliberate re-run.")
    else:
        add("\nNo non-empty cell disagreed with the fresh value.")
    add("")

    add("## 4. Class distribution before and after\n")
    add("### Eligible rows only (`training_eligible == True`)\n")
    add("| class | sessions v2 | sessions v3 | subjects v2 | subjects v3 |")
    add("|---|---:|---:|---:|---:|")
    classes = list(CLASS_ORDER)
    for cls in classes:
        add(f"| {cls} | {v2_counts['eligible_sessions'].get(cls, 0)} | "
            f"{v3_counts['eligible_sessions'].get(cls, 0)} | "
            f"{v2_counts['eligible_subjects_by_class'].get(cls, 0)} | "
            f"{v3_counts['eligible_subjects_by_class'].get(cls, 0)} |")
    add(f"| **total** | **{v2_counts['eligible_rows']}** | **{v3_counts['eligible_rows']}** | "
        f"**{v2_counts['eligible_subjects']}** | **{v3_counts['eligible_subjects']}** |\n")
    add("`IMPAIRED_NOT_MCI` and `OTHER_DEMENTIA` never appear above because the policy makes them "
        "ineligible; they are shown below.\n")
    add("### All labelled rows (eligible or not)\n")
    add("| class | sessions v2 | sessions v3 | subjects v2 | subjects v3 |")
    add("|---|---:|---:|---:|---:|")
    for cls in sorted(set(v2_counts["labelled_sessions"]) | set(v3_counts["labelled_sessions"]),
                      key=lambda c: (classes.index(c) if c in classes else 99, c)):
        add(f"| {cls} | {v2_counts['labelled_sessions'].get(cls, 0)} | "
            f"{v3_counts['labelled_sessions'].get(cls, 0)} | "
            f"{v2_counts['labelled_subjects'].get(cls, 0)} | "
            f"{v3_counts['labelled_subjects'].get(cls, 0)} |")
    add(f"| **labelled rows** | **{v2_counts['labelled_rows']}** | **{v3_counts['labelled_rows']}** | | |")
    add(f"| unlabelled rows | {v2_counts['unlabelled_rows']} | {v3_counts['unlabelled_rows']} | | |\n")

    add("## 5. Subjects with more than one class\n")
    for name, key in (("eligible rows", "multi_class_eligible"), ("all labelled rows", "multi_class_labelled")):
        for tag, counts in (("v2", v2_counts), ("v3", v3_counts)):
            multi = counts[key]
            combos = Counter(multi.values())
            add(f"- {name}, {tag}: **{len(multi)}** subject(s)"
                + (" - " + "; ".join(f"{'+'.join(k)} x{n}" for k, n in combos.most_common()) if combos else ""))
    add("\nSubject identifiers (eligible rows, v3): "
        + (", ".join(sorted(v3_counts["multi_class_eligible"])) or "none") + "\n")

    add("## 6. Regression test\n")
    add(f"- Rows with a `supervised_label` in v2: **{verification['regression_rows_identical']}**; "
        f"**all identical, column by column** in v3 (checked on the re-read file).")
    add("- Only empty cells were filled on the other rows; no radiomic, morphometric or "
        "merge-provenance column changed (checked).")
    add("- No row labelled by the backfill lies outside the clinical window (checked).")
    add(f"- The production path reproduces the {verification['regression_rows_identical']} "
        f"already-labelled rows with **{result.reproduction_mismatch_cells}** differing cell(s)"
        + (" in " + ", ".join(f"`{c}` x{n}" for c, n in result.reproduction_mismatch_columns.most_common(6))
           if result.reproduction_mismatch_cells else " (after mirroring the numeric style of each column)") + ".")
    if crosscheck:
        add(f"- Cross-check against `{Path(crosscheck['reference']).name}` (same pipeline, earlier run): "
            f"byte-identical = **{crosscheck['byte_identical']}**, "
            f"rows with any difference = {crosscheck['rows_with_any_difference']} "
            f"of {crosscheck['reference_rows']}.")
    add("")

    add("## 7. Independent field `primary_diagnosis` (informational, not used)\n")
    add("`primary_diagnosis` comes from an earlier merge and is independent of this policy. "
        "It is **not** used and not changed. Policy label (rows) vs that field, for the sessions "
        "that received a label:\n")
    add("| policy label | " + " | ".join(sorted({k for v in primary_diagnosis_table.values() for k in v})) + " |")
    cols = sorted({k for v in primary_diagnosis_table.values() for k in v})
    add("|---|" + "---:|" * len(cols))
    for label, counts in sorted(primary_diagnosis_table.items()):
        add(f"| {label} | " + " | ".join(str(counts.get(c, 0)) for c in cols) + " |")
    add("")

    add("## 8. Rows that changed\n")
    by_cat_ids: dict[str, list[str]] = defaultdict(list)
    for sid, cat in result.changed.items():
        by_cat_ids[cat].append(sid)
    for cat in CATEGORIES:
        ids = sorted(by_cat_ids.get(cat, []))
        if ids:
            add(f"**{cat}** ({len(ids)}): " + ", ".join(ids) + "\n")
    unchanged_unlabelled = [t.session_id for t in result.triage if t.session_id not in result.changed]
    if unchanged_unlabelled:
        add(f"Unlabelled and **unchanged** ({len(unchanged_unlabelled)}): " + ", ".join(sorted(unchanged_unlabelled)) + "\n")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consolidated", type=Path, required=True, help="v2 consolidated CSV (never modified).")
    parser.add_argument("--mri-catalog", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--b4", type=Path, required=True)
    parser.add_argument("--c1", type=Path, default=None)
    parser.add_argument("--label-policy", type=Path, default=None, help="supervised_labels.yaml")
    parser.add_argument("--clinical-window-days", type=int, default=DEFAULT_CLINICAL_WINDOW_DAYS)
    parser.add_argument("--crosscheck-labels", type=Path, default=None,
                        help="Earlier supervised_radiomics_sessions.csv of the full catalogue.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--output-name", default="clinical_radiomics_CONSOLIDATED_v3.csv")
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch directory (default: temporary).")
    parser.add_argument("--dry-run", action="store_true", help="Compute and verify; write nothing to --output-dir.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for noisy in ("oasis_radiomics",):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    assert_lossless_roundtrip(args.consolidated)
    header, rows = read_text_table(args.consolidated)

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="backfill_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    fresh = run_production_path(args, work_dir)

    missing = [c for c in (*PAIRING_COLUMNS, *DEMOGRAPHIC_COLUMNS, LABEL_COLUMN) if c not in fresh.columns]
    if missing:
        raise BackfillError(f"production output lacks expected columns: {missing}")

    result = apply_backfill(header, rows, fresh)
    if result.unmatched_catalogue:
        raise BackfillError(
            f"{len(result.unmatched_catalogue)} unlabelled session(s) are absent from the catalogue run: "
            + ", ".join(result.unmatched_catalogue[:10])
        )

    by_category = Counter(t.category for t in result.triage)
    logger.info("Unlabelled sessions triaged: %d", len(result.triage))
    for cat in CATEGORIES:
        logger.info("  category %s: %d", cat, by_category.get(cat, 0))
    logger.info("Rows that would change: %d | cells filled: %d", len(result.changed), sum(result.updated_cells.values()))
    logger.info("Reproduction of the already-labelled rows by the production path: %d differing cell(s)",
                result.reproduction_mismatch_cells)

    out_dir = Path(args.output_dir)
    v3_path = out_dir / args.output_name
    if args.dry_run:
        verify_backfill(header, rows, result.columns, result.rows, fresh.window_days)
        logger.info("Dry run: all invariants hold on the in-memory table; nothing written to %s", out_dir)
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    v3_path.write_bytes(serialise_table(result.columns, result.rows))

    triage_path = out_dir / "sem_rotulo_triagem.csv"
    write_simple(triage_path, ("session_id", "subject_id", "categoria", "detalhe"),
                 [{"session_id": t.session_id, "subject_id": t.subject_id, "categoria": t.category,
                   "detalhe": t.detail} for t in result.triage])
    candidates_path = out_dir / "candidatos_fora_da_janela.csv"
    write_simple(candidates_path,
                 ("session_id", "subject_id", "mri_day", "nearest_clinical_day", "signed_gap_days",
                  "abs_gap_days", "window_days", "nearest_dx1", "probable_label", "label_rule_id", "note"),
                 result.candidates)

    # Everything below is computed from the files as written.
    v2_header, v2_rows = read_text_table(args.consolidated)
    v3_header, v3_rows = read_text_table(v3_path)
    verification = verify_backfill(v2_header, v2_rows, v3_header, v3_rows, fresh.window_days)
    v2_counts, v3_counts = compute_counts(v2_rows), compute_counts(v3_rows)
    if v3_counts != compute_counts(result.rows):
        raise BackfillError("counts recomputed from the written v3 differ from the in-memory table")

    eligible_gaps = [float(r["clinical_mri_abs_gap_days"]) for r in v2_rows
                     if r.get("training_eligible") == "True" and r.get("clinical_mri_abs_gap_days")]
    outside_labelled = sum(1 for r in v2_rows if r.get(LABEL_COLUMN) and
                           r.get("training_exclusion_reason") == EXCLUSION_OUTSIDE_WINDOW)

    primary: dict[str, Counter] = defaultdict(Counter)
    for row in v3_rows:
        if row[NOTE_COLUMN] and row[LABEL_COLUMN]:
            primary[row[LABEL_COLUMN]][row.get("primary_diagnosis") or "(empty)"] += 1

    crosscheck = crosscheck_labels(fresh, args.crosscheck_labels) if args.crosscheck_labels else None
    report = build_report(
        args=args, fresh=fresh, result=result, verification=verification,
        v2_counts=v2_counts, v3_counts=v3_counts, crosscheck=crosscheck,
        observed_max_gap=max(eligible_gaps) if eligible_gaps else None,
        v2_outside_window_labelled=outside_labelled,
        primary_diagnosis_table={k: dict(v) for k, v in primary.items()},
    )
    (out_dir / "backfill_report.md").write_text(report, encoding="utf-8")

    logger.info("Wrote %s (%d rows x %d columns)", v3_path, len(v3_rows), len(v3_header))
    logger.info("Regression: %d already-labelled rows identical", verification["regression_rows_identical"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
