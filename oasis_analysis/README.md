# OASIS-3 cohort analysis

This folder contains the downstream OASIS-3 cohort preparation and demographic
baseline analysis. It is part of the parent `radiomics` project, but keeps its
scripts, environment, inputs, logs, and outputs isolated from the acquisition
and feature-extraction pipeline.

## Scope

The workflow currently performs two steps:

1. `filter_supervised_labels.py` prepares one consistently labelled CN or AD
   session per participant. It removes participants with conflicting CN/AD
   labels and selects the eligible session with the smallest absolute
   `clinical_mri_gap_days`.
2. `demographic_baseline.py` evaluates a leakage-safe logistic-regression
   baseline using only age and sex, with out-of-fold predictions from
   `StratifiedGroupKFold`.

The input used for the copied analysis is `input/radiomics_sessions.csv`. The
filtered cohort is written to
`output/filter_supervised_labels/radiomics_sessions_cn_ad_simplified.csv` and
is the default input for the demographic baseline.

## Folder layout

```text
oasis_analysis/
├── cli.py
├── filter_supervised_labels.py
├── demographic_baseline.py
├── requirements.txt
├── input/
├── logs/
└── output/
```

Each script owns an output directory named `output/<script_name>/`. The CLI's
`clean` command removes only that directory for a known script.

## Environment

Use a separate virtual environment for this folder. The parent radiomics
pipeline pins NumPy 1.26.4 for compatibility with PyRadiomics 3.0.1, whereas
this analysis has its own dependency set.

From the root of the `radiomics` repository:

```bash
python3 -m venv oasis_analysis/.venv
oasis_analysis/.venv/bin/pip install -r oasis_analysis/requirements.txt
```

## Running the analysis

The folder CLI discovers the top-level analysis scripts and always runs them
with `oasis_analysis/` as their working directory. Commands can therefore be
issued from the repository root while the scripts continue to use their local
`input/`, `output/`, and `logs/` paths.

```bash
python oasis_analysis/cli.py list
python oasis_analysis/cli.py run filter_supervised_labels -- input/radiomics_sessions.csv
python oasis_analysis/cli.py run demographic_baseline
python oasis_analysis/cli.py clean demographic_baseline
```

Script options follow `--`:

```bash
python oasis_analysis/cli.py run demographic_baseline -- --n-splits 5 --random-state 42
```

## Demographic baseline

The default model uses:

- predictors: `age_at_mri` and `sex`;
- validation: five shuffled `StratifiedGroupKFold` folds grouped by
  `subject_id`, with `random_state=42`;
- preprocessing inside each fold's pipeline: `StandardScaler` for age and a
  fixed-category `OneHotEncoder` for sex;
- classifier: balanced logistic regression with a 0.5 decision threshold;
- primary metric: ROC-AUC.

The validation fold is used only for prediction. Scaling, encoding, and model
fitting use the corresponding training folds, and every participant receives
exactly one out-of-fold prediction.

Generated artifacts include OOF predictions, fold assignments, train and
validation tables, fitted fold pipelines, coefficients, metrics, split audits,
hash manifests, the cohort summary, ROC/precision-recall curves, and the OOF
confusion matrix.

## Parent-CLI integration

The root `cli.py` exposes a namespaced `analysis` command that forwards all
remaining arguments to this folder's CLI:

```bash
python cli.py analysis list
python cli.py analysis run filter_supervised_labels -- input/radiomics_sessions.csv
python cli.py analysis run demographic_baseline
python cli.py analysis clean demographic_baseline
```

The integration uses subprocess delegation rather than importing this file
into the root CLI:

1. The `analysis` subcommand captures the remaining arguments unchanged.
2. It resolves the analysis interpreter in this order:
   `oasis_analysis/.venv/bin/python`, its Windows equivalent, then the current
   interpreter as an explicit fallback.
3. It executes `oasis_analysis/cli.py` with the forwarded arguments and preserves
   its exit code and terminal streams.
4. The folder CLI retains ownership of script discovery, its working
   directory, and its narrowly scoped `clean` behavior.
5. Root-level tests cover argument forwarding, interpreter selection, and
   exit-code propagation.

This namespace avoids colliding with the root CLI's existing `run` command.
Process isolation also prevents the analysis dependency set from being loaded
into the PyRadiomics process.
