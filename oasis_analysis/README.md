# OASIS-3 cohort analysis

This folder contains the downstream OASIS-3 cohort preparation, demographic
baseline and morphometric model. It is part of the parent `radiomics` project, but keeps its
scripts, environment, inputs, logs, and outputs isolated from the acquisition
and feature-extraction pipeline.

## Scope

The workflow currently performs these steps:

1. `filter_supervised_labels.py` prepares one consistently labelled CN or AD
   session per participant. It removes participants with conflicting CN/AD
   labels, drops sessions whose `training_eligible` flag is false (when the
   column is present), and selects the eligible session with the smallest
   absolute `clinical_mri_gap_days`.
2. `demographic_baseline.py` evaluates a leakage-safe logistic-regression
   baseline using only age and sex, with out-of-fold predictions from
   `StratifiedGroupKFold`.
3. `prepare_morphological_features.py` exports auditable, fold-specific eTIV
   adjusted measurements from the same one-session CN/AD cohort.
4. `morphological_model.py` performs nested, subject-grouped model selection
   and evaluation using the original measurements, refitting all preprocessing
   inside each training split.
5. `compare_models.py` compares two models' out-of-fold predictions on the same
   participants with DeLong's paired test and a paired bootstrap.

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
├── prepare_morphological_features.py
├── morphological_model.py
├── compare_models.py
├── requirements.txt
├── input/
├── logs/
└── output/
```

Each script owns an output directory named `output/<script_name>/`. The CLI's
`clean` command removes only that directory for a known script.
Default output paths are anchored to `oasis_analysis/`, including when a script
is executed directly from the repository root.

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
python oasis_analysis/cli.py run prepare_morphological_features -- --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv
python oasis_analysis/cli.py run morphological_model -- --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv
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

The demographic pipeline also implements the same alternative classifiers as
the morphological pipeline: Elastic Net logistic regression, SVM, Random
Forest, and XGBoost. Select one with `--classifier elasticnet`,
`--classifier svm`, `--classifier rf`, or `--classifier xgboost`. The original
plain logistic model remains the default as `--classifier logistic` for
backward compatibility. Elastic Net searches
`C ∈ {0.003, 0.01, 0.03, 0.1, 1}` with `l1_ratio = 0.5`, the same grid as the
morphological model; XGBoost sets `scale_pos_weight` to the CN/AD ratio of each
outer training fold, matching the balanced class weights of the other models.

Alternative classifiers use nested validation: within each outer training
fold, a three-fold `StratifiedGroupKFold` search selects the same
hyperparameters and ROC-AUC objective as the morphological pipeline. The
held-out outer fold is never used for model selection. Set the inner fold count
with `--inner-folds`.

```bash
python oasis_analysis/cli.py run demographic_baseline -- \
  --input output/filter_supervised_labels/clinical_radiomic_cn_ad_simplified.csv \
  --classifier elasticnet --inner-folds 3
```

The validation fold is used only for prediction. Scaling, encoding, and model
fitting use the corresponding training folds, and every participant receives
exactly one out-of-fold prediction.

Generated artifacts include OOF predictions, fold assignments, train and
validation tables, fitted fold pipelines, coefficients or feature importances,
inner-search tables for tuned classifiers, metrics, split audits, hash
manifests, the cohort summary, ROC/precision-recall curves, and the OOF
confusion matrix. Classification metrics use `recall` for the AD positive-class
recall; no separate sensitivity alias is written.

## Morphological feature preparation

Run `filter_supervised_labels` on the **full** consolidated table first; the
simplified output does not contain imaging measurements:

```bash
python oasis_analysis/cli.py run filter_supervised_labels -- input/clinical_radiomics_CONSOLIDATED_v2.csv
python oasis_analysis/cli.py run prepare_morphological_features -- --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv
```

The second command requires an explicit `--input` path; `--output` can override
the output path. By default, it adjusts the eight left/right mask volumes
already in the consolidated CSV: hippocampus, amygdala, entorhinal and
parahippocampal. The output also carries IDs, label, validation fold and eTIV
for audit; raw volumes are not exported as model predictors.
Only `EstimatedTotalIntraCranialVol` must be added to the filtered CSV for
this default mode. The other missing measurements are not required by default.

Use `--extended-data` to opt into the stricter specification:

```bash
python oasis_analysis/cli.py run prepare_morphological_features -- --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv --extended-data
```

This mode uses FreeSurfer `aseg.stats` volumes for hippocampus and amygdala,
adds bilateral inferior lateral ventricle volumes, and uses `ThickAvg` for
bilateral entorhinal and parahippocampal cortices. Its required flattened
source columns are:

| Source | Required input columns |
| --- | --- |
| `aseg.stats` | `EstimatedTotalIntraCranialVol`, `aseg.Left-Hippocampus.Volume_mm3`, `aseg.Right-Hippocampus.Volume_mm3`, `aseg.Left-Amygdala.Volume_mm3`, `aseg.Right-Amygdala.Volume_mm3`, `aseg.Left-Inf-Lat-Vent.Volume_mm3`, `aseg.Right-Inf-Lat-Vent.Volume_mm3` |
| `lh.aparc.stats` | `lh.aparc.entorhinal.ThickAvg`, `lh.aparc.parahippocampal.ThickAvg` |
| `rh.aparc.stats` | `rh.aparc.entorhinal.ThickAvg`, `rh.aparc.parahippocampal.ThickAvg` |

The current `input/clinical_radiomics_CONSOLIDATED_v2.csv` does **not** contain
eTIV or the FreeSurfer stats needed for the extended mode. The preparation
script reports required missing columns and stops without creating an
incomplete table. Merge the relevant stats by `session_id` first.

Residual adjustment is the default. For every fold and every volume, the script
fits `volume = intercept + slope × eTIV` using **only CN rows in that fold's
training subset**. It exports `volume − predicted_volume` for both train and
validation. `--proportional` instead exports `volume / eTIV`, without fitting
parameters. Cortical thickness is copied unchanged in either mode.

The main output CSV contains one out-of-fold adjusted row per participant.
Adjacent `<output_stem>_folds/fold_XX_train.csv` and `fold_XX_validation.csv`
files contain the fold-specific features needed for model training. Residual
mode also saves each fold's CN regression coefficients. The manifest records
the exact source mapping, adjustment method, predictor names and fold files.
The default is five subject-level, stratified, reproducible folds with seed 42.
Use `--fold-assignments path/to/fold_assignments.csv` to reuse existing
`subject_id`, `session_id`, `validation_fold` assignments, such as those from
the demographic baseline, ensuring the comparisons use identical splits.

```bash
python oasis_analysis/cli.py run prepare_morphological_features -- \
  --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv \
  --proportional
```

Use `--asymmetry` to add `(left - right) / (left + right)` for each bilateral
structure: four indices in the default mode and five in the extended mode
(including inferior lateral ventricle). The extended cortical indices use
left/right thickness, while the default cortical indices use mask volume.
These dimensionless indices are computed from the measured values before any
training-fold eTIV adjustment. Without the flag, no asymmetry columns appear.
Lateral ventricle and total cortical gray matter volumes remain excluded from
both modes.

### Synthetic eTIV input for a pipeline smoke test

`create_synthetic_etiv_input.py` makes a separate, ignored test CSV containing
`EstimatedTotalIntraCranialVol`. It uses four regions outside the model's
selected ROIs: fusiform, inferior temporal, middle temporal and precuneus.
For each region it sums the left and right `mask_volume_mm3` columns and divides
by a published bilateral region-to-ICV ratio. The final synthetic eTIV is the
median of the four estimates. These ratios come from sex-weighted mean cortical
volumes and mean ICV among 406 cognitively normal older adults, using FreeSurfer;
the paper averages the two hemispheres, so the reference volume is doubled for
our bilateral sums ([Voevodskaya et al., 2014, Tables 1 and 4](https://www.frontiersin.org/journals/aging-neuroscience/articles/10.3389/fnagi.2014.00264/full)).
The script records all four ratios, formula and source in a `.metadata.json`
file and refuses to overwrite an input that already contains eTIV.

```bash
python oasis_analysis/cli.py run create_synthetic_etiv_input -- \
  --input input/clinical_radiomics_CONSOLIDATED_v2.csv \
  --output input/clinical_radiomics_CONSOLIDATED_v2_SYNTHETIC_ETIV_TEST_ONLY.csv
```

These four cortical volumes may still vary with disease, so this synthetic
value cannot validate head-size correction or model performance. Use the copy
only to check data flow, and replace it with measured FreeSurfer eTIV for any
actual model analysis.

## Morphological model and nested evaluation

Pass the **filtered full CSV with original ROI measurements and eTIV** to
`morphological_model.py`. The prepared feature CSV is an audit artifact and
must not be used for nested model selection: its residual fits belong to a
particular outer fold and cannot be reused for inner folds.

```bash
python oasis_analysis/cli.py run morphological_model -- \
  --input output/filter_supervised_labels/clinical_radiomics_CONSOLIDATED_v2_cn_ad.csv \
  --fold-assignments output/demographic_baseline/fold_assignments.csv
```

The default is Elastic Net logistic regression, residual eTIV adjustment and
five outer by three inner `StratifiedGroupKFold` splits grouped by `subject_id`.
Within every inner split, the CN-only eTIV regressions, z-score means/scales
and classifier are fitted on that split's training rows. ROC-AUC selects
hyperparameters. The chosen pipeline is then refitted on the whole outer
training set and predicts its held-out participants. The optional
`--fold-assignments` reuses the demographic baseline's outer folds; without
it, the same five-fold splitter and seed 42 are used.

`--adjust-age-sex` replaces the eTIV-only residuals with a CN-normative
(W-score style) adjustment: every volume is regressed on eTIV, age and sex, and
every cortical thickness on age and sex, using only the CN rows of the training
split. Without it, age-related atrophy is read as disease: in the 222-participant
CN/AD cohort the eTIV-only model's false positives were mostly older CN and its
false negatives younger AD participants. `--include-demographics` adds
`age_at_mri` and a male indicator as features. Both flags need `age_at_mri` and
`sex` (`F`/`M`) in the input; `--proportional` combined with
`--adjust-age-sex` regresses `volume / eTIV` on age and sex.

The Elastic Net grid is `C ∈ {0.003, 0.01, 0.03, 0.1, 1}` with `l1_ratio = 0.5`.
On the CN/AD cohort the earlier grid (`C ∈ {0.1, 1, 10}`, `l1_ratio ∈ {0.1, 0.5,
0.9}`) selected its smallest C in every fold, and `l1_ratio` changed the inner
AUC by at most 0.008.

`--classifier svm`, `--classifier rf` and `--classifier xgboost` select the
secondary classifiers. XGBoost receives `scale_pos_weight` equal to the CN/AD
ratio of each outer training fold. XGBoost is installed through this folder's requirements.
`--proportional` chooses `volume / eTIV`; `--extended-data` requires the
FreeSurfer stats columns; `--asymmetry` adds the bilateral indices. These
flags can be combined. `--outer-folds`, `--inner-folds`, `--random-state`,
`--threshold` and `--bootstrap-resamples` control evaluation details.

The output directory contains pooled outer-fold predictions, per-fold model
files, inner search tables, fold metrics, a calibration table and plot, and
`metrics.json` with pooled ROC-AUC and a stratified subject-bootstrap 95% CI,
the mean and SD of the per-fold ROC-AUC,
balanced accuracy, AD recall, specificity, Brier score
and expected calibration error. `performance_curves.png` shows the pooled ROC and
precision-recall curves, and `confusion_matrix.png` shows the pooled out-of-fold
classifications. `run_config.json` records each fold's hyperparameters,
CN-only fits and z-score parameters. Files from an earlier run
(`fold_*_model.joblib`, `fold_*_inner_search.csv`, `fold_*_validation.csv`) are
removed first, so a run with fewer folds leaves no stale results.

Pass the demographic baseline's `fold_assignments.csv` explicitly for any run
that will be compared with it. The configuration chosen for the closed
morphological model is:

```bash
python oasis_analysis/cli.py run morphological_model -- \
  --input output/filter_supervised_labels/<cohort>_cn_ad.csv \
  --extended-data --adjust-age-sex --include-demographics \
  --fold-assignments output/demographic_baseline/fold_assignments.csv
```

## Paired model comparison

`compare_models.py` reads two `oof_predictions.csv` files, aligns them by
`subject_id` and `session_id`, and refuses files with different participants
or labels. It reports DeLong's test for correlated ROC-AUCs (difference A − B,
95% CI and p-value) and a stratified participant bootstrap of the same
difference, and warns when the two runs used different folds.

```bash
python oasis_analysis/cli.py run compare_models -- \
  --a output/morphological_model/oof_predictions.csv --name-a morphological \
  --b output/demographic_baseline/oof_predictions.csv --name-b demographic
```

The result is written to `output/compare_models/<name-a>_vs_<name-b>.json`. Synthetic eTIV input is for pipeline
tests only; its metrics must not be interpreted as scientific performance.

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
