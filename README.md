# OASIS-3 Longitudinal Hippocampal Radiomics

Pipeline that turns the FreeSurfer outputs shipped with OASIS-3 into a
**longitudinal** radiomic dataset: one row per subject/session, plus per-subject
deltas and annualised slopes, ready for later work on CN → MCI → AD progression.

```
OASIS-3 / NITRC-IR
      ↓  (FreeSurfer already processed by OASIS-3)
T1.mgz + aseg.mgz
      ↓  labels 17 / 53
left + right hippocampus masks   ← always separate, never a union mask
      ↓  NIfTI
PyRadiomics (107 Original-image features per side)
      ↓
radiomics_features_long.csv        (subject × session × ROI)
      ↓  bilateral derivation
radiomics_features_wide.csv        (subject × session; _left/_right/_mean/_total/_asymmetry)
      ↓  temporal ordering
radiomics_longitudinal_deltas.csv  (Δ and Δ/year between visits)
radiomics_longitudinal_slopes.csv  (OLS slope, intercept, r², per subject × feature)
```

> **Status.** This is research *infrastructure*, not a finished protocol. No
> machine learning, feature selection, harmonisation (ComBat) or scanner
> correction is implemented — those come later, once the dataset itself is
> trustworthy. Every open methodological question is marked
> `TODO: methodological decision required` in `radiomics_config.yaml`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install numpy==1.26.4
pip install --no-build-isolation -r requirements.txt
```

PyRadiomics 3.0.1 imports NumPy in its own `setup.py` without declaring it as a
build dependency, hence the two-step install with `--no-build-isolation`.

> ### ⚠️ Do not upgrade to NumPy 2.x
> NumPy 2.x with PyRadiomics 3.0.1 caused a **segmentation fault** during
> feature extraction in this project's validated environment. PyRadiomics 3.0.1
> is compiled against the NumPy 1.x C ABI. Keep `numpy==1.26.4`.

Conda users: `conda env create -f environment.yml`.

## Usage

```bash
# full pipeline: extraction + longitudinal derivation + provenance
python cli.py run --input oasis3_radiomics_smoketest/freesurfer --output results/

# stages individually
python cli.py extract      --input oasis3_radiomics_smoketest/freesurfer --output results/
python cli.py longitudinal --features results/radiomics_features_long.csv --output results/
python cli.py qc           --input oasis3_radiomics_smoketest/freesurfer --output results/

# download (explicit only — nothing is ever fetched automatically)
python cli.py download --ids freesurfer_ids.csv --nitrc-user <your-nitrc-user>
```

Global options: `--config <path>`, `--log-level {DEBUG,INFO,WARNING,ERROR}`,
`--log-file <path>`.

The original entry point still works unchanged:

```bash
python oasis3_segmented_to_radiomics.py --skip-download
```

It now delegates to the package and writes the same
`oasis3_radiomics_smoketest/radiomics_features.csv`.

## Outputs

| File | Grain | Contents |
|---|---|---|
| `radiomics_features_long.csv` | subject × session × ROI | raw PyRadiomics output, one row per hemisphere |
| `radiomics_features_wide.csv` | subject × session | `_left`, `_right`, `_mean`, `_total`, `_asymmetry` |
| `radiomics_longitudinal_deltas.csv` | subject × visit pair | `delta_<f>`, `slope_<f>` (per year), `delta_days`, `delta_years` |
| `radiomics_longitudinal_slopes.csv` | subject × feature | `feature_slope`, `feature_intercept`, `feature_r2`, `n_sessions`, `followup_years` |
| `quality_control.csv` | subject × session | voxel counts, volumes, `qc_status`, `qc_warning`, `qc_outlier` |
| `run_metadata.json` | run | timestamp, all library versions, resolved config, counts |

## Design decisions worth knowing

**No radiomics on a bilateral union mask.** `Left-Hippocampus ∪
Right-Hippocampus` is two spatially disconnected objects. Shape features
(sphericity, surface-to-volume ratio, axis lengths, mesh volume) computed on
such a mask describe a nonexistent structure. Left and right are extracted
separately and bilateral quantities are derived *tabularly*:

```
mean      = (left + right) / 2
total     = left + right            # only for additive features (volume, surface area)
asymmetry = (left - right) / (left + right)
```

The legacy behaviour is still reachable via
`python oasis3_segmented_to_radiomics.py --legacy-bilateral` or
`bilateral.extract_union_mask: true`, but it is off by default.

**Asymmetry of signed features is NaN.** The normalised difference is only
interpretable for strictly positive quantities: for signed features such as
`firstorder_Skewness`, `glcm_ClusterShade` or `glcm_Imc1` the denominator
collapses near zero and the index can even invert its sign. Under the default
`asymmetry_mode: positive_only` these get NaN rather than a plausible-looking
number. `always` restores the naive behaviour.

**Slopes use every timepoint.** With more than two visits,
`(last - first) / span` throws away the middle. `radiomics_longitudinal_slopes.csv`
fits `feature = β₀ + β₁·years` by least squares (`scipy.stats.linregress`) and
reports `r²` so a non-linear trajectory is visible. With exactly two visits the
fit is exact and reduces to the annualised delta, which keeps a two-session
pilot comparable with later multi-visit runs.

**QC flags, never filters.** Sessions are annotated with `qc_status`,
`qc_warning` and `qc_outlier` (robust MAD z-score or IQR). Nothing is removed
automatically — excluding a subject is a scientific decision. Cohort-relative
outlier detection is skipped below `min_samples` sessions and says so
(`insufficient_samples(n=2, required=8)`) so a small pilot is never mistaken for
a clean cohort.

**No automatic downloads.** `extract`, `run`, `qc` and `longitudinal` operate
exclusively on data already on disk. Only `cli.py download` contacts NITRC-IR,
and only when invoked explicitly with `--nitrc-user`.

## Repository layout

```
oasis_radiomics/
├── config.py               typed config, loaded from radiomics_config.yaml
├── ids.py                  subject / session / days parsing and grouping
├── discovery.py            find T1.mgz + aseg.mgz on disk
├── masks.py                aseg labels 17 / 53 → binary masks → NIfTI
├── radiomics_extractor.py  PyRadiomics wrapper
├── quality_control.py      per-session checks + robust outlier flagging
├── longitudinal.py         wide table, bilateral derivation, deltas, slopes
├── download_oasis.py       official NrgXnat/oasis-scripts wrapper
├── metadata.py             run_metadata.json
├── tables.py               CSV IO with a stable column order
├── logging_setup.py        logging configuration
├── pipeline.py             stage orchestration
└── cli.py                  command line interface
```

The package is deliberately **not** called `radiomics`: that name belongs to
PyRadiomics, and a local package with the same name shadows it whenever the
repository root lands on `sys.path`.

## Tests

```bash
pytest -q
```

Unit tests are pure-Python and always run. `tests/test_integration.py` exercises
the full pipeline on the two pilot sessions (`OAS30001_MR_d0129`,
`OAS30001_MR_d0757`) and **skips itself** when the data is absent — OASIS-3 is
credentialed and is not distributed with this repository.

## Data

OASIS-3 requires credentials from [nitrc.org](https://www.nitrc.org/) /
[oasis-brains.org](https://www.oasis-brains.org/). Do not commit `.mgz`,
`.nii` or `.nii.gz` files; `.gitignore` blocks them.
