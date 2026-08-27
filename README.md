# OASIS-3 Alzheimer Radiomics Acquisition

Reproducible pipeline for acquiring **already processed FreeSurfer outputs from OASIS-3** and transforming structural T1 MRI into a longitudinal radiomics dataset.

This repository currently stops at **data acquisition, ROI extraction, radiomic feature extraction, longitudinal tabulation and quality control**. No machine learning, feature selection, SMOTE, ComBat or predictive modelling is part of the acquisition stage.

## Frozen acquisition protocol

Protocol identifier:

```text
OASIS3-AD-RADIOMICS-v1.0
```

Pipeline:

```text
OASIS-3 / NITRC-IR
        ↓
FreeSurfer outputs already supplied by OASIS-3
        ↓
T1.mgz + aparc+aseg.mgz
        ↓
16 Alzheimer-related ROIs
        ↓
Original T1 + PyRadiomics 3.0.1
        ↓
107 frozen radiomic features / ROI
        ↓
16 × 107 = 1,712 raw radiomic features / session
        ↓
radiomics_features_long.csv
        ↓
QC + acquisition validation gate
        ↓
longitudinal wide / delta / slope tables
```

The frozen constants and exact feature names are in [`oasis_radiomics/protocol.py`](oasis_radiomics/protocol.py).

## Why 107 features instead of the 104 written in the preliminary project?

The preliminary methodology estimated:

```text
18 first-order
14 shape
21 GLCM
16 GLRLM
16 GLSZM
14 GLDM
5 NGTDM
----------------
104 / ROI
```

The validated environment actually returns **24 non-deprecated GLCM features** with PyRadiomics 3.0.1, giving:

```text
18 first-order
14 shape
24 GLCM
16 GLRLM
16 GLSZM
14 GLDM
5 NGTDM
----------------
107 / ROI
```

The acquisition protocol therefore records a methodological amendment rather than deleting three valid GLCM measurements only to preserve the old arithmetic. The exact 107 names are explicitly enabled and every ROI is checked against that schema. A library upgrade cannot silently change the dataset.

With 16 ROIs, the raw per-session count is consequently:

```text
16 × 107 = 1,712 radiomic features
```

Derived left/right means, totals, asymmetry indices, deltas and slopes are **derived variables** and are not counted as additional raw PyRadiomics features.

## Alzheimer ROIs

Eight bilateral anatomical regions are extracted from `aparc+aseg.mgz`:

| Region | Left label | Right label |
|---|---:|---:|
| Hippocampus | 17 | 53 |
| Amygdala | 18 | 54 |
| Entorhinal cortex | 1006 | 2006 |
| Fusiform | 1007 | 2007 |
| Inferior temporal | 1009 | 2009 |
| Middle temporal | 1015 | 2015 |
| Parahippocampal | 1016 | 2016 |
| Precuneus | 1025 | 2025 |

Left and right masks are always separate. A disconnected bilateral union mask is never sent to PyRadiomics.

`aparc+aseg.mgz` is mandatory for the final protocol. `aseg.mgz` alone is insufficient because it does not provide the cortical atlas labels required by the study.

## Validated environment

The pilot was validated on macOS ARM64 with:

```text
Python       3.10.x
NumPy        1.26.4
PyRadiomics  3.0.1
SimpleITK    2.x
```

### Do not upgrade to NumPy 2.x in this environment

NumPy 2.0.1 + PyRadiomics 3.0.1 produced a segmentation fault during the pilot. Keep `numpy==1.26.4` for this acquisition protocol.

Install with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4
python -m pip install --no-build-isolation -r requirements.txt
```

Conda users can use `environment.yml`.

## Final acquisition workflow

### 1. Obtain a FreeSurfer catalogue

From the authorized OASIS-3/NITRC environment, obtain/export a CSV containing the available processed FreeSurfer identifiers. The only required column for cohort preparation is:

```csv
freesurfer_id
OAS30001_Freesurfer53_d0129
OAS30001_Freesurfer53_d0757
...
```

Do not commit credentialed OASIS-3 data or metadata that is restricted by the data-use agreement.

### 2. Build a longitudinal acquisition manifest

The selection unit is the **participant**, not the MRI session.

For the full eligible longitudinal cohort:

```bash
python prepare_acquisition.py \
  --catalog oasis3_freesurfer_catalog.csv \
  --min-sessions 2 \
  --output acquisition/
```

If the study protocol defines a final participant target `N`, acquire with a QC/file-attrition margin:

```bash
python prepare_acquisition.py \
  --catalog oasis3_freesurfer_catalog.csv \
  --min-sessions 2 \
  --target-subjects <N_DO_PROJETO> \
  --oversample 1.20 \
  --seed 2026 \
  --output acquisition/
```

This writes:

```text
acquisition/acquisition_freesurfer_ids.csv
acquisition/acquisition_subjects.csv
```

The 20% margin is only an acquisition buffer for missing files and QC attrition. It is **not** a replacement for a formal sample-size calculation.

### 3. Download only the selected processed sessions

```bash
python cli.py download \
  --ids acquisition/acquisition_freesurfer_ids.csv \
  --nitrc-user luanmantegazine \
  --output oasis3_data/
```

The password is requested by the official OASIS/NITRC downloader and is not stored by this project.

### 4. Run extraction + longitudinal tables + QC

```bash
python cli.py run \
  --input oasis3_data/freesurfer \
  --output results/ \
  --config radiomics_config.yaml
```

For a small validation batch first:

```bash
python cli.py run \
  --input oasis3_data/freesurfer \
  --output results_pilot/ \
  --config radiomics_config.yaml \
  --max-sessions 20
```

### 5. Run the acquisition gate

A run is not considered an acquisition success merely because PyRadiomics finished. It must pass:

```bash
python validate_acquisition.py \
  --features results/radiomics_features_long.csv \
  --output results/acquisition_validation.json
```

A valid session must contain:

```text
16/16 expected ROIs
107/107 expected radiomic features in every ROI
no duplicated ROI rows
no unexpected radiomic columns
no non-finite radiomic values
```

The command exits with a non-zero status when any session violates the contract.

## Recommended staged acquisition

Do not begin by downloading the entire database blindly.

```text
Stage A — existing smoke test
1 participant / 2 sessions
purpose: image → segmentation → ROI → PyRadiomics

Stage B — protocol pilot
10–20 participants with ≥2 sessions
purpose: validate all 16 ROIs, schema stability, storage/time and QC

Stage C — definitive acquisition
N defined by the study + acquisition margin
purpose: produce the frozen raw longitudinal radiomics dataset
```

After Stage B passes the acquisition gate, the protocol should not change during Stage C. Any change to ROI labels, bin width, normalization, image type or feature list requires a new protocol version and complete re-extraction.

## Outputs

| File | Grain | Contents |
|---|---|---|
| `radiomics_features_long.csv` | subject × session × ROI | 107 raw PyRadiomics features per ROI |
| `radiomics_features_wide.csv` | subject × session | side-specific/derived columns plus other ROI features |
| `radiomics_longitudinal_deltas.csv` | subject × visit pair | absolute and annualized changes |
| `radiomics_longitudinal_slopes.csv` | subject × feature | OLS longitudinal slope/intercept/r² |
| `quality_control.csv` | subject × session | anatomical/geometry QC flags |
| `run_metadata.json` | run | environment, resolved config and counts |
| `acquisition_validation.json` | acquisition gate | session-level schema/completeness failures |

## Bilateral handling

The two hippocampi are never merged before feature extraction. For the paired hippocampal variables, derived values are computed tabularly:

```text
mean      = (left + right) / 2
total     = left + right       # only physically additive variables
asymmetry = (left - right) / (left + right)
```

Signed variables use the conservative `positive_only` asymmetry mode by default.

## Longitudinal handling

Session time is parsed from identifiers such as:

```text
OAS30001_MR_d0129 → subject OAS30001, day 129
OAS30001_MR_d0757 → subject OAS30001, day 757
```

For two visits, the pipeline calculates delta and annualized rate. For three or more visits it also fits:

```text
feature = beta0 + beta1 × years
```

using all available timepoints and reports slope, intercept and r².

## Quality-control policy

QC **flags and reports**; it does not silently delete participants. The final investigator-facing cohort must preserve the reason for every exclusion.

The acquisition gate is stricter than the anatomical QC table: it guarantees the raw data matrix has the exact protocol structure before the acquisition phase is declared complete.

## Repository layout

```text
oasis_radiomics/
├── acquisition.py          cohort manifest + final acquisition validator
├── protocol.py             frozen 16-ROI / 107-feature contract
├── config.py
├── discovery.py            requires T1.mgz + aparc+aseg.mgz
├── download_oasis.py
├── ids.py
├── masks.py
├── radiomics_extractor.py  explicit feature-name extraction
├── quality_control.py
├── longitudinal.py
├── metadata.py
├── tables.py
└── pipeline.py

prepare_acquisition.py      catalogue → reproducible download manifest
validate_acquisition.py     final 16 × 107 acquisition gate
radiomics_config.yaml       frozen acquisition settings
ACQUISITION_PROTOCOL.md     methodological protocol/checklist
```

## Tests

```bash
pytest -q
```

Pure-Python tests run without OASIS data. The integration test uses the two pilot sessions when they are present locally and otherwise skips itself.

## Data governance

No OASIS-3 imaging data should be committed to this repository. The dataset is credentialed and must be obtained through the authorized OASIS/NITRC workflow. `.gitignore` must continue to exclude FreeSurfer/MRI data and result directories as appropriate.
