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

## Clinical data integration

A dedicated clinical layer (`oasis_radiomics/clinical/`) links OASIS-3 clinical
assessments to MRI sessions so radiomic features can later be analysed against
cognitive status and progression. It contains **no machine learning** and does
not touch the frozen 16-ROI / 107-feature radiomics protocol; the radiomics
outputs are read-only inputs.

The full specification lives in **[`CLINICAL_LINKAGE_PROTOCOL.md`](CLINICAL_LINKAGE_PROTOCOL.md)**.

### Required OASIS files

| File | Role |
|---|---|
| `OASIS3_UDSd1_diagnoses.csv` | **primary diagnostic source** (NORMCOG, DEMENTED, MCI\*, PROBAD, POSSAD, ...) |
| `OASIS3_UDSb4_cdr.csv` | CDR (`CDRTOT`, `CDRSUM`), `MMSE` and free-text `dx1`..`dx5`; complements D1 |
| `OASIS3_UDSc1_cognitive_assessments.csv` | **optional** psychometrics (fluency, Trail Making, logical memory, ...) |
| MRI session catalogue | `Label`, `Subject`, `M/F`, `Age`, `Scanner` |

Source files are read-only and are never modified.

### Usage

```bash
python cli.py clinical-link \
  --mri-catalog oasis3_mri_catalog.csv \
  --d1 diagnostic/OASIS3_UDSd1_diagnoses.csv \
  --b4 diagnostic/OASIS3_UDSb4_cdr.csv \
  --c1 diagnostic/OASIS3_UDSc1_cognitive_assessments.csv \
  --clinical-window-days 180 \
  --output clinical_results/

python cli.py clinical-radiomics \
  --clinical clinical_results/clinical_imaging_master.csv \
  --radiomics results/radiomics_features_wide.csv \
  --deltas results/radiomics_longitudinal_deltas.csv \
  --slopes results/radiomics_longitudinal_slopes.csv \
  --output dataset/
```

### Generated datasets

```
clinical_results/
├── clinical_visits.csv                 one row per subject x clinical day (D1+B4 merged)
├── clinical_imaging_master.csv         one row per MRI session
└── clinical_linkage_validation.json    parameters, summary and every issue found

dataset/
├── clinical_radiomics_sessions.csv     clinical + all session-level radiomic features
├── clinical_radiomics_deltas.csv       radiomic deltas with clinical status at t0/t1
└── clinical_radiomics_subjects.csv     radiomic slopes with the subject's trajectory
```

### MRI-clinical temporal matching

MRI and clinical visits do not share a calendar, so each MRI session is linked
to the **nearest in time** clinical visit **of the same subject**:

```
clinical_mri_gap_days = clinical_day - mri_day      (negative = visit before the scan)
```

* default window **±180 days**, configurable with `--clinical-window-days`;
* equidistant visits resolve deterministically to the **earlier** one and are
  flagged `ambiguous_equal_distance`;
* **nothing is silently discarded** - a session outside the window is kept with
  `clinical_match_valid = False`, and a subject with no clinical data at all is
  kept with `clinical_match_reason = no_clinical_visit`;
* C1 is matched **independently**, with its own day, gap and validity columns,
  because psychometric testing rarely falls on the diagnostic visit day.

### Diagnosis at MRI vs. future conversion

> A diagnosis recorded **after** an MRI never relabels that MRI.

`diagnosis_at_mri` comes only from the visit the scan was linked to. Later
information is exposed separately so conversion can be *predicted* rather than
leaked:

```
clinical d0500 = MCI,  d1200 = MCI,  d2000 = AD        MRI at d0800
    diagnosis_at_mri   = MCI     <- not AD
    future_diagnosis   = AD
    conversion_event   = MCI_to_AD
    conversion_day     = 2000
    days_to_conversion = 1200
```

### ⚠️ D1 code mappings require the official data dictionary

The numeric semantics of the D1 variables are defined by the official NACC UDS
Data Element Dictionary, which is **not part of this repository**. This project
does not guess at them.

All clinical meaning lives in `clinical_classification.yaml`, which ships
**unfrozen**. Until it is frozen every visit is reported as:

```
cognitive_status      = UNKNOWN
ad_etiology           = UNKNOWN
classification_status = unresolved_codebook
```

while **every raw D1/B4/C1 variable is still written to the outputs** (145 + 19
+ 103 columns), so nothing is lost and no unverified assumption enters the
dataset. `UNKNOWN` never means "normal". To activate classification, obtain the
official dictionary, write the rules with a `reference` citation for each, and
set `codebook_frozen: true`. An empty frozen codebook, an uncited rule, and a
value outside the controlled vocabulary are all rejected.

## Supervised labels

A separately versioned layer turns the linked clinical information into
supervised-learning targets. **No machine learning and no feature selection** -
labels only. Full specification: **[`SUPERVISED_LABELING_PROTOCOL.md`](SUPERVISED_LABELING_PROTOCOL.md)**.

Two targets, deliberately kept apart:

```
Target A   X(MRI)            ->  y in {CN, MCI, AD}
Target B   X(MRI while MCI)  ->  y in {MCI_STABLE, MCI_TO_AD}   (+ CENSORED, excluded)
```

```bash
python cli.py supervised-labels \
  --clinical-radiomics dataset_full/clinical_radiomics_sessions.csv \
  --clinical-visits clinical_results/clinical_visits.csv \
  --label-policy supervised_labels.yaml \
  --clinical-window-days 180 \
  --progression-horizon-days 1095 \
  --output supervised_dataset/
```

```
supervised_dataset/
├── supervised_radiomics_sessions.csv   Target A: one row per MRI + all features
├── supervised_mci_progression.csv      Target B: MCI sessions + outcome
├── diagnosis_vocabulary.csv            every dx1..dx5 string and its mapping
└── supervised_label_audit.json         counts by session AND by subject
```

### Labels come from D1, validated against the B4 text

`supervised_labels.yaml` v2.0 uses **D1 as the primary source** and keeps B4 as
**auxiliary validation**. v1.0 labelled from B4 and found it carries no MCI
label at all; D1 has explicit MCI variables and an aetiology block, so it
expresses the full taxonomy:

| D1 | label |
|---|---|
| `NORMCOG` | `CN` |
| `MCIAMEM` / `MCIAPLUS` / `MCINON1` / `MCINON2` | `MCI` |
| `IMPNOMCI` | `IMPAIRED_NOT_MCI` |
| `DEMENTED` + `PROBAD`/`POSSAD`/`alzdis` | `AD` |
| `DEMENTED` + `DLB`/`VASC`/`FTD`/… | `OTHER_DEMENTIA` |
| `DEMENTED`, aetiology not established | `DEMENTIA_UNKNOWN_ETIOLOGY` |

The D1 coding is **not assumed** - it is validated against the independent B4
text over the 8475 visits carrying both instruments (`NORMCOG` 99.7 % CN,
`DEMENTED` 92.6 % dementia, `PROBAD` 98.1 % AD, `FTD` 100 % non-AD). Every row
records the outcome in `b4_agreement`; on the real cohort **93.4 %** agree.
B4 never overrides D1 - a disagreement is warned, not applied.

Values outside `{0, 1}` are never guessed (D1 has three such cells). Two
cognitive states firing at once give `CONFLICTING`, not a priority win.

Aetiology spans both UDS generations (`PROBAD`/`POSSAD` and `alzdis`), which are
disjoint in OASIS-3. Their paired `…IF` qualifiers are **not** binary — domain
`{0, 1, 2}`, where 2 means AD *contributes* rather than causes — so a
contributing-only aetiology does not enter the `AD` class. MCI subtype and
impaired domains are recorded in `mci_subtype` / `mci_domains`.

CDR and MMSE remain covariates and cross-checks only. `CDRTOT == 0.5` does
**not** mean MCI.

### Two rules that must not be relaxed

**`CENSORED` is not `MCI_STABLE`.** Stability requires a non-AD sighting at or
after the horizon. A participant who leaves follow-up six months after an MCI
scan has an unobserved outcome, not a stable one.

**Split by `subject_id`, never by row.** One participant contributes several MRI
sessions; row-wise splitting puts `OAS30001` in both train and test. Use
`subject_groups(rows)` as `groups` for `GroupKFold`, and
`assert_no_subject_leakage(train, test)` to verify.

## Tests

```bash
pytest -q
```

Pure-Python tests run without OASIS data. The integration test uses the two pilot sessions when they are present locally and otherwise skips itself.

## Data governance

No OASIS-3 imaging data should be committed to this repository. The dataset is credentialed and must be obtained through the authorized OASIS/NITRC workflow. `.gitignore` must continue to exclude FreeSurfer/MRI data and result directories as appropriate.
