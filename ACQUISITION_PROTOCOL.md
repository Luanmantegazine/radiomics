# Final Data Acquisition Protocol — OASIS-3 Alzheimer Radiomics

**Protocol:** `OASIS3-AD-RADIOMICS-v1.0`  
**Scope:** acquisition, FreeSurfer ROI masks, radiomic extraction, QC and longitudinal tabulation.  
**Out of scope:** feature selection and machine-learning modelling.

## 1. Frozen acquisition unit

The statistical/acquisition unit is the **participant**. MRI sessions are repeated observations belonging to that participant.

For longitudinal inclusion, the acquisition manifest requires at least **two processed FreeSurfer sessions per participant**. The pipeline retains all selected visits in chronological order.

If the approved study design defines a final target of `N` participants, the downloader manifest may include an additional acquisition margin (default 20%) to compensate for unavailable files and QC losses. This margin does not alter the target sample size and must not be reported as a power calculation.

## 2. Required OASIS-3 files and processing cohort

Every session used by the final protocol must contain:

```text
mri/T1.mgz
mri/aparc+aseg.mgz
```

`aseg.mgz` alone is insufficient for the final study because the protocol includes cortical Desikan-Killiany labels in addition to subcortical structures.

No new FreeSurfer reconstruction is performed. The study consumes the processed FreeSurfer products distributed with OASIS-3.

### 2.1 FreeSurfer 5.3 / 3T restriction

Protocol v1.0 defaults to identifiers containing:

```text
_Freesurfer53_
```

OASIS-3 documents two relevant processing regimes: 1.5T MRI was reprocessed with FreeSurfer 5.0/5.1, while 3.0T MRI was reprocessed with FreeSurfer 5.3-HCP-patch. The final raw radiomics cohort is therefore restricted to `Freesurfer53` sessions so the acquisition does not intentionally mix the 1.5T/FS5.0–5.1 and 3T/FS5.3 regimes.

`prepare_acquisition.py` enforces version `53` by default. Changing this filter creates a different acquisition protocol and must be documented/versioned.

## 3. Regions of interest

The protocol uses 8 bilateral structures, therefore 16 independent masks per MRI session.

| ROI | Left | Right | Rationale in the acquisition design |
|---|---:|---:|---|
| Hippocampus | 17 | 53 | medial temporal neurodegeneration / atrophy |
| Amygdala | 18 | 54 | medial temporal involvement |
| Entorhinal cortex | 1006 | 2006 | early medial temporal cortical involvement |
| Fusiform | 1007 | 2007 | temporal cortical involvement |
| Inferior temporal | 1009 | 2009 | AD-related temporal degeneration |
| Middle temporal | 1015 | 2015 | temporal association cortex |
| Parahippocampal | 1016 | 2016 | medial temporal memory network |
| Precuneus | 1025 | 2025 | posterior association/default-mode involvement |

The labels are read from `aparc+aseg.mgz` and are kept as separate left/right masks. A bilateral union mask is not used for direct radiomics because disconnected anatomy makes shape descriptors difficult to interpret.

## 4. MRI/radiomics preprocessing frozen for acquisition

The final acquisition settings are recorded in `radiomics_config.yaml`:

```text
image type        Original T1 only
normalization     PyRadiomics normalize=true
normalize scale   100
bin width         25
resampling        no second acquisition-time resampling
mask correction   enabled
C extensions      disabled in validated environment
```

No Wavelet or LoG features are enabled in protocol v1.0.

Any future change to these parameters requires a new protocol version and complete re-extraction of the cohort.

## 5. Feature signature

The preliminary project text estimated 104 features per ROI. During the validated pilot, PyRadiomics 3.0.1 produced 107 non-deprecated features:

| Family | Count |
|---|---:|
| First order | 18 |
| Shape | 14 |
| GLCM | 24 |
| GLRLM | 16 |
| GLSZM | 16 |
| GLDM | 14 |
| NGTDM | 5 |
| **Total** | **107** |

The exact feature names are frozen in `oasis_radiomics/protocol.py` and are enabled explicitly by name. The acquisition gate fails if PyRadiomics adds, removes or renames a feature.

Therefore the raw feature count per MRI session is:

```text
16 ROIs × 107 features = 1,712 raw radiomic features/session
```

This is an explicit methodological amendment from the preliminary estimate of 1,664 (`16 × 104`). It avoids arbitrary post-hoc deletion of valid GLCM features merely to reproduce the old total.

## 6. Acquisition sequence

### Phase A — validated smoke test

Existing pilot:

```text
OAS30001_MR_d0129
OAS30001_MR_d0757
```

This established that OASIS FreeSurfer outputs can be converted into ROI masks and processed by PyRadiomics in the pinned environment.

### Phase B — protocol pilot

Acquire **10–20 longitudinal participants** first. The purpose is operational validation, not statistical inference:

- confirm `aparc+aseg.mgz` availability;
- confirm all 16 masks are present;
- confirm the selected sessions are `Freesurfer53`;
- measure extraction time/storage requirements;
- inspect QC distributions;
- verify the exact 107-feature schema across participants.

No protocol parameter should be changed after this phase without incrementing the protocol version.

### Phase C — definitive acquisition

Build the participant list from the complete authorized OASIS-3 FreeSurfer catalogue and acquire the study target plus the predefined QC attrition margin.

## 7. Commands

### Prepare participant/session manifest

```bash
python prepare_acquisition.py \
  --catalog oasis3_freesurfer_catalog.csv \
  --freesurfer-version 53 \
  --min-sessions 2 \
  --target-subjects <N_DO_PROJETO> \
  --oversample 1.20 \
  --seed 2026 \
  --output acquisition/
```

For all longitudinally eligible `Freesurfer53` participants, omit `--target-subjects`.

### Download selected FreeSurfer outputs

```bash
python cli.py download \
  --ids acquisition/acquisition_freesurfer_ids.csv \
  --nitrc-user luanmantegazine \
  --output oasis3_data/
```

### Extract features and longitudinal tables

```bash
python cli.py run \
  --input oasis3_data/freesurfer \
  --output results/ \
  --config radiomics_config.yaml
```

### Mandatory acquisition gate

```bash
python validate_acquisition.py \
  --features results/radiomics_features_long.csv \
  --output results/acquisition_validation.json
```

Acquisition is considered structurally complete only when this command returns exit code 0.

## 8. Acceptance criteria

A session passes the acquisition gate when:

- it belongs to the frozen `Freesurfer53` acquisition cohort;
- `T1.mgz` and `aparc+aseg.mgz` are available;
- all 16 protocol ROIs produce one and only one row;
- every ROI contains all 107 expected features;
- no unexpected `original_*` feature appears;
- radiomic values are finite;
- the session is traceable to a subject and acquisition day;
- QC results are preserved, including warnings/outlier flags.

A definitive cohort is accepted when the number of **participants** satisfying the study inclusion criteria reaches the target specified in the approved project after QC. Sessions are never counted as independent participants.

## 9. Outputs retained at the end of acquisition

```text
acquisition/acquisition_subjects.csv
acquisition/acquisition_freesurfer_ids.csv
results/radiomics_features_long.csv
results/radiomics_features_wide.csv
results/radiomics_longitudinal_deltas.csv
results/radiomics_longitudinal_slopes.csv
results/quality_control.csv
results/run_metadata.json
results/acquisition_validation.json
```

The raw MRI/FreeSurfer files remain local under the credentialed data workflow and are not committed to Git.

## 10. Reproducibility freeze

Validated dependency core:

```text
Python       3.10.x
NumPy        1.26.4
PyRadiomics  3.0.1
```

NumPy 2.x is not permitted in protocol v1.0 because NumPy 2.0.1 produced a segmentation fault with the pilot environment.

Once definitive acquisition begins, changes to any of the following require a protocol version increment and complete cohort re-extraction:

- field-strength/FreeSurfer processing cohort;
- FreeSurfer segmentation source;
- ROI list/labels;
- image normalization;
- discretization/bin width;
- resampling;
- image filters;
- PyRadiomics feature list;
- PyRadiomics/NumPy versions affecting feature computation.
