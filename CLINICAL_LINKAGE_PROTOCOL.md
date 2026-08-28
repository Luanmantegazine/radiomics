# OASIS-3 Clinical Linkage Protocol

`CLINICAL_LINKAGE_VERSION = "oasis3-clinical-linkage-v1.0"`

This document is the frozen specification of how OASIS-3 clinical assessments
are attached to MRI sessions. It exists so that a reviewer can reconstruct, from
the outputs alone, exactly why any given scan carries the clinical row it does.

The linkage does **not** change the radiomics protocol
(`ACQUISITION_PROTOCOL.md`, `oasis_radiomics/protocol.py`): 16 ROIs,
107 features per ROI, 1712 raw features per session, Original image only.
Radiomics outputs are read-only inputs to this layer.

---

## 1. Sources

| Instrument | File | Role |
|---|---|---|
| **D1** | `OASIS3_UDSd1_diagnoses.csv` | **primary diagnostic source** (NORMCOG, DEMENTED, MCI\*, PROBAD, POSSAD, ...) |
| **B4** | `OASIS3_UDSb4_cdr.csv` | CDR (`CDRTOT`, `CDRSUM`), `MMSE`, free-text `dx1`..`dx5`; complementary to D1 |
| **C1** | `OASIS3_UDSc1_cognitive_assessments.csv` | optional psychometrics (fluency, Trail Making, logical memory, ...) |
| MRI catalogue | OASIS session export | `Label`, `Subject`, `M/F`, `Age`, `Scanner` |

Source files are opened read-only and are never rewritten.

## 2. Identifier and time normalisation

| Source column | Internal name |
|---|---|
| `OASISID` | `subject_id` |
| `OASIS_session_label` | `clinical_session_id` |
| `days_to_visit` | `clinical_day` |
| `age at visit` | `age_at_clinical_visit` |
| catalogue `Label` | `session_id`, and `subject_id` / `mri_day` re-derived from it |

Two conventions in the raw data are normalised:

* `days_to_visit` is zero-padded in D1 (`"0339"`) and plain in B4 (`"339"`);
  both parse to the same integer.
* `""` and `"."` both mean "absent" and both become `None`.

Subject and day for an MRI session are re-derived from the `Label` with the
repository's existing parser rather than trusting the catalogue's `Subject`
column; a disagreement raises `mri_subject_mismatch`.

## 3. Clinical visit construction (D1 + B4)

A **clinical visit** is one `(subject_id, clinical_day)` pair.

* D1 and B4 are joined on `OASISID` **and** `days_to_visit`.
* Either side may be missing. A D1 diagnosis is never dropped because B4 is
  absent (`missing_b4`), and vice versa (`missing_d1`).
* Two rows of the same instrument on the same day are **not** silently merged:
  the record with the lexicographically smallest session id is kept so the
  result is deterministic, and `duplicate_clinical_visit` names both.

Observed on the real files: 8650 visits — 8475 with D1+B4, 24 D1-only,
151 B4-only.

## 4. MRI ↔ clinical matching

MRI and clinical visits do not share a calendar. For each MRI session:

```
clinical_mri_gap_days     = clinical_day - mri_day      (signed)
clinical_mri_abs_gap_days = abs(clinical_day - mri_day)
```

**Rules**

1. **Same subject only.** Candidates are restricted to the identical
   `subject_id`. Participants are never merged across OASIS ids.
2. **Nearest wins.** The visit minimising `clinical_mri_abs_gap_days` is
   selected.
3. **Ties prefer the earlier visit.** When two visits are equidistant the
   earlier one is chosen and `clinical_match_ambiguous = True` with
   `clinical_match_reason = ambiguous_equal_distance`. Preferring the past is
   the conservative choice: it never lets a later assessment describe an
   earlier scan. The result does not depend on input ordering.
4. **Window.** `|gap| <= --clinical-window-days` (default **±180 days**,
   boundary inclusive) sets `clinical_match_valid`.
5. **Nothing is discarded.** A session whose nearest visit is outside the
   window is kept with `clinical_match_valid = False` and
   `reason = outside_window`. A session whose subject has no clinical data at
   all is kept with `reason = no_clinical_visit`.

`matching_strategy = "nearest-absolute-gap;ties-prefer-earlier"`.

### C1 is matched separately

Psychometric testing rarely falls on the diagnostic visit day, so C1 gets its
own nearest-match pass and its own columns (`cognitive_day`,
`cognitive_mri_gap_days`, `cognitive_match_valid`). Forcing C1 onto the D1/B4
day would silently misstate both gaps.

## 5. Classification — and why it is inactive by default

The numeric semantics of the D1 variables are defined by the official NACC UDS
Data Element Dictionary, which is **not part of this repository**. This project
therefore does not guess at them.

All clinical meaning lives in `clinical_classification.yaml`. Nothing is
hard-coded in Python. While `codebook_frozen: false`:

```
cognitive_status      = UNKNOWN
ad_etiology           = UNKNOWN
classification_status = unresolved_codebook
```

and **every raw D1/B4/C1 variable is still written to the outputs** (145 + 19 +
103 columns), so no information is lost and no unverified assumption enters the
dataset. `UNKNOWN` never means "normal".

Controlled vocabularies: `cognitive_status ∈ {CN, MCI, DEMENTIA, UNKNOWN}`,
`ad_etiology ∈ {AD, NON_AD, UNCERTAIN, UNKNOWN}`.

Freezing is deliberately hard to do by accident:

* an empty frozen codebook is rejected;
* every rule in a frozen codebook must carry a `reference` citation;
* values outside the controlled vocabularies are rejected;
* when two matching rules disagree the visit becomes
  `classification_status = conflicting` — the disagreement is reported, never
  resolved.

The `version` string is recorded in every output as `classification_version`.

## 6. Trajectories and the anti-leakage rule

> **A diagnosis recorded after an MRI never relabels that MRI.**

Visits are ordered by `clinical_day`. Per subject the pipeline derives
`baseline_diagnosis`, `last_diagnosis`, a collapsed path
(`CN -> MCI -> AD`), and **every** progression, not just the first — a
participant can convert twice, and an MRI taken between the two needs the
second one.

For each MRI session:

| Column | Meaning |
|---|---|
| `diagnosis_at_mri` | status at the visit **this scan was linked to**, and nothing else |
| `future_diagnosis` | status at the last visit strictly **after** `mri_day` |
| `conversion_event` | next progression strictly after `mri_day` (e.g. `MCI_to_AD`) |
| `conversion_day`, `days_to_conversion` | when it happens, relative to the scan |

Worked example (clinical d0500 = MCI, d1200 = MCI, d2000 = AD; MRI at d0800):

```
diagnosis_at_mri   = MCI      <- NOT AD
future_diagnosis   = AD
conversion_event   = MCI_to_AD
conversion_day     = 2000
days_to_conversion = 1200
```

`diagnosis_label()` is a naming convention only: `DEMENTIA` + `AD` is written
`AD`, `DEMENTIA` + anything else stays `DEMENTIA`. It interprets nothing.

A history whose recorded severity *decreases* is flagged
(`non_monotonic_trajectory`) and left exactly as recorded.

## 7. Radiomics join

Exact join on `subject_id` + `session_id` — no temporal tolerance; an MRI
session is itself.

| Output | Grain | Built from |
|---|---|---|
| `clinical_radiomics_sessions.csv` | one row per MRI session | `radiomics_features_wide.csv` |
| `clinical_radiomics_deltas.csv` | one row per visit pair | `radiomics_longitudinal_deltas.csv` |
| `clinical_radiomics_subjects.csv` | one row per subject × feature | `radiomics_longitudinal_slopes.csv` |

The session join is driven by the radiomics table: a radiomics session absent
from the clinical master is kept with `clinical_master_found = False`, never
dropped. Radiomic columns are written last, so a clinical column can never
shadow a feature.

`conversion_between_visits` compares the conversion day against the interval of
the two **matched clinical days**, not the MRI days. Using MRI days would miss a
conversion recorded at the very visit `t1` was matched to whenever that visit
falls a few days after the scan — the common case.

## 8. Validation

`clinical_linkage_validation.json` carries the run `parameters`, a `summary`,
and the issue list. Codes:

| Code | Severity | Meaning |
|---|---|---|
| `duplicate_clinical_visit` | warning | two rows of one instrument on one day |
| `duplicate_mri_session` | warning | repeated session id in the catalogue |
| `mri_subject_mismatch` | error | `Subject` column disagrees with the `Label` |
| `invalid_day_value` | warning | `days_to_visit` unparsable; row skipped |
| `implausible_day_value` | warning | `days_to_visit` negative; row kept and flagged |
| `missing_d1` / `missing_b4` | info | visit covered by only one instrument |
| `no_clinical_visit` | warning | subject has no clinical data |
| `outside_window` | warning | nearest visit beyond the window; kept, `valid=False` |
| `ambiguous_equal_distance` | warning | tie resolved toward the earlier visit |
| `non_monotonic_trajectory` | warning | recorded severity improves over time |
| `conflicting_diagnosis_same_day` | warning | codebook rules disagree |
| `duplicate_master_row` | error | more than one row per `subject_id + session_id` |

Nothing is ever removed on the strength of an issue. Exclusion is a scientific
decision and stays with the investigator.

`implausible_day_value` is not hypothetical: OASIS-3 contains negative
`days_to_visit`, e.g. `OAS30753_UDSb4_d-39520`.

## 9. Reproducibility

Recorded in `clinical_linkage_validation.json` → `parameters`:

```
clinical_linkage_version   oasis3-clinical-linkage-v1.0
oasis_radiomics_version
clinical_window_days
cognitive_window_days
matching_strategy          nearest-absolute-gap;ties-prefer-earlier
classification_version
classification_frozen
mri_catalog / d1 / b4 / c1 (absolute paths)
```

Changing a matching rule requires bumping `CLINICAL_LINKAGE_VERSION`. Matching
behaviour must not change silently.
