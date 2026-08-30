# OASIS-3 Supervised Labeling Protocol

`SUPERVISED_LABEL_POLICY = "oasis3-supervised-labels-v1.0"` (`supervised_labels.yaml`)

Specification of how OASIS-3 clinical information becomes the `y` of a
supervised-learning experiment. No machine learning is performed here, and no
feature selection: this layer produces labels only.

A reviewer must be able to ask *"why does `OAS30001_MR_d0129` carry label CN?"*
and get an exact answer from the dataset row alone — clinical visit, time gap,
raw `dx1`, normalised `dx1`, matching rule, and policy version are all columns.

---

## 1. Clinical source

Labels come from **B4 free-text diagnoses** (`OASIS3_UDSb4_cdr.csv`, `dx1`..`dx5`).

This is a deliberate separation from `clinical_classification.yaml`, which
governs the **D1 numeric** codebook. D1's numeric semantics are not documented
in this repository, so that codebook stays unfrozen and derives nothing. The B4
text is clinically readable, so it carries its own frozen, versioned policy.
A supervised label must never rest on an unverified numeric mapping.

## 2. `dx1` mapping

`dx1` is the primary diagnosis and the sole driver of the current-state label.
Matching is **exact after normalisation** (strip, collapse whitespace,
lowercase). The original string is always preserved in the output.

**No substring rules.** `"AD dem cannot be primary"` contains `"AD"` and would
be caught by `if "AD" in dx1`, yet it states that AD is *not* the primary
etiology. It is classified `OTHER_DEMENTIA`.

All **53** distinct `dx1` values observed in the 8626-row B4 file are enumerated:

| Label | n strings | Examples |
|---|---:|---|
| `CN` | 2 | `Cognitively normal`, `No dementia` |
| `MCI` | **0** | — see §8 |
| `AD` | 32 | `AD Dementia`, `DAT`, `AD dem w/CVD contribut`, `DAT Language dysf after` |
| `OTHER_DEMENTIA` | 12 | `Vascular Demt, primary`, `DLBD, primary`, `Frontotemporal demt. prim`, `AD dem cannot be primary` |
| `UNCERTAIN` | 6 | `uncertain dementia`, `Unc: ques. Impairment`, `0.5 in memory only` |
| `NON_DIAGNOSTIC` | 1 | `Q` |

`DAT` (Dementia of the Alzheimer Type) is the WashU/ADRC synonym for AD dementia
and is mapped to `AD`, together with its qualified variants.

A string that is **not** enumerated becomes `UNMAPPED` and is excluded. There is
no fallback, so a future OASIS release cannot silently acquire a label, and an
unknown diagnosis can never default to `CN`.

## 3. `dx2`–`dx5` (secondary diagnoses)

Secondary diagnoses are used **only** for conflict detection and audit. 95
strings are enumerated across `AD`, `OTHER_DEMENTIA`, `UNCERTAIN`,
`COMORBIDITY` and `NON_DIAGNOSTIC` (`A`/`B` are NACC list selectors, not
diagnoses).

* `secondary_ad_promotes_primary: false` — a secondary AD **never** promotes a
  session into the AD class.
* A `CN` primary with a dementia secondary → `label_status = conflicting`,
  excluded from training. The contradiction is reported, not resolved.
* An `AD` primary with a non-AD dementia secondary keeps the `AD` label and is
  annotated *mixed dementia etiology*.

> Source-data note: `other mental retarAD demion` is a corrupted string in the
> B4 file — a find/replace turned `dat` into `AD dem` inside `retardation`. It
> is enumerated verbatim so it matches rather than becoming `UNMAPPED`.

## 4. Role of D1

D1 variables (`NORMCOG`, `DEMENTED`, `MCIAMEM`, `PROBAD`, `POSSAD`, `alzdis`, …)
are carried through as **raw corroborating evidence** and are never used to
assign or override a label, because their numeric semantics are undocumented
here. `label_source` records provenance as `B4_dx1`; the architecture allows a
future `B4_dx1+D1` once a verified D1 codebook is frozen.

## 5. Role of CDR and MMSE

Covariates and consistency checks — **never** labels.

`CDRTOT == 0.5` does **not** mean MCI under this policy. Disagreements are
flagged in `label_warnings` (`diagnosis_cdr_disagreement`) and never overwrite a
diagnosis:

* `CN` with `CDRTOT > 0`
* `AD` with `CDRTOT < 0.5`

## 6. Clinical–MRI matching window

Labels are only trustworthy when the clinical visit is close to the scan. A
session is training-eligible only when

```
clinical_match_valid == True   AND   clinical_mri_abs_gap_days <= 180
```

(`--clinical-window-days`, boundary inclusive). The temporal gate is applied
**before** the diagnostic one. Ineligible rows are kept in the audit dataset
with `training_eligible = False` and a `training_exclusion_reason` from:

`outside_clinical_window` · `unmapped_diagnosis` · `uncertain_diagnosis` ·
`other_dementia` · `conflicting_diagnosis` · `missing_diagnosis` ·
`non_diagnostic_value`

## 7. Definitions — CN and AD

**CN** — `dx1` states absence of dementia (`Cognitively normal`, `No dementia`),
no dementia diagnosis among `dx2`–`dx5`, and a valid clinical match.

**AD** — `dx1` names Alzheimer dementia as the **primary** etiology, including
the `DAT` synonym and the qualified `AD dem …` / `DAT …` variants. Explicitly
**excluded**: `AD dem cannot be primary`, and vascular, frontotemporal, Lewy
body and Parkinson dementias, which are `OTHER_DEMENTIA` and never folded into
`AD`.

## 8. Definition of MCI — the central finding of v1.0

**OASIS-3 B4 `dx1`–`dx5` contains no MCI label.** No value anywhere in the file
matches `/\bMCI\b/` or `mild cognitive`. `current_state.MCI.dx1_exact` is
therefore **empty**, and the MCI class and Target B are both empty.

That is a property of the data under an honest policy, not a pipeline failure.

The MCI-like categories are the WashU/ADRC *uncertain* family, classified
`UNCERTAIN` and excluded pending clinical review:

| Candidate `dx1` | B4 rows |
|---|---:|
| `uncertain dementia` | 475 |
| `Unc: ques. Impairment` | 80 |
| `0.5 in memory only` | 23 |
| `Incipient demt PTP` | 23 |
| `uncertain, possible NON AD dem` | 18 |
| `Unc: impair reversible` | 7 |
| **total** | **626** |

471 of the 475 `uncertain dementia` rows carry `CDRTOT = 0.5`, which is
*consistent* with MCI — but CDR is not a diagnosis (§5), and mapping these to
MCI is a clinical decision this repository is not entitled to make on its own.

**To define MCI**: move the chosen strings from `excluded.UNCERTAIN.dx1_exact`
into `current_state.MCI.dx1_exact`, add a `reference` justifying each, and bump
`version`. Nothing else changes — the whole Target-B machinery is implemented
and tested and activates on that edit.

`Incipient Non-AD dem` (4 rows) is `OTHER_DEMENTIA`, not an MCI candidate,
because it names a non-Alzheimer etiology.

## 9. Excluded dementia etiologies

`OTHER_DEMENTIA` covers vascular, Lewy body (DLBD), frontotemporal,
Parkinson-associated and "other non-AD" dementias, plus `AD dem cannot be
primary` and `ProAph w/o dement`. None enters the CN/MCI/AD cohort. Assuming all
dementia is Alzheimer's would be the single most damaging error available here.

## 10. Target B — `MCI_TO_AD` / `MCI_STABLE` / `CENSORED`

For a session with `diagnosis_at_mri == MCI` at day `t`, with horizon `H`
(`--progression-horizon-days`, default **1095** ≈ 3 years) and deadline
`t + H`, using **only** clinical visits of the same subject at days `> t`:

**`MCI_TO_AD`** — an `AD` diagnosis occurs in `(t, t+H]`.

**`MCI_STABLE`** — no `AD` in `(t, t+H]`, **and** the participant was observed
still non-AD at a visit on day `>= t+H`.

**`CENSORED`** (`progression_eligible = False`) — anything else.

### Why stability requires a confirming sighting

It is not enough that *some* later visit exists. Consider a scan at day 1000
with `H = 730` (deadline 1730), an MCI visit at day 1500 and an AD visit at day
2500. Follow-up extends far past the deadline, yet the last time the participant
was seen non-AD was day 1500 — the conversion may have occurred at day 1600,
inside the horizon. The outcome is unobserved, so the session is `CENSORED`.

A participant who leaves follow-up six months after an MCI scan is **not**
stable MCI. Collapsing censoring into `MCI_STABLE` injects label noise straight
into `y`.

## 11. Leakage prevention

> **Future information may define `y`. It may never be an input `x`.**

`diagnosis_at_mri` comes only from the clinical visit the scan was linked to. A
later AD diagnosis never relabels an earlier MCI scan.

These columns are registered in `FUTURE_INFORMATION_COLUMNS` and are removed
from the predictor block by `predictor_columns()`, which the dataset builder
asserts on every run:

```
future_diagnosis        conversion_event       conversion_day
days_to_conversion      progression_label      last_diagnosis
clinical_trajectory     last_followup_day      followup_days_after_mri
progression_eligible    progression_exclusion_reason
```

Permitted predictors at `t0`: radiomics measured at or before `t0`, age at `t0`,
sex, scanner, and baseline clinical covariates at or before `t0`.

### Longitudinal radiomics carry their own restriction

* A radiomic delta `t0 → t1` may only predict an outcome occurring **after
  `t1`**, not after `t0`.
* `clinical_radiomics_subjects.csv` holds **full-history** slopes computed over
  every session, including sessions recorded *after* a conversion. Using it for
  conversion prediction leaks the outcome into the feature.

Neither table is wired into the supervised datasets. Any future use must first
truncate the history at the prediction time.

## 12. Subject-level grouping is mandatory

This is a longitudinal study: one participant contributes several MRI sessions.

```
OAS30001 MRI t0 -> train
OAS30001 MRI t1 -> test     <-- SUBJECT LEAKAGE
```

Train/validation/test splits must be grouped by `subject_id`, never by session
row. Use `subject_groups(rows)` as the `groups` argument to `GroupKFold` /
`GroupShuffleSplit`, and `assert_no_subject_leakage(train, test)` to verify.

## 13. Outputs

```
supervised_dataset/
├── supervised_radiomics_sessions.csv   Target A: one row per MRI + all features
├── supervised_mci_progression.csv      Target B: MCI sessions + outcome
├── diagnosis_vocabulary.csv            every dx1..dx5 string and its mapping
└── supervised_label_audit.json         counts by session AND by subject
```

`dataset_is_final` in the audit is `true` only when no diagnosis string is
`UNMAPPED` **and** the MCI class is non-empty. Under v1.0 it is `false`, by §8.

## 14. Reproducibility

Recorded in `supervised_label_audit.json` → `parameters`:

```
label_policy_version   oasis3-supervised-labels-v1.0
label_policy_path      policy_defines_mci
clinical_window_days   progression_horizon_days
oasis_radiomics_version
clinical_radiomics / clinical_visits (absolute paths)
n_predictor_columns
```

Any change to a mapping or a rule requires bumping `version` in
`supervised_labels.yaml`. Label semantics must never change silently.
