# OASIS-3 Supervised Labeling Protocol

`SUPERVISED_LABEL_POLICY = "oasis3-supervised-labels-v2.1"` (`supervised_labels.yaml`)

> **v2.0 changes the primary label source from B4 free text to the D1 diagnosis
> form.** v1.0 found that B4 `dx1`..`dx5` carries no MCI label at all, which left
> the MCI class and the whole MCI→AD target empty. D1 has explicit MCI variables
> and an explicit aetiology block, so it expresses the full taxonomy. B4 is
> retained as **auxiliary validation** and now populates `b4_agreement`.

Specification of how OASIS-3 clinical information becomes the `y` of a
supervised-learning experiment. No machine learning is performed here, and no
feature selection: this layer produces labels only.

A reviewer must be able to ask *"why does `OAS30001_MR_d0129` carry label CN?"*
and get an exact answer from the dataset row alone — clinical visit, time gap,
raw `dx1`, normalised `dx1`, matching rule, and policy version are all columns.

---

## 1. Clinical source

Labels come from **D1** (`OASIS3_UDSd1_diagnoses.csv`). **B4** free text
(`dx1`..`dx5`) is auxiliary validation only.

### The D1 coding is validated, not assumed

v1.0 refused to read D1 because its numeric semantics are not documented in this
repository. v2.0 does not assume them either — it **validates** them against the
independent B4 free text over the 8475 visits carrying both instruments. Every
`== 1` flag, against the v1.0 B4 text mapping:

| D1 flag | n | agreement with B4 |
|---|---:|---|
| `NORMCOG` | 5785 | **99.7 %** CN |
| `DEMENTED` | 1371 | **92.6 %** dementia (AD + non-AD) |
| `MCIAMEM` | 83 | 86.7 % B4 "uncertain" family |
| `MCIAPLUS` | 115 | 68.7 % B4 "uncertain" family |
| `IMPNOMCI` | 255 | **91.4 %** B4 "uncertain" family |
| `PROBAD` | 643 | **98.1 %** AD |
| `alzdis` | 520 | **95.8 %** AD |
| `FTD` | 9 | **100 %** non-AD dementia |

This closes the v1.0 finding: the B4 "uncertain dementia" family that v1.0
refused to call MCI is precisely what D1 codes as MCI / impaired-not-MCI.

Values are binary. Anything outside `{0, 1}` is treated as unusable and never
guessed — D1 contains three such cells (`DEMENTED=2` once, `IMPNOMCI=2` twice).
Blank cells are NACC skip patterns, not zeros.

`clinical_classification.yaml` is unaffected and remains unfrozen: it governs a
different derivation (`cognitive_status` in the clinical linkage layer). The
validation above is what licenses D1 here, and it is recorded per row in
`b4_agreement`.

### Taxonomy

`cognitive_status` × `ad_etiology` → `supervised_label`:

| D1 cognitive state | aetiology | label |
|---|---|---|
| `NORMCOG` | — | `CN` |
| `MCIAMEM` / `MCIAPLUS` / `MCINON1` / `MCINON2` | any | `MCI` |
| `IMPNOMCI` | — | `IMPAIRED_NOT_MCI` |
| `DEMENTED` | AD | `AD` |
| `DEMENTED` | non-AD or mixed | `OTHER_DEMENTIA` |
| `DEMENTED` | none established | `DEMENTIA_UNKNOWN_ETIOLOGY` |
| two states at once | — | `CONFLICTING` |
| no state set | — | `MISSING` |

Training cohort: **`CN` / `MCI` / `AD`**. Aetiology stays in its own column, so
"MCI due to AD" remains visible without changing the label.

### Aetiology across UDS generations

UDS v1/v2 (`PROBAD`, `POSSAD`) and UDS v3 (`alzdis`) are version-dependent
representations of the same concept. They are disjoint in OASIS-3 — never both
filled on one visit — and neither is required to exist.

### The paired "IF" fields are not binary

Each AD flag has a companion qualifier — `PROBADIF`, `POSSADIF`, `alzdisif` —
whose domain is **`{0, 1, 2}`**, where `1` = the aetiology is the **primary**
cause and `2` = it merely **contributes**. Observed in OASIS-3: `PROBADIF=2`
×9, `POSSADIF=2` ×39, `alzdisif=2` ×6.

By default `ad_etiology_roles_accepted: [primary, unspecified]`. A
contributing-only AD aetiology therefore does **not** put a demented visit in
the `AD` class — AD is present but is not the cause — mirroring the exclusion
of the B4 string `AD dem cannot be primary`. A blank qualifier leaves the role
`unspecified` and does **not** demote the flag; a value outside the domain is
recorded as `unspecified`, never guessed.

The role is written to every row as `ad_etiology_role`, so the decision is
auditable and the policy can be relaxed without touching code.

> On the current cohort this rule changes **no** label: the 7 contributing
> sessions are `IMPAIRED_NOT_MCI` (2), `OTHER_DEMENTIA` (3, where a non-AD
> aetiology dominates) and `MCI` (2). None is a demented visit whose only AD
> evidence is contributing.

### MCI subtype qualifiers

The four core indicators decide *whether* a visit is MCI. The companion domain
fields describe *which kind* and never change the label:

| core indicator | subtype | domain fields |
|---|---|---|
| `MCIAMEM` | `amnestic_single_domain` | — |
| `MCIAPLUS` | `amnestic_multi_domain` | `MCIAPLAN` `MCIAPATT` `MCIAPEX` `MCIAPVIS` |
| `MCINON1` | `non_amnestic_single_domain` | `MCIN1LAN` `MCIN1ATT` `MCIN1EX` `MCIN1VIS` |
| `MCINON2` | `non_amnestic_multi_domain` | `MCIN2LAN` `MCIN2ATT` `MCIN2EX` `MCIN2VIS` |

Written to `mci_subtype` and `mci_domains`. An orphaned domain flag (set
without its core indicator — 2 such cells in OASIS-3, both `MCIAPEX`) never
manufactures an MCI label.

### B4 is an independent comparison source

B4 no longer determines any label. `derive_b4_comparison_label()` derives a
label from `dx1` **without seeing the D1 result**, and three columns record the
comparison:

| column | meaning |
|---|---|
| `b4_label` | the label B4 `dx1` would give on its own |
| `b4_agreement` | `agree` / `disagree` / `b4_unavailable` / `not_comparable` |
| `b4_disagreement_reason` | why, when they differ |

A disagreement is warned in `label_warnings`, never applied.

Multiple cognitive states firing at once produce `CONFLICTING`, **not** a
priority win — 14 OASIS-3 visits are affected.

## 2. `dx1` mapping (auxiliary validation)

Under v2.0 this mapping no longer assigns labels. It powers `b4_agreement`, and
remains the active path when `primary_source: B4` (the v1.0 behaviour, still
tested).
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

## 8. Definition of MCI — resolved in v2.0

**OASIS-3 B4 `dx1`–`dx5` contains no MCI label.** No value anywhere in the file
matches `/\bMCI\b/` or `mild cognitive`. That was v1.0's blocker.

**v2.0 resolves it from D1**, which carries explicit MCI variables:
`MCIAMEM` and `MCIAPLUS` (amnestic) and `MCINON1` / `MCINON2` (non-amnestic).
Any of them set to 1 yields `MCI`. `IMPNOMCI` yields the separate
`IMPAIRED_NOT_MCI` class, which is **not** MCI and is excluded from training.

The B4 cross-check confirms the reading: 77–91 % of the D1 MCI and
impaired-not-MCI visits fall in the B4 "uncertain" family — exactly the strings
v1.0 declined to interpret.

The historical v1.0 finding is preserved below, because it still constrains what
B4 alone can express.

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
