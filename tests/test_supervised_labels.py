"""Tests for the supervised label policy and Target A / Target B derivation."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from oasis_radiomics.clinical.labels import (
    EXCLUSION_OTHER_DEMENTIA,
    EXCLUSION_OUTSIDE_WINDOW,
    EXCLUSION_UNCERTAIN,
    EXCLUSION_UNMAPPED,
    FUTURE_INFORMATION_COLUMNS,
    LABEL_AD,
    LABEL_CN,
    LABEL_MCI,
    LABEL_OTHER_DEMENTIA,
    LABEL_UNCERTAIN,
    LABEL_UNMAPPED,
    PROGRESSION_CENSORED,
    PROGRESSION_STABLE,
    PROGRESSION_TO_AD,
    STATUS_CONFLICTING,
    STATUS_MISSING,
    STATUS_UNMAPPED,
    LabelPolicy,
    LabelPolicyError,
    derive_current_label,
    derive_progression_label,
    leaking_columns,
    normalise_diagnosis,
    training_eligibility,
)


@pytest.fixture(scope="module")
def shipped_policy() -> LabelPolicy:
    """The repository's shipped v2.0 policy: D1 primary, B4 auxiliary."""
    return LabelPolicy.load(None)


@pytest.fixture(scope="module")
def policy() -> LabelPolicy:
    """The shipped policy forced to B4-primary.

    The B4 text mapping is still a live code path in v2.0 - it drives the
    auxiliary cross-check - so it keeps its own tests. Flipping
    ``primary_source`` is also the v1.0 behaviour, which must not rot.
    """
    import yaml

    raw = yaml.safe_load(Path("supervised_labels.yaml").read_text())
    raw["primary_source"] = "B4"
    return LabelPolicy.from_mapping(raw)


def _d1(**flags) -> dict:
    """A D1 visit row."""
    return {key: "1" for key in flags if flags[key]}


def _mci_policy() -> LabelPolicy:
    """A policy that DOES define MCI, for exercising Target B."""
    return LabelPolicy.from_mapping(
        {
            "version": "test-mci-v1",
            "current_state": {
                "CN": {"rule_id": "cn", "dx1_exact": ["Cognitively normal"]},
                "MCI": {"rule_id": "mci", "dx1_exact": ["uncertain dementia"]},
                "AD": {"rule_id": "ad", "dx1_exact": ["AD Dementia", "DAT"]},
            },
            "excluded": {
                "OTHER_DEMENTIA": {"rule_id": "other", "dx1_exact": ["Vascular Demt, primary"]},
                "UNCERTAIN": {"rule_id": "unc", "dx1_exact": []},
                "NON_DIAGNOSTIC": {"rule_id": "nd", "dx1_exact": ["Q"]},
            },
        }
    )


def _visits(*pairs: tuple[int, str]) -> list[dict]:
    return [{"clinical_day": day, "dx1": dx1} for day, dx1 in pairs]


# --- normalisation ---------------------------------------------------------
def test_normalisation_collapses_and_lowercases() -> None:
    assert normalise_diagnosis("  Cognitively   NORMAL ") == "cognitively normal"


@pytest.mark.parametrize("missing", ["", ".", "nan", "NaN", "none", "N/A", None])
def test_missing_tokens_normalise_to_none(missing) -> None:
    assert normalise_diagnosis(missing) is None


def test_float_nan_normalises_to_none() -> None:
    """An empty CSV cell becomes float NaN after a pandas round-trip."""
    assert normalise_diagnosis(float("nan")) is None


# --- policy integrity ------------------------------------------------------
def test_shipped_policy_maps_every_dx1_value(shipped_policy: LabelPolicy) -> None:
    assert len(shipped_policy.primary_map) == 53
    assert shipped_policy.version == "oasis3-supervised-labels-v2.1"


def test_shipped_policy_is_d1_primary(shipped_policy: LabelPolicy) -> None:
    """v2.0 change: D1 decides, B4 only cross-checks."""
    assert shipped_policy.primary_source == "D1"
    assert shipped_policy.b4_validation_enabled is True


def test_shipped_policy_defines_mci_via_d1(shipped_policy: LabelPolicy) -> None:
    """v1.0's blocker is resolved: D1 carries explicit MCI variables."""
    assert shipped_policy.defines_mci is True
    assert not shipped_policy.current_state_mci() if hasattr(
        shipped_policy, "current_state_mci"
    ) else True


def test_b4_primary_still_has_no_mci(policy: LabelPolicy) -> None:
    """The v1.0 finding stands: the B4 text alone cannot express MCI."""
    assert policy.defines_mci is False


def test_d1_primary_requires_d1_rules() -> None:
    with pytest.raises(LabelPolicyError):
        LabelPolicy.from_mapping(
            {"primary_source": "D1", "current_state": {"CN": {"dx1_exact": ["x"]}}}
        )


def test_policy_rejects_contradictory_duplicate() -> None:
    with pytest.raises(LabelPolicyError):
        LabelPolicy.from_mapping(
            {
                "current_state": {
                    "CN": {"dx1_exact": ["X"]},
                    "AD": {"dx1_exact": ["X"]},
                }
            }
        )


def test_policy_rejects_empty_mapping() -> None:
    with pytest.raises(LabelPolicyError):
        LabelPolicy.from_mapping({"current_state": {}})


def test_missing_policy_file_raises() -> None:
    with pytest.raises(LabelPolicyError):
        LabelPolicy.from_yaml(Path("does/not/exist.yaml"))


# --- Target A: the required cases -----------------------------------------
def test_cn_from_known_text(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "Cognitively normal"}, policy)
    assert label.label == LABEL_CN
    assert label.is_training_label


def test_no_dementia_is_also_cn(policy: LabelPolicy) -> None:
    assert derive_current_label({"dx1": "No dementia"}, policy)[0].label == LABEL_CN


def test_ad_from_known_text(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "AD Dementia"}, policy)
    assert label.label == LABEL_AD


def test_dat_is_ad(policy: LabelPolicy) -> None:
    """DAT (Dementia of the Alzheimer Type) is the ADRC synonym for AD."""
    assert derive_current_label({"dx1": "DAT"}, policy)[0].label == LABEL_AD


@pytest.mark.parametrize(
    "dx1",
    [
        "AD dem w/depresss, not contribut",
        "AD dem Language dysf with",
        "AD dem w/CVD contribut",
        "DAT w/depresss, contribut",
        "AD dem visuospatial, prior",
    ],
)
def test_ad_variants_are_ad(policy: LabelPolicy, dx1: str) -> None:
    assert derive_current_label({"dx1": dx1}, policy)[0].label == LABEL_AD


def test_ad_not_primary_is_not_ad(policy: LabelPolicy) -> None:
    """The substring trap: 'AD dem cannot be primary' contains 'AD'."""
    label, _ = derive_current_label({"dx1": "AD dem cannot be primary"}, policy)
    assert label.label == LABEL_OTHER_DEMENTIA
    assert label.label != LABEL_AD


@pytest.mark.parametrize(
    "dx1",
    ["Vascular Demt, primary", "DLBD, primary", "Frontotemporal demt. prim", "Dementia/PD, primary"],
)
def test_other_dementias_are_excluded(policy: LabelPolicy, dx1: str) -> None:
    label, _ = derive_current_label({"dx1": dx1}, policy)
    assert label.label == LABEL_OTHER_DEMENTIA
    eligible, reason = training_eligibility(label, True, 0, 180)
    assert eligible is False
    assert reason == EXCLUSION_OTHER_DEMENTIA


def test_uncertain_family_is_not_mci(policy: LabelPolicy) -> None:
    """The v1.0 decision: 'uncertain dementia' is UNCERTAIN, never MCI."""
    label, _ = derive_current_label({"dx1": "uncertain dementia"}, policy)
    assert label.label == LABEL_UNCERTAIN
    assert label.label != LABEL_MCI
    assert training_eligibility(label, True, 0, 180) == (False, EXCLUSION_UNCERTAIN)


def test_unknown_string_is_unmapped_never_cn(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "some diagnosis nobody has seen"}, policy)
    assert label.label == LABEL_UNMAPPED
    assert label.label != LABEL_CN
    assert label.status == STATUS_UNMAPPED
    assert training_eligibility(label, True, 0, 180) == (False, EXCLUSION_UNMAPPED)


def test_missing_diagnosis_is_reported(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": None}, policy)
    assert label.status == STATUS_MISSING
    assert not label.is_training_label


def test_label_carries_full_provenance(policy: LabelPolicy) -> None:
    """A reviewer must be able to answer 'why this label?' from the row alone."""
    label, _ = derive_current_label({"dx1": "  AD Dementia "}, policy)
    row = label.as_row()
    assert row["supervised_label"] == LABEL_AD
    assert row["label_source"] == "B4_dx1"
    assert row["label_rule_id"] == "ad_b4_dx1_exact"
    assert row["label_policy_version"] == "oasis3-supervised-labels-v2.1"
    assert row["dx1_normalized"] == "ad dementia"
    assert "AD Dementia" in row["label_reason"]


# --- secondary diagnoses ---------------------------------------------------
def test_secondary_ad_does_not_promote_a_cn_primary(policy: LabelPolicy) -> None:
    """A secondary AD must never turn a session into the AD class."""
    label, _ = derive_current_label(
        {"dx1": "Cognitively normal", "dx2": "AD dem w/depresss, contribut"}, policy
    )
    assert label.label != LABEL_AD
    assert label.status == STATUS_CONFLICTING


def test_cn_with_dementia_secondary_is_conflicting(policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"dx1": "Cognitively normal", "dx2": "Vascular Demt, secondary"}, policy
    )
    assert label.status == STATUS_CONFLICTING
    assert training_eligibility(label, True, 0, 180)[0] is False


def test_comorbidity_secondary_does_not_conflict(policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"dx1": "Cognitively normal", "dx2": "Active Mood disorder"}, policy
    )
    assert label.label == LABEL_CN
    assert label.status != STATUS_CONFLICTING


def test_ad_with_other_dementia_secondary_keeps_ad(policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"dx1": "AD Dementia", "dx2": "Vascular Demt, secondary"}, policy
    )
    assert label.label == LABEL_AD
    assert "mixed dementia etiology" in label.reason


# --- CDR / MMSE are checks, never labels ----------------------------------
def test_cdr_never_assigns_a_label(policy: LabelPolicy) -> None:
    """CDRTOT 0.5 must not create MCI."""
    label, _ = derive_current_label({"dx1": "Cognitively normal", "CDRTOT": "0.5"}, policy)
    assert label.label == LABEL_CN


def test_cdr_disagreement_is_warned_not_applied(policy: LabelPolicy) -> None:
    label, warnings = derive_current_label(
        {"dx1": "Cognitively normal", "CDRTOT": "0.5"}, policy
    )
    assert any("diagnosis_cdr_disagreement" in warning for warning in warnings)
    assert label.label == LABEL_CN


# --- temporal eligibility --------------------------------------------------
def test_invalid_clinical_match_is_not_trainable(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "Cognitively normal"}, policy)
    assert training_eligibility(label, False, 10, 180) == (False, EXCLUSION_OUTSIDE_WINDOW)


def test_gap_beyond_window_is_not_trainable(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "Cognitively normal"}, policy)
    assert training_eligibility(label, True, 300, 180) == (False, EXCLUSION_OUTSIDE_WINDOW)


def test_gap_at_the_boundary_is_trainable(policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"dx1": "Cognitively normal"}, policy)
    assert training_eligibility(label, True, 180, 180) == (True, None)


# --- Target B: progression, horizon and censoring --------------------------
def test_progression_requires_mci() -> None:
    result = derive_progression_label(1000, LABEL_CN, _visits((2000, "AD Dementia")), _mci_policy(), 1095)
    assert result.eligible is False
    assert result.exclusion_reason == "not_mci_at_mri"


def test_future_conversion_within_horizon() -> None:
    """MRI d1000, MCI at scan, AD at d1500, horizon 1095 -> MCI_TO_AD."""
    result = derive_progression_label(
        1000, LABEL_MCI, _visits((1000, "uncertain dementia"), (1500, "AD Dementia")),
        _mci_policy(), 1095,
    )
    assert result.label == PROGRESSION_TO_AD
    assert result.eligible is True
    assert result.conversion_day == 1500
    assert result.days_to_conversion == 500
    assert result.conversion_event == "MCI_to_AD"


def test_conversion_after_the_horizon_is_not_a_conversion() -> None:
    """AD beyond the horizon, with follow-up covering it -> stable in-horizon."""
    result = derive_progression_label(
        1000, LABEL_MCI,
        _visits((1500, "uncertain dementia"), (2200, "uncertain dementia"), (3000, "AD Dementia")),
        _mci_policy(), 1095,
    )
    assert result.label == PROGRESSION_STABLE


def test_stable_mci_requires_followup_through_the_horizon() -> None:
    """MRI d1000, MCI through d2200, horizon 1095 -> MCI_STABLE."""
    result = derive_progression_label(
        1000, LABEL_MCI,
        _visits((1500, "uncertain dementia"), (2200, "uncertain dementia")),
        _mci_policy(), 1095,
    )
    assert result.label == PROGRESSION_STABLE
    assert result.eligible is True
    assert result.last_followup_day == 2200


def test_short_followup_is_censored_not_stable() -> None:
    """The critical rule: MRI d1000, last visit d1500, no AD, horizon 1095."""
    result = derive_progression_label(
        1000, LABEL_MCI, _visits((1500, "uncertain dementia")), _mci_policy(), 1095
    )
    assert result.label == PROGRESSION_CENSORED
    assert result.label != PROGRESSION_STABLE
    assert result.eligible is False
    assert result.exclusion_reason == "insufficient_followup"


def test_no_followup_at_all_is_censored() -> None:
    result = derive_progression_label(1000, LABEL_MCI, _visits((500, "uncertain dementia")), _mci_policy(), 1095)
    assert result.label == PROGRESSION_CENSORED
    assert result.exclusion_reason == "no_followup_after_mri"


def test_followup_exactly_at_the_horizon_is_stable() -> None:
    result = derive_progression_label(
        1000, LABEL_MCI, _visits((2095, "uncertain dementia")), _mci_policy(), 1095
    )
    assert result.label == PROGRESSION_STABLE


def test_conversion_exactly_at_the_horizon_counts() -> None:
    result = derive_progression_label(
        1000, LABEL_MCI, _visits((2095, "AD Dementia")), _mci_policy(), 1095
    )
    assert result.label == PROGRESSION_TO_AD
    assert result.days_to_conversion == 1095


def test_visits_before_the_mri_are_ignored_for_the_outcome() -> None:
    """A past AD diagnosis cannot be the future outcome of this scan."""
    result = derive_progression_label(
        1000, LABEL_MCI,
        _visits((200, "AD Dementia"), (2200, "uncertain dementia")),
        _mci_policy(), 1095,
    )
    assert result.label == PROGRESSION_STABLE


def test_longer_horizon_increases_both_conversions_and_censoring() -> None:
    visits = _visits((1500, "uncertain dementia"), (2500, "AD Dementia"))
    # Horizon 730 -> deadline 1730. No AD inside it, but the last non-AD sighting
    # was day 1500: the conversion may have happened at day 1600. Unobserved.
    assert derive_progression_label(1000, LABEL_MCI, visits, _mci_policy(), 730).label == (
        PROGRESSION_CENSORED
    )
    assert derive_progression_label(1000, LABEL_MCI, visits, _mci_policy(), 1825).label == (
        PROGRESSION_TO_AD
    )


def test_stability_needs_a_non_ad_sighting_past_the_horizon() -> None:
    """Long follow-up is not stability if the next sighting is already AD."""
    visits = _visits((1500, "uncertain dementia"), (2500, "AD Dementia"))
    result = derive_progression_label(1000, LABEL_MCI, visits, _mci_policy(), 1095)
    assert result.label == PROGRESSION_CENSORED
    assert result.last_followup_day == 2500  # follow-up is long...
    assert result.eligible is False          # ...but the outcome is unobserved


# --- leakage ---------------------------------------------------------------
def test_future_information_columns_are_recognised() -> None:
    columns = ["original_glcm_Contrast_left", "age_at_mri", "future_diagnosis", "conversion_day"]
    assert leaking_columns(columns) == ["conversion_day", "future_diagnosis"]


def test_clean_feature_block_has_no_leaks() -> None:
    assert leaking_columns(["original_firstorder_Mean_left", "sex", "MMSE"]) == []


def test_every_future_column_is_registered() -> None:
    for column in ("future_diagnosis", "conversion_day", "days_to_conversion", "progression_label"):
        assert column in FUTURE_INFORMATION_COLUMNS


# ---------------------------------------------------------------------------
# D1 as the primary source (v2.0)
# ---------------------------------------------------------------------------
def test_d1_normcog_is_cn(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(NORMCOG=True), shipped_policy)
    assert label.label == "CN"
    assert label.source in ("D1", "D1+B4")


@pytest.mark.parametrize("variable", ["MCIAMEM", "MCIAPLUS", "MCINON1", "MCINON2"])
def test_every_d1_mci_variable_yields_mci(shipped_policy: LabelPolicy, variable: str) -> None:
    label, _ = derive_current_label({variable: "1"}, shipped_policy)
    assert label.label == LABEL_MCI


def test_d1_dementia_with_ad_etiology_is_ad(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(DEMENTED=True, PROBAD=True), shipped_policy)
    assert label.label == LABEL_AD
    assert label.etiology == "AD"


def test_d1_uds_v3_etiology_variable_also_works(shipped_policy: LabelPolicy) -> None:
    """PROBAD/POSSAD (UDS v2) and alzdis (UDS v3) are disjoint in OASIS-3."""
    label, _ = derive_current_label(_d1(DEMENTED=True, alzdis=True), shipped_policy)
    assert label.label == LABEL_AD


def test_d1_dementia_with_non_ad_etiology(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(DEMENTED=True, FTD=True), shipped_policy)
    assert label.label == "OTHER_DEMENTIA"
    assert label.etiology == "NON_AD"


def test_d1_mixed_etiology_is_not_ad(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(DEMENTED=True, alzdis=True, VASC=True), shipped_policy)
    assert label.etiology == "MIXED"
    assert label.label != LABEL_AD


def test_d1_dementia_without_etiology_is_unspecified(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(DEMENTED=True), shipped_policy)
    assert label.label == "DEMENTIA_UNKNOWN_ETIOLOGY"
    assert training_eligibility(label, True, 0, 180, shipped_policy.training_labels)[0] is False


def test_d1_impaired_not_mci(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(IMPNOMCI=True), shipped_policy)
    assert label.label == "IMPAIRED_NOT_MCI"
    assert training_eligibility(label, True, 0, 180, shipped_policy.training_labels)[0] is False


def test_d1_mci_can_carry_an_ad_etiology(shipped_policy: LabelPolicy) -> None:
    """"MCI due to AD" stays MCI, with the aetiology visible alongside."""
    label, _ = derive_current_label(_d1(MCIAPLUS=True, alzdis=True), shipped_policy)
    assert label.label == LABEL_MCI
    assert label.etiology == "AD"


def test_d1_contradiction_is_reported_not_prioritised(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(_d1(NORMCOG=True, DEMENTED=True), shipped_policy)
    assert label.label == "CONFLICTING"
    assert label.status == STATUS_CONFLICTING


def test_d1_out_of_range_value_is_not_guessed(shipped_policy: LabelPolicy) -> None:
    """OASIS-3 contains DEMENTED=2 once and IMPNOMCI=2 twice."""
    label, _ = derive_current_label({"DEMENTED": "2"}, shipped_policy)
    assert label.status == STATUS_MISSING


def test_d1_blank_form_is_missing(shipped_policy: LabelPolicy) -> None:
    assert derive_current_label({}, shipped_policy)[0].status == STATUS_MISSING


def test_d1_accepts_pandas_floats(shipped_policy: LabelPolicy) -> None:
    """A CSV round-trip turns "1" into 1.0."""
    assert derive_current_label({"NORMCOG": 1.0}, shipped_policy)[0].label == "CN"


# --- the auxiliary B4 cross-check -----------------------------------------
def test_b4_agreement_is_recorded(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"NORMCOG": "1", "dx1": "Cognitively normal"}, shipped_policy
    )
    assert label.b4_agreement == "agree"
    assert label.source == "D1+B4"


def test_b4_disagreement_warns_but_never_overrides(shipped_policy: LabelPolicy) -> None:
    label, warnings = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "dx1": "Cognitively normal"}, shipped_policy
    )
    assert label.label == LABEL_AD          # D1 wins
    assert label.b4_agreement == "disagree"
    assert any("d1_b4_disagreement" in warning for warning in warnings)


def test_missing_b4_leaves_the_d1_label_intact(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"NORMCOG": "1"}, shipped_policy)
    assert label.label == "CN"
    assert label.b4_agreement == "b4_unavailable"
    assert label.source == "D1"


# --- Target B now reads future visits through D1 ---------------------------
def test_progression_reads_future_visits_via_d1(shipped_policy: LabelPolicy) -> None:
    visits = [
        {"clinical_day": 1500, "MCIAMEM": "1"},
        {"clinical_day": 2000, "DEMENTED": "1", "alzdis": "1"},
    ]
    result = derive_progression_label(1000, LABEL_MCI, visits, shipped_policy, 1095)
    assert result.label == PROGRESSION_TO_AD
    assert result.conversion_day == 2000
    assert result.days_to_conversion == 1000


def test_progression_non_ad_dementia_is_not_a_conversion(shipped_policy: LabelPolicy) -> None:
    """Converting to vascular dementia is not MCI_TO_AD."""
    visits = [
        {"clinical_day": 1500, "DEMENTED": "1", "VASC": "1"},
        {"clinical_day": 2500, "DEMENTED": "1", "VASC": "1"},
    ]
    result = derive_progression_label(1000, LABEL_MCI, visits, shipped_policy, 1095)
    assert result.label != PROGRESSION_TO_AD


# ---------------------------------------------------------------------------
# D1 aetiology roles: the "IF" fields are {0, 1, 2}, not binary
# ---------------------------------------------------------------------------
def test_if_field_marks_a_primary_aetiology(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "PROBADIF": "1"}, shipped_policy
    )
    assert label.label == LABEL_AD
    assert label.ad_etiology_role == "primary"


def test_contributing_aetiology_does_not_make_a_visit_ad(shipped_policy: LabelPolicy) -> None:
    """IF == 2 means AD contributes but is not the primary cause."""
    label, _ = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "PROBADIF": "2"}, shipped_policy
    )
    assert label.ad_etiology_role == "contributing"
    assert label.label != LABEL_AD
    assert label.label == "DEMENTIA_UNKNOWN_ETIOLOGY"


def test_blank_if_field_leaves_the_role_unspecified(shipped_policy: LabelPolicy) -> None:
    """A missing qualifier must not silently demote a flagged aetiology."""
    label, _ = derive_current_label({"DEMENTED": "1", "PROBAD": "1"}, shipped_policy)
    assert label.ad_etiology_role == "unspecified"
    assert label.label == LABEL_AD


def test_out_of_domain_if_value_is_not_guessed(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "PROBADIF": "9"}, shipped_policy
    )
    assert label.ad_etiology_role == "unspecified"


@pytest.mark.parametrize(
    "flag, qualifier", [("PROBAD", "PROBADIF"), ("POSSAD", "POSSADIF"), ("alzdis", "alzdisif")]
)
def test_both_uds_generations_carry_their_qualifier(
    shipped_policy: LabelPolicy, flag: str, qualifier: str
) -> None:
    """v1/v2 and v3 are disjoint representations; neither is required."""
    label, _ = derive_current_label(
        {"DEMENTED": "1", flag: "1", qualifier: "1"}, shipped_policy
    )
    assert label.label == LABEL_AD
    assert label.ad_etiology_role == "primary"


def test_primary_wins_over_contributing_across_flags(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "PROBADIF": "2", "POSSAD": "1", "POSSADIF": "1"},
        shipped_policy,
    )
    assert label.ad_etiology_role == "primary"
    assert label.label == LABEL_AD


# ---------------------------------------------------------------------------
# MCI subtypes: qualifiers describe, they never decide
# ---------------------------------------------------------------------------
def test_amnestic_single_domain_subtype(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label({"MCIAMEM": "1"}, shipped_policy)
    assert label.label == LABEL_MCI
    assert label.mci_subtype == "amnestic_single_domain"


def test_amnestic_multi_domain_records_its_domains(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"MCIAPLUS": "1", "MCIAPEX": "1", "MCIAPLAN": "1"}, shipped_policy
    )
    assert label.mci_subtype == "amnestic_multi_domain"
    assert label.mci_domains == "executive, language"


@pytest.mark.parametrize(
    "core, subtype",
    [
        ("MCINON1", "non_amnestic_single_domain"),
        ("MCINON2", "non_amnestic_multi_domain"),
    ],
)
def test_non_amnestic_subtypes(shipped_policy: LabelPolicy, core: str, subtype: str) -> None:
    assert derive_current_label({core: "1"}, shipped_policy)[0].mci_subtype == subtype


def test_subtype_qualifier_alone_does_not_create_mci(shipped_policy: LabelPolicy) -> None:
    """An orphaned domain flag must not manufacture an MCI label."""
    label, _ = derive_current_label({"MCIAPEX": "1"}, shipped_policy)
    assert label.label != LABEL_MCI
    assert label.status == STATUS_MISSING


def test_mci_can_carry_an_ad_aetiology_with_its_role(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"MCIAPLUS": "1", "alzdis": "1", "alzdisif": "1"}, shipped_policy
    )
    assert label.label == LABEL_MCI
    assert label.etiology == "AD"
    assert label.ad_etiology_role == "primary"


# ---------------------------------------------------------------------------
# B4 as an independent comparison source
# ---------------------------------------------------------------------------
def test_b4_label_is_emitted_independently(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"NORMCOG": "1", "dx1": "Cognitively normal"}, shipped_policy
    )
    assert label.b4_label == "CN"
    assert label.b4_agreement == "agree"
    assert label.b4_disagreement_reason is None


def test_b4_disagreement_reason_is_recorded(shipped_policy: LabelPolicy) -> None:
    label, _ = derive_current_label(
        {"DEMENTED": "1", "PROBAD": "1", "dx1": "Cognitively normal"}, shipped_policy
    )
    assert label.label == LABEL_AD                 # D1 decides
    assert label.b4_label == "CN"
    assert label.b4_agreement == "disagree"
    assert "B4=CN" in label.b4_disagreement_reason


def test_b4_comparison_label_is_derived_without_d1(shipped_policy: LabelPolicy) -> None:
    from oasis_radiomics.clinical.labels import derive_b4_comparison_label

    assert derive_b4_comparison_label({"dx1": "AD Dementia"}, shipped_policy) == "AD"
    assert derive_b4_comparison_label({"dx1": "never seen"}, shipped_policy) == "UNMAPPED"
    assert derive_b4_comparison_label({}, shipped_policy) is None
