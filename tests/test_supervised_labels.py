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
def policy() -> LabelPolicy:
    """The repository's shipped v1.0 policy."""
    return LabelPolicy.load(None)


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
def test_shipped_policy_maps_every_dx1_value(policy: LabelPolicy) -> None:
    assert len(policy.primary_map) == 53
    assert policy.version == "oasis3-supervised-labels-v1.0"


def test_shipped_policy_defines_no_mci(policy: LabelPolicy) -> None:
    """v1.0 finding: OASIS-3 B4 dx1 carries no MCI label."""
    assert policy.defines_mci is False
    assert not policy.current_state_mci() if hasattr(policy, "current_state_mci") else True


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
    assert row["label_policy_version"] == "oasis3-supervised-labels-v1.0"
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
