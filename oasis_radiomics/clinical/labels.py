"""Supervised-learning labels derived from the OASIS-3 B4 textual diagnoses.

Two independent targets, deliberately not mixed:

**Target A - current clinical state.** ``CN`` / ``MCI`` / ``AD`` for one MRI
session, from the diagnosis at the clinical visit that session was linked to.

**Target B - future progression.** ``MCI_TO_AD`` / ``MCI_STABLE`` / ``CENSORED``
for sessions whose current state is MCI, from clinical visits occurring strictly
*after* the scan.

Relationship to :mod:`oasis_radiomics.clinical.classification`
--------------------------------------------------------------
That module governs the D1 **numeric** codebook, whose semantics are not
documented in this repository and therefore stay unfrozen. This module governs
the B4 **textual** diagnoses, which are clinically readable, and carries its own
version (``SUPERVISED_LABEL_POLICY`` in ``supervised_labels.yaml``). The two are
separate on purpose: a supervised label must never depend on an unverified
numeric mapping, and D1 remains available as raw corroborating evidence.

Guarantees
----------
* Matching is **exact after normalisation**. No substring rules: ``"AD dem
  cannot be primary"`` contains ``"AD"`` yet explicitly means AD is *not* the
  primary etiology.
* An unlisted string becomes ``UNMAPPED`` and is excluded. There is no fallback,
  so an unknown diagnosis can never silently become ``CN``.
* Raw values always survive alongside the derived label.
* A diagnosis recorded after an MRI never changes that MRI's current-state
  label; it can only contribute to Target B.
* Absence of a later AD diagnosis is **not** stability. Without follow-up
  reaching the horizon the session is ``CENSORED``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_POLICY_FILENAME = "supervised_labels.yaml"

# --- Target A vocabulary ---------------------------------------------------
LABEL_CN = "CN"
LABEL_MCI = "MCI"
LABEL_AD = "AD"
TRAINING_LABELS = (LABEL_CN, LABEL_MCI, LABEL_AD)

LABEL_OTHER_DEMENTIA_D1 = "OTHER_DEMENTIA"
LABEL_IMPAIRED_NOT_MCI = "IMPAIRED_NOT_MCI"
LABEL_DEMENTIA_UNKNOWN_ETIOLOGY = "DEMENTIA_UNKNOWN_ETIOLOGY"
LABEL_CONFLICTING = "CONFLICTING"
LABEL_OTHER_DEMENTIA = "OTHER_DEMENTIA"
LABEL_UNCERTAIN = "UNCERTAIN"
LABEL_NON_DIAGNOSTIC = "NON_DIAGNOSTIC"
LABEL_UNMAPPED = "UNMAPPED"
LABEL_MISSING = "MISSING"
ALL_LABELS = TRAINING_LABELS + (
    LABEL_IMPAIRED_NOT_MCI,
    LABEL_DEMENTIA_UNKNOWN_ETIOLOGY,
    LABEL_CONFLICTING,
    LABEL_OTHER_DEMENTIA,
    LABEL_UNCERTAIN,
    LABEL_NON_DIAGNOSTIC,
    LABEL_UNMAPPED,
    LABEL_MISSING,
)

# --- Target B vocabulary ---------------------------------------------------
PROGRESSION_TO_AD = "MCI_TO_AD"
PROGRESSION_STABLE = "MCI_STABLE"
PROGRESSION_CENSORED = "CENSORED"

# --- statuses and exclusion reasons ----------------------------------------
#: D1-derived cognitive states, in the order the policy evaluates them.
STATE_DEMENTIA = "DEMENTIA"
STATE_MCI = "MCI"
STATE_IMPAIRED_NOT_MCI = "IMPAIRED_NOT_MCI"
STATE_CN = "CN"

ETIOLOGY_AD = "AD"
ETIOLOGY_NON_AD = "NON_AD"
ETIOLOGY_MIXED = "MIXED"
ETIOLOGY_UNKNOWN = "UNKNOWN"

#: Role a flagged aetiology plays, from the paired NACC "IF" field. Its domain
#: is {0, 1, 2} - NOT binary: 1 = primary cause, 2 = contributing cause.
ROLE_PRIMARY = "primary"
ROLE_CONTRIBUTING = "contributing"
ROLE_UNSPECIFIED = "unspecified"
ROLE_CODES = {"1": ROLE_PRIMARY, "2": ROLE_CONTRIBUTING}

SOURCE_D1 = "D1"
SOURCE_D1_B4 = "D1+B4"

AGREEMENT_AGREE = "agree"
AGREEMENT_DISAGREE = "disagree"
AGREEMENT_UNAVAILABLE = "b4_unavailable"
AGREEMENT_NOT_COMPARABLE = "not_comparable"

STATUS_LABELLED = "labelled"
STATUS_CONFLICTING = "conflicting"
STATUS_UNMAPPED = "unmapped"
STATUS_MISSING = "missing_diagnosis"

EXCLUSION_OUTSIDE_WINDOW = "outside_clinical_window"
EXCLUSION_UNMAPPED = "unmapped_diagnosis"
EXCLUSION_UNCERTAIN = "uncertain_diagnosis"
EXCLUSION_OTHER_DEMENTIA = "other_dementia"
EXCLUSION_CONFLICTING = "conflicting_diagnosis"
EXCLUSION_MISSING = "missing_diagnosis"
EXCLUSION_NON_DIAGNOSTIC = "non_diagnostic_value"
EXCLUSION_OTHER_DEMENTIA_D1 = "other_dementia"
EXCLUSION_IMPAIRED_NOT_MCI = "impaired_not_mci"
EXCLUSION_DEMENTIA_UNKNOWN_ETIOLOGY = "dementia_unknown_etiology"
EXCLUSION_MIXED_ETIOLOGY = "mixed_etiology"

#: Columns that encode information from *after* the scan. They may define y and
#: must never be handed to a model as x. Enforced by :func:`leaking_columns`.
FUTURE_INFORMATION_COLUMNS = frozenset(
    {
        "future_diagnosis",
        "conversion_event",
        "conversion_day",
        "days_to_conversion",
        "progression_label",
        "last_diagnosis",
        "clinical_trajectory",
        "last_followup_day",
        "followup_days_after_mri",
        "progression_eligible",
        "progression_exclusion_reason",
        "conversion_visit_dx1",
    }
)


class LabelPolicyError(ValueError):
    """Raised when the supervised label policy is structurally invalid."""


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------
#: Everything that means "no diagnosis recorded". ``"."`` is the OASIS missing
#: token; ``"nan"``/``"none"`` appear once a table has been through pandas,
#: which turns an empty CSV cell into a float NaN.
MISSING_DIAGNOSIS_TOKENS = frozenset({"", ".", "nan", "none", "na", "n/a", "null"})


def normalise_diagnosis(value: Any, case_insensitive: bool = True) -> str | None:
    """Normalise a raw diagnosis string for exact matching.

    Strips, collapses internal whitespace and (by default) lowercases. Anything
    meaning "not recorded" becomes ``None`` - including a float NaN, which is
    what an empty CSV cell becomes after a pandas round-trip and which would
    otherwise be stringified into a bogus ``"nan"`` diagnosis. The original
    value is never modified in place; callers keep it for the outputs.
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    text = " ".join(str(value).split())
    if text.lower() in MISSING_DIAGNOSIS_TOKENS:
        return None
    return text.lower() if case_insensitive else text


# ---------------------------------------------------------------------------
# typed results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SupervisedLabel:
    """One derived supervised label with its full provenance.

    Carries enough to answer "why does this session have this label?" without
    re-running anything: the raw string, its normalised form, the rule that
    matched, and the policy version.
    """

    label: str
    source: str
    rule_id: str
    confidence: str
    reason: str
    status: str = STATUS_LABELLED
    policy_version: str | None = None
    raw_value: str | None = None
    normalized_value: str | None = None
    #: D1 aetiology behind the label: AD / NON_AD / MIXED / UNKNOWN. Kept
    #: separate so "MCI due to AD" stays visible even though it collapses into
    #: the MCI label.
    etiology: str = ETIOLOGY_UNKNOWN
    #: Role the AD aetiology plays: primary / contributing / unspecified.
    ad_etiology_role: str | None = None
    #: MCI subtype from the D1 qualifier fields, e.g. amnestic_multi_domain.
    mci_subtype: str | None = None
    #: Impaired cognitive domains recorded alongside the MCI subtype.
    mci_domains: str | None = None
    #: Independent comparison label derived from B4 dx1.
    b4_label: str | None = None
    #: Outcome of the auxiliary B4 cross-check.
    b4_agreement: str = AGREEMENT_NOT_COMPARABLE
    #: Why D1 and B4 disagree, when they do.
    b4_disagreement_reason: str | None = None

    @property
    def is_training_label(self) -> bool:
        """Whether this label belongs to the CN/MCI/AD cohort."""
        return self.label in TRAINING_LABELS and self.status == STATUS_LABELLED

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the supervised dataset."""
        return {
            "supervised_label": self.label,
            "label_source": self.source,
            "label_rule_id": self.rule_id,
            "label_status": self.status,
            "label_confidence": self.confidence,
            "label_reason": self.reason,
            "label_policy_version": self.policy_version,
            "dx1_normalized": self.normalized_value,
            "ad_etiology": self.etiology,
            "ad_etiology_role": self.ad_etiology_role,
            "mci_subtype": self.mci_subtype,
            "mci_domains": self.mci_domains,
            "b4_label": self.b4_label,
            "b4_agreement": self.b4_agreement,
            "b4_disagreement_reason": self.b4_disagreement_reason,
        }


@dataclass(frozen=True)
class ProgressionLabel:
    """Target-B outcome for one MCI session."""

    label: str | None
    eligible: bool
    exclusion_reason: str | None
    conversion_event: str | None = None
    conversion_day: int | None = None
    days_to_conversion: int | None = None
    last_followup_day: int | None = None
    followup_days_after_mri: int | None = None
    horizon_days: int | None = None
    reason: str = ""

    def as_row(self) -> dict[str, Any]:
        """Flat mapping for the progression dataset."""
        return {
            "progression_label": self.label,
            "progression_eligible": self.eligible,
            "progression_exclusion_reason": self.exclusion_reason,
            "conversion_event": self.conversion_event,
            "conversion_day": self.conversion_day,
            "days_to_conversion": self.days_to_conversion,
            "last_followup_day": self.last_followup_day,
            "followup_days_after_mri": self.followup_days_after_mri,
            "progression_horizon_days": self.horizon_days,
            "progression_reason": self.reason,
        }


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LabelPolicy:
    """The parsed ``supervised_labels.yaml``."""

    version: str
    primary_column: str = "dx1"
    secondary_columns: tuple[str, ...] = ("dx2", "dx3", "dx4", "dx5")
    case_insensitive: bool = True
    #: normalised primary string -> (label, rule_id)
    primary_map: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    #: normalised secondary string -> (label, rule_id)
    secondary_map: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    secondary_ad_promotes_primary: bool = False
    flag_cn_with_dementia_secondary: bool = True
    flag_mixed_dementia_etiology: bool = True
    consistency_enabled: bool = True
    cn_max_cdrtot: float | None = 0.0
    ad_min_cdrtot: float | None = 0.5
    source_path: Path | None = None
    #: "D1" (v2.0 default) or "B4" (v1.0 behaviour).
    primary_source: str = "D1"
    #: cognitive state -> (rule_id, (variable, ...)), in evaluation order.
    d1_states: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    #: aetiology -> (rule_id, (variable, ...)).
    d1_etiologies: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    d1_truth_value: str = "1"
    #: AD flag -> its paired "IF" role qualifier, e.g. ``{"PROBAD": "PROBADIF"}``.
    d1_role_qualifiers: Mapping[str, str] = field(default_factory=dict)
    #: Which aetiology roles admit a demented visit into the AD class.
    ad_roles_accepted: tuple[str, ...] = (ROLE_PRIMARY, ROLE_UNSPECIFIED)
    #: MCI core indicator -> {"label": subtype, "domains": {var: domain}}.
    d1_mci_subtypes: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    label_composition: Mapping[str, str] = field(default_factory=dict)
    training_labels: tuple[str, ...] = TRAINING_LABELS
    b4_validation_enabled: bool = True
    b4_concordance: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    multiple_states_conflict: bool = True
    mixed_etiology_excluded: bool = True

    @property
    def defines_mci(self) -> bool:
        """Whether the policy can produce an MCI label at all.

        Under ``primary_source: D1`` this asks whether the D1 MCI variables are
        configured; under B4 it asks whether any dx1 string maps to MCI.
        """
        if self.primary_source == "D1":
            return any(name == STATE_MCI and variables for name, _, variables in self.d1_states)
        return any(label == LABEL_MCI for label, _ in self.primary_map.values())

    # -- construction ------------------------------------------------------
    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source_path: Path | None = None
    ) -> "LabelPolicy":
        """Build and validate a policy from a parsed mapping."""
        if not isinstance(raw, Mapping):
            raise LabelPolicyError("Label policy root must be a mapping.")

        version = str(raw.get("version") or "unversioned")
        normalization = raw.get("normalization") or {}
        case_insensitive = bool(normalization.get("case_insensitive", True))

        primary_map: dict[str, tuple[str, str]] = {}
        for section, labels in (
            ("current_state", (LABEL_CN, LABEL_MCI, LABEL_AD)),
            (
                "excluded",
                (LABEL_OTHER_DEMENTIA, LABEL_UNCERTAIN, LABEL_NON_DIAGNOSTIC),
            ),
        ):
            block = raw.get(section) or {}
            if not isinstance(block, Mapping):
                raise LabelPolicyError(f"Section {section!r} must be a mapping.")
            for label in labels:
                entry = block.get(label) or {}
                if not isinstance(entry, Mapping):
                    raise LabelPolicyError(f"{section}.{label} must be a mapping.")
                rule_id = str(entry.get("rule_id") or f"{label.lower()}_exact")
                for value in entry.get("dx1_exact") or []:
                    _register(primary_map, value, label, rule_id, case_insensitive,
                              context=f"{section}.{label}")

        secondary_map: dict[str, tuple[str, str]] = {}
        for label, entry in (raw.get("secondary") or {}).items():
            if not isinstance(entry, Mapping):
                raise LabelPolicyError(f"secondary.{label} must be a mapping.")
            rule_id = str(entry.get("rule_id") or f"{str(label).lower()}_secondary")
            for value in entry.get("values") or []:
                _register(secondary_map, value, str(label), rule_id, case_insensitive,
                          context=f"secondary.{label}")

        if not primary_map:
            raise LabelPolicyError(
                "The label policy maps no primary diagnosis string at all; every "
                "session would be UNMAPPED."
            )

        conflict = raw.get("conflict_policy") or {}
        checks = raw.get("consistency_checks") or {}

        states_block = raw.get("d1_cognitive_status") or {}
        truth = str(states_block.get("truth_value", "1"))
        d1_states = tuple(
            (name, str((states_block[name] or {}).get("rule_id") or f"d1_{name.lower()}"),
             tuple((states_block[name] or {}).get("any_of") or ()))
            for name in (STATE_DEMENTIA, STATE_MCI, STATE_IMPAIRED_NOT_MCI, STATE_CN)
            if isinstance(states_block.get(name), Mapping)
        )
        etiology_block = raw.get("d1_ad_etiology") or {}
        d1_etiologies = tuple(
            (name, str((etiology_block[name] or {}).get("rule_id") or f"d1_{name.lower()}"),
             tuple((etiology_block[name] or {}).get("any_of") or ()))
            for name in (ETIOLOGY_AD, ETIOLOGY_NON_AD)
            if isinstance(etiology_block.get(name), Mapping)
        )
        ad_block = etiology_block.get(ETIOLOGY_AD) or {}
        d1_role_qualifiers = {
            str(flag): str(qualifier)
            for flag, qualifier in (ad_block.get("role_qualifiers") or {}).items()
        }
        ad_roles_accepted = tuple(
            str(role) for role in (raw.get("ad_etiology_roles_accepted") or [ROLE_PRIMARY, ROLE_UNSPECIFIED])
        )
        mci_block = states_block.get(STATE_MCI) or {}
        d1_mci_subtypes = {
            str(core): dict(spec or {})
            for core, spec in (mci_block.get("subtypes") or {}).items()
        }
        validation = raw.get("b4_validation") or {}
        primary_source = str(raw.get("primary_source") or "B4").upper()
        if primary_source == "D1" and not d1_states:
            raise LabelPolicyError(
                "primary_source is D1 but the policy defines no d1_cognitive_status rules."
            )

        return cls(
            primary_source=primary_source,
            d1_states=d1_states,
            d1_etiologies=d1_etiologies,
            d1_truth_value=truth,
            d1_role_qualifiers=d1_role_qualifiers,
            ad_roles_accepted=ad_roles_accepted,
            d1_mci_subtypes=d1_mci_subtypes,
            label_composition=dict(raw.get("label_composition") or {}),
            training_labels=tuple(raw.get("training_labels") or TRAINING_LABELS),
            b4_validation_enabled=bool(validation.get("enabled", True)),
            b4_concordance={
                str(key): tuple(value or ())
                for key, value in (validation.get("concordance") or {}).items()
            },
            multiple_states_conflict=bool(conflict.get("multiple_cognitive_states_conflict", True)),
            mixed_etiology_excluded=bool(conflict.get("mixed_etiology_excluded", True)),
            version=version,
            primary_column=str(raw.get("primary_column") or "dx1"),
            secondary_columns=tuple(raw.get("secondary_columns") or ("dx2", "dx3", "dx4", "dx5")),
            case_insensitive=case_insensitive,
            primary_map=primary_map,
            secondary_map=secondary_map,
            secondary_ad_promotes_primary=bool(
                conflict.get("secondary_ad_promotes_primary", False)
            ),
            flag_cn_with_dementia_secondary=bool(
                conflict.get("flag_cn_with_dementia_secondary", True)
            ),
            flag_mixed_dementia_etiology=bool(
                conflict.get("flag_mixed_dementia_etiology", True)
            ),
            consistency_enabled=bool(checks.get("enabled", True)),
            cn_max_cdrtot=_optional_float(checks.get("cn_max_cdrtot", 0.0)),
            ad_min_cdrtot=_optional_float(checks.get("ad_min_cdrtot", 0.5)),
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "LabelPolicy":
        """Load a policy from YAML."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise LabelPolicyError(f"Supervised label policy not found: {path}")
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        policy = cls.from_mapping(raw, source_path=path.resolve())
        logger.info(
            "Loaded supervised label policy %s from %s (%d primary, %d secondary strings).",
            policy.version,
            path,
            len(policy.primary_map),
            len(policy.secondary_map),
        )
        if not policy.defines_mci:
            logger.warning(
                "Label policy %s defines NO MCI mapping: the MCI class will be empty "
                "and the MCI->AD progression target will have no candidates. This is "
                "expected for v1.0 - OASIS-3 B4 dx1 carries no MCI label. See "
                "SUPERVISED_LABELING_PROTOCOL.md.",
                policy.version,
            )
        return policy

    @classmethod
    def load(cls, path: Path | str | None) -> "LabelPolicy":
        """Load ``path``, or the repository default."""
        if path is not None:
            return cls.from_yaml(path)
        default = Path(__file__).resolve().parents[2] / DEFAULT_POLICY_FILENAME
        if not default.exists():
            raise LabelPolicyError(
                f"No supervised label policy found at {default}; pass --label-policy."
            )
        return cls.from_yaml(default)

    # -- lookup ------------------------------------------------------------
    def normalise(self, value: Any) -> str | None:
        """Normalise ``value`` under this policy's rules."""
        return normalise_diagnosis(value, self.case_insensitive)

    def lookup_primary(self, value: Any) -> tuple[str, str] | None:
        """``(label, rule_id)`` for a primary diagnosis, or ``None`` if unmapped."""
        key = self.normalise(value)
        return self.primary_map.get(key) if key is not None else None

    def lookup_secondary(self, value: Any) -> tuple[str, str] | None:
        """``(label, rule_id)`` for a secondary diagnosis, or ``None`` if unmapped."""
        key = self.normalise(value)
        return self.secondary_map.get(key) if key is not None else None


def _register(
    target: dict[str, tuple[str, str]],
    value: Any,
    label: str,
    rule_id: str,
    case_insensitive: bool,
    context: str,
) -> None:
    """Add one string to a lookup table, rejecting contradictory duplicates."""
    key = normalise_diagnosis(value, case_insensitive)
    if key is None:
        raise LabelPolicyError(f"{context}: empty diagnosis string is not allowed.")
    existing = target.get(key)
    if existing is not None and existing[0] != label:
        raise LabelPolicyError(
            f"{context}: {value!r} is already mapped to {existing[0]}; a string "
            "cannot carry two labels."
        )
    target[key] = (label, rule_id)


def _optional_float(value: Any) -> float | None:
    """Float or ``None``."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# D1: the primary label source (v2.0)
# ---------------------------------------------------------------------------
def _d1_flag(row: Mapping[str, Any], variable: str, truth: str) -> bool | None:
    """Whether ``variable`` is set in ``row``.

    Returns ``None`` when the cell is blank (a NACC skip pattern) **or** carries
    a value outside ``{truth, "0"}``. D1 contains three such cells
    (``DEMENTED=2`` once, ``IMPNOMCI=2`` twice); they are reported as unusable
    rather than guessed at.
    """
    raw = row.get(variable)
    if raw is None or (isinstance(raw, float) and raw != raw):
        return None
    text = str(raw).strip()
    if text in ("", "."):
        return None
    if text == truth:
        return True
    if text in ("0", "0.0"):
        return False
    try:  # pandas turns "1" into 1.0
        number = float(text)
    except ValueError:
        return None
    if number == float(truth):
        return True
    return False if number == 0 else None


def derive_d1_cognitive_state(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[list[str], list[str]]:
    """Cognitive states asserted by D1 for one visit.

    Returns ``(states, rule_ids)``. More than one state means the form
    contradicts itself; the caller reports that rather than applying a priority
    order (14 OASIS-3 visits are affected).
    """
    states: list[str] = []
    rules: list[str] = []
    for name, rule_id, variables in policy.d1_states:
        if any(_d1_flag(row, variable, policy.d1_truth_value) for variable in variables):
            states.append(name)
            rules.append(rule_id)
    return states, rules


def etiology_role(row: Mapping[str, Any], flag: str, policy: LabelPolicy) -> str | None:
    """Role the aetiology ``flag`` plays, from its paired NACC "IF" field.

    The "IF" domain is ``{0, 1, 2}``, **not** binary: 1 means the aetiology is
    the primary cause, 2 means it merely contributes. A blank field leaves the
    role ``unspecified``; a value outside the domain is reported as unspecified
    rather than guessed at.

    Returns ``None`` when the flag itself is not set.
    """
    if not _d1_flag(row, flag, policy.d1_truth_value):
        return None
    qualifier = policy.d1_role_qualifiers.get(flag)
    if not qualifier:
        return ROLE_UNSPECIFIED
    raw = row.get(qualifier)
    if raw is None or (isinstance(raw, float) and raw != raw):
        return ROLE_UNSPECIFIED
    text = str(raw).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return ROLE_CODES.get(text, ROLE_UNSPECIFIED)


def derive_d1_etiology(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[str, str | None]:
    """AD / NON_AD / MIXED / UNKNOWN aetiology asserted by D1.

    Applies to MCI as well as dementia: D1 records aetiology for both, so "MCI
    due to AD" is expressible. The two form generations are handled together as
    version-dependent representations of one concept - UDS v1/v2
    ``PROBAD``/``POSSAD`` and UDS v3 ``alzdis`` are disjoint in OASIS-3 and
    neither is required to exist.

    An AD flag only counts when its role is in ``policy.ad_roles_accepted``.
    By default a *contributing* AD aetiology (``IF == 2``) does not make the
    visit AD: AD is present but is not the primary cause, exactly like the B4
    string "AD dem cannot be primary".

    Returns
    -------
    tuple
        ``(etiology, ad_role)``; ``ad_role`` is ``None`` when no AD flag is set.
    """
    ad_roles = [
        role
        for name, _, variables in policy.d1_etiologies
        if name == ETIOLOGY_AD
        for variable in variables
        for role in (etiology_role(row, variable, policy),)
        if role is not None
    ]
    has_non_ad = any(
        _d1_flag(row, variable, policy.d1_truth_value)
        for name, _, variables in policy.d1_etiologies
        if name == ETIOLOGY_NON_AD
        for variable in variables
    )

    ad_role = _dominant_role(ad_roles)
    has_ad = any(role in policy.ad_roles_accepted for role in ad_roles)

    if has_ad and has_non_ad:
        return ETIOLOGY_MIXED, ad_role
    if has_ad:
        return ETIOLOGY_AD, ad_role
    if has_non_ad:
        return ETIOLOGY_NON_AD, ad_role
    return ETIOLOGY_UNKNOWN, ad_role


def _dominant_role(roles: Sequence[str]) -> str | None:
    """Strongest role among several AD flags: primary > unspecified > contributing."""
    for role in (ROLE_PRIMARY, ROLE_UNSPECIFIED, ROLE_CONTRIBUTING):
        if role in roles:
            return role
    return None


def derive_mci_subtype(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[str | None, tuple[str, ...]]:
    """MCI subtype and impaired domains from the D1 qualifier fields.

    The four core indicators decide *whether* a visit is MCI; these companion
    fields describe *which kind*. They never change the label, and orphaned
    qualifiers (a domain set without its core indicator) are ignored here and
    reported by validation.

    Returns ``(subtype, domains)``, both empty when no core indicator is set.
    """
    for core, spec in policy.d1_mci_subtypes.items():
        if not _d1_flag(row, core, policy.d1_truth_value):
            continue
        domains = tuple(
            sorted(
                name
                for variable, name in (spec.get("domains") or {}).items()
                if _d1_flag(row, variable, policy.d1_truth_value)
            )
        )
        return str(spec.get("label") or core), domains
    return None, ()


def compose_label(state: str, etiology: str, policy: LabelPolicy) -> str:
    """Combine a cognitive state and an aetiology into the supervised label."""
    composition = policy.label_composition
    for key in (f"{state}+{etiology}", state):
        if key in composition:
            return str(composition[key])
    return state


def derive_d1_label(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[SupervisedLabel, str]:
    """Derive the Target-A label for one visit from D1.

    Returns ``(label, etiology)``. The aetiology is surfaced separately so that
    "MCI due to AD" and "dementia of mixed aetiology" remain visible in the
    output even though they collapse into one label.
    """
    states, rules = derive_d1_cognitive_state(row, policy)

    if not states:
        return (
            SupervisedLabel(
                label=LABEL_MISSING,
                source=SOURCE_D1,
                rule_id="d1_no_cognitive_status",
                confidence="none",
                reason=(
                    "No D1 cognitive-status variable is set for the linked visit "
                    "(blank form or an out-of-range value)."
                ),
                status=STATUS_MISSING,
                policy_version=policy.version,
            ),
            ETIOLOGY_UNKNOWN,
        )

    etiology, ad_role = derive_d1_etiology(row, policy)

    if len(states) > 1 and policy.multiple_states_conflict:
        return (
            SupervisedLabel(
                label=LABEL_CONFLICTING,
                source=SOURCE_D1,
                rule_id="d1_multiple_cognitive_states",
                confidence="none",
                reason=(
                    f"D1 asserts several cognitive states at once: {sorted(states)}. "
                    "Reported, not resolved by priority."
                ),
                status=STATUS_CONFLICTING,
                policy_version=policy.version,
            ),
            etiology,
        )

    state = states[0]
    label = compose_label(state, etiology, policy)
    subtype, domains = derive_mci_subtype(row, policy)

    reason = f"D1 {state} via {rules[0]}"
    if state in (STATE_DEMENTIA, STATE_MCI):
        reason = f"{reason}; aetiology {etiology}"
        if ad_role:
            reason = f"{reason} (role {ad_role})"
    if subtype:
        reason = f"{reason}; subtype {subtype}" + (
            f" [{', '.join(domains)}]" if domains else ""
        )

    return (
        SupervisedLabel(
            label=label,
            source=SOURCE_D1,
            rule_id=rules[0],
            confidence="high" if etiology != ETIOLOGY_MIXED else "low",
            reason=reason,
            status=STATUS_LABELLED,
            policy_version=policy.version,
            ad_etiology_role=ad_role,
            mci_subtype=subtype,
            mci_domains=", ".join(domains) or None,
        ),
        etiology,
    )


def derive_b4_comparison_label(
    row: Mapping[str, Any], policy: LabelPolicy
) -> str | None:
    """Independent comparison label from the B4 ``dx1`` free text.

    Derived on its own terms, with no knowledge of the D1 label, so the
    cross-check is genuinely independent. ``None`` when B4 records nothing.
    """
    matched = policy.lookup_primary(row.get(policy.primary_column))
    if matched is not None:
        return matched[0]
    return LABEL_UNMAPPED if policy.normalise(row.get(policy.primary_column)) else None


def validate_against_b4(
    label: SupervisedLabel, row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[str | None, str, str | None]:
    """Cross-check a D1 label against the independent B4 free text.

    B4 never assigns or overrides a label: this only populates ``b4_label``,
    ``b4_agreement`` and ``b4_disagreement_reason`` so systematic disagreement
    stays visible and auditable.

    Returns ``(b4_label, agreement, disagreement_reason)``.
    """
    if not policy.b4_validation_enabled:
        return None, AGREEMENT_NOT_COMPARABLE, None

    b4_label = derive_b4_comparison_label(row, policy)
    if b4_label is None:
        return None, AGREEMENT_UNAVAILABLE, None
    if b4_label == LABEL_UNMAPPED:
        return b4_label, AGREEMENT_NOT_COMPARABLE, "B4 dx1 is not enumerated by the policy"

    concordant = policy.b4_concordance.get(b4_label)
    if concordant is None:
        return b4_label, AGREEMENT_NOT_COMPARABLE, f"B4={b4_label} has no concordance entry"
    if not concordant:
        return b4_label, AGREEMENT_NOT_COMPARABLE, f"B4={b4_label} is not a diagnosis"
    if label.label in concordant:
        return b4_label, AGREEMENT_AGREE, None
    return (
        b4_label,
        AGREEMENT_DISAGREE,
        f"D1={label.label} is not concordant with B4={b4_label}",
    )


# ---------------------------------------------------------------------------
# Target A: current clinical state
# ---------------------------------------------------------------------------
def derive_current_label(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[SupervisedLabel, list[str]]:
    """Derive the Target-A label for one MRI session.

    Dispatches on ``policy.primary_source``:

    ``D1`` (v2.0 default)
        the D1 diagnosis form decides, and the B4 free text is used only as a
        cross-check recorded in ``b4_agreement``;
    ``B4`` (v1.0 behaviour)
        the ``dx1`` free text decides.

    Returns
    -------
    tuple
        ``(label, warnings)``. Warnings are consistency findings such as
        ``diagnosis_cdr_disagreement``; they never alter the label.
    """
    if policy.primary_source == "D1":
        return _derive_from_d1(row, policy)
    return _derive_from_b4(row, policy)


def _derive_from_d1(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[SupervisedLabel, list[str]]:
    """D1-primary derivation, with the B4 text as an auxiliary cross-check."""
    label, etiology = derive_d1_label(row, policy)
    b4_label, agreement, detail = validate_against_b4(label, row, policy)

    warnings = _consistency_warnings(label.label, row, policy)
    if agreement == AGREEMENT_DISAGREE and detail:
        warnings.append(f"d1_b4_disagreement({detail})")

    enriched = replace(
        label,
        source=SOURCE_D1_B4 if agreement in (AGREEMENT_AGREE, AGREEMENT_DISAGREE) else SOURCE_D1,
        raw_value=_optional_str(row.get(policy.primary_column)),
        normalized_value=policy.normalise(row.get(policy.primary_column)),
        etiology=etiology,
        b4_label=b4_label,
        b4_agreement=agreement,
        b4_disagreement_reason=detail if agreement == AGREEMENT_DISAGREE else None,
        reason=f"{label.reason}; B4 cross-check: {agreement}"
        + (f" ({detail})" if detail else ""),
    )
    return enriched, warnings


def _optional_str(value: Any) -> str | None:
    """String form of a raw value, or ``None`` when it means "absent"."""
    return None if normalise_diagnosis(value) is None else str(value)


def _derive_from_b4(
    row: Mapping[str, Any], policy: LabelPolicy
) -> tuple[SupervisedLabel, list[str]]:
    """v1.0 behaviour: the B4 free text decides the label."""
    raw = row.get(policy.primary_column)
    normalized = policy.normalise(raw)

    if normalized is None:
        return (
            SupervisedLabel(
                label=LABEL_MISSING,
                source=f"B4_{policy.primary_column}",
                rule_id="missing_primary_diagnosis",
                confidence="none",
                reason=f"{policy.primary_column} is absent for the linked clinical visit.",
                status=STATUS_MISSING,
                policy_version=policy.version,
                raw_value=None,
                normalized_value=None,
            ),
            [],
        )

    matched = policy.primary_map.get(normalized)
    if matched is None:
        return (
            SupervisedLabel(
                label=LABEL_UNMAPPED,
                source=f"B4_{policy.primary_column}",
                rule_id="unmapped_primary_diagnosis",
                confidence="none",
                reason=(
                    f"{policy.primary_column}={raw!r} is not enumerated in policy "
                    f"{policy.version}. Excluded; never defaulted to CN."
                ),
                status=STATUS_UNMAPPED,
                policy_version=policy.version,
                raw_value=str(raw),
                normalized_value=normalized,
            ),
            [],
        )

    label, rule_id = matched
    secondary = _secondary_labels(row, policy)
    status, conflict_reason = _detect_conflict(label, secondary, policy)
    warnings = _consistency_warnings(label, row, policy)

    reason = f"{policy.primary_column}={raw!r} -> {label} via {rule_id}"
    if conflict_reason:
        reason = f"{reason}; {conflict_reason}"

    return (
        SupervisedLabel(
            label=LABEL_UNCERTAIN if status == STATUS_CONFLICTING else label,
            source=f"B4_{policy.primary_column}",
            rule_id=rule_id,
            confidence="high" if status == STATUS_LABELLED else "low",
            reason=reason,
            status=status,
            policy_version=policy.version,
            raw_value=str(raw),
            normalized_value=normalized,
        ),
        warnings,
    )


def _secondary_labels(row: Mapping[str, Any], policy: LabelPolicy) -> list[str]:
    """Labels contributed by ``dx2``..``dx5``, ignoring unmapped values."""
    labels: list[str] = []
    for column in policy.secondary_columns:
        matched = policy.lookup_secondary(row.get(column))
        if matched is not None:
            labels.append(matched[0])
    return labels


def _detect_conflict(
    primary: str, secondary: Sequence[str], policy: LabelPolicy
) -> tuple[str, str | None]:
    """Decide whether the secondary diagnoses contradict the primary one."""
    dementia_secondary = {
        label for label in secondary if label in (LABEL_AD, LABEL_OTHER_DEMENTIA)
    }

    if (
        primary == LABEL_CN
        and dementia_secondary
        and policy.flag_cn_with_dementia_secondary
    ):
        return STATUS_CONFLICTING, (
            f"primary is CN but secondary diagnoses include {sorted(dementia_secondary)}"
        )

    if (
        primary == LABEL_AD
        and LABEL_OTHER_DEMENTIA in dementia_secondary
        and policy.flag_mixed_dementia_etiology
    ):
        # Mixed etiology is reported but does not unmake the AD primary.
        return STATUS_LABELLED, "mixed dementia etiology: non-AD dementia also recorded"

    return STATUS_LABELLED, None


def _consistency_warnings(
    label: str, row: Mapping[str, Any], policy: LabelPolicy
) -> list[str]:
    """CDR cross-checks. These never overwrite the diagnosis."""
    if not policy.consistency_enabled:
        return []

    cdr = _optional_float(row.get("CDRTOT"))
    if cdr is None:
        return []

    if label == LABEL_CN and policy.cn_max_cdrtot is not None and cdr > policy.cn_max_cdrtot:
        return [f"diagnosis_cdr_disagreement(CN with CDRTOT={cdr:g})"]
    if label == LABEL_AD and policy.ad_min_cdrtot is not None and cdr < policy.ad_min_cdrtot:
        return [f"diagnosis_cdr_disagreement(AD with CDRTOT={cdr:g})"]
    return []


def training_eligibility(
    label: SupervisedLabel,
    clinical_match_valid: bool,
    abs_gap_days: float | None,
    window_days: int,
    training_labels: Sequence[str] | None = None,
) -> tuple[bool, str | None]:
    """Whether a session may enter the default CN/MCI/AD training cohort.

    The temporal gate is applied first: a label derived from a visit that is too
    far from the scan is not trustworthy regardless of how clean the string was.
    """
    if not clinical_match_valid or (
        abs_gap_days is not None and abs_gap_days > window_days
    ):
        return False, EXCLUSION_OUTSIDE_WINDOW

    if label.status == STATUS_CONFLICTING:
        return False, EXCLUSION_CONFLICTING
    if label.status == STATUS_MISSING:
        return False, EXCLUSION_MISSING
    if label.status == STATUS_UNMAPPED:
        return False, EXCLUSION_UNMAPPED

    reasons = {
        LABEL_OTHER_DEMENTIA: EXCLUSION_OTHER_DEMENTIA,
        LABEL_UNCERTAIN: EXCLUSION_UNCERTAIN,
        LABEL_NON_DIAGNOSTIC: EXCLUSION_NON_DIAGNOSTIC,
        LABEL_IMPAIRED_NOT_MCI: EXCLUSION_IMPAIRED_NOT_MCI,
        LABEL_DEMENTIA_UNKNOWN_ETIOLOGY: EXCLUSION_DEMENTIA_UNKNOWN_ETIOLOGY,
        LABEL_CONFLICTING: EXCLUSION_CONFLICTING,
    }
    if label.label in reasons:
        return False, reasons[label.label]

    if label.label in (training_labels or TRAINING_LABELS):
        return True, None
    return False, EXCLUSION_UNMAPPED


# ---------------------------------------------------------------------------
# Target B: MCI -> AD progression
# ---------------------------------------------------------------------------
def derive_progression_label(
    mri_day: int,
    current_label: str,
    future_visits: Sequence[Mapping[str, Any]],
    policy: LabelPolicy,
    horizon_days: int,
) -> ProgressionLabel:
    """Derive ``MCI_TO_AD`` / ``MCI_STABLE`` / ``CENSORED`` for one session.

    Parameters
    ----------
    mri_day:
        Day of the MRI session.
    current_label:
        Target-A label at the scan. Only ``MCI`` sessions are eligible.
    future_visits:
        Clinical visits of the **same subject**, each a mapping with
        ``clinical_day`` and the policy's primary column. Visits at or before
        ``mri_day`` are ignored here; the caller may pass the whole history.
    horizon_days:
        Prediction horizon. AD within ``(mri_day, mri_day + horizon]`` is a
        conversion.

    Censoring
    ---------
    Absence of a later AD diagnosis is **not** stability. ``MCI_STABLE`` requires
    a **non-AD observation at or after** ``mri_day + horizon``: the participant
    must actually have been seen, and seen non-converted, once the horizon had
    elapsed.

    Merely having *some* later visit is not enough. Consider a scan at day 1000
    with a horizon of 730 (deadline 1730), an MCI visit at 1500 and an AD visit
    at 2500. Follow-up extends well past the deadline, yet the last time the
    participant was observed non-AD was day 1500 - the conversion may have
    happened at day 1600, inside the horizon. The outcome is unobserved, so the
    session is ``CENSORED``. Collapsing that into ``MCI_STABLE`` would inject
    label noise straight into ``y``.
    """
    if current_label != LABEL_MCI:
        return ProgressionLabel(
            label=None,
            eligible=False,
            exclusion_reason="not_mci_at_mri",
            horizon_days=horizon_days,
            reason=f"current label is {current_label}, not MCI.",
        )

    later = sorted(
        (visit for visit in future_visits if _visit_day(visit) > mri_day),
        key=_visit_day,
    )
    if not later:
        return ProgressionLabel(
            label=PROGRESSION_CENSORED,
            eligible=False,
            exclusion_reason="no_followup_after_mri",
            last_followup_day=None,
            followup_days_after_mri=0,
            horizon_days=horizon_days,
            reason="No clinical visit after the MRI; conversion status unobservable.",
        )

    last_day = _visit_day(later[-1])
    followup = last_day - mri_day
    deadline = mri_day + horizon_days

    for visit in later:
        day = _visit_day(visit)
        if day > deadline:
            break
        if _visit_label(visit, policy) == LABEL_AD:
            return ProgressionLabel(
                label=PROGRESSION_TO_AD,
                eligible=True,
                exclusion_reason=None,
                conversion_event="MCI_to_AD",
                conversion_day=day,
                days_to_conversion=day - mri_day,
                last_followup_day=last_day,
                followup_days_after_mri=followup,
                horizon_days=horizon_days,
                reason=(
                    f"AD recorded at day {day}, within the {horizon_days}-day horizon."
                ),
            )

    # Stability must be *observed*, not inferred from silence: we need a visit at
    # or after the deadline that is still non-AD.
    confirming_day = next(
        (
            _visit_day(visit)
            for visit in later
            if _visit_day(visit) >= deadline
            and _visit_label(visit, policy) != LABEL_AD
        ),
        None,
    )
    if confirming_day is not None:
        return ProgressionLabel(
            label=PROGRESSION_STABLE,
            eligible=True,
            exclusion_reason=None,
            last_followup_day=last_day,
            followup_days_after_mri=followup,
            horizon_days=horizon_days,
            reason=(
                f"No AD through day {deadline}, and the participant was still "
                f"non-AD when seen at day {confirming_day}, at or after the horizon."
            ),
        )

    return ProgressionLabel(
        label=PROGRESSION_CENSORED,
        eligible=False,
        exclusion_reason="insufficient_followup",
        last_followup_day=last_day,
        followup_days_after_mri=followup,
        horizon_days=horizon_days,
        reason=(
            f"No AD recorded within the horizon, but the participant was never "
            f"observed non-AD at or after day {deadline} (last visit: day "
            f"{last_day}). Outcome unobserved - censored, not stable."
        ),
    )


def _visit_label(visit: Mapping[str, Any], policy: LabelPolicy) -> str:
    """Target-A label of a clinical visit, under the policy's primary source.

    Target B must read future visits with the same instrument that produced the
    label at the scan, or the outcome would be defined on a different scale from
    the exposure.
    """
    return derive_current_label(visit, policy)[0].label


def _visit_day(visit: Mapping[str, Any]) -> int:
    """``clinical_day`` of a visit mapping, as an int."""
    try:
        return int(float(visit["clinical_day"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LabelPolicyError(f"Clinical visit has no usable clinical_day: {visit!r}") from exc


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------
def leaking_columns(columns: Iterable[str]) -> list[str]:
    """Columns in ``columns`` that carry post-scan information.

    Any of these may be used to build ``y``; none may be handed to a model as
    ``x``. Used by the dataset builder and asserted in the tests.
    """
    return sorted(column for column in columns if column in FUTURE_INFORMATION_COLUMNS)
