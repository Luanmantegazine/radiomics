"""Codebook-driven derivation of cognitive status and AD etiology.

All clinical meaning lives in ``clinical_classification.yaml``; this module only
evaluates it. That separation is deliberate and is the core scientific guard
rail of the clinical layer:

* the numeric semantics of the NACC/UDS D1 variables are not documented in this
  repository, so nothing here assumes them;
* while the codebook is unfrozen every visit is reported as
  ``unresolved_codebook`` with ``UNKNOWN`` derived values, and every raw
  variable is still passed through to the outputs;
* freezing the codebook is an explicit, versioned act, recorded in every run's
  metadata via ``classification_version``.

``UNKNOWN`` never means "normal".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import (
    AD_ETIOLOGY_VALUES,
    COGNITIVE_STATUS_VALUES,
    ClinicalClassification,
    ClinicalVisit,
)

logger = logging.getLogger(__name__)

DEFAULT_CODEBOOK_FILENAME = "clinical_classification.yaml"

STATUS_CLASSIFIED = "classified"
STATUS_UNRESOLVED = "unresolved_codebook"
STATUS_NO_DATA = "no_clinical_data"
STATUS_CONFLICTING = "conflicting"

UNKNOWN = "UNKNOWN"


class CodebookError(ValueError):
    """Raised when the classification codebook is structurally invalid."""


@dataclass(frozen=True)
class ClassificationRule:
    """One codebook rule, evaluated against a merged clinical visit."""

    id: str
    source: str
    when: Mapping[str, Any]
    cognitive_status: str = UNKNOWN
    ad_etiology: str = UNKNOWN
    confidence: str = "medium"
    reference: str | None = None

    def matches(self, visit: ClinicalVisit) -> bool:
        """Whether every condition in ``when`` holds for ``visit``."""
        lookup = visit.d1_value if self.source == "d1" else visit.b4_value
        for variable, expected in self.when.items():
            actual = lookup(variable)
            if expected is None:
                if actual is not None:
                    return False
            elif actual is None or str(actual).strip() != str(expected).strip():
                return False
        return True


@dataclass(frozen=True)
class ClassificationCodebook:
    """A parsed, validated codebook.

    ``frozen`` gates every derivation: an unfrozen codebook classifies nothing.
    """

    version: str
    frozen: bool = False
    rules: tuple[ClassificationRule, ...] = ()
    b4_dx_text: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    cdr_global: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def is_active(self) -> bool:
        """Whether the codebook can actually derive anything."""
        return self.frozen and bool(self.rules or self.b4_dx_text or self.cdr_global)

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], source_path: Path | None = None
    ) -> "ClassificationCodebook":
        """Build and validate a codebook from a parsed mapping."""
        if not isinstance(raw, Mapping):
            raise CodebookError("Codebook root must be a mapping.")

        version = str(raw.get("version") or "unversioned")
        frozen = bool(raw.get("codebook_frozen", False))

        rules = tuple(_parse_rule(entry, index) for index, entry in enumerate(raw.get("rules") or []))
        b4_dx_text = _parse_outcome_map(raw.get("b4_dx_text") or {}, "b4_dx_text", lower_keys=True)
        cdr_global = _parse_outcome_map(raw.get("cdr_global") or {}, "cdr_global")

        if frozen and not (rules or b4_dx_text or cdr_global):
            raise CodebookError(
                "codebook_frozen is true but no rules, b4_dx_text or cdr_global "
                "mappings are defined. Freezing an empty codebook would silently "
                "classify every visit as UNKNOWN."
            )
        if frozen:
            _require_references(rules)

        return cls(
            version=version,
            frozen=frozen,
            rules=rules,
            b4_dx_text=b4_dx_text,
            cdr_global=cdr_global,
            source_path=source_path,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ClassificationCodebook":
        """Load a codebook from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise CodebookError(f"Classification codebook not found: {path}")
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        codebook = cls.from_mapping(raw, source_path=path.resolve())
        logger.info(
            "Loaded classification codebook %s from %s (frozen=%s, %d rule(s)).",
            codebook.version,
            path,
            codebook.frozen,
            len(codebook.rules),
        )
        if not codebook.is_active:
            logger.warning(
                "Classification codebook is NOT frozen: every visit will be reported "
                "as '%s' with UNKNOWN status. Raw D1/B4 variables are preserved. "
                "See %s for how to freeze it.",
                STATUS_UNRESOLVED,
                path.name,
            )
        return codebook

    @classmethod
    def load(cls, path: Path | str | None) -> "ClassificationCodebook":
        """Load ``path``, or the repository default, or an empty unfrozen codebook."""
        if path is not None:
            return cls.from_yaml(path)

        default = Path(__file__).resolve().parents[2] / DEFAULT_CODEBOOK_FILENAME
        if default.exists():
            return cls.from_yaml(default)

        logger.warning("No classification codebook found; classification disabled.")
        return cls(version="absent", frozen=False)


def _parse_rule(entry: Any, index: int) -> ClassificationRule:
    """Parse and validate one rule entry."""
    if not isinstance(entry, Mapping):
        raise CodebookError(f"rules[{index}] must be a mapping.")

    rule_id = str(entry.get("id") or f"rule_{index}")
    source = str(entry.get("source") or "d1").lower()
    if source not in ("d1", "b4"):
        raise CodebookError(f"rules[{index}] ({rule_id}): source must be 'd1' or 'b4'.")

    when = entry.get("when")
    if not isinstance(when, Mapping) or not when:
        raise CodebookError(f"rules[{index}] ({rule_id}): 'when' must be a non-empty mapping.")

    cognitive_status = str(entry.get("cognitive_status", UNKNOWN)).upper()
    ad_etiology = str(entry.get("ad_etiology", UNKNOWN)).upper()
    _validate_vocabulary(rule_id, cognitive_status, ad_etiology)

    return ClassificationRule(
        id=rule_id,
        source=source,
        when=dict(when),
        cognitive_status=cognitive_status,
        ad_etiology=ad_etiology,
        confidence=str(entry.get("confidence", "medium")).lower(),
        reference=entry.get("reference"),
    )


def _parse_outcome_map(
    raw: Mapping[str, Any], name: str, lower_keys: bool = False
) -> dict[str, dict[str, str]]:
    """Parse a ``{key: {cognitive_status, ad_etiology}}`` mapping."""
    if not isinstance(raw, Mapping):
        raise CodebookError(f"{name} must be a mapping.")

    parsed: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise CodebookError(f"{name}[{key!r}] must be a mapping of outcomes.")
        cognitive_status = str(value.get("cognitive_status", UNKNOWN)).upper()
        ad_etiology = str(value.get("ad_etiology", UNKNOWN)).upper()
        _validate_vocabulary(f"{name}[{key!r}]", cognitive_status, ad_etiology)
        normalised_key = " ".join(str(key).split())
        parsed[normalised_key.lower() if lower_keys else normalised_key] = {
            "cognitive_status": cognitive_status,
            "ad_etiology": ad_etiology,
            "confidence": str(value.get("confidence", "medium")).lower(),
        }
    return parsed


def _validate_vocabulary(context: str, cognitive_status: str, ad_etiology: str) -> None:
    """Reject values outside the controlled vocabularies."""
    if cognitive_status not in COGNITIVE_STATUS_VALUES:
        raise CodebookError(
            f"{context}: cognitive_status must be one of {COGNITIVE_STATUS_VALUES}, "
            f"got {cognitive_status!r}."
        )
    if ad_etiology not in AD_ETIOLOGY_VALUES:
        raise CodebookError(
            f"{context}: ad_etiology must be one of {AD_ETIOLOGY_VALUES}, got {ad_etiology!r}."
        )


def _require_references(rules: Sequence[ClassificationRule]) -> None:
    """A frozen codebook must cite a source for every rule."""
    unreferenced = [rule.id for rule in rules if not rule.reference]
    if unreferenced:
        raise CodebookError(
            "A frozen codebook must cite a 'reference' for every rule "
            f"(missing on: {', '.join(unreferenced)})."
        )


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def classify_clinical_visit(
    visit: ClinicalVisit | None, codebook: ClassificationCodebook
) -> ClinicalClassification:
    """Derive cognitive status and AD etiology for one clinical visit.

    Returns a :class:`ClinicalClassification` whose ``status`` explains how the
    values were obtained:

    ``no_clinical_data``
        ``visit`` is ``None``.
    ``unresolved_codebook``
        the codebook is not frozen, so nothing is derived. The raw variables
        remain available on the visit.
    ``conflicting``
        two or more matching sources disagree on the cognitive status; the
        disagreement is reported rather than resolved.
    ``classified``
        a single consistent outcome was derived.
    """
    if visit is None:
        return ClinicalClassification(
            status=STATUS_NO_DATA,
            reason="No clinical visit was linked to this MRI session.",
            codebook_version=codebook.version,
        )

    if not codebook.is_active:
        return ClinicalClassification(
            status=STATUS_UNRESOLVED,
            reason=(
                "Classification codebook is not frozen; D1 numeric semantics are "
                "not documented in this repository. Raw variables preserved."
            ),
            codebook_version=codebook.version,
        )

    outcomes = _collect_outcomes(visit, codebook)
    if not outcomes:
        return ClinicalClassification(
            status=STATUS_UNRESOLVED,
            reason="No codebook rule matched this visit.",
            codebook_version=codebook.version,
        )

    statuses = {outcome["cognitive_status"] for outcome in outcomes}
    etiologies = {outcome["ad_etiology"] for outcome in outcomes if outcome["ad_etiology"] != UNKNOWN}
    sources = ", ".join(sorted(outcome["origin"] for outcome in outcomes))

    if len(statuses) > 1:
        return ClinicalClassification(
            status=STATUS_CONFLICTING,
            reason=f"Conflicting cognitive_status {sorted(statuses)} from: {sources}.",
            codebook_version=codebook.version,
        )

    return ClinicalClassification(
        cognitive_status=next(iter(statuses)),
        ad_etiology=next(iter(etiologies)) if len(etiologies) == 1 else (
            "UNCERTAIN" if len(etiologies) > 1 else UNKNOWN
        ),
        status=STATUS_CLASSIFIED,
        confidence=_worst_confidence(outcomes),
        reason=f"Matched: {sources}.",
        codebook_version=codebook.version,
    )


def _collect_outcomes(
    visit: ClinicalVisit, codebook: ClassificationCodebook
) -> list[dict[str, str]]:
    """Every outcome the codebook produces for ``visit``, with its origin."""
    outcomes: list[dict[str, str]] = []

    for rule in codebook.rules:
        if rule.matches(visit):
            outcomes.append(
                {
                    "cognitive_status": rule.cognitive_status,
                    "ad_etiology": rule.ad_etiology,
                    "confidence": rule.confidence,
                    "origin": f"rule:{rule.id}",
                }
            )

    dx1 = visit.b4_value("dx1")
    if codebook.b4_dx_text and dx1:
        mapped = codebook.b4_dx_text.get(" ".join(str(dx1).split()).lower())
        if mapped:
            outcomes.append({**mapped, "origin": "b4_dx_text:dx1"})

    cdr = visit.b4_value("CDRTOT")
    if codebook.cdr_global and cdr is not None:
        mapped = codebook.cdr_global.get(str(cdr).strip())
        if mapped:
            outcomes.append({**mapped, "origin": f"cdr_global:{cdr}"})

    return outcomes


def _worst_confidence(outcomes: Sequence[Mapping[str, str]]) -> str:
    """Lowest confidence among the contributing outcomes."""
    order = ("low", "medium", "high")
    levels = [outcome.get("confidence", "medium") for outcome in outcomes]
    return min(levels, key=lambda level: order.index(level) if level in order else 0)
