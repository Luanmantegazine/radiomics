"""Disclosure-control rules applied on both sides of the federated boundary.

The same rules run at the client (egress, before bytes leave the site) and at
the coordinator (ingress, before an update is aggregated): defence in depth.

Hypotheses and limitations
--------------------------
* Threat model: honest-but-curious coordinator, honest clients. These checks
  prevent *accidental* disclosure by our own code; they are not a defence
  against a malicious client implementation.
* The minimum cohort size is a necessary, not sufficient, protection. Small
  cohorts and repeated queries can still leak information through aggregates;
  secure aggregation / differential privacy are out of scope for Stage 1.
"""

from __future__ import annotations

from collections.abc import Iterable

from shared.protocol import ClientUpdate, PrivacyPolicy, ReasonCode, UpdateStatus


class DisclosureViolation(Exception):
    """An outgoing or incoming message would breach the disclosure policy."""

    def __init__(self, code: ReasonCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


def check_update(update: ClientUpdate, policy: PrivacyPolicy) -> None:
    """Validate a client update against the disclosure policy.

    Structural rules (closed schema, feature-indexed arrays) are enforced by the
    wire models themselves; this adds the policy-dependent rules.
    """
    if update.status is not UpdateStatus.COMPLETED or update.payload is None:
        return
    n_samples = getattr(update.payload, "n_samples", None)
    if n_samples is not None and n_samples < policy.min_cohort_size:
        raise DisclosureViolation(
            ReasonCode.COHORT_TOO_SMALL,
            f"statistics on {n_samples} participants (< {policy.min_cohort_size})",
        )


def scan_for_identifiers(raw: bytes, identifiers: Iterable[str]) -> None:
    """Fail if any local identifier appears verbatim in serialized bytes.

    Client-side last line of defence: the site knows its own participant
    identifiers and refuses to emit a message that contains one of them.
    """
    text = raw.decode("utf-8", errors="replace")
    for identifier in identifiers:
        if identifier and identifier in text:
            # The identifier itself is deliberately not echoed in the error.
            raise DisclosureViolation(
                ReasonCode.DISCLOSURE_BLOCKED, "a local identifier was found in an outgoing message"
            )
