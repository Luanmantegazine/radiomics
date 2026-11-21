"""Egress guard: the single exit point for bytes leaving a client."""

from __future__ import annotations

from collections.abc import Iterable

from shared.privacy import DisclosureViolation, check_update, scan_for_identifiers
from shared.protocol import MAX_MESSAGE_BYTES, ClientUpdate, PrivacyPolicy, ReasonCode, encode


class EgressGuard:
    """Authorizes and serializes outgoing updates.

    Checks, in order: disclosure policy (minimum cohort size), message size,
    and absence of any local participant identifier in the serialized bytes.
    """

    def __init__(self, local_identifiers: Iterable[str]) -> None:
        self._identifiers = frozenset(local_identifiers)

    def authorize(self, update: ClientUpdate, policy: PrivacyPolicy) -> bytes:
        """Return wire bytes, or raise :class:`DisclosureViolation`."""
        check_update(update, policy)
        raw = encode(update)
        if len(raw) > MAX_MESSAGE_BYTES:
            raise DisclosureViolation(ReasonCode.DISCLOSURE_BLOCKED, "outgoing message too large")
        scan_for_identifiers(raw, self._identifiers)
        return raw
