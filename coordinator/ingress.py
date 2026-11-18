"""Ingress validation: nothing is aggregated before passing these checks."""

from __future__ import annotations

from shared.privacy import DisclosureViolation, check_update
from shared.protocol import ClientUpdate, ReasonCode, RoundConfig, UpdateStatus, decode


class IngressRejected(Exception):
    """An incoming update was discarded."""


def validate_update(raw: bytes, *, sender: str, config: RoundConfig) -> ClientUpdate:
    """Parse and validate an update received from ``sender`` for ``config``.

    Enforces the closed wire schema, that the update answers *this* round and
    comes from the expected site, the disclosure policy, and that the reported
    features are exactly the ones requested.
    """
    try:
        update = decode(raw, ClientUpdate)
    except ValueError as exc:
        raise IngressRejected(f"schema violation: {type(exc).__name__}") from exc
    if update.client_id != sender:
        raise IngressRejected("sender mismatch")
    if (update.experiment_id, update.round_id, update.task) != (
        config.experiment_id,
        config.round_id,
        config.task,
    ):
        raise IngressRejected("update does not answer the current round")
    try:
        check_update(update, config.privacy)
    except DisclosureViolation as exc:
        raise IngressRejected(str(exc)) from exc
    if update.status is not UpdateStatus.COMPLETED:
        return update
    payload = update.payload
    if (
        config.feature_names is not None
        and hasattr(payload, "feature_names")
        and payload.feature_names != config.feature_names
    ):
        raise IngressRejected(ReasonCode.MISSING_FEATURES.value)
    if hasattr(payload, "n_parameters") and (
        config.model is None or payload.n_parameters != len(config.model.coefficients)
    ):
        raise IngressRejected("parameter count does not match the round's model")
    if hasattr(payload, "intercept") and (
        config.model is None or payload.horizon_months != config.model.horizon_months
    ):
        raise IngressRejected("horizon does not match the round's request")
    if hasattr(payload, "cumulative_hazard") and (
        config.model is None or payload.horizons_months != config.model.horizons_months
    ):
        raise IngressRejected("horizons do not match the round's request")
    return update
