"""The wire schema is closed: only authorized, feature-indexed content fits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    ClientUpdate,
    FeatureSummaryPayload,
    decode,
    encode,
)

BASE = {
    "protocol_version": PROTOCOL_VERSION,
    "experiment_id": "exp-1",
    "round_id": "exp-1-r001",
    "client_id": "client-01",
    "task": "feature_summary",
    "status": "completed",
    "payload": {
        "kind": "feature_summary",
        "feature_names": ["a", "b"],
        "n_samples": 20,
        "mean": [0.1, 0.2],
        "variance": [1.0, 2.0],
    },
}


def test_valid_update_round_trips():
    update = ClientUpdate.model_validate(BASE)
    assert decode(encode(update), ClientUpdate) == update


@pytest.mark.parametrize(
    "field", ["participant_ids", "time_months", "event", "features", "labels", "notes"]
)
def test_unknown_top_level_fields_are_rejected(field):
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, field: [1, 2, 3]})


def test_unknown_payload_fields_are_rejected():
    payload = {**BASE["payload"], "per_participant_values": [[0.1, 0.2]] * 20}
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, "payload": payload})


def test_unknown_payload_kind_is_rejected():
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, "payload": {"kind": "raw_features", "rows": []}})


def test_arrays_must_be_indexed_by_feature_not_participant():
    # 20 values (one per participant) for 2 features cannot be expressed.
    with pytest.raises(ValidationError):
        FeatureSummaryPayload(
            feature_names=("a", "b"), n_samples=20, mean=tuple(range(20)), variance=(1.0, 2.0)
        )


def test_free_text_cannot_be_used_as_reason():
    refused = {**BASE, "status": "refused", "payload": None}
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**refused, "reason_code": "patient OAS30001 converted"})
    assert ClientUpdate.model_validate({**refused, "reason_code": "cohort_too_small"})


def test_refused_update_cannot_carry_payload():
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, "status": "refused", "reason_code": "internal_error"})


def test_payload_must_match_requested_task():
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, "task": "feature_inventory"})


def test_client_identifier_must_be_pseudonymised():
    with pytest.raises(ValidationError):
        ClientUpdate.model_validate({**BASE, "client_id": "Hospital Sao Lucas"})


def test_oversized_messages_are_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        decode(b" " * (MAX_MESSAGE_BYTES + 1), ClientUpdate)
