"""Distributed evaluation: equivalence, variants, external validation, isolation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from coordinator.transport import Direction, InProcessTransport
from experiments.centralized import AuthorizationRequired, centralized_reference
from experiments.simulation import build_clients, run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from shared.protocol import TaskKind
from tests.federated.test_harmonization import STAGE6


@pytest.fixture(scope="module")
def run():
    config = load_experiment_config(STAGE6)
    transport = InProcessTransport(capture=True)
    outcome = run_simulation("exp-s6", config, transport=transport)
    return config, outcome, transport


def _evaluations(outcome, **filters):
    return [e for e in outcome.evaluation.evaluations
            if all(getattr(e, k) == v for k, v in filters.items())]


def test_pipeline_completes_with_every_phase(run):
    _, outcome, _ = run
    assert outcome.completed
    phases = {r.phase.value for r in outcome.rounds}
    assert phases == {"harmonization", "feature_selection", "training", "evaluation"}
    assert outcome.harmonization.method == "combat"
    internal = _evaluations(outcome, scope="internal")
    assert {(e.model_name, e.variant) for e in internal} >= {
        ("cox_selected", "released"), ("cox_selected", "site_baseline"),
        ("cox_selected", "local"), ("cox_clinical_reference", "released"),
    }
    assert all(e.n_clients == 3 for e in internal)
    assert outcome.evaluation_s > 0 and outcome.harmonization_s > 0


def test_distributed_evaluation_equals_centralized_stratified_evaluation(run):
    config, outcome, _ = run
    reference = centralized_reference(config, outcome, "synthetic data, test authorization")
    assert reference["authorization"] == "synthetic data, test authorization"
    by_model = {m["model_name"]: m for m in reference["models"]}
    for evaluation in _evaluations(outcome, scope="internal", variant="released"):
        central = by_model[evaluation.model_name]["released_federated_model"][
            "stratified_evaluation"]
        for federated, centralized in zip(evaluation.pooled, central, strict=True):
            for metric in ("auc", "c_index", "brier", "o_e", "n", "n_events"):
                assert getattr(federated, metric) == pytest.approx(centralized[metric],
                                                                   rel=1e-10), metric


def test_centralized_reference_requires_authorization(run):
    config, outcome, _ = run
    for authorization in ("", "   "):
        with pytest.raises(AuthorizationRequired):
            centralized_reference(config, outcome, authorization)


def test_evaluation_uses_the_held_out_partition_only(run):
    config, outcome, _ = run
    from client.splits import partition
    from shared.protocol import PartitionSpec

    test_sizes = sum(
        partition(c._data_source.load(), PartitionSpec(name="test", test_fraction=0.3, seed=7))
        .n_participants for c in build_clients("exp-s6", config)
    )
    released = _evaluations(outcome, scope="internal", variant="released")[0]
    assert all(h.n == test_sizes for h in released.pooled)


def test_variants_behave_as_defined(run):
    _, outcome, _ = run
    released = _evaluations(outcome, model_name="cox_selected", scope="internal",
                            variant="released")[0]
    site = _evaluations(outcome, model_name="cox_selected", scope="internal",
                        variant="site_baseline")[0]
    local = _evaluations(outcome, model_name="cox_selected", scope="internal",
                         variant="local")[0]
    for a, b in zip(released.pooled, site.pooled, strict=True):
        # Same coefficients: within-site ranking (AUC, C) identical; calibration differs.
        assert a.auc == pytest.approx(b.auc) and a.c_index == pytest.approx(b.c_index)
    assert [a.brier for a in released.pooled] != [b.brier for b in site.pooled]
    assert [a.auc for a in released.pooled] != [c.auc for c in local.pooled]
    common = _evaluations(outcome, model_name="logistic_ipcw_site", scope="internal")
    assert {e.variant for e in common} == {"released", "site_baseline", "local"}


def test_metrics_are_complete_and_sites_released(run):
    _, outcome, _ = run
    evaluation = _evaluations(outcome, model_name="cox_reference", scope="internal",
                              variant="released")[0]
    for horizon in evaluation.pooled:
        assert horizon.available and 0.5 < horizon.auc < 1 and 0 < horizon.brier < 0.25
        assert horizon.calibration and horizon.decision_curve
        assert all(b["n"] == 0 or b["n"] >= 10 for b in horizon.calibration)
    assert {s.client_id for s in evaluation.sites} == {"client-01", "client-02", "client-03"}


def test_external_validation_exposes_a_non_transportable_feature(run):
    """Without client-01, feat_mixed looks consistent (two sites, same sign) and is
    selected; its effect is reversed at client-01, so the model does not transport."""
    _, outcome, _ = run
    external = _evaluations(outcome, scope="external:client-01", model_name="cox_selected")[0]
    assert external.n_clients == 1 and "feat_mixed" in external.feature_names
    assert external.pooled[1].auc < 0.6
    reference = _evaluations(outcome, scope="external:client-01", model_name="cox_reference")[0]
    assert reference.pooled[1].auc > 0.65  # fixed features without feat_mixed transport
    scopes = {e.scope for e in outcome.evaluation.evaluations}
    assert scopes == {"internal", "external:client-01", "external:client-02",
                      "external:client-03"}


def test_no_prediction_or_individual_value_leaves_a_site(run):
    config, _, transport = run
    from client.models.prediction import predict_conversion

    datasets = [c._data_source.load() for c in build_clients("exp-s6", config)]
    wire = b"\n".join(r.raw for r in transport.log.records)
    assert not any(pid.encode() in wire for d in datasets for pid in d.participant_ids)
    sent = []
    for record in transport.log.records:
        if record.direction is not Direction.TO_COORDINATOR:
            continue
        payload = json.loads(record.raw).get("payload") or {}
        if payload.get("kind") == "model_evaluation":
            for horizon in payload["horizons"]:
                sent += [v for k, v in horizon.items() if isinstance(v, float)]
                for key in ("bin_n", "bin_w", "bin_wy", "bin_wp"):
                    sent += horizon[key]
                assert all(n == 0 or n >= 10 for n in horizon["bin_n"])
    model = run[1].training.models[0].model
    _, probabilities = predict_conversion(model, datasets[0].feature_names, datasets[0].features)
    individual = np.r_[probabilities.ravel(), datasets[0].time_months]
    assert np.abs(np.array(sent)[:, None] - individual[None, :]).min() > 1e-9


def test_per_site_metrics_can_be_withheld(run):
    raw = load_experiment_config(STAGE6).model_dump(mode="json")
    raw["evaluation"].update(release_site_metrics=False, external_validation="none",
                             local_models=False, site_baseline_variant=False)
    outcome = run_simulation("exp-s6", ExperimentConfig.model_validate(raw))
    assert all(e.sites == () for e in outcome.evaluation.evaluations)
    assert {e.variant for e in outcome.evaluation.evaluations} == {"released"}
    assert sum(r.task is TaskKind.MODEL_EVALUATION for r in outcome.rounds) == 5
