"""Distributed evaluation of trained models.

One ``model_evaluation`` round per (model, variant). Each site evaluates on its
own evaluation partition and returns additive statistics
(:mod:`evaluation.metrics`); the coordinator adds them (site-stratified pooled
metrics) and, if configured, also releases each site's metrics.

Variants:

``released``       the global model as released (pooled baseline / intercept);
``site_baseline``  coefficients unchanged, each site's own baseline hazard
                   (Cox) or intercept (site-specific logistic), estimated on its
                   training partition - a local recalibration;
``local``          each site's own model, same configuration and features,
                   trained on its training partition only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from coordinator.aggregation import AggregationError
from coordinator.selection.protocol import RoundRequest
from evaluation.metrics import HorizonStatistics, pool, summarize
from shared.evaluation import EvaluationConfig
from shared.models import CoxModel, TrainedModel
from shared.protocol import (
    ClientUpdate,
    EvaluationRequest,
    FederatedPhase,
    HorizonEvaluation,
    MetricsSummary,
    ModelEvaluationAggregate,
    ModelEvaluationPayload,
    SiteMetrics,
    TaskKind,
    WireModel,
)
from shared.training import TrainingConfig


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluations: tuple[ModelEvaluationAggregate, ...]


def _statistics(entry: HorizonEvaluation) -> HorizonStatistics:
    return HorizonStatistics(
        horizon=entry.horizon_months, n=entry.n, n_events=entry.n_events,
        weight_sum=entry.weight_sum, weighted_events=entry.weighted_events,
        brier_sum=entry.brier_sum, auc_num=entry.auc_num, auc_den=entry.auc_den,
        c_num=entry.c_num, c_den=entry.c_den, edges=entry.edges,
        bin_n=np.array(entry.bin_n), bin_w=np.array(entry.bin_w),
        bin_wy=np.array(entry.bin_wy), bin_wp=np.array(entry.bin_wp),
    )


def _summaries(per_horizon: list[list[HorizonEvaluation]], thresholds) -> tuple:
    out = []
    for entries in per_horizon:
        horizon = entries[0].horizon_months
        if not all(e.available for e in entries):
            out.append(MetricsSummary(horizon_months=horizon, available=False))
            continue
        metrics = summarize(pool([_statistics(e) for e in entries]), thresholds)
        out.append(MetricsSummary(
            available=True,
            **{**metrics, "calibration": tuple(metrics["calibration"]),
               "decision_curve": tuple(metrics["decision_curve"])},
        ))
    return tuple(out)


class FederatedEvaluationProtocol:
    def __init__(
        self,
        config: EvaluationConfig,
        training: TrainingConfig,
        models: Sequence[TrainedModel],
        *,
        scope: str = "internal",
        clients: tuple[str, ...] | None = None,
        variants: tuple[str, ...] | None = None,
        evaluation_partition: str | None = None,
    ) -> None:
        self._config = config
        self._scope = scope
        self._clients = clients
        self._partition = evaluation_partition or config.partition
        specs = {entry.model.name: entry.model for entry in training.models}
        self._queue: list[RoundRequest] = []
        for trained in models:
            if trained.status != "trained":
                continue
            for variant in variants or self._variants(trained):
                self._queue.append(RoundRequest(
                    FederatedPhase.EVALUATION, TaskKind.MODEL_EVALUATION,
                    trained.model.feature_names, {"model_name": trained.name},
                    f"pooled_{variant}",
                    evaluation=EvaluationRequest(
                        variant=variant, model=trained.model,
                        model_spec=specs[trained.name] if variant == "local" else None,
                        evaluation_partition=self._partition, grid=config.grid,
                    ),
                    clients=clients,
                    min_clients=1 if clients is not None and len(clients) == 1 else None,
                ))
        self._results: list[ModelEvaluationAggregate] = []

    def _variants(self, trained: TrainedModel) -> tuple[str, ...]:
        variants = ["released"]
        model = trained.model
        if self._config.site_baseline_variant and (
            isinstance(model, CoxModel) or model.intercept == "site_specific"
        ):
            variants.append("site_baseline")
        if self._config.local_models:
            variants.append("local")
        return tuple(variants)

    def next_request(self) -> RoundRequest | None:
        return self._queue[0] if self._queue else None

    def on_round_failed(self, request: RoundRequest, reason: str) -> bool:
        self._queue.pop(0)  # that evaluation is unavailable; continue with the next
        return True

    def aggregate(self, request: RoundRequest, updates: Sequence[ClientUpdate]) -> WireModel:
        self._queue.pop(0)
        payloads = [(u.client_id, u.payload) for u in updates
                    if isinstance(u.payload, ModelEvaluationPayload)]
        if len(payloads) != len(updates):
            raise AggregationError("expected model_evaluation payloads")
        horizons = len(payloads[0][1].horizons)
        if any(len(p.horizons) != horizons for _, p in payloads):
            raise AggregationError("sites evaluated different horizons")
        thresholds = self._config.decision_thresholds
        pooled = _summaries([[p.horizons[h] for _, p in payloads] for h in range(horizons)],
                            thresholds)
        sites = ()
        if self._config.release_site_metrics:
            sites = tuple(
                SiteMetrics(client_id=cid,
                            horizons=_summaries([[h] for h in p.horizons], thresholds))
                for cid, p in payloads
            )
        aggregate = ModelEvaluationAggregate(
            model_name=request.params["model_name"], variant=request.evaluation.variant,
            scope=self._scope, n_clients=len(payloads), pooled=pooled, sites=sites,
            feature_names=request.evaluation.model.feature_names,
        )
        self._results.append(aggregate)
        return aggregate

    def result(self) -> EvaluationResult:
        return EvaluationResult(evaluations=tuple(self._results))
