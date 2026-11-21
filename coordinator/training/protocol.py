"""Federated training protocol (coordinator side).

For each configured model:

1. ``feature_summary`` - pooled mean and SD of the model's features (training
   partition), used to standardise every site identically. Reused across
   models with the same features.
2. ``model_update`` (repeated) - every site returns its log-likelihood,
   gradient and information at the current global parameters; the coordinator
   sums them and takes a penalised Newton step (with step halving).
   ``cox`` is a Cox model *stratified by site*: shared coefficients, one
   baseline hazard per site; its partial likelihood is the sum of the sites'.
   ``horizon_logistic`` fits one weighted logistic regression per horizon with
   site-specific censoring weights, with a common intercept or one intercept per
   site (each site profiles out its own; only the shared slopes travel).
   Both equal the corresponding centralized fit exactly; neither averages
   locally trained parameters (no FedAvg).
3. ``baseline_hazard`` (Cox) / ``site_intercept`` (site-specific logistic) -
   every site returns its own baseline at the final coefficients; the released
   baseline is their average weighted by site size.

A model whose round fails is reported as failed and the next model is trained.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from coordinator.aggregation import AggregationError, PooledMoments
from coordinator.selection.protocol import RoundRequest
from coordinator.selection.results import FederatedSelectionResult
from shared.models import (
    CoxModel,
    FitReport,
    HorizonFit,
    HorizonLogisticModel,
    TrainedModel,
)
from shared.newton import NewtonOptimizer, NewtonSettings
from shared.protocol import (
    BaselineHazardAggregate,
    BaselineHazardPayload,
    ClientUpdate,
    FederatedPhase,
    ModelIterationAggregate,
    ModelState,
    ModelUpdatePayload,
    SiteInterceptAggregate,
    SiteInterceptPayload,
    TaskKind,
    WireModel,
)
from shared.training import CoxSpec, ExplicitFeatures, HorizonLogisticSpec, TrainingConfig


class TrainingResult(WireModel):
    models: tuple[TrainedModel, ...]


@dataclass
class _Job:
    """Training state of one configured model."""

    name: str
    spec: CoxSpec | HorizonLogisticSpec
    features: tuple[str, ...]
    source: str
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    horizon_index: int = 0
    optimizer: NewtonOptimizer | None = None
    last_round: tuple[int, int, int] = (0, 0, 0)  # clients, samples, events
    sites: tuple[str, ...] | None = None
    """Sites of the first Newton round: every later Newton/baseline round must
    have the same ones, or the objective would change between iterations."""
    fits: list[HorizonFit] = field(default_factory=list)
    cox: dict | None = None
    awaiting_baseline: bool = False
    pending_fit: dict | None = None  # site-specific logistic, before its intercept round
    done: bool = False
    failure: str | None = None


def _unpack(upper: Sequence[float], p: int) -> np.ndarray:
    matrix = np.zeros((p, p))
    matrix[np.triu_indices(p)] = upper
    return matrix + np.triu(matrix, 1).T


class FederatedTrainingProtocol:
    def __init__(
        self, config: TrainingConfig, selection: FederatedSelectionResult | None
    ) -> None:
        self._config = config
        self._horizons = tuple(config.horizons_months)
        self._standardization: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._jobs = [self._job(entry, selection) for entry in config.models]
        self._results: list[TrainedModel] = []

    def _job(self, entry, selection: FederatedSelectionResult | None) -> _Job:
        if isinstance(entry.features, ExplicitFeatures):
            return _Job(entry.model.name, entry.model, entry.features.names, "explicit")
        source = f"strategy:{entry.features.from_strategy}"
        strategy = next(
            (s for s in (selection.strategies if selection else ()) if
             s.name == entry.features.from_strategy),
            None,
        )
        job = _Job(entry.model.name, entry.model, (), source)
        if strategy is None or not strategy.completed:
            job.failure, job.done = "feature_selection_unavailable", True
        elif not strategy.selected:
            job.failure, job.done = "no_feature_selected", True
        else:
            job.features = tuple(f.feature for f in strategy.selected)
        return job

    # -- planning ------------------------------------------------------------ #
    def _current(self) -> _Job | None:
        while self._jobs and self._jobs[0].done:
            self._finish(self._jobs.pop(0))
        return self._jobs[0] if self._jobs else None

    def next_request(self) -> RoundRequest | None:
        job = self._current()
        if job is None:
            return None
        phase = FederatedPhase.TRAINING
        if job.center is None:
            cached = self._standardization.get(job.features)
            if cached is None:
                return RoundRequest(
                    phase, TaskKind.FEATURE_SUMMARY, job.features, {}, "pooled_moments"
                )
            job.center, job.scale = cached
        if job.awaiting_baseline:
            return RoundRequest(
                phase, TaskKind.BASELINE_HAZARD, job.features, {}, "site_weighted_baseline",
                model=self._state(job, job.optimizer.best, baseline=True),
            )
        if job.pending_fit is not None:
            return RoundRequest(
                phase, TaskKind.SITE_INTERCEPT, job.features, {}, "site_weighted_intercept",
                model=self._state(job, np.array(job.pending_fit["coefficients"])),
            )
        if job.optimizer is None:
            job.optimizer = self._new_optimizer(job)
        return RoundRequest(
            phase, TaskKind.MODEL_UPDATE, job.features, {}, "summed_newton",
            model=self._state(job, job.optimizer.proposal),
        )

    def _new_optimizer(self, job: _Job) -> NewtonOptimizer:
        p = len(job.features)
        optimizer = job.spec.optimizer
        settings = NewtonSettings(optimizer.ridge, optimizer.max_iterations, optimizer.tolerance)
        if isinstance(job.spec, CoxSpec) or job.spec.intercept == "site_specific":
            return NewtonOptimizer(p, settings)
        return NewtonOptimizer(p + 1, settings, np.r_[False, np.ones(p, dtype=bool)])

    def _state(self, job: _Job, coefficients: np.ndarray, baseline: bool = False) -> ModelState:
        logistic = isinstance(job.spec, HorizonLogisticSpec)
        return ModelState(
            kind=job.spec.kind,
            center=tuple(job.center.tolist()),
            scale=tuple(job.scale.tolist()),
            coefficients=tuple(np.asarray(coefficients).tolist()),
            horizon_months=self._horizons[job.horizon_index] if logistic else None,
            censoring=job.spec.censoring if logistic else None,
            intercept=job.spec.intercept if logistic else None,
            horizons_months=self._horizons if baseline else (),
        )

    # -- aggregation --------------------------------------------------------- #
    def aggregate(self, request: RoundRequest, updates: Sequence[ClientUpdate]) -> WireModel:
        job = self._current()
        if request.task is not TaskKind.FEATURE_SUMMARY:
            # The standardisation is a fixed affine map; only the rounds that
            # evaluate the objective (and the final baseline) must share sites.
            sites = tuple(sorted(u.client_id for u in updates))
            if job.sites is None:
                job.sites = sites
            elif sites != job.sites:
                raise AggregationError("the set of contributing sites changed during training")
        if request.task is TaskKind.FEATURE_SUMMARY:
            aggregate = PooledMoments().aggregate(updates)
            scale = np.sqrt(np.array(aggregate.variance))
            if (scale <= 0).any():
                raise AggregationError("a model feature has no variation")
            job.center, job.scale = np.array(aggregate.mean), scale
            self._standardization[job.features] = (job.center, job.scale)
            return aggregate
        if request.task is TaskKind.MODEL_UPDATE:
            return self._aggregate_update(job, updates)
        if request.task is TaskKind.BASELINE_HAZARD:
            return self._aggregate_baseline(job, updates)
        if request.task is TaskKind.SITE_INTERCEPT:
            return self._aggregate_intercept(job, updates)
        raise AggregationError(f"unexpected task {request.task}")  # pragma: no cover

    def _aggregate_update(self, job: _Job, updates) -> ModelIterationAggregate:
        payloads = [u.payload for u in updates if isinstance(u.payload, ModelUpdatePayload)]
        p = job.optimizer.n_parameters
        if len(payloads) != len(updates) or any(x.n_parameters != p for x in payloads):
            raise AggregationError("expected model_update payloads of the current model")
        coefficients = job.optimizer.proposal.copy()
        logistic = isinstance(job.spec, HorizonLogisticSpec)
        horizon = self._horizons[job.horizon_index] if logistic else None
        loglik = float(sum(x.loglik for x in payloads))
        job.optimizer.observe(
            loglik,
            np.sum([x.gradient for x in payloads], axis=0),
            np.sum([_unpack(x.information_upper, p) for x in payloads], axis=0),
        )
        job.last_round = (
            len(payloads), sum(x.n_samples for x in payloads), sum(x.n_events for x in payloads)
        )
        if job.optimizer.finished:
            self._optimizer_finished(job)
        return ModelIterationAggregate(
            model_name=job.name,
            horizon_months=horizon,
            n_clients=job.last_round[0],
            n_samples=job.last_round[1],
            n_events=job.last_round[2],
            coefficients=tuple(coefficients.tolist()),
            loglik=loglik,
        )

    def _report(self, job: _Job, optimizer: NewtonOptimizer) -> FitReport:
        return FitReport(
            converged=optimizer.converged,
            failure_reason=optimizer.failed,
            iterations=len(optimizer.history),
            evaluations=optimizer.evaluations,
            penalized_loglik_history=tuple(optimizer.history),
            n_clients=job.last_round[0],
            n_samples=job.last_round[1],
            n_events=job.last_round[2],
        )

    def _optimizer_finished(self, job: _Job) -> None:
        optimizer = job.optimizer
        if optimizer.best is None or not optimizer.converged:
            job.failure, job.done = f"optimizer_{optimizer.failed}", True
            return
        errors = optimizer.standard_errors()
        if isinstance(job.spec, CoxSpec):
            job.cox = {
                "coefficients": tuple(optimizer.best.tolist()),
                "standard_errors": tuple(errors.tolist()),
                "fit": self._report(job, optimizer),
            }
            job.awaiting_baseline = True
            return
        if job.spec.intercept == "site_specific":
            job.pending_fit = {
                "coefficients": tuple(optimizer.best.tolist()),
                "standard_errors": tuple(errors.tolist()),
                "fit": self._report(job, optimizer),
            }
            return
        self._append_horizon(
            job, float(optimizer.best[0]), tuple(optimizer.best[1:].tolist()),
            tuple(errors[1:].tolist()), self._report(job, optimizer),
        )

    def _append_horizon(self, job: _Job, intercept, coefficients, errors, report) -> None:
        job.fits.append(
            HorizonFit(
                horizon_months=self._horizons[job.horizon_index],
                intercept=intercept,
                coefficients=coefficients,
                standard_errors=errors,
                fit=report,
            )
        )
        job.horizon_index += 1
        job.optimizer = None
        job.pending_fit = None
        if job.horizon_index == len(self._horizons):
            job.done = True

    def _aggregate_intercept(self, job: _Job, updates) -> SiteInterceptAggregate:
        payloads = [u.payload for u in updates if isinstance(u.payload, SiteInterceptPayload)]
        if len(payloads) != len(updates):
            raise AggregationError("expected site_intercept payloads")
        weights = np.array([x.n_samples for x in payloads], dtype=float)
        pooled = float(np.dot(weights, [x.intercept for x in payloads]) / weights.sum())
        horizon = self._horizons[job.horizon_index]
        pending = job.pending_fit
        self._append_horizon(
            job, pooled, pending["coefficients"], pending["standard_errors"], pending["fit"]
        )
        return SiteInterceptAggregate(
            model_name=job.name, horizon_months=horizon, n_clients=len(payloads),
            intercept=pooled,
        )

    def _aggregate_baseline(self, job: _Job, updates) -> BaselineHazardAggregate:
        payloads = [u.payload for u in updates if isinstance(u.payload, BaselineHazardPayload)]
        if len(payloads) != len(updates):
            raise AggregationError("expected baseline_hazard payloads")
        pooled: list[float | None] = []
        for h in range(len(self._horizons)):
            defined = [(x.n_samples, x.cumulative_hazard[h]) for x in payloads
                       if x.cumulative_hazard[h] is not None]
            if len(defined) < len(payloads):
                # Averaging only the sites with follow-up would mix populations
                # silently; the horizon is reported as undefined instead.
                pooled.append(None)
            else:
                weights = np.array([n for n, _ in defined], dtype=float)
                pooled.append(float(np.dot(weights, [v for _, v in defined]) / weights.sum()))
        job.cox["baseline"] = tuple(pooled)
        job.awaiting_baseline, job.done = False, True
        return BaselineHazardAggregate(
            model_name=job.name,
            n_clients=len(payloads),
            horizons_months=self._horizons,
            cumulative_hazard=tuple(pooled),
        )

    def on_round_failed(self, request: RoundRequest, reason: str) -> bool:
        """A failed round fails the current model only; training continues."""
        job = self._current()
        if job is not None:
            job.failure, job.done = f"{request.task.value}: {reason}", True
        return True

    # -- results ------------------------------------------------------------- #
    def _finish(self, job: _Job) -> None:
        if job.failure is not None:
            self._results.append(
                TrainedModel(name=job.name, status="failed", failure_reason=job.failure,
                             features_source=job.source)
            )
            return
        common = {
            "feature_names": job.features,
            "center": tuple(job.center.tolist()),
            "scale": tuple(job.scale.tolist()),
        }
        if isinstance(job.spec, CoxSpec):
            model = CoxModel(
                **common,
                coefficients=job.cox["coefficients"],
                standard_errors=job.cox["standard_errors"],
                horizons_months=self._horizons,
                baseline_cumulative_hazard=job.cox["baseline"],
                baseline="pooled_sites",
                fit=job.cox["fit"],
            )
        else:
            model = HorizonLogisticModel(
                **common, censoring=job.spec.censoring, intercept=job.spec.intercept,
                horizons=tuple(job.fits),
            )
        self._results.append(
            TrainedModel(name=job.name, status="trained", features_source=job.source, model=model)
        )

    def result(self) -> TrainingResult:
        if self._current() is not None:
            raise AggregationError("training rounds are still pending")
        return TrainingResult(models=tuple(self._results))
