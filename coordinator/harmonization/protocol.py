"""Federated ComBat (d-ComBat) and site-effect assessment.

Rounds (training partitions):

1. ``feature_inventory``         (only if no schema was agreed yet)
2. ``harmonization_statistics``  regression sums + cohort profile per site.
   The coordinator solves the covariate effects, assesses site effects before
   harmonization and how much sites differ clinically.
3. ``harmonization_variance``    (combat) residual sums of squares given the
   shared covariate effects -> pooled variance.
4. ``harmonization_apply``       (combat) each site estimates and installs its
   own adjustment and returns sums of the harmonized data -> site effect after.

With ``method: none`` only steps 1-2 run (assessment without adjustment).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from coordinator.aggregation import AggregationError, SchemaIntersection
from coordinator.selection.protocol import RoundRequest
from harmonization import assessment
from harmonization.combat import SiteSums, solve_regression
from shared.evaluation import HarmonizationConfig
from shared.protocol import (
    ClientUpdate,
    FederatedPhase,
    HarmonizationApplyPayload,
    HarmonizationState,
    HarmonizationStatisticsPayload,
    HarmonizationVariancePayload,
    PrivacyPolicy,
    SiteEffectAggregate,
    TaskKind,
    WireModel,
)


class SiteProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    client_id: str
    n: int
    event_rate: float
    median_follow_up: float | None


class HarmonizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str
    covariates: tuple[str, ...]
    n_clients: int
    harmonized_features: int
    site_effect_before: dict
    site_effect_after: dict | None
    covariate_eta_squared: dict[str, float | None]
    """How much each covariate differs between sites (a clinical difference)."""
    event_rate_chi2: float
    follow_up_range_months: tuple[float, float] | None
    """Smallest and largest site median follow-up."""
    sites: tuple[SiteProfile, ...] = ()
    """Per-site profile, only when ``release_site_summaries`` is set."""


def _optional(values: np.ndarray) -> tuple[float | None, ...]:
    return tuple(float(v) if np.isfinite(v) else None for v in values)


class FederatedHarmonizationProtocol:
    def __init__(
        self, config: HarmonizationConfig, privacy: PrivacyPolicy,
        feature_names: tuple[str, ...] | None,
    ) -> None:
        self._config = config
        self._privacy = privacy
        self._schema = feature_names
        self._done: set[TaskKind] = set()
        self._beta: np.ndarray | None = None
        self._grand_mean: np.ndarray | None = None
        self._var_pooled: np.ndarray | None = None
        self._result: dict = {}

    @property
    def schema(self) -> tuple[str, ...] | None:
        return self._schema

    @property
    def _features(self) -> tuple[str, ...]:
        return tuple(n for n in self._schema if n not in self._config.covariates)

    def _state(self, **arrays) -> HarmonizationState:
        return HarmonizationState(
            covariates=self._config.covariates,
            empirical_bayes=self._config.empirical_bayes,
            eb_tolerance=self._config.eb_tolerance,
            **{k: tuple(np.ravel(v).tolist()) for k, v in arrays.items()},
        )

    def next_request(self) -> RoundRequest | None:
        phase = FederatedPhase.INFRASTRUCTURE
        if self._schema is None:
            return RoundRequest(phase, TaskKind.FEATURE_INVENTORY, None, {},
                                "schema_intersection")
        if TaskKind.HARMONIZATION_STATISTICS not in self._done:
            missing = set(self._config.covariates) - set(self._schema)
            if missing:
                raise AggregationError(f"covariates not in the agreed schema: {sorted(missing)}")
            return RoundRequest(phase, TaskKind.HARMONIZATION_STATISTICS, self._features, {},
                                "combat_regression", harmonization=self._state())
        if self._config.method == "none":
            return None
        if TaskKind.HARMONIZATION_VARIANCE not in self._done:
            return RoundRequest(phase, TaskKind.HARMONIZATION_VARIANCE, self._features, {},
                                "combat_pooled_variance", harmonization=self._state(
                                    beta=self._beta))
        if TaskKind.HARMONIZATION_APPLY not in self._done:
            return RoundRequest(phase, TaskKind.HARMONIZATION_APPLY, self._features, {},
                                "combat_site_adjustment", harmonization=self._state(
                                    beta=self._beta, grand_mean=self._grand_mean,
                                    var_pooled=self._var_pooled))
        return None

    def on_round_failed(self, request: RoundRequest, reason: str) -> bool:
        return False  # every harmonization round is required

    def aggregate(self, request: RoundRequest, updates: Sequence[ClientUpdate]) -> WireModel:
        self._done.add(request.task)
        if request.task is TaskKind.FEATURE_INVENTORY:
            aggregate = SchemaIntersection().aggregate(updates)
            self._schema = aggregate.feature_names
            return aggregate
        if request.task is TaskKind.HARMONIZATION_STATISTICS:
            return self._statistics(updates)
        if request.task is TaskKind.HARMONIZATION_VARIANCE:
            payloads = [u.payload for u in updates
                        if isinstance(u.payload, HarmonizationVariancePayload)]
            n = sum(p.n_samples for p in payloads)
            self._var_pooled = np.sum([p.residual_ss for p in payloads], axis=0) / n
            if (self._var_pooled <= 0).any():
                raise AggregationError("a feature has no residual variance")
            return SiteEffectAggregate(
                stage="variance", n_clients=len(payloads), n_samples=n,
                feature_names=self._features, eta_squared=(None,) * len(self._features),
            )
        if request.task is TaskKind.HARMONIZATION_APPLY:
            payloads = [u.payload for u in updates
                        if isinstance(u.payload, HarmonizationApplyPayload)]
            effect = assessment.site_effect(
                np.array([p.n_samples for p in payloads]),
                np.array([p.sum_y for p in payloads]), np.array([p.sum_yy for p in payloads]),
            )
            self._result["site_effect_after"] = assessment.summarize_effect(effect["eta_squared"])
            return SiteEffectAggregate(
                stage="after", n_clients=len(payloads),
                n_samples=sum(p.n_samples for p in payloads), feature_names=self._features,
                eta_squared=_optional(effect["eta_squared"]),
            )
        raise AggregationError(f"unexpected task {request.task}")  # pragma: no cover

    def _statistics(self, updates) -> SiteEffectAggregate:
        payloads = [(u.client_id, u.payload) for u in updates
                    if isinstance(u.payload, HarmonizationStatisticsPayload)]
        if len(payloads) != len(updates):
            raise AggregationError("expected harmonization_statistics payloads")
        c, p = len(self._config.covariates), len(self._features)
        sums = []
        for _, x in payloads:
            xx = np.zeros((c, c))
            if c:
                xx[np.triu_indices(c)] = x.sum_xx_upper
                xx = xx + np.triu(xx, 1).T
            sums.append(SiteSums(x.n_samples, np.array(x.sum_x), xx, np.array(x.sum_y),
                                 np.array(x.sum_yy), np.array(x.sum_xy).reshape(c, p)))
        n = np.array([s.n for s in sums], dtype=float)
        before = assessment.site_effect(n, np.array([s.sum_y for s in sums]),
                                        np.array([s.sum_yy for s in sums]))
        covariate_eta = {}
        for j, name in enumerate(self._config.covariates):
            sum_x = np.array([[s.sum_x[j]] for s in sums])
            sum_xx = np.array([[s.sum_xx[j, j]] for s in sums])
            eta = assessment.site_effect(n, sum_x, sum_xx)["eta_squared"][0]
            covariate_eta[name] = float(eta) if np.isfinite(eta) else None
        follow = [x.median_follow_up for _, x in payloads if x.median_follow_up is not None]
        self._result.update(
            n_clients=len(payloads),
            site_effect_before=assessment.summarize_effect(before["eta_squared"]),
            covariate_eta_squared=covariate_eta,
            event_rate_chi2=assessment.event_rate_chi2(
                n, np.array([x.n_converted for _, x in payloads])),
            follow_up_range_months=(min(follow), max(follow)) if follow else None,
            sites=tuple(
                SiteProfile(client_id=cid, n=x.n_samples, event_rate=x.n_converted / x.n_samples,
                            median_follow_up=x.median_follow_up)
                for cid, x in payloads
            ) if self._config.release_site_summaries else (),
        )
        if self._config.method == "combat":
            try:
                _, self._beta, _ = solve_regression(sums)
            except np.linalg.LinAlgError as exc:
                raise AggregationError("ComBat regression is singular") from exc
            total = n.sum()
            sum_y = np.sum([s.sum_y for s in sums], axis=0)
            sum_x = np.sum([s.sum_x for s in sums], axis=0)
            # Grand mean = size-weighted site intercepts = mean of (y - x'beta).
            self._grand_mean = (sum_y - sum_x @ self._beta) / total
        return SiteEffectAggregate(
            stage="before", n_clients=len(payloads), n_samples=int(n.sum()),
            feature_names=self._features, eta_squared=_optional(before["eta_squared"]),
        )

    def result(self) -> HarmonizationResult:
        return HarmonizationResult(
            method=self._config.method,
            covariates=self._config.covariates,
            harmonized_features=len(self._features) if self._config.method == "combat" else 0,
            site_effect_after=self._result.get("site_effect_after"),
            **{k: v for k, v in self._result.items() if k != "site_effect_after"},
        )
