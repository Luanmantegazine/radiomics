"""Wire contracts between federated clients and the coordinator.

These Pydantic models are the *only* things allowed to cross the
client/coordinator boundary. The design is allowlist-based:

* every model forbids unknown fields (``extra="forbid"``), so a field such as
  ``participant_ids`` or ``time_months`` cannot be smuggled into a message;
* there is no free-text field: refusal reasons are enum codes and every string
  is pattern-constrained;
* payload arrays are indexed by *feature*, never by participant. Each payload
  validates that its array lengths equal ``len(feature_names)``.

Limitations (documented, not solved, in Stage 1):

* The coordinator sees per-client aggregates (honest-but-curious model). Secure
  aggregation and differential privacy are not implemented.
* Shape checks cannot prove that an array of length ``n_features`` does not
  encode individual values; this is mitigated by the minimum cohort size rule
  (:mod:`shared.privacy`) and by code review of the client tasks, which are the
  only producers of payloads.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from shared.models import ReleasedModel
from shared.training import CoxSpec, HorizonLogisticSpec

#: Version of the client/coordinator protocol. Bump on any contract change.
PROTOCOL_VERSION = "1.4"

#: Upper bound on a single serialized message (coarse exfiltration/volume guard).
MAX_MESSAGE_BYTES = 1_048_576

ClientId = Annotated[str, StringConstraints(pattern=r"^client-[a-z0-9]{2,32}$")]
FeatureName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{1,128}$")]
Token = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.+\-]{1,64}$")]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,96}$")]
JsonScalar = bool | int | float | str | None


class WireModel(BaseModel):
    """Base class of every message: immutable and closed to unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FederatedPhase(StrEnum):
    """Kind of federated activity a round belongs to.

    The three scientific phases are kept separate because they need different
    aggregation strategies. ``INFRASTRUCTURE`` is the only phase with tasks
    implemented in Stage 1.
    """

    INFRASTRUCTURE = "infrastructure"
    HARMONIZATION = "harmonization"
    FEATURE_SELECTION = "feature_selection"
    TRAINING = "training"
    EVALUATION = "evaluation"


class TaskKind(StrEnum):
    """Local computations a client can be asked to run."""

    FEATURE_INVENTORY = "feature_inventory"
    FEATURE_SUMMARY = "feature_summary"
    FEATURE_RELIABILITY = "feature_reliability"
    FEATURE_RELEVANCE = "feature_relevance"
    FEATURE_COVARIANCE = "feature_covariance"
    MODEL_UPDATE = "model_update"
    BASELINE_HAZARD = "baseline_hazard"
    SITE_INTERCEPT = "site_intercept"
    HARMONIZATION_STATISTICS = "harmonization_statistics"
    HARMONIZATION_VARIANCE = "harmonization_variance"
    HARMONIZATION_APPLY = "harmonization_apply"
    MODEL_EVALUATION = "model_evaluation"


class DataOrigin(StrEnum):
    """Provenance of the data behind a result. Synthetic results are never science."""

    SYNTHETIC = "synthetic"
    REAL = "real"


class UpdateStatus(StrEnum):
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"


class ReasonCode(StrEnum):
    """Closed vocabulary for refusals/failures (free text would be a leak channel)."""

    COHORT_TOO_SMALL = "cohort_too_small"
    UNSUPPORTED_TASK = "unsupported_task"
    MISSING_FEATURES = "missing_features"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    EXPERIMENT_MISMATCH = "experiment_mismatch"
    NOT_REGISTERED = "not_registered"
    DISCLOSURE_BLOCKED = "disclosure_blocked"
    UNKNOWN_CLIENT = "unknown_client"
    DUPLICATE_CLIENT = "duplicate_client"
    DATA_ORIGIN_MISMATCH = "data_origin_mismatch"
    RELIABILITY_UNAVAILABLE = "reliability_unavailable"
    TOO_FEW_EVENTS = "too_few_events"
    NO_RESPONSE = "no_response"
    INTERNAL_ERROR = "internal_error"


class PrivacyPolicy(WireModel):
    """Disclosure-control thresholds, distributed with every round."""

    min_cohort_size: int = Field(default=10, ge=2)
    """A client never reports statistics computed on fewer participants."""
    min_clients_per_aggregate: int = Field(default=2, ge=2)
    """The coordinator never releases an aggregate built from fewer clients."""
    min_events: int = Field(default=5, ge=1)
    """A client never reports outcome-derived statistics computed from fewer
    observed events: with a single event, the Cox score is that participant's
    feature value minus the risk-set mean."""


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class ClientRegistration(WireModel):
    """Client -> coordinator. Announces a pseudonymised participant site."""

    message_type: Literal["client_registration"] = "client_registration"
    protocol_version: Token
    experiment_id: Identifier
    client_id: ClientId
    supported_tasks: tuple[TaskKind, ...]
    data_origin: DataOrigin
    software: dict[Token, Token] = Field(default_factory=dict)
    """Library name -> version, for provenance."""


class RegistrationAck(WireModel):
    """Coordinator -> client."""

    message_type: Literal["registration_ack"] = "registration_ack"
    protocol_version: Token
    experiment_id: Identifier
    client_id: ClientId
    accepted: bool
    reason_code: ReasonCode | None = None


class PartitionSpec(WireModel):
    """Which local partition a task must use (the split is computed by the site)."""

    name: Literal["train", "test"]
    test_fraction: float = Field(gt=0, lt=1)
    seed: int = Field(ge=0)


class ModelState(WireModel):
    """Global model parameters at which sites evaluate their local contribution."""

    kind: Literal["cox", "horizon_logistic"]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    """For ``horizon_logistic`` the intercept comes first."""
    horizon_months: float | None = Field(default=None, gt=0)
    censoring: Literal["ipcw", "complete_case"] | None = None
    intercept: Literal["common", "site_specific"] | None = None
    """Horizon logistic only. ``site_specific``: each site profiles out its own
    intercept, so ``coefficients`` holds the shared slopes only."""
    horizons_months: tuple[float, ...] = ()
    """Horizons at which a baseline is requested (``baseline_hazard`` task)."""

    @model_validator(mode="after")
    def _shape(self) -> ModelState:
        p = len(self.center)
        if len(self.scale) != p or any(v <= 0 for v in self.scale):
            raise ValueError("scale must be positive, one entry per feature")
        if self.kind == "horizon_logistic" and (
            self.horizon_months is None or self.censoring is None or self.intercept is None
        ):
            raise ValueError("a horizon logistic model needs horizon, censoring and intercept")
        expected = p + (1 if self.intercept == "common" else 0)
        if len(self.coefficients) != expected:
            raise ValueError("wrong number of coefficients")
        if not all(math.isfinite(v) for v in (*self.center, *self.coefficients)):
            raise ValueError("model state must be finite")
        return self


class HarmonizationState(WireModel):
    """Global ComBat quantities sent to sites (no site-specific value)."""

    covariates: tuple[FeatureName, ...]
    beta: tuple[float, ...] = ()
    """Covariate effects, row-major (n_covariates x n_features)."""
    grand_mean: tuple[float, ...] = ()
    var_pooled: tuple[float, ...] = ()
    empirical_bayes: bool = True
    eb_tolerance: float = Field(default=1e-4, gt=0, lt=1)

    @model_validator(mode="after")
    def _finite(self) -> HarmonizationState:
        if not all(math.isfinite(v) for v in (*self.beta, *self.grand_mean, *self.var_pooled)):
            raise ValueError("harmonization state must be finite")
        if any(v <= 0 for v in self.var_pooled):
            raise ValueError("pooled variance must be positive")
        return self


class EvaluationRequest(WireModel):
    """What a site must evaluate in a ``model_evaluation`` round."""

    variant: Literal["released", "site_baseline", "local"]
    model: ReleasedModel
    """Global released model (coefficients, standardisation, pooled baseline)."""
    model_spec: CoxSpec | HorizonLogisticSpec | None = None
    """For the ``local`` variant: the configuration to refit on the site alone."""
    evaluation_partition: Literal["test", "all"]
    grid: tuple[float, ...] = Field(min_length=3)


class SessionEnd(WireModel):
    """Coordinator -> client. The experiment is over; the site may disconnect."""

    message_type: Literal["session_end"] = "session_end"
    protocol_version: Token
    experiment_id: Identifier
    completed: bool


# --------------------------------------------------------------------------- #
# Round configuration
# --------------------------------------------------------------------------- #
class RoundConfig(WireModel):
    """Coordinator -> client. Everything a client needs to run one round."""

    message_type: Literal["round_config"] = "round_config"
    protocol_version: Token
    experiment_id: Identifier
    round_id: Identifier
    round_number: int = Field(ge=1)
    phase: FederatedPhase
    task: TaskKind
    seed: int = Field(ge=0)
    config_hash: Token
    """Hash of the versioned experiment configuration this round derives from."""
    privacy: PrivacyPolicy
    feature_names: tuple[FeatureName, ...] | None = None
    """Agreed feature schema, when the task needs one."""
    params: dict[Token, JsonScalar] = Field(default_factory=dict)
    partition: PartitionSpec | None = None
    """Local partition to use; None = all local participants."""
    model: ModelState | None = None
    harmonization: HarmonizationState | None = None
    evaluation: EvaluationRequest | None = None


# --------------------------------------------------------------------------- #
# Authorized payloads (client -> coordinator)
# --------------------------------------------------------------------------- #
class FeatureInventoryPayload(WireModel):
    """Names of the features available at a site. Contains no values."""

    kind: Literal["feature_inventory"] = "feature_inventory"
    feature_names: tuple[FeatureName, ...]


class FeatureSummaryPayload(WireModel):
    """Per-feature sufficient statistics (count, mean, unbiased variance).

    Infrastructure payload of Stage 1: it exercises a real, verifiable
    aggregation (pooled moments) without carrying any scientific claim.
    """

    kind: Literal["feature_summary"] = "feature_summary"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    n_samples: int = Field(ge=2)
    mean: tuple[float, ...]
    variance: tuple[float, ...]

    @model_validator(mode="after")
    def _feature_indexed_and_finite(self) -> FeatureSummaryPayload:
        n = len(self.feature_names)
        if len(self.mean) != n or len(self.variance) != n:
            raise ValueError("mean/variance must have exactly one entry per feature")
        if len(set(self.feature_names)) != n:
            raise ValueError("feature_names must be unique")
        if not all(math.isfinite(v) for v in (*self.mean, *self.variance)):
            raise ValueError("statistics must be finite")
        if any(v < 0 for v in self.variance):
            raise ValueError("variance cannot be negative")
        return self


def _check_feature_indexed(names: tuple[str, ...], *arrays: tuple) -> None:
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    for array in arrays:
        if len(array) != len(names):
            raise ValueError("arrays must have exactly one entry per feature")


def _check_finite(*arrays: tuple) -> None:
    if not all(math.isfinite(v) for array in arrays for v in array if v is not None):
        raise ValueError("statistics must be finite")


class FeatureReliabilityPayload(WireModel):
    """Per-feature reliability estimated locally from repeated observations.

    ``values`` holds one summary per feature (e.g. the site's minimum ICC over
    its reliability families); ``None`` means undefined at this site.
    """

    kind: Literal["feature_reliability"] = "feature_reliability"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    column: Literal["min_icc", "min_ci_low"]
    n_subjects: int = Field(ge=2)
    values: tuple[float | None, ...]

    @model_validator(mode="after")
    def _shape(self) -> FeatureReliabilityPayload:
        _check_feature_indexed(self.feature_names, self.values)
        _check_finite(self.values)
        if any(v is not None and not -1.0 <= v <= 1.0 for v in self.values):
            raise ValueError("reliability values must lie in [-1, 1]")
        return self


class FeatureRelevancePayload(WireModel):
    """Per-feature components of the univariable Cox score test at beta = 0.

    ``score`` (U) and ``information`` (I) are computed on the site's own risk
    sets, so summing them over sites gives the site-stratified score test.
    """

    kind: Literal["feature_relevance"] = "feature_relevance"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    score: tuple[float, ...]
    information: tuple[float, ...]

    @model_validator(mode="after")
    def _shape(self) -> FeatureRelevancePayload:
        _check_feature_indexed(self.feature_names, self.score, self.information)
        _check_finite(self.score, self.information)
        if any(v < 0 for v in self.information):
            raise ValueError("information cannot be negative")
        if self.n_events > self.n_samples:
            raise ValueError("n_events cannot exceed n_samples")
        return self


class FeatureCovariancePayload(WireModel):
    """Within-site scatter matrix of a candidate feature set.

    ``scatter_upper`` is the row-major upper triangle (diagonal included) of
    sum_i (x_i - mean)(x_i - mean)^T over the site's participants, of length
    p(p+1)/2. With ``rank_transformed`` the columns were replaced by their
    within-site ranks first.
    """

    kind: Literal["feature_covariance"] = "feature_covariance"
    feature_names: tuple[FeatureName, ...] = Field(min_length=2, max_length=200)
    n_samples: int = Field(ge=2)
    rank_transformed: bool
    scatter_upper: tuple[float, ...]

    @model_validator(mode="after")
    def _shape(self) -> FeatureCovariancePayload:
        _check_feature_indexed(self.feature_names)
        p = len(self.feature_names)
        if len(self.scatter_upper) != p * (p + 1) // 2:
            raise ValueError("scatter_upper must have p(p+1)/2 entries")
        _check_finite(self.scatter_upper)
        diagonal = [self.scatter_upper[i * p - i * (i - 1) // 2] for i in range(p)]
        if any(v < 0 for v in diagonal):
            raise ValueError("scatter diagonal cannot be negative")
        return self


class ModelUpdatePayload(WireModel):
    """Local log-likelihood, gradient and information at the global parameters.

    For ``cox`` the site's own risk sets are used (stratified by site); for
    ``horizon_logistic`` the site's own censoring weights. Summing over sites
    gives the corresponding pooled quantities exactly.
    """

    kind: Literal["model_update"] = "model_update"
    n_parameters: int = Field(ge=1, le=200)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    loglik: float
    gradient: tuple[float, ...]
    information_upper: tuple[float, ...]

    @model_validator(mode="after")
    def _shape(self) -> ModelUpdatePayload:
        p = self.n_parameters
        if len(self.gradient) != p or len(self.information_upper) != p * (p + 1) // 2:
            raise ValueError("gradient needs p entries and information_upper p(p+1)/2")
        _check_finite((self.loglik,), self.gradient, self.information_upper)
        if self.n_events > self.n_samples:
            raise ValueError("n_events cannot exceed n_samples")
        return self


class BaselineHazardPayload(WireModel):
    """Breslow cumulative baseline hazard at the requested horizons.

    ``None`` where the site has nobody at risk at that horizon (no extrapolation).
    """

    kind: Literal["baseline_hazard"] = "baseline_hazard"
    horizons_months: tuple[float, ...] = Field(min_length=1)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    cumulative_hazard: tuple[float | None, ...]

    @model_validator(mode="after")
    def _shape(self) -> BaselineHazardPayload:
        if len(self.cumulative_hazard) != len(self.horizons_months):
            raise ValueError("one cumulative hazard per horizon")
        _check_finite(self.cumulative_hazard)
        if any(v is not None and v < 0 for v in self.cumulative_hazard):
            raise ValueError("cumulative hazard cannot be negative")
        return self


class SiteInterceptPayload(WireModel):
    """The site's own intercept of a site-specific horizon logistic model."""

    kind: Literal["site_intercept"] = "site_intercept"
    horizon_months: float = Field(gt=0)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    intercept: float

    @model_validator(mode="after")
    def _finite(self) -> SiteInterceptPayload:
        _check_finite((self.intercept,))
        return self


class HarmonizationStatisticsPayload(WireModel):
    """Regression sums for ComBat and the site's cohort profile."""

    kind: Literal["harmonization_statistics"] = "harmonization_statistics"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    covariate_names: tuple[FeatureName, ...]
    n_samples: int = Field(ge=2)
    n_converted: int = Field(ge=0)
    """Observed conversions (any time) - a count, for the cohort profile."""
    median_follow_up: float | None
    """Reverse Kaplan-Meier median follow-up, rounded to whole months (an exact
    median would be one participant's own follow-up time). None if not reached."""
    sum_x: tuple[float, ...]
    sum_xx_upper: tuple[float, ...]
    sum_y: tuple[float, ...]
    sum_yy: tuple[float, ...]
    sum_xy: tuple[float, ...]
    """Row-major (n_covariates x n_features)."""

    @model_validator(mode="after")
    def _shape(self) -> HarmonizationStatisticsPayload:
        _check_feature_indexed(self.feature_names, self.sum_y, self.sum_yy)
        c, p = len(self.covariate_names), len(self.feature_names)
        if len(self.sum_x) != c or len(self.sum_xx_upper) != c * (c + 1) // 2:
            raise ValueError("covariate sums have the wrong length")
        if len(self.sum_xy) != c * p:
            raise ValueError("sum_xy must have n_covariates x n_features entries")
        _check_finite(self.sum_x, self.sum_xx_upper, self.sum_y, self.sum_yy, self.sum_xy,
                      (self.median_follow_up,))
        if self.n_converted > self.n_samples:
            raise ValueError("n_converted cannot exceed n_samples")
        if self.median_follow_up is not None and self.median_follow_up != round(
            self.median_follow_up
        ):
            raise ValueError("median_follow_up must be rounded to whole months")
        return self


class HarmonizationVariancePayload(WireModel):
    kind: Literal["harmonization_variance"] = "harmonization_variance"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    n_samples: int = Field(ge=2)
    residual_ss: tuple[float, ...]

    @model_validator(mode="after")
    def _shape(self) -> HarmonizationVariancePayload:
        _check_feature_indexed(self.feature_names, self.residual_ss)
        _check_finite(self.residual_ss)
        return self


class HarmonizationApplyPayload(WireModel):
    """Sums of the harmonized training data (to measure the remaining site effect)."""

    kind: Literal["harmonization_apply"] = "harmonization_apply"
    feature_names: tuple[FeatureName, ...] = Field(min_length=1)
    n_samples: int = Field(ge=2)
    sum_y: tuple[float, ...]
    sum_yy: tuple[float, ...]

    @model_validator(mode="after")
    def _shape(self) -> HarmonizationApplyPayload:
        _check_feature_indexed(self.feature_names, self.sum_y, self.sum_yy)
        _check_finite(self.sum_y, self.sum_yy)
        return self


class HorizonEvaluation(WireModel):
    """Additive evaluation statistics at one horizon (see ``evaluation.metrics``)."""

    horizon_months: float = Field(gt=0)
    available: bool
    """False when the model gives no prediction at this horizon (beyond follow-up)."""
    n: int = Field(default=0, ge=0)
    n_events: int = Field(default=0, ge=0)
    weight_sum: float = 0.0
    weighted_events: float = 0.0
    brier_sum: float = 0.0
    auc_num: float = 0.0
    auc_den: float = 0.0
    c_num: float = 0.0
    c_den: float = 0.0
    edges: tuple[float, ...] = ()
    bin_n: tuple[float, ...] = ()
    bin_w: tuple[float, ...] = ()
    bin_wy: tuple[float, ...] = ()
    bin_wp: tuple[float, ...] = ()

    @model_validator(mode="after")
    def _shape(self) -> HorizonEvaluation:
        bins = max(len(self.edges) - 1, 0)
        if any(len(v) != bins for v in (self.bin_n, self.bin_w, self.bin_wy, self.bin_wp)):
            raise ValueError("one value per bin")
        _check_finite(self.edges, self.bin_n, self.bin_w, self.bin_wy, self.bin_wp,
                      (self.weight_sum, self.weighted_events, self.brier_sum, self.auc_num,
                       self.auc_den, self.c_num, self.c_den))
        return self


class ModelEvaluationPayload(WireModel):
    kind: Literal["model_evaluation"] = "model_evaluation"
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    horizons: tuple[HorizonEvaluation, ...] = Field(min_length=1)


AuthorizedPayload = Annotated[
    FeatureInventoryPayload
    | FeatureSummaryPayload
    | FeatureReliabilityPayload
    | FeatureRelevancePayload
    | FeatureCovariancePayload
    | ModelUpdatePayload
    | BaselineHazardPayload
    | SiteInterceptPayload
    | HarmonizationStatisticsPayload
    | HarmonizationVariancePayload
    | HarmonizationApplyPayload
    | ModelEvaluationPayload,
    Field(discriminator="kind"),
]


class CostReport(WireModel):
    """Client-side computational cost of one round (no data-derived content)."""

    durations_s: dict[Token, float] = Field(default_factory=dict)
    peak_memory_bytes: int | None = Field(default=None, ge=0)


class ClientUpdate(WireModel):
    """Client -> coordinator. Result of one round."""

    message_type: Literal["client_update"] = "client_update"
    protocol_version: Token
    experiment_id: Identifier
    round_id: Identifier
    client_id: ClientId
    task: TaskKind
    status: UpdateStatus
    reason_code: ReasonCode | None = None
    payload: AuthorizedPayload | None = None
    costs: CostReport = Field(default_factory=CostReport)

    @model_validator(mode="after")
    def _status_consistency(self) -> ClientUpdate:
        if self.status is UpdateStatus.COMPLETED:
            if self.payload is None:
                raise ValueError("a completed update must carry a payload")
            if self.payload.kind != self.task.value:
                raise ValueError("payload kind does not match the requested task")
        else:
            if self.payload is not None:
                raise ValueError("a refused/failed update must not carry a payload")
            if self.reason_code is None:
                raise ValueError("a refused/failed update must carry a reason code")
        return self


# --------------------------------------------------------------------------- #
# Global aggregates (produced by the coordinator)
# --------------------------------------------------------------------------- #
class FeatureInventoryAggregate(WireModel):
    """Features available at every participating site."""

    kind: Literal["feature_inventory"] = "feature_inventory"
    n_clients: int = Field(ge=2)
    feature_names: tuple[FeatureName, ...]


class FeatureSummaryAggregate(WireModel):
    """Pooled per-feature moments over all participating sites."""

    kind: Literal["feature_summary"] = "feature_summary"
    n_clients: int = Field(ge=2)
    n_samples: int = Field(ge=2)
    feature_names: tuple[FeatureName, ...]
    mean: tuple[float, ...]
    variance: tuple[float, ...]


class FeatureReliabilityAggregate(WireModel):
    """Reliability per feature combined over the reporting sites.

    Limitation: ``minimum`` is by construction the value of one site.
    """

    kind: Literal["feature_reliability"] = "feature_reliability"
    n_clients: int = Field(ge=2)
    column: Literal["min_icc", "min_ci_low"]
    feature_names: tuple[FeatureName, ...]
    minimum: tuple[float | None, ...]
    weighted_mean: tuple[float | None, ...]
    """Mean weighted by the number of subjects of each site's reliability study."""


class FeatureRelevanceAggregate(WireModel):
    """Site-stratified relevance and between-site agreement per feature."""

    kind: Literal["feature_relevance"] = "feature_relevance"
    n_clients: int = Field(ge=2)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    feature_names: tuple[FeatureName, ...]
    stratified_z: tuple[float | None, ...]
    i_squared: tuple[float | None, ...]
    sign_consistency: tuple[float | None, ...]


class FeatureCorrelationAggregate(WireModel):
    """Pooled within-site correlation of the candidate features (upper triangle)."""

    kind: Literal["feature_covariance"] = "feature_covariance"
    n_clients: int = Field(ge=2)
    n_samples: int = Field(ge=2)
    method: Literal["pearson", "spearman"]
    feature_names: tuple[FeatureName, ...]
    correlation_upper: tuple[float | None, ...]


class ModelIterationAggregate(WireModel):
    """Global evaluation of one Newton iterate (no per-site quantity)."""

    kind: Literal["model_update"] = "model_update"
    model_name: str
    horizon_months: float | None
    n_clients: int = Field(ge=2)
    n_samples: int = Field(ge=2)
    n_events: int = Field(ge=1)
    coefficients: tuple[float, ...]
    loglik: float


class BaselineHazardAggregate(WireModel):
    kind: Literal["baseline_hazard"] = "baseline_hazard"
    model_name: str
    n_clients: int = Field(ge=2)
    horizons_months: tuple[float, ...]
    cumulative_hazard: tuple[float | None, ...]
    """Average of the site baselines weighted by site size (see docs)."""


class SiteInterceptAggregate(WireModel):
    kind: Literal["site_intercept"] = "site_intercept"
    model_name: str
    horizon_months: float
    n_clients: int = Field(ge=2)
    intercept: float
    """Average of the site intercepts weighted by site size (see docs)."""


class SiteEffectAggregate(WireModel):
    """Between-site share of variance per feature (before or after harmonization)."""

    kind: Literal["site_effect"] = "site_effect"
    stage: Literal["before", "after", "variance"]
    n_clients: int = Field(ge=2)
    n_samples: int = Field(ge=2)
    feature_names: tuple[FeatureName, ...]
    eta_squared: tuple[float | None, ...]


class MetricsSummary(WireModel):
    horizon_months: float
    available: bool
    n: int = 0
    n_events: int = 0
    observed_incidence: float | None = None
    brier: float | None = None
    scaled_brier: float | None = None
    auc: float | None = None
    c_index: float | None = None
    o_e: float | None = None
    calibration_error: float | None = None
    calibration: tuple[dict, ...] = ()
    decision_curve: tuple[dict, ...] = ()


class SiteMetrics(WireModel):
    client_id: ClientId
    horizons: tuple[MetricsSummary, ...]


class ModelEvaluationAggregate(WireModel):
    kind: Literal["model_evaluation"] = "model_evaluation"
    model_name: str
    variant: Literal["released", "site_baseline", "local"]
    scope: str
    """``internal`` or ``external:<client_id>``."""
    feature_names: tuple[FeatureName, ...] = ()
    """Features of the evaluated model (they can differ between external folds)."""
    n_clients: int = Field(ge=1)
    pooled: tuple[MetricsSummary, ...]
    sites: tuple[SiteMetrics, ...] = ()


GlobalAggregate = Annotated[
    FeatureInventoryAggregate
    | FeatureSummaryAggregate
    | FeatureReliabilityAggregate
    | FeatureRelevanceAggregate
    | FeatureCorrelationAggregate
    | ModelIterationAggregate
    | BaselineHazardAggregate
    | SiteInterceptAggregate
    | SiteEffectAggregate
    | ModelEvaluationAggregate,
    Field(discriminator="kind"),
]


def encode(message: WireModel) -> bytes:
    """Serialize a message to its canonical wire form (UTF-8 JSON)."""
    return message.model_dump_json().encode("utf-8")


def decode[T: WireModel](raw: bytes, model: type[T]) -> T:
    """Parse and validate wire bytes. Raises ``ValueError`` on any violation."""
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message exceeds {MAX_MESSAGE_BYTES} bytes")
    return model.model_validate_json(raw)
