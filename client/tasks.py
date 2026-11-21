"""Local computations. A task is the only place where individual-level data is
turned into an authorized, aggregate payload.

Stage 1 ships two infrastructure tasks. Scientific tasks (reliability,
prognostic relevance, redundancy, local training) are added in later stages by
registering new :class:`LocalTask` implementations.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from client.data import LocalDataset
from shared.privacy import DisclosureViolation
from shared.protocol import (
    BaselineHazardPayload,
    FeatureCovariancePayload,
    FeatureInventoryPayload,
    FeatureRelevancePayload,
    FeatureReliabilityPayload,
    FeatureSummaryPayload,
    HarmonizationApplyPayload,
    HarmonizationStatisticsPayload,
    HarmonizationVariancePayload,
    HorizonEvaluation,
    ModelEvaluationPayload,
    ModelState,
    ModelUpdatePayload,
    ReasonCode,
    RoundConfig,
    SiteInterceptPayload,
    TaskKind,
    WireModel,
)


class TaskRefused(Exception):
    """The task cannot be run locally; reported to the coordinator as a code."""

    def __init__(self, code: ReasonCode) -> None:
        super().__init__(code.value)
        self.code = code


class LocalTask(Protocol):
    kind: TaskKind

    def run(self, dataset: LocalDataset, config: RoundConfig) -> WireModel: ...


class FeatureInventoryTask:
    """Report which features exist locally (names only)."""

    kind = TaskKind.FEATURE_INVENTORY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureInventoryPayload:
        return FeatureInventoryPayload(feature_names=dataset.feature_names)


class FeatureSummaryTask:
    """Per-feature count, mean and unbiased variance over the local cohort.

    Assumes complete data (no missing values); refuses when the cohort is
    smaller than the policy threshold or a requested feature is unavailable.
    """

    kind = TaskKind.FEATURE_SUMMARY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureSummaryPayload:
        if not config.feature_names:
            raise TaskRefused(ReasonCode.MISSING_FEATURES)
        if dataset.n_participants < config.privacy.min_cohort_size:
            raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "local cohort below threshold")
        index = {name: i for i, name in enumerate(dataset.feature_names)}
        if any(name not in index for name in config.feature_names):
            raise TaskRefused(ReasonCode.MISSING_FEATURES)
        columns = dataset.features[:, [index[name] for name in config.feature_names]]
        return FeatureSummaryPayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            mean=tuple(np.mean(columns, axis=0).tolist()),
            variance=tuple(np.var(columns, axis=0, ddof=1).tolist()),
        )


def _columns(dataset: LocalDataset, config: RoundConfig) -> np.ndarray:
    """Columns of the requested (agreed) features, after the cohort-size rule."""
    if not config.feature_names:
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    if dataset.n_participants < config.privacy.min_cohort_size:
        raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "local cohort below threshold")
    index = {name: i for i, name in enumerate(dataset.feature_names)}
    if any(name not in index for name in config.feature_names):
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    return dataset.features[:, [index[name] for name in config.feature_names]]


class FeatureReliabilityTask:
    """Report the site's reliability summary for the agreed features.

    Refuses when the site has no reliability study: reliability is never
    reported without repeated or alternative observations behind it.
    """

    kind = TaskKind.FEATURE_RELIABILITY

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureReliabilityPayload:
        if dataset.reliability is None:
            raise TaskRefused(ReasonCode.RELIABILITY_UNAVAILABLE)
        column = config.params.get("reliability_column")
        table = dataset.reliability.values.get(str(column))
        if table is None or not config.feature_names:
            raise TaskRefused(ReasonCode.RELIABILITY_UNAVAILABLE)
        if dataset.reliability.n_subjects < config.privacy.min_cohort_size:
            raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "reliability study too small")
        values = []
        for name in config.feature_names:
            value = float(table.get(name, float("nan")))
            values.append(value if np.isfinite(value) else None)
        return FeatureReliabilityPayload(
            feature_names=config.feature_names,
            column=column,
            n_subjects=dataset.reliability.n_subjects,
            values=tuple(values),
        )


class FeatureRelevanceTask:
    """Per-feature Cox score U and information I on the site's own risk sets.

    Uses follow-up time and the event indicator: censored participants count
    while at risk and are never treated as non-converters.
    """

    kind = TaskKind.FEATURE_RELEVANCE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureRelevancePayload:
        from client.relevance import cox_score_components  # needs the science extra

        columns = _columns(dataset, config)
        n_events = int(dataset.event.sum())
        if n_events < config.privacy.min_events:
            raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few local events")
        score, information = cox_score_components(columns, dataset.time_months, dataset.event)
        return FeatureRelevancePayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            n_events=n_events,
            score=tuple(score.tolist()),
            information=tuple(information.tolist()),
        )


class FeatureCovarianceTask:
    """Within-site scatter matrix of the candidate features (no outcome used).

    With ``rank_transform`` each column is replaced by its within-site ranks
    (average ranks for ties), which makes the pooled result a within-site
    Spearman correlation.
    """

    kind = TaskKind.FEATURE_COVARIANCE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> FeatureCovariancePayload:
        columns = _columns(dataset, config)
        rank = bool(config.params.get("rank_transform", False))
        if rank:
            from scipy.stats import rankdata  # needs the science extra

            columns = rankdata(columns, axis=0)
        centred = columns - columns.mean(axis=0)
        scatter = centred.T @ centred
        upper = scatter[np.triu_indices(scatter.shape[0])]
        return FeatureCovariancePayload(
            feature_names=config.feature_names,
            n_samples=dataset.n_participants,
            rank_transformed=rank,
            scatter_upper=tuple(upper.tolist()),
        )


def _model_inputs(dataset: LocalDataset, config: RoundConfig) -> tuple[ModelState, np.ndarray]:
    """Standardised model features (global centre/scale from the round)."""
    if config.model is None:
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    columns = _columns(dataset, config)
    if columns.shape[1] != len(config.model.center):
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    return config.model, (columns - np.array(config.model.center)) / np.array(config.model.scale)


def _upper(matrix: np.ndarray) -> tuple[float, ...]:
    return tuple(matrix[np.triu_indices(matrix.shape[0])].tolist())


class ModelUpdateTask:
    """Local log-likelihood, gradient and information at the global parameters.

    ``cox``: the site's own risk sets (a site-stratified Cox model overall).
    ``horizon_logistic``: labels and censoring weights computed by the site from
    its own follow-up; participants censored before the horizon are never
    labelled non-converters.
    """

    kind = TaskKind.MODEL_UPDATE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> ModelUpdatePayload:
        from client.models import cox, logistic  # needs the science extra

        model, z = _model_inputs(dataset, config)
        beta = np.array(model.coefficients)
        if model.kind == "cox":
            n_events = int(dataset.event.sum())
            if n_events < config.privacy.min_events:
                raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few local events")
            loglik, gradient, information = cox.partial_likelihood(
                z, dataset.time_months, dataset.event, beta
            )
            n_samples = dataset.n_participants
        else:
            y, weights = logistic.horizon_labels(
                dataset.time_months, dataset.event, model.horizon_months, model.censoring
            )
            n_events, negatives = logistic.counts(y, weights)
            if min(n_events, negatives) < config.privacy.min_events:
                raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few known outcomes")
            n_samples = n_events + negatives
            if n_samples < config.privacy.min_cohort_size:
                raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "too few known outcomes")
            if model.intercept == "site_specific":
                loglik, gradient, information, _ = logistic.profile_logistic(
                    z, y, weights, beta
                )
            else:
                design = np.column_stack([np.ones(len(z)), z])
                loglik, gradient, information = logistic.weighted_logistic(
                    design, y, weights, beta
                )
        return ModelUpdatePayload(
            n_parameters=len(beta),
            n_samples=n_samples,
            n_events=n_events,
            loglik=loglik,
            gradient=tuple(np.asarray(gradient).tolist()),
            information_upper=_upper(np.asarray(information)),
        )


class BaselineHazardTask:
    """Breslow cumulative baseline hazard of the site at the requested horizons."""

    kind = TaskKind.BASELINE_HAZARD

    def run(self, dataset: LocalDataset, config: RoundConfig) -> BaselineHazardPayload:
        from client.models import cox

        model, z = _model_inputs(dataset, config)
        if model.kind != "cox" or not model.horizons_months:
            raise TaskRefused(ReasonCode.UNSUPPORTED_TASK)
        n_events = int(dataset.event.sum())
        if n_events < config.privacy.min_events:
            raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few local events")
        return BaselineHazardPayload(
            horizons_months=model.horizons_months,
            n_samples=dataset.n_participants,
            n_events=n_events,
            cumulative_hazard=cox.breslow_cumulative_hazard(
                z, dataset.time_months, dataset.event, np.array(model.coefficients),
                model.horizons_months,
            ),
        )


class SiteInterceptTask:
    """The site's own intercept at the final slopes (site-specific logistic)."""

    kind = TaskKind.SITE_INTERCEPT

    def run(self, dataset: LocalDataset, config: RoundConfig) -> SiteInterceptPayload:
        from client.models import logistic

        model, z = _model_inputs(dataset, config)
        if model.kind != "horizon_logistic" or model.intercept != "site_specific":
            raise TaskRefused(ReasonCode.UNSUPPORTED_TASK)
        y, weights = logistic.horizon_labels(
            dataset.time_months, dataset.event, model.horizon_months, model.censoring
        )
        positives, negatives = logistic.counts(y, weights)
        if min(positives, negatives) < config.privacy.min_events:
            raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few known outcomes")
        return SiteInterceptPayload(
            horizon_months=model.horizon_months,
            n_samples=positives + negatives,
            n_events=positives,
            intercept=logistic.profile_intercept(z, y, weights, np.array(model.coefficients)),
        )


# --------------------------------------------------------------------------- #
# Harmonization (ComBat)
# --------------------------------------------------------------------------- #
def _harmonization_inputs(dataset: LocalDataset, config: RoundConfig):
    from client.harmonize import columns

    state = config.harmonization
    if state is None or not config.feature_names:
        raise TaskRefused(ReasonCode.MISSING_FEATURES)
    if dataset.n_participants < config.privacy.min_cohort_size:
        raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "local cohort below threshold")
    try:
        y = columns(dataset, config.feature_names)
        x = columns(dataset, state.covariates)
    except KeyError:
        raise TaskRefused(ReasonCode.MISSING_FEATURES) from None
    return state, y, x


def _global_combat(state, p: int):
    from harmonization.combat import GlobalCombat

    c = len(state.covariates)
    return GlobalCombat(
        grand_mean=np.array(state.grand_mean),
        beta=np.array(state.beta).reshape(c, p),
        var_pooled=np.array(state.var_pooled),
    )


class HarmonizationStatisticsTask:
    """Sums for the ComBat regression, plus the site's cohort profile."""

    kind = TaskKind.HARMONIZATION_STATISTICS

    def run(self, dataset: LocalDataset, config: RoundConfig) -> HarmonizationStatisticsPayload:
        from evaluation.censoring import censoring_survival
        from harmonization.combat import site_sums

        state, y, x = _harmonization_inputs(dataset, config)
        sums = site_sums(y, x)
        times, values = censoring_survival(dataset.time_months, dataset.event)
        reached = np.flatnonzero(values <= 0.5)
        return HarmonizationStatisticsPayload(
            feature_names=config.feature_names,
            covariate_names=state.covariates,
            n_samples=sums.n,
            n_converted=int(dataset.event.sum()),
            # A median is one participant's own follow-up time: report it rounded to
            # whole months so that no individual time leaves the site.
            median_follow_up=float(round(times[reached[0]])) if reached.size else None,
            sum_x=tuple(sums.sum_x.tolist()),
            sum_xx_upper=_upper(sums.sum_xx) if sums.sum_xx.size else (),
            sum_y=tuple(sums.sum_y.tolist()),
            sum_yy=tuple(sums.sum_yy.tolist()),
            sum_xy=tuple(sums.sum_xy.ravel().tolist()),
        )


class HarmonizationVarianceTask:
    kind = TaskKind.HARMONIZATION_VARIANCE

    def run(self, dataset: LocalDataset, config: RoundConfig) -> HarmonizationVariancePayload:
        from harmonization.combat import residual_sum_of_squares

        state, y, x = _harmonization_inputs(dataset, config)
        beta = np.array(state.beta).reshape(len(state.covariates), y.shape[1])
        return HarmonizationVariancePayload(
            feature_names=config.feature_names,
            n_samples=len(y),
            residual_ss=tuple(residual_sum_of_squares(y, x, beta).tolist()),
        )


class HarmonizationApplyTask:
    """Estimate the site's own ComBat adjustment and install it locally."""

    kind = TaskKind.HARMONIZATION_APPLY

    def run(self, dataset: LocalDataset, config: RoundConfig):
        from client.harmonize import LocalStateUpdate, SiteHarmonizer
        from harmonization.combat import fit_site

        state, y, x = _harmonization_inputs(dataset, config)
        model = fit_site(y, x, _global_combat(state, y.shape[1]),
                         empirical_bayes=state.empirical_bayes, tolerance=state.eb_tolerance)
        adjusted = model.apply(y, x)
        return LocalStateUpdate(
            payload=HarmonizationApplyPayload(
                feature_names=config.feature_names,
                n_samples=len(y),
                sum_y=tuple(adjusted.sum(0).tolist()),
                sum_yy=tuple((adjusted**2).sum(0).tolist()),
            ),
            harmonizer=SiteHarmonizer(config.feature_names, state.covariates, model),
        )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class ModelEvaluationTask:
    """Evaluate a model on the site's evaluation partition (sums only).

    Receives the full local dataset: the ``site_baseline`` and ``local``
    variants estimate site quantities on the training partition and evaluate
    on the held-out one.
    """

    kind = TaskKind.MODEL_EVALUATION
    needs_full_dataset = True

    def run(self, dataset: LocalDataset, config: RoundConfig) -> ModelEvaluationPayload:
        from client.models import cox, logistic
        from client.models.prediction import predict_conversion
        from client.splits import partition
        from evaluation.metrics import horizon_statistics, merge_bins
        from shared.models import CoxModel
        from shared.newton import NewtonSettings
        from shared.protocol import PartitionSpec
        from shared.training import CoxSpec

        request = config.evaluation
        if request is None:
            raise TaskRefused(ReasonCode.UNSUPPORTED_TASK)
        split = config.partition
        if request.evaluation_partition == "test":
            if split is None:
                raise TaskRefused(ReasonCode.UNSUPPORTED_TASK)
            evaluated = partition(dataset, PartitionSpec(
                name="test", test_fraction=split.test_fraction, seed=split.seed))
            train = partition(dataset, PartitionSpec(
                name="train", test_fraction=split.test_fraction, seed=split.seed))
        else:
            evaluated, train = dataset, dataset
        n_events = int(evaluated.event.sum())
        if evaluated.n_participants < config.privacy.min_cohort_size:
            raise DisclosureViolation(ReasonCode.COHORT_TOO_SMALL, "evaluation set too small")
        if n_events < config.privacy.min_events:
            raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "too few events to evaluate")

        model = request.model
        names = model.feature_names
        try:
            x_train = train.features[:, [train.feature_names.index(n) for n in names]]
        except ValueError:
            raise TaskRefused(ReasonCode.MISSING_FEATURES) from None
        z_train = (x_train - np.array(model.center)) / np.array(model.scale)
        if request.variant == "site_baseline":
            if isinstance(model, CoxModel):
                model = model.model_copy(update={"baseline_cumulative_hazard":
                    cox.breslow_cumulative_hazard(z_train, train.time_months, train.event,
                                                  np.array(model.coefficients),
                                                  model.horizons_months)})
            else:
                fits = []
                for fit in model.horizons:
                    y, w = logistic.horizon_labels(train.time_months, train.event,
                                                   fit.horizon_months, model.censoring)
                    fits.append(fit.model_copy(update={"intercept": logistic.profile_intercept(
                        z_train, y, w, np.array(fit.coefficients))}))
                model = model.model_copy(update={"horizons": tuple(fits)})
        elif request.variant == "local":
            spec = request.model_spec
            o = spec.optimizer
            settings = NewtonSettings(o.ridge, o.max_iterations, o.tolerance)
            horizons = (model.horizons_months if isinstance(model, CoxModel)
                        else tuple(f.horizon_months for f in model.horizons))
            try:
                if isinstance(spec, CoxSpec):
                    model = cox.fit_cox(names, x_train, train.time_months, train.event,
                                        settings, horizons, min_events=config.privacy.min_events)
                else:
                    model = logistic.fit_horizon_logistic(
                        names, x_train, train.time_months, train.event, horizons,
                        spec.censoring, settings, min_events=config.privacy.min_events)
            except ValueError:
                raise DisclosureViolation(ReasonCode.TOO_FEW_EVENTS, "local fit failed") from None

        horizons, probabilities = predict_conversion(model, evaluated.feature_names,
                                                     evaluated.features)
        grid = np.array(request.grid)
        entries = []
        for j, horizon in enumerate(horizons):
            p = probabilities[:, j]
            if not np.isfinite(p).all():
                entries.append(HorizonEvaluation(horizon_months=horizon, available=False))
                continue
            stats = merge_bins(
                horizon_statistics(evaluated.time_months, evaluated.event, p, horizon, grid),
                config.privacy.min_cohort_size,
            )
            entries.append(HorizonEvaluation(
                horizon_months=horizon, available=True, n=stats.n, n_events=stats.n_events,
                weight_sum=stats.weight_sum, weighted_events=stats.weighted_events,
                brier_sum=stats.brier_sum, auc_num=stats.auc_num, auc_den=stats.auc_den,
                c_num=stats.c_num, c_den=stats.c_den, edges=stats.edges,
                bin_n=tuple(stats.bin_n.tolist()), bin_w=tuple(stats.bin_w.tolist()),
                bin_wy=tuple(stats.bin_wy.tolist()), bin_wp=tuple(stats.bin_wp.tolist()),
            ))
        return ModelEvaluationPayload(
            n_samples=evaluated.n_participants, n_events=n_events, horizons=tuple(entries)
        )


def default_tasks() -> dict[TaskKind, LocalTask]:
    tasks: list[LocalTask] = [
        FeatureInventoryTask(),
        FeatureSummaryTask(),
        FeatureReliabilityTask(),
        FeatureRelevanceTask(),
        FeatureCovarianceTask(),
        ModelUpdateTask(),
        BaselineHazardTask(),
        SiteInterceptTask(),
        HarmonizationStatisticsTask(),
        HarmonizationVarianceTask(),
        HarmonizationApplyTask(),
        ModelEvaluationTask(),
    ]
    return {task.kind: task for task in tasks}
