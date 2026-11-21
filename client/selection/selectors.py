"""Local feature selectors.

All selectors share the same pre-filter (features that are constant or
non-finite in the cohort are excluded) and return features ranked from most to
least preferred.

Hypotheses and limitations (all selectors)
------------------------------------------
* Selection uses the outcome. Selecting on a cohort and evaluating a model on
  the same cohort is optimistically biased; call these inside the training
  split of any later evaluation.
* ``n_features`` is an input, not something estimated from the data.
* With few events the ranking is unstable; check the bootstrap stability
  reported by :mod:`client.selection.study` before interpreting a selection.

Selector-specific notes
-----------------------
``component``
    Hard thresholds on reliability and correlation. Greedy pruning keeps the
    highest-ranked feature of each correlated group, so the result depends on
    small differences in relevance between correlated features.
``correlation_filter``
    Same pruning without the reliability step; the classical filter baseline.
``lasso_cox``
    Multivariable, but among correlated features the LASSO picks one almost
    arbitrarily. The penalty is chosen as the strongest one on the path that
    keeps at most ``n_features`` features - not by cross-validation - so the
    comparison with the filters is at equal size, not at optimal size.
``graph_representatives``
    Groups are connected components (single linkage): chains of correlated
    features merge into large groups, so far fewer than ``n_features``
    features may be returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from client.redundancy import correlation_matrix, redundancy_groups
from client.relevance import RelevanceUnavailable, get_relevance
from client.selection.config import (
    ComponentSelector,
    CorrelationFilterSelector,
    GraphSelector,
    LassoCoxSelector,
    SelectorSpec,
)


class SelectionError(Exception):
    """The selector cannot run on the inputs provided."""


@dataclass(frozen=True, slots=True)
class SelectionInputs:
    feature_names: tuple[str, ...]
    features: np.ndarray  # (n, p)
    time: np.ndarray
    event: np.ndarray
    reliability: dict[str, dict[str, float]] | None = None
    """Reliability column -> (feature -> value), or None when not estimated."""


@dataclass(frozen=True, slots=True)
class SelectionResult:
    name: str
    method: str
    selected: tuple[str, ...]
    """Ranked, most preferred first."""
    scores: dict[str, float] = field(default_factory=dict)
    """Ranking score of each selected feature."""
    excluded: dict[str, str] = field(default_factory=dict)
    """Feature -> reason, for features removed before ranking."""


def _usable(inputs: SelectionInputs, excluded: dict[str, str]) -> np.ndarray:
    x = inputs.features
    ok = np.isfinite(x).all(axis=0)
    constant = ok & (np.ptp(np.where(np.isfinite(x), x, 0.0), axis=0) == 0)
    for index in np.flatnonzero(~ok):
        excluded[inputs.feature_names[index]] = "non_finite"
    for index in np.flatnonzero(constant):
        excluded[inputs.feature_names[index]] = "constant"
    return ok & ~constant


def _relevance(inputs: SelectionInputs, method: str, keep: np.ndarray, min_events: int):
    scores = np.full(len(inputs.feature_names), np.nan)
    try:
        result = get_relevance(method)(
            inputs.features[:, keep], inputs.time, inputs.event, min_events=min_events
        )
    except RelevanceUnavailable as exc:
        raise SelectionError(str(exc)) from exc
    scores[keep] = result.score
    return scores


def _ranked(scores: np.ndarray, keep: np.ndarray) -> list[int]:
    """Indices of kept features with a defined score, best first (stable ties)."""
    candidates = np.flatnonzero(keep & np.isfinite(scores))
    return list(candidates[np.argsort(-scores[candidates], kind="stable")])


def _prune(
    inputs: SelectionInputs, order: list[int], correlation: str | None, limit: float, k: int
) -> list[int]:
    if correlation is None:
        return order[:k]
    corr = correlation_matrix(inputs.features[:, order], correlation)
    chosen: list[int] = []
    for position in range(len(order)):
        if all(corr[position, other] < limit for other in chosen):
            chosen.append(position)
            if len(chosen) == k:
                break
    return [order[position] for position in chosen]


def _result(spec: SelectorSpec, inputs: SelectionInputs, chosen, scores, excluded):
    names = inputs.feature_names
    return SelectionResult(
        name=spec.name,
        method=spec.method,
        selected=tuple(names[i] for i in chosen),
        scores={names[i]: float(scores[i]) for i in chosen},
        excluded=excluded,
    )


def _component(spec: ComponentSelector, inputs: SelectionInputs, min_events: int):
    excluded: dict[str, str] = {}
    keep = _usable(inputs, excluded)
    if spec.reliability is not None:
        table = (inputs.reliability or {}).get(spec.reliability.column)
        if table is None:
            raise SelectionError(
                f"selector {spec.name!r} filters on reliability ({spec.reliability.column}) but "
                "no reliability estimates were provided; run a reliability study or set "
                "'reliability: null' explicitly"
            )
        for index in np.flatnonzero(keep):
            value = table.get(inputs.feature_names[index], float("nan"))
            if not np.isfinite(value):
                excluded[inputs.feature_names[index]] = "reliability_unknown"
                keep[index] = False
            elif value < spec.reliability.min_value:
                excluded[inputs.feature_names[index]] = "below_reliability_threshold"
                keep[index] = False
    if not keep.any():
        return _result(spec, inputs, [], np.array([]), excluded)
    scores = _relevance(inputs, spec.relevance, keep, min_events)
    redundancy = spec.redundancy
    chosen = _prune(
        inputs,
        _ranked(scores, keep),
        redundancy.correlation if redundancy else None,
        redundancy.max_abs_correlation if redundancy else 1.0,
        spec.n_features,
    )
    return _result(spec, inputs, chosen, scores, excluded)


def _correlation_filter(spec: CorrelationFilterSelector, inputs: SelectionInputs, min_events: int):
    excluded: dict[str, str] = {}
    keep = _usable(inputs, excluded)
    scores = _relevance(inputs, spec.relevance, keep, min_events)
    chosen = _prune(
        inputs, _ranked(scores, keep), spec.correlation, spec.max_abs_correlation, spec.n_features
    )
    return _result(spec, inputs, chosen, scores, excluded)


def _graph(spec: GraphSelector, inputs: SelectionInputs, min_events: int):
    excluded: dict[str, str] = {}
    keep = _usable(inputs, excluded)
    scores = _relevance(inputs, spec.relevance, keep, min_events)
    order = _ranked(scores, keep)
    if not order:
        return _result(spec, inputs, [], scores, excluded)
    groups = redundancy_groups(
        correlation_matrix(inputs.features[:, order], spec.correlation), spec.edge_threshold
    )
    representatives, seen = [], set()
    for position, group in enumerate(groups):  # `order` is already best-first
        if group not in seen:
            seen.add(group)
            representatives.append(order[position])
    return _result(spec, inputs, representatives[: spec.n_features], scores, excluded)


def _lasso_cox(spec: LassoCoxSelector, inputs: SelectionInputs, min_events: int):
    excluded: dict[str, str] = {}
    keep = _usable(inputs, excluded)
    if int(np.sum(inputs.event)) < min_events:
        raise SelectionError(
            f"{int(np.sum(inputs.event))} events observed, at least {min_events} required"
        )
    columns = np.flatnonzero(keep)
    x = inputs.features[:, columns]
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0, n_alphas=spec.n_alphas, alpha_min_ratio=spec.alpha_min_ratio,
        max_iter=100_000,
    )
    try:
        model.fit(x, Surv.from_arrays(event=inputs.event.astype(bool), time=inputs.time))
    except (ArithmeticError, ValueError) as exc:
        raise SelectionError(f"penalised Cox model failed: {exc}") from exc
    non_zero = (model.coef_ != 0).sum(axis=0)
    admissible = np.flatnonzero(non_zero <= spec.n_features)
    coefficients = model.coef_[:, admissible[-1]] if admissible.size else np.zeros(len(columns))
    magnitude = np.zeros(len(inputs.feature_names))
    magnitude[columns] = np.abs(coefficients)
    chosen = [i for i in np.argsort(-magnitude, kind="stable") if magnitude[i] > 0]
    return _result(spec, inputs, chosen, magnitude, excluded)


def run_selector(spec: SelectorSpec, inputs: SelectionInputs, *, min_events: int):
    """Run one configured selector."""
    if isinstance(spec, ComponentSelector):
        return _component(spec, inputs, min_events)
    if isinstance(spec, CorrelationFilterSelector):
        return _correlation_filter(spec, inputs, min_events)
    if isinstance(spec, GraphSelector):
        return _graph(spec, inputs, min_events)
    if isinstance(spec, LassoCoxSelector):
        return _lasso_cox(spec, inputs, min_events)
    raise SelectionError(f"unsupported selector {spec!r}")  # pragma: no cover
