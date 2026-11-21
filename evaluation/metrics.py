"""Censoring-aware performance metrics at a horizon, from additive statistics.

Every metric is computed from sums (:class:`HorizonStatistics`). One site
computes its sums from its own participants; pooling sites is adding the sums.
The pooled metric is then the *site-stratified* version:

* discrimination (AUC, C) counts only pairs of participants **of the same
  site** - comparing a participant with one from another site would need
  individual predictions to leave a site;
* Brier score, calibration and net benefit pool exactly;
* censoring weights use each site's own Kaplan-Meier censoring distribution,
  estimated on the evaluated participants.

Definitions at horizon tau (y = converted by tau, w = IPCW weight, p = predicted
probability of conversion by tau):

``brier``          sum w (y - p)^2 / sum w                       (Graf et al. 1999)
``scaled_brier``   1 - brier / (pi (1 - pi)), pi = sum w y / sum w  (IPA)
``auc``            cumulative/dynamic AUC: cases T <= tau with weight 1/G(T-),
                   controls T > tau with weight 1/G(tau); ties count 1/2
                   (Uno et al. 2007; Hung & Chiang 2010)
``c_index``        Uno's C truncated at tau, pairs T_i < T_j, event i, T_i < tau,
                   weight 1/G(T_i-)^2, risk = p (Uno et al. 2011)
``o_e``            sum w y / sum w p (calibration in the large)
calibration bins   per bin of p: sum w, sum w y, sum w p
``net_benefit(t)`` (sum_{p>=t} w y - sum_{p>=t} w (1-y) t/(1-t)) / sum w (Vickers 2006)

Hypotheses and limitations
--------------------------
* Censoring independent of the outcome and of the predictors (G does not
  depend on covariates); estimated on the evaluated set.
* Pooled discrimination ignores between-site pairs; it answers "how well does
  the model rank participants within a site". It is not the AUC of the pooled
  population when sites differ in risk.
* No confidence intervals here (they need resampling of individuals, which is
  site-local; see the documentation).
* Calibration uses fixed bins; bins too small for disclosure are merged
  (:func:`merge_bins`), which changes the curve's resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from evaluation.censoring import _evaluate, censoring_survival, horizon_labels


@dataclass
class HorizonStatistics:
    """Additive sufficient statistics of one model at one horizon."""

    horizon: float
    n: int = 0
    n_events: int = 0
    weight_sum: float = 0.0
    weighted_events: float = 0.0
    brier_sum: float = 0.0
    auc_num: float = 0.0
    auc_den: float = 0.0
    c_num: float = 0.0
    c_den: float = 0.0
    edges: tuple[float, ...] = ()
    """Bin edges (a subset of the global grid); bins are [e_k, e_k+1), last closed."""
    bin_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bin_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bin_wy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bin_wp: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _pair_sums(case_score, case_w, control_score, control_w) -> tuple[float, float]:
    """sum_i sum_j w_i w_j [I(s_i > s_j) + I(s_i = s_j)/2] and sum_i sum_j w_i w_j."""
    if len(case_score) == 0 or len(control_score) == 0:
        return 0.0, 0.0
    order = np.argsort(control_score, kind="stable")
    sorted_scores = control_score[order]
    cumulative = np.r_[0.0, np.cumsum(control_w[order])]
    below = cumulative[np.searchsorted(sorted_scores, case_score, side="left")]
    up_to = cumulative[np.searchsorted(sorted_scores, case_score, side="right")]
    numerator = float(np.sum(case_w * (below + 0.5 * (up_to - below))))
    return numerator, float(case_w.sum() * control_w.sum())


def bin_index(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of each prediction for [e_k, e_k+1) bins, last bin closed."""
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)


def horizon_statistics(
    time: np.ndarray, event: np.ndarray, predicted: np.ndarray, horizon: float,
    grid: np.ndarray,
) -> HorizonStatistics:
    """Sufficient statistics on one set of participants (e.g. one site's test set)."""
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    p = np.asarray(predicted, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("predicted probabilities must be finite and within [0, 1]")
    y, w = horizon_labels(time, event, horizon, "ipcw")
    stats = HorizonStatistics(
        horizon=horizon, n=len(time), n_events=int(((w > 0) & (y == 1)).sum())
    )
    stats.weight_sum = float(w.sum())
    stats.weighted_events = float((w * y).sum())
    stats.brier_sum = float((w * (y - p) ** 2).sum())
    cases, controls = (w > 0) & (y == 1), (w > 0) & (y == 0)
    stats.auc_num, stats.auc_den = _pair_sums(p[cases], w[cases], p[controls], w[controls])
    # Uno's C: for each event i before tau, compare with everyone still event-free after T_i.
    times, values = censoring_survival(time, event)
    g = _evaluate(times, values, time, left=True)
    for i in np.flatnonzero(event & (time < horizon)):
        later = time > time[i]
        if not later.any() or g[i] <= 0:
            continue
        weight = 1.0 / g[i] ** 2
        numerator, count = _pair_sums(
            p[i : i + 1], np.array([weight]), p[later], np.ones(int(later.sum()))
        )
        stats.c_num += numerator
        stats.c_den += count
    edges = np.asarray(grid, dtype=float)
    index = bin_index(p, edges)
    k = len(edges) - 1
    stats.edges = tuple(edges.tolist())
    stats.bin_n = np.bincount(index, minlength=k).astype(float)
    stats.bin_w = np.bincount(index, weights=w, minlength=k)
    stats.bin_wy = np.bincount(index, weights=w * y, minlength=k)
    stats.bin_wp = np.bincount(index, weights=w * p, minlength=k)
    return stats


def merge_bins(stats: HorizonStatistics, min_count: int) -> HorizonStatistics:
    """Merge adjacent bins until each holds 0 or at least ``min_count`` participants.

    Disclosure control for binned outputs: a bin with one participant would
    reveal that participant's predicted risk range and outcome.
    """
    edges = list(stats.edges)
    counts = np.asarray(stats.bin_n, dtype=float)
    keep = [0]
    running = 0.0
    for k, count in enumerate(counts):
        running += count
        if running == 0 or running >= min_count:
            keep.append(k + 1)
            running = 0.0
    if keep[-1] != len(edges) - 1:
        keep.append(len(edges) - 1)
    # A small leftover at the end is merged backwards until it joins a large bin
    # (merging it with empty bins alone would leave it small).
    while len(keep) > 2:
        last = counts[keep[-2]:keep[-1]].sum()
        if last == 0 or last >= min_count:
            break
        del keep[-2]

    def regroup(values: np.ndarray) -> np.ndarray:
        return np.array([values[a:b].sum() for a, b in zip(keep[:-1], keep[1:], strict=True)])

    out = HorizonStatistics(**{**stats.__dict__})
    out.edges = tuple(edges[i] for i in keep)
    out.bin_n, out.bin_w = regroup(stats.bin_n), regroup(stats.bin_w)
    out.bin_wy, out.bin_wp = regroup(stats.bin_wy), regroup(stats.bin_wp)
    return out


def _restrict(stats: HorizonStatistics, edges: tuple[float, ...]) -> HorizonStatistics:
    """Re-bin onto a coarser edge set (``edges`` must be a subset of ``stats.edges``)."""
    position = {edge: i for i, edge in enumerate(stats.edges)}
    cuts = [position[e] for e in edges]
    out = HorizonStatistics(**{**stats.__dict__})
    out.edges = edges
    for name in ("bin_n", "bin_w", "bin_wy", "bin_wp"):
        values = getattr(stats, name)
        setattr(out, name, np.array([values[a:b].sum() for a, b in zip(cuts[:-1], cuts[1:],
                                                                         strict=True)]))
    return out


def pool(parts: list[HorizonStatistics]) -> HorizonStatistics:
    """Add statistics of several sites; bins are pooled on the edges common to all."""
    if not parts:
        raise ValueError("nothing to pool")
    common = tuple(sorted(set.intersection(*(set(p.edges) for p in parts))))
    parts = [_restrict(p, common) for p in parts]
    total = HorizonStatistics(horizon=parts[0].horizon, edges=common)
    for name in ("n", "n_events", "weight_sum", "weighted_events", "brier_sum", "auc_num",
                 "auc_den", "c_num", "c_den"):
        setattr(total, name, sum(getattr(p, name) for p in parts))
    for name in ("bin_n", "bin_w", "bin_wy", "bin_wp"):
        setattr(total, name, np.sum([getattr(p, name) for p in parts], axis=0))
    return total


def _ratio(a: float, b: float) -> float | None:
    return float(a / b) if b > 0 else None


def summarize(stats: HorizonStatistics, thresholds: tuple[float, ...]) -> dict:
    """Metrics from (possibly pooled) statistics."""
    incidence = _ratio(stats.weighted_events, stats.weight_sum)
    brier = _ratio(stats.brier_sum, stats.weight_sum)
    scaled = None
    if brier is not None and incidence is not None and 0 < incidence < 1:
        scaled = 1 - brier / (incidence * (1 - incidence))
    predicted_total = float(stats.bin_wp.sum())
    calibration = [
        {"lower": stats.edges[k], "upper": stats.edges[k + 1], "n": int(stats.bin_n[k]),
         "observed": _ratio(stats.bin_wy[k], stats.bin_w[k]),
         "predicted": _ratio(stats.bin_wp[k], stats.bin_w[k])}
        for k in range(len(stats.bin_n))
    ]
    usable = [b for b in calibration if b["observed"] is not None]
    ici = (
        float(np.average([abs(b["observed"] - b["predicted"]) for b in usable],
                         weights=[stats.bin_w[calibration.index(b)] for b in usable]))
        if usable else None
    )
    return {
        "horizon_months": stats.horizon,
        "n": int(stats.n),
        "n_events": int(stats.n_events),
        "observed_incidence": incidence,
        "brier": brier,
        "scaled_brier": scaled,
        "auc": _ratio(stats.auc_num, stats.auc_den),
        "c_index": _ratio(stats.c_num, stats.c_den),
        "o_e": _ratio(stats.weighted_events, predicted_total),
        "calibration_error": ici,
        "calibration": calibration,
        "decision_curve": decision_curve(stats, thresholds),
    }


def decision_curve(stats: HorizonStatistics, thresholds: tuple[float, ...]) -> list[dict]:
    """Net benefit of the model, of treating everyone and of treating no one.

    Only thresholds that are bin edges can be evaluated exactly; others are
    reported as unavailable (None).
    """
    total = stats.weight_sum
    incidence = stats.weighted_events / total if total > 0 else None
    out = []
    for t in thresholds:
        if t not in stats.edges or total <= 0 or not 0 < t < 1:
            out.append({"threshold": t, "model": None, "treat_all": None, "treat_none": 0.0})
            continue
        start = stats.edges.index(t)
        tp = stats.bin_wy[start:].sum()
        fp = stats.bin_w[start:].sum() - tp
        odds = t / (1 - t)
        out.append({
            "threshold": t,
            "model": float((tp - fp * odds) / total),
            "treat_all": float(incidence - (1 - incidence) * odds),
            "treat_none": 0.0,
        })
    return out
