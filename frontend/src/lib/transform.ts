// Pure reshaping of API results into tables and chart series (no statistics).
import type { EvaluationRow, FeatureSelection, MetricsSummary, Results } from "../api/types";

export interface CalibrationBin {
  lower: number;
  upper: number;
  n: number;
  observed: number | null;
  predicted: number | null;
}

export interface DecisionPoint {
  threshold: number;
  model: number | null;
  treat_all: number | null;
  treat_none: number | null;
}

export interface MetricRow {
  model: string;
  variant: string;
  scope: string;
  horizon: number;
  available: boolean;
  n: number;
  events: number;
  auc: number | null;
  c: number | null;
  brier: number | null;
  scaledBrier: number | null;
  oe: number | null;
  calibrationError: number | null;
}

export function metricRows(rows: EvaluationRow[]): MetricRow[] {
  return rows.flatMap((row) =>
    row.pooled.map((h) => ({
      model: row.model_name,
      variant: row.variant,
      scope: row.scope,
      horizon: h.horizon_months,
      available: h.available,
      n: h.n ?? 0,
      events: h.n_events ?? 0,
      auc: h.auc ?? null,
      c: h.c_index ?? null,
      brier: h.brier ?? null,
      scaledBrier: h.scaled_brier ?? null,
      oe: h.o_e ?? null,
      calibrationError: h.calibration_error ?? null,
    })),
  );
}

export interface Series {
  name: string;
  x: (number | string)[];
  y: (number | null)[];
}

/** One series per model: a metric across horizons, for one scope and variant. */
export function metricByHorizon(
  rows: EvaluationRow[],
  scope: string,
  variant: string,
  metric: keyof Pick<MetricsSummary, "auc" | "c_index" | "brier" | "scaled_brier" | "o_e">,
): Series[] {
  return rows
    .filter((r) => r.scope === scope && r.variant === variant)
    .map((r) => ({
      name: r.model_name,
      x: r.pooled.map((h) => h.horizon_months),
      y: r.pooled.map((h) => (h.available ? (h[metric] ?? null) : null)),
    }));
}

/** External validation: one series per model, AUC at ``horizon`` by held-out site. */
export function externalAuc(rows: EvaluationRow[], horizon: number): Series[] {
  const external = rows.filter((r) => r.scope.startsWith("external:"));
  const sites = [...new Set(external.map((r) => r.scope.slice(9)))].sort();
  const models = [...new Set(external.map((r) => r.model_name))];
  return models.map((model) => ({
    name: model,
    x: sites,
    y: sites.map((site) => {
      const row = external.find((r) => r.model_name === model && r.scope === `external:${site}`);
      const h = row?.pooled.find((p) => p.horizon_months === horizon);
      return h?.available ? (h.auc ?? null) : null;
    }),
  }));
}

export function calibrationBins(metrics: MetricsSummary): CalibrationBin[] {
  return (metrics.calibration ?? []) as unknown as CalibrationBin[];
}

export function decisionCurve(metrics: MetricsSummary): DecisionPoint[] {
  return ((metrics.decision_curve ?? []) as unknown as DecisionPoint[]).filter(
    (p) => p.model !== null,
  );
}

/** Per-feature between-site share of variance, before and after harmonization. */
export function siteEffects(results: Results): {
  features: string[];
  before: (number | null)[];
  after: (number | null)[] | null;
} | null {
  const aggregates = results.rounds
    .map((r) => r.aggregate as Record<string, unknown> | null)
    .filter((a): a is Record<string, unknown> => !!a && a.kind === "site_effect");
  const before = aggregates.find((a) => a.stage === "before");
  if (!before) return null;
  const after = aggregates.find((a) => a.stage === "after");
  return {
    features: before.feature_names as string[],
    before: before.eta_squared as (number | null)[],
    after: after ? (after.eta_squared as (number | null)[]) : null,
  };
}

/** Pooled relevance (z) and disagreement (I^2) of every feature, from the relevance round. */
export function relevance(results: Results): {
  features: string[];
  z: (number | null)[];
  iSquared: (number | null)[];
} | null {
  const aggregate = results.rounds
    .map((r) => r.aggregate as Record<string, unknown> | null)
    .find((a) => !!a && a.kind === "feature_relevance");
  if (!aggregate) return null;
  return {
    features: aggregate.feature_names as string[],
    z: aggregate.stratified_z as (number | null)[],
    iSquared: aggregate.i_squared as (number | null)[],
  };
}

/** Symmetric Jaccard matrix between strategies (diagonal 1). */
export function jaccardMatrix(selection: FeatureSelection): { names: string[]; z: number[][] } {
  const names = selection.strategies.map((s) => s.name);
  const pairs = selection.jaccard_between_strategies;
  const z = names.map((a) =>
    names.map((b) => (a === b ? 1 : (pairs[`${a}|${b}`] ?? pairs[`${b}|${a}`] ?? NaN))),
  );
  return { names, z };
}

export interface CoefficientRow {
  horizon: number | null;
  feature: string;
  estimate: number;
  lower: number;
  upper: number;
}

/** Coefficients with Wald 95% intervals from a released model's parameters. */
export function coefficients(parameters: Record<string, unknown> | null | undefined): CoefficientRow[] {
  if (!parameters) return [];
  const features = parameters.feature_names as string[];
  const fits: { horizon: number | null; coefficients: number[]; errors: number[] }[] =
    parameters.kind === "cox"
      ? [{ horizon: null, coefficients: parameters.coefficients as number[],
           errors: parameters.standard_errors as number[] }]
      : (parameters.horizons as Record<string, unknown>[]).map((h) => ({
          horizon: h.horizon_months as number,
          coefficients: h.coefficients as number[],
          errors: h.standard_errors as number[],
        }));
  return fits.flatMap((fit) =>
    features.map((feature, j) => ({
      horizon: fit.horizon,
      feature,
      estimate: fit.coefficients[j],
      lower: fit.coefficients[j] - 1.96 * fit.errors[j],
      upper: fit.coefficients[j] + 1.96 * fit.errors[j],
    })),
  );
}

/** Penalised log-likelihood histories (one per fit) of a released model. */
export function convergence(parameters: Record<string, unknown> | null | undefined): Series[] {
  if (!parameters) return [];
  const fits =
    parameters.kind === "cox"
      ? [{ name: "Cox", fit: parameters.fit as Record<string, unknown> }]
      : (parameters.horizons as Record<string, unknown>[]).map((h) => ({
          name: `${h.horizon_months} meses`,
          fit: h.fit as Record<string, unknown>,
        }));
  return fits.map(({ name, fit }) => {
    const history = fit.penalized_loglik_history as number[];
    return { name, x: history.map((_, i) => i + 1), y: history };
  });
}
