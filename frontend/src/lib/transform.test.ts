import { fixtures } from "../test/fixtures";
import {
  coefficients, decisionCurve, externalAuc, jaccardMatrix, metricByHorizon, metricRows,
} from "./transform";

const rows = fixtures.evaluation.evaluations;

describe("evaluation transforms", () => {
  it("flattens pooled metrics into one row per model and horizon", () => {
    const internal = rows.filter((r) => r.scope === "internal" && r.variant === "released");
    const table = metricRows(internal);
    expect(table).toHaveLength(internal.length * 3);
    const first = table[0];
    expect(first.horizon).toBe(12);
    expect(first.auc).toBeCloseTo(internal[0].pooled[0].auc!, 12);
  });

  it("builds one series per model across horizons", () => {
    const series = metricByHorizon(rows, "internal", "released", "auc");
    expect(series.map((s) => s.name).sort()).toEqual(["cox_reference", "cox_selected"]);
    expect(series[0].x).toEqual([12, 24, 36]);
  });

  it("arranges external validation by held-out site", () => {
    const series = externalAuc(rows, 24);
    expect(series[0].x).toEqual(["client-01", "client-02", "client-03"]);
    const selected = series.find((s) => s.name === "cox_selected")!;
    const row = rows.find((r) => r.scope === "external:client-01" && r.model_name === "cox_selected")!;
    expect(selected.y[0]).toBeCloseTo(row.pooled[1].auc!, 12);
  });

  it("keeps only evaluable thresholds of the decision curve", () => {
    const points = decisionCurve(rows[0].pooled[1]);
    expect(points.length).toBeGreaterThan(0);
    expect(points.every((p) => p.model !== null)).toBe(true);
  });
});

describe("selection and model transforms", () => {
  it("makes a symmetric Jaccard matrix with a unit diagonal", () => {
    const { names, z } = jaccardMatrix(fixtures.selection);
    expect(names).toHaveLength(fixtures.selection.strategies.length);
    names.forEach((_, i) => {
      expect(z[i][i]).toBe(1);
      names.forEach((__, j) => expect(z[i][j]).toBe(z[j][i]));
    });
  });

  it("derives Wald intervals from coefficients and standard errors", () => {
    const rowsOut = coefficients({
      kind: "cox", feature_names: ["a", "b"], coefficients: [0.5, -0.2],
      standard_errors: [0.1, 0.05],
    });
    expect(rowsOut[0]).toMatchObject({ feature: "a", horizon: null });
    expect(rowsOut[0].lower).toBeCloseTo(0.5 - 0.196, 12);
    expect(rowsOut[1].upper).toBeCloseTo(-0.2 + 0.098, 12);
    expect(coefficients(null)).toEqual([]);
  });
});
