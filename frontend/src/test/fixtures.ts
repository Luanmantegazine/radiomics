import type {
  Evaluation, ExperimentDetail, ExperimentSummary, FeatureSelection, Template,
} from "../api/types";
import detail from "./fixtures/detail.json";
import evaluation from "./fixtures/evaluation.json";
import selection from "./fixtures/selection.json";
import summary from "./fixtures/summary.json";
import templates from "./fixtures/templates.json";

// Captured from the real API (stage 6 synthetic experiment run on PostgreSQL).
export const fixtures = {
  summary: summary as unknown as ExperimentSummary,
  detail: detail as unknown as ExperimentDetail,
  evaluation: evaluation as unknown as Evaluation,
  selection: selection as unknown as FeatureSelection,
  templates: templates as unknown as Template[],
};
