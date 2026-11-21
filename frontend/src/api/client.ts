import axios, { type InternalAxiosRequestConfig } from "axios";

import type {
  Client,
  CostSummary,
  Evaluation,
  Experiment,
  ExperimentDetail,
  ExperimentSummary,
  FeatureSelection,
  Harmonization,
  Model,
  ModelDetail,
  Results,
  Round,
  RoundDetail,
  SelectedFeature,
  Template,
  Validation,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/api";
const TOKEN_KEY = "fedrad.apiToken";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setToken(token: string): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: the token lasts for this page only */
  }
}

/** Adds the optional bearer token configured in the dashboard. */
export function withToken(config: InternalAxiosRequestConfig): InternalAxiosRequestConfig {
  const token = getToken();
  if (token) config.headers.set("Authorization", `Bearer ${token}`);
  return config;
}

export const http = axios.create({ baseURL: API_BASE });
http.interceptors.request.use(withToken);

/** Human-readable message for any API error. */
export function errorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (error.response?.status === 401) return "Token de acesso ausente ou inválido.";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((d) => d.msg ?? String(d)).join("; ");
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

const get = async <T>(url: string, params?: Record<string, unknown>) =>
  (await http.get<T>(url, { params })).data;

export const api = {
  experiments: (params?: { status?: string }) => get<Experiment[]>("/experiments", params),
  experiment: (id: string) => get<ExperimentDetail>(`/experiments/${id}`),
  summary: (id: string) => get<ExperimentSummary>(`/experiments/${id}/summary`),
  templates: () => get<Template[]>("/experiments/templates"),
  validate: async (config: unknown) =>
    (await http.post<Validation>("/experiments/validate", config)).data,
  create: async (config: unknown) =>
    (await http.post<ExperimentDetail>("/experiments", config)).data,
  run: async (id: string) => (await http.post<Experiment>(`/experiments/${id}/run`)).data,
  clients: (id: string) => get<Client[]>("/clients", { experiment_id: id }),
  rounds: (id: string) => get<Round[]>("/federated-rounds", { experiment_id: id }),
  round: (roundId: string) => get<RoundDetail>(`/federated-rounds/${roundId}`),
  results: (id: string) => get<Results>(`/results/${id}`),
  selection: (id: string) => get<FeatureSelection>(`/results/${id}/feature-selection`),
  selectedFeatures: (id: string) => get<SelectedFeature[]>(`/results/${id}/selected-features`),
  harmonization: (id: string) => get<Harmonization>(`/results/${id}/harmonization`),
  evaluation: (id: string) => get<Evaluation>(`/results/${id}/evaluation`),
  costs: (id: string) => get<CostSummary>(`/results/${id}/costs/summary`),
  models: (id: string) => get<Model[]>("/models", { experiment_id: id }),
  model: (modelId: number) => get<ModelDetail>(`/models/${modelId}`),
};

/** Downloads through the API (so the token header is sent) and saves the file. */
export async function download(path: string, filename: string): Promise<void> {
  const response = await http.get<Blob>(path, { responseType: "blob" });
  const url = URL.createObjectURL(response.data);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
