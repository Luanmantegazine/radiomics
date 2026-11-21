export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function formatBytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let v = value;
  let unit = 0;
  while (v >= 1024 && unit < units.length - 1) {
    v /= 1024;
    unit += 1;
  }
  return `${v.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)} ms`;
  if (value < 120) return `${value.toFixed(1)} s`;
  return `${(value / 60).toFixed(1)} min`;
}

export function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  return new Date(value).toLocaleString("pt-BR");
}

export const PHASE_LABELS: Record<string, string> = {
  infrastructure: "Infraestrutura",
  harmonization: "Harmonização",
  feature_selection: "Seleção",
  training: "Treinamento",
  evaluation: "Avaliação",
};

export const VARIANT_LABELS: Record<string, string> = {
  released: "Federado (liberado)",
  site_baseline: "Federado, basal do site",
  local: "Local (cada site)",
};

export function scopeLabel(scope: string): string {
  if (scope === "internal") return "Interna (teste)";
  return scope.startsWith("external:") ? `Externa: ${scope.slice(9)}` : scope;
}
