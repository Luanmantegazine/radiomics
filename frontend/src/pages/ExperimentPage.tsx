import { Alert, Box, Button, LinearProgress, Stack, Tab, Tabs, Typography } from "@mui/material";
import { lazy, Suspense, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import { ErrorState, Loading, OriginChip, StatusChip, SyntheticBanner } from "../components/common";
import { useApi } from "../lib/useApi";
import { ExportTab } from "./tabs/ExportTab";
import { SitesRoundsTab } from "./tabs/SitesRoundsTab";

// Tabs with charts load Plotly on demand.
const OverviewTab = lazy(() => import("./tabs/OverviewTab").then((m) => ({ default: m.OverviewTab })));
const HarmonizationTab = lazy(() =>
  import("./tabs/HarmonizationTab").then((m) => ({ default: m.HarmonizationTab })));
const SelectionTab = lazy(() =>
  import("./tabs/SelectionTab").then((m) => ({ default: m.SelectionTab })));
const ModelsTab = lazy(() => import("./tabs/ModelsTab").then((m) => ({ default: m.ModelsTab })));
const EvaluationTab = lazy(() =>
  import("./tabs/EvaluationTab").then((m) => ({ default: m.EvaluationTab })));
const CostsTab = lazy(() => import("./tabs/CostsTab").then((m) => ({ default: m.CostsTab })));

const ACTIVE = new Set(["queued", "running"]);

export function ExperimentPage() {
  const { id = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const [runError, setRunError] = useState<string>();
  const [polling, setPolling] = useState(true);
  const summary = useApi(async () => {
    const value = await api.summary(id);
    setPolling(ACTIVE.has(value.experiment.status));
    return value;
  }, [id], polling ? 2000 : undefined);
  const detail = useApi(() => api.experiment(id), [id]);

  if (summary.error && !summary.data) return <ErrorState message={summary.error} />;
  if (!summary.data || !detail.data) return <Loading />;
  const { experiment, available_results: available } = summary.data;

  const tabs: { key: string; label: string; show: boolean }[] = [
    { key: "overview", label: "Visão geral", show: true },
    { key: "sites", label: "Sites e rodadas", show: true },
    { key: "harmonization", label: "Harmonização", show: available.harmonization },
    { key: "selection", label: "Seleção", show: available.feature_selection },
    { key: "models", label: "Modelos", show: available.models },
    { key: "evaluation", label: "Avaliação", show: available.evaluation },
    { key: "costs", label: "Custos", show: true },
    { key: "export", label: "Exportar", show: true },
  ];
  const visible = tabs.filter((t) => t.show);
  const current = visible.find((t) => t.key === params.get("tab"))?.key ?? "overview";

  const run = async () => {
    setRunError(undefined);
    try {
      await api.run(id);
      setPolling(true);
      summary.reload();
    } catch (error) {
      setRunError(errorMessage(error));
    }
  };

  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 1, flexWrap: "wrap" }}>
        <Typography variant="h5">{experiment.name}</Typography>
        <StatusChip status={experiment.status} />
        <OriginChip origin={experiment.data_origin} />
        <Box sx={{ flexGrow: 1 }} />
        {experiment.status === "created" &&
          <Button variant="contained" onClick={run}>Executar</Button>}
      </Box>
      <Stack direction="row" spacing={2} sx={{ mb: 2, color: "text.secondary" }} useFlexGap
        flexWrap="wrap">
        <Typography variant="body2">id <code>{experiment.id}</code></Typography>
        <Typography variant="body2">configuração <code>{experiment.config_hash}</code></Typography>
        <Typography variant="body2">protocolo {experiment.protocol_version}</Typography>
        {experiment.mlflow_run_id &&
          <Typography variant="body2">MLflow <code>{experiment.mlflow_run_id}</code></Typography>}
      </Stack>
      <SyntheticBanner origin={experiment.data_origin} />
      {runError && <Alert severity="error" sx={{ mb: 2 }}>{runError}</Alert>}
      {experiment.status === "failed" && experiment.error &&
        <Alert severity="error" sx={{ mb: 2 }}>Falha: {experiment.error.slice(0, 600)}</Alert>}
      {ACTIVE.has(experiment.status) && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            Em execução · {summary.data.rounds.total} rodadas concluídas
          </Typography>
          <LinearProgress sx={{ mt: 0.5 }} />
        </Box>
      )}
      <Tabs value={current} onChange={(_, value) => setParams({ tab: value })}
        variant="scrollable" sx={{ borderBottom: 1, borderColor: "divider", mb: 2 }}>
        {visible.map((t) => <Tab key={t.key} value={t.key} label={t.label} />)}
      </Tabs>
      <Suspense fallback={<Loading />}>
      {current === "overview" && <OverviewTab summary={summary.data} detail={detail.data} />}
      {current === "sites" && <SitesRoundsTab id={id} refreshKey={summary.data.rounds.total} />}
      {current === "harmonization" && <HarmonizationTab id={id} />}
      {current === "selection" && <SelectionTab id={id} />}
      {current === "models" && <ModelsTab id={id} />}
      {current === "evaluation" && <EvaluationTab id={id} />}
      {current === "costs" && <CostsTab id={id} refreshKey={summary.data.rounds.total} />}
      {current === "export" && <ExportTab id={id} />}
      </Suspense>
    </>
  );
}
