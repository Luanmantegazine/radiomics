import { Box, Grid, Typography } from "@mui/material";
import yaml from "js-yaml";

import type { ExperimentDetail, ExperimentSummary } from "../../api/types";
import { Panel, StatCard } from "../../components/common";
import { Plot } from "../../components/Plot";
import { formatBytes, formatDate, formatSeconds, PHASE_LABELS } from "../../lib/format";

export function OverviewTab({ summary, detail }: {
  summary: ExperimentSummary; detail: ExperimentDetail;
}) {
  const phases = summary.planned_phases.length
    ? summary.planned_phases
    : Object.keys(summary.rounds.by_phase);
  return (
    <>
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid item xs={6} md={2}><StatCard label="Sites registrados"
          value={summary.registered_clients}
          hint={summary.rejected_clients ? `${summary.rejected_clients} recusados` : undefined} />
        </Grid>
        <Grid item xs={6} md={2}><StatCard label="Rodadas" value={summary.rounds.total}
          hint={summary.rounds.failed ? `${summary.rounds.failed} com falha` : "nenhuma falha"} />
        </Grid>
        <Grid item xs={6} md={2}>
          <StatCard label="Tráfego total" value={formatBytes(summary.total_bytes)} />
        </Grid>
        <Grid item xs={6} md={2}>
          <StatCard label="Duração" value={formatSeconds(summary.duration_s)} />
        </Grid>
        <Grid item xs={6} md={2}>
          <StatCard label="Início" value={formatDate(detail.started_at)} />
        </Grid>
        <Grid item xs={6} md={2}>
          <StatCard label="Fim" value={formatDate(detail.finished_at)} />
        </Grid>
      </Grid>
      <Panel title="Rodadas por fase" subtitle="cada rodada é uma ida e volta com todos os sites">
        <Plot filename={`${detail.id}-rodadas-por-fase`} height={260}
          data={[{
            type: "bar",
            x: phases.map((p) => PHASE_LABELS[p] ?? p),
            y: phases.map((p) => summary.rounds.by_phase[p] ?? 0),
            hovertemplate: "%{x}: %{y} rodadas<extra></extra>",
          }]}
          layout={{ yaxis: { title: { text: "rodadas" }, gridcolor: "#e6eaef" } }} />
      </Panel>
      <Panel title="Configuração versionada" subtitle={`hash ${detail.config_hash}`}>
        <Box component="pre" sx={{ m: 0, p: 1.5, bgcolor: "#f4f6f9", borderRadius: 1,
          maxHeight: 420, overflow: "auto", fontSize: 12 }}>
          {yaml.dump(detail.config, { lineWidth: 100, noRefs: true })}
        </Box>
        <Typography variant="caption" color="text.secondary">
          Qualquer mudança nesta configuração muda o hash e é registrada como outro experimento.
        </Typography>
      </Panel>
    </>
  );
}
