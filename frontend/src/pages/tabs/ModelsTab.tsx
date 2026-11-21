import {
  Grid, Table, TableBody, TableCell, TableHead, TableRow, Typography,
} from "@mui/material";
import { useState } from "react";

import { api } from "../../api/client";
import { ErrorState, Loading, Panel, StatusChip } from "../../components/common";
import { PALETTE, Plot } from "../../components/Plot";
import { coefficients, convergence } from "../../lib/transform";
import { useApi } from "../../lib/useApi";

function ModelDetailPanel({ modelId }: { modelId: number }) {
  const { data, error } = useApi(() => api.model(modelId), [modelId]);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  if (!data.parameters) {
    return <Panel title={data.name}><Typography>Falhou: {data.failure_reason}</Typography></Panel>;
  }
  const rows = coefficients(data.parameters);
  const horizons = [...new Set(rows.map((r) => r.horizon))];
  const p = data.parameters as Record<string, unknown>;
  const baseline = p.kind === "cox"
    ? (p.baseline_cumulative_hazard as (number | null)[]).map((v, i) => ({
        horizon: (p.horizons_months as number[])[i], value: v }))
    : null;
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} lg={7}>
        <Panel title="Coeficientes" subtitle={p.kind === "cox"
          ? "log hazard ratio por desvio-padrão, IC 95% (Wald)"
          : "log odds por desvio-padrão, por horizonte, IC 95% (Wald)"}>
          <Plot filename={`${data.name}-coeficientes`} height={Math.max(260, 60 * rows.length / horizons.length + 80)}
            data={horizons.map((h, k) => {
              const subset = rows.filter((r) => r.horizon === h);
              return {
                type: "scatter", mode: "markers",
                name: h === null ? "Cox" : `${h} meses`,
                x: subset.map((r) => r.estimate), y: subset.map((r) => r.feature),
                error_x: { type: "data", symmetric: false,
                           array: subset.map((r) => r.upper - r.estimate),
                           arrayminus: subset.map((r) => r.estimate - r.lower) },
                marker: { color: PALETTE[k], size: 9 },
                hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
              };
            })}
            layout={{ shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1,
                                 line: { color: "#9aa8b6", dash: "dot" } }],
                      margin: { l: 170, r: 16, t: 8, b: 48 } }} />
        </Panel>
      </Grid>
      <Grid item xs={12} lg={5}>
        <Panel title="Convergência" subtitle="log-verossimilhança penalizada por iteração de Newton">
          <Plot filename={`${data.name}-convergencia`} height={260}
            data={convergence(data.parameters).map((s) => ({
              type: "scatter", mode: "lines+markers", name: s.name, x: s.x, y: s.y }))}
            layout={{ xaxis: { title: { text: "iteração" }, dtick: 1 } }} />
        </Panel>
        {baseline && (
          <Panel title="Risco basal acumulado liberado"
            subtitle="média dos sites ponderada pelo tamanho">
            <Table size="small">
              <TableBody>
                {baseline.map((b) => (
                  <TableRow key={b.horizon}>
                    <TableCell>{b.horizon} meses</TableCell>
                    <TableCell align="right">{b.value === null ? "indefinido" : b.value.toFixed(4)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Panel>
        )}
      </Grid>
    </Grid>
  );
}

export function ModelsTab({ id }: { id: string }) {
  const models = useApi(() => api.models(id), [id]);
  const [selected, setSelected] = useState<number>();
  if (models.error) return <ErrorState message={models.error} />;
  if (!models.data) return <Loading />;
  const current = selected ?? models.data.find((m) => m.status === "trained")?.id;
  return (
    <>
      <Panel title="Modelos treinados" subtitle="clique para ver coeficientes e convergência">
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>Modelo</TableCell><TableCell>Tipo</TableCell><TableCell>Status</TableCell>
            <TableCell>Origem dos atributos</TableCell><TableCell>Atributos</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {models.data.map((m) => (
              <TableRow key={m.id} hover selected={m.id === current} sx={{ cursor: "pointer" }}
                onClick={() => setSelected(m.id)}>
                <TableCell>{m.name}</TableCell>
                <TableCell>{m.kind}</TableCell>
                <TableCell><StatusChip status={m.status} /></TableCell>
                <TableCell>{m.features_source}</TableCell>
                <TableCell>{m.feature_names.join(", ") || m.failure_reason}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Panel>
      {current !== undefined && <ModelDetailPanel modelId={current} />}
    </>
  );
}
