import {
  Alert, Grid, MenuItem, Stack, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, TextField,
} from "@mui/material";
import { useState } from "react";

import { api } from "../../api/client";
import type { EvaluationRow } from "../../api/types";
import { Empty, ErrorState, Loading, Panel } from "../../components/common";
import { PALETTE, Plot } from "../../components/Plot";
import { formatNumber, scopeLabel, VARIANT_LABELS } from "../../lib/format";
import {
  calibrationBins, decisionCurve, externalAuc, metricByHorizon, metricRows,
} from "../../lib/transform";
import { useApi } from "../../lib/useApi";

const METRICS = [
  { key: "auc", label: "AUC dependente do tempo" },
  { key: "c_index", label: "C de Uno" },
  { key: "brier", label: "Brier (IPCW)" },
  { key: "scaled_brier", label: "Brier escalado" },
  { key: "o_e", label: "Observado / esperado" },
] as const;

function Select({ label, value, options, onChange }: {
  label: string; value: string; options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <TextField select size="small" label={label} value={value} sx={{ minWidth: 200 }}
      onChange={(e) => onChange(e.target.value)}>
      {options.map((o) => <MenuItem key={o.value} value={o.value}>{o.label}</MenuItem>)}
    </TextField>
  );
}

function CurvePanels({ row, horizon, id }: { row: EvaluationRow; horizon: number; id: string }) {
  const metrics = row.pooled.find((h) => h.horizon_months === horizon);
  if (!metrics?.available) return <Empty>Sem predição neste horizonte (além do seguimento).</Empty>;
  const bins = calibrationBins(metrics).filter((b) => b.observed !== null);
  const decision = decisionCurve(metrics);
  const configured = metrics.decision_curve?.length ?? 0;
  const merged = configured - decision.length;
  const name = `${row.model_name}-${row.variant}-${horizon}m`;
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Panel title="Calibração" subtitle="observado (IPCW) × previsto, por faixa de risco">
          <Plot filename={`${id}-${name}-calibracao`} height={330}
            data={[
              { type: "scatter", mode: "lines", name: "ideal", x: [0, 1], y: [0, 1],
                line: { color: "#9aa8b6", dash: "dot" }, hoverinfo: "skip" },
              { type: "scatter", mode: "lines+markers", name: "modelo",
                x: bins.map((b) => b.predicted), y: bins.map((b) => b.observed),
                text: bins.map((b) => `n = ${b.n}`), marker: { size: 8, color: PALETTE[0] },
                hovertemplate: "previsto %{x:.2f} · observado %{y:.2f} · %{text}<extra></extra>" },
            ]}
            layout={{ xaxis: { title: { text: "probabilidade prevista" }, range: [0, 1] },
                      yaxis: { title: { text: "proporção observada" }, range: [0, 1] } }} />
        </Panel>
      </Grid>
      <Grid item xs={12} md={6}>
        <Panel title="Curva de decisão" subtitle={merged > 0
          ? `benefício líquido por limiar; ${merged} limiar(es) indisponível(is): faixas fundidas por controle de divulgação`
          : "benefício líquido por limiar de risco"}>
          <Plot filename={`${id}-${name}-decisao`} height={330}
            data={[
              { type: "scatter", mode: "lines+markers", name: "modelo",
                x: decision.map((d) => d.threshold), y: decision.map((d) => d.model),
                line: { color: PALETTE[0] } },
              { type: "scatter", mode: "lines", name: "tratar todos",
                x: decision.map((d) => d.threshold), y: decision.map((d) => d.treat_all),
                line: { color: PALETTE[1], dash: "dash" } },
              { type: "scatter", mode: "lines", name: "não tratar",
                x: decision.map((d) => d.threshold), y: decision.map(() => 0),
                line: { color: "#9aa8b6", dash: "dot" } },
            ]}
            layout={{ xaxis: { title: { text: "limiar de probabilidade" } },
                      yaxis: { title: { text: "benefício líquido" } } }} />
        </Panel>
      </Grid>
    </Grid>
  );
}

export function EvaluationTab({ id }: { id: string }) {
  const { data, error } = useApi(() => api.evaluation(id), [id]);
  const [scope, setScope] = useState("internal");
  const [variant, setVariant] = useState("released");
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("auc");
  const [model, setModel] = useState<string>();
  const [horizon, setHorizon] = useState<number>();
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  const rows = data.evaluations;
  const scopes = [...new Set(rows.map((r) => r.scope))];
  const variants = [...new Set(rows.filter((r) => r.scope === scope).map((r) => r.variant))];
  const activeVariant = variants.includes(variant) ? variant : variants[0];
  const inScope = rows.filter((r) => r.scope === scope && r.variant === activeVariant);
  const horizons = inScope[0]?.pooled.map((h) => h.horizon_months) ?? [];
  const activeHorizon = horizon ?? horizons[1] ?? horizons[0];
  const activeModel = inScope.find((r) => r.model_name === model) ?? inScope[0];
  const series = metricByHorizon(rows, scope, activeVariant, metric);
  const table = metricRows(inScope);
  const hasExternal = scopes.some((s) => s.startsWith("external:"));
  return (
    <>
      <Alert severity="info" variant="outlined" sx={{ mb: 2 }}>
        Métricas agregadas entre sites: AUC e C comparam apenas participantes do mesmo site.
        Avaliação interna usa a partição de teste reservada de cada site. Sem intervalos de
        confiança.
      </Alert>
      <Stack direction="row" spacing={1.5} sx={{ mb: 2 }} useFlexGap flexWrap="wrap">
        <Select label="Escopo" value={scope} onChange={setScope}
          options={scopes.map((s) => ({ value: s, label: scopeLabel(s) }))} />
        <Select label="Variante" value={activeVariant} onChange={setVariant}
          options={variants.map((v) => ({ value: v, label: VARIANT_LABELS[v] ?? v }))} />
        <Select label="Métrica" value={metric} onChange={(v) => setMetric(v as typeof metric)}
          options={METRICS.map((m) => ({ value: m.key, label: m.label }))} />
      </Stack>
      <Grid container spacing={2}>
        <Grid item xs={12} lg={6}>
          <Panel title={METRICS.find((m) => m.key === metric)!.label} subtitle="por horizonte">
            <Plot filename={`${id}-${metric}-por-horizonte`} height={340}
              data={series.map((s) => ({ type: "scatter", mode: "lines+markers", name: s.name,
                                         x: s.x, y: s.y }))}
              layout={{ xaxis: { title: { text: "horizonte (meses)" }, tickvals: horizons } }} />
          </Panel>
        </Grid>
        <Grid item xs={12} lg={6}>
          {hasExternal ? (
            <Panel title="Validação externa" subtitle={`AUC em ${activeHorizon} meses no site deixado de fora`}>
              <Plot filename={`${id}-validacao-externa`} height={340}
                data={externalAuc(rows, activeHorizon).map((s) => ({
                  type: "bar", name: s.name, x: s.x, y: s.y }))}
                layout={{ barmode: "group", yaxis: { title: { text: "AUC" }, range: [0, 1] },
                          shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0.5, y1: 0.5,
                                     line: { color: "#9aa8b6", dash: "dot" } }] }} />
            </Panel>
          ) : (
            <Panel title="Validação externa"><Empty>Não configurada neste experimento.</Empty></Panel>
          )}
        </Grid>
      </Grid>
      <Panel title="Métricas por modelo e horizonte">
        <TableContainer>
          <Table size="small">
            <TableHead><TableRow>
              <TableCell>Modelo</TableCell><TableCell align="right">Horizonte</TableCell>
              <TableCell align="right">n</TableCell><TableCell align="right">Eventos</TableCell>
              <TableCell align="right">AUC</TableCell><TableCell align="right">C</TableCell>
              <TableCell align="right">Brier</TableCell><TableCell align="right">Brier escalado</TableCell>
              <TableCell align="right">O/E</TableCell><TableCell align="right">Erro de calibração</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {table.map((r) => (
                <TableRow key={`${r.model}-${r.horizon}`}>
                  <TableCell>{r.model}</TableCell>
                  <TableCell align="right">{r.horizon}</TableCell>
                  <TableCell align="right">{r.n}</TableCell>
                  <TableCell align="right">{r.events}</TableCell>
                  <TableCell align="right">{formatNumber(r.auc)}</TableCell>
                  <TableCell align="right">{formatNumber(r.c)}</TableCell>
                  <TableCell align="right">{formatNumber(r.brier)}</TableCell>
                  <TableCell align="right">{formatNumber(r.scaledBrier)}</TableCell>
                  <TableCell align="right">{formatNumber(r.oe, 2)}</TableCell>
                  <TableCell align="right">{formatNumber(r.calibrationError)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Panel>
      {activeModel && (
        <>
          <Stack direction="row" spacing={1.5} sx={{ mb: 2 }}>
            <Select label="Modelo" value={activeModel.model_name} onChange={setModel}
              options={inScope.map((r) => ({ value: r.model_name, label: r.model_name }))} />
            <Select label="Horizonte" value={String(activeHorizon)}
              onChange={(v) => setHorizon(Number(v))}
              options={horizons.map((h) => ({ value: String(h), label: `${h} meses` }))} />
          </Stack>
          <CurvePanels row={activeModel} horizon={activeHorizon} id={id} />
          {activeModel.sites.length > 0 && (
            <Panel title="Desempenho por site" subtitle={`${activeModel.model_name}, ${activeHorizon} meses`}>
              <Plot filename={`${id}-por-site`} height={280}
                data={[{
                  type: "bar", x: activeModel.sites.map((s) => s.client_id),
                  y: activeModel.sites.map((s) =>
                    s.horizons.find((h) => h.horizon_months === activeHorizon)?.auc ?? null),
                  marker: { color: PALETTE[0] }, name: "AUC",
                }]}
                layout={{ yaxis: { title: { text: "AUC" }, range: [0, 1] } }} />
            </Panel>
          )}
        </>
      )}
    </>
  );
}
