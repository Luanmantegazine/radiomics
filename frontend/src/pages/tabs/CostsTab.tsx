import { Grid, Table, TableBody, TableCell, TableHead, TableRow } from "@mui/material";

import { api } from "../../api/client";
import { ErrorState, Loading, Panel, StatCard } from "../../components/common";
import { PALETTE, Plot } from "../../components/Plot";
import { formatBytes, formatSeconds, PHASE_LABELS } from "../../lib/format";
import { useApi } from "../../lib/useApi";

const PHASE_TIMES = [
  ["harmonization_time", "Harmonização"],
  ["selection_time", "Seleção (estratégias)"],
  ["training_time", "Treinamento"],
  ["evaluation_time", "Avaliação"],
  ["total_time", "Total"],
] as const;

export function CostsTab({ id, refreshKey }: { id: string; refreshKey: number }) {
  const { data, error } = useApi(() => api.costs(id), [id, refreshKey]);
  if (error) return <ErrorState message={error} />;
  if (!data) return <Loading />;
  const phases = [...new Set(data.by_round.map((r) => r.phase as string))];
  return (
    <>
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        {PHASE_TIMES.filter(([key]) => data.totals[key] !== undefined).map(([key, label]) => (
          <Grid item xs={6} md={2} key={key}>
            <StatCard label={label} value={formatSeconds(data.totals[key])}
              hint={key === "evaluation_time" ? "inclui refazer seleção e treino na validação externa"
                : key === "selection_time" ? "só a comparação de estratégias" : undefined} />
          </Grid>
        ))}
        <Grid item xs={6} md={2}>
          <StatCard label="Tráfego total" value={formatBytes(data.totals.total_bytes)} />
        </Grid>
      </Grid>
      <Panel title="Tráfego por rodada" subtitle="bytes enviados pelos sites + recebidos por eles">
        <Plot filename={`${id}-trafego-por-rodada`} height={320}
          data={phases.map((phase, k) => {
            const rows = data.by_round.filter((r) => r.phase === phase);
            return {
              type: "bar", name: PHASE_LABELS[phase] ?? phase,
              x: rows.map((r) => r.round_number as number),
              y: rows.map((r) => (r.bytes_from_clients as number) + (r.bytes_to_clients as number)),
              marker: { color: PALETTE[k] },
              hovertemplate: "rodada %{x}: %{y:,.0f} bytes<extra></extra>",
            };
          })}
          layout={{ barmode: "stack", bargap: 0.05, xaxis: { title: { text: "rodada" } },
                    yaxis: { title: { text: "bytes" } } }} />
      </Panel>
      <Grid container spacing={2}>
        <Grid item xs={12} lg={6}>
          <Panel title="Por fase">
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>Fase</TableCell><TableCell align="right">Rodadas</TableCell>
                <TableCell align="right">Dos sites</TableCell><TableCell align="right">Para os sites</TableCell>
                <TableCell align="right">Comunicação</TableCell><TableCell align="right">Agregação</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {data.by_phase.map((p) => (
                  <TableRow key={p.phase}>
                    <TableCell>{PHASE_LABELS[p.phase] ?? p.phase}</TableCell>
                    <TableCell align="right">{p.rounds}</TableCell>
                    <TableCell align="right">{formatBytes(p.bytes_from_clients)}</TableCell>
                    <TableCell align="right">{formatBytes(p.bytes_to_clients)}</TableCell>
                    <TableCell align="right">{formatSeconds(p.communication_s)}</TableCell>
                    <TableCell align="right">{formatSeconds(p.aggregation_s)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Panel>
        </Grid>
        <Grid item xs={12} lg={6}>
          <Panel title="Por site" subtitle="memória: pico do processo (na simulação, compartilhado)">
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>Site</TableCell><TableCell align="right">Rodadas</TableCell>
                <TableCell align="right">Enviado</TableCell><TableCell align="right">Recebido</TableCell>
                <TableCell align="right">Execução local</TableCell><TableCell align="right">Memória</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {data.by_client.map((c) => (
                  <TableRow key={c.client_id}>
                    <TableCell>{c.client_id}</TableCell>
                    <TableCell align="right">{c.rounds}</TableCell>
                    <TableCell align="right">{formatBytes(c.bytes_sent)}</TableCell>
                    <TableCell align="right">{formatBytes(c.bytes_received)}</TableCell>
                    <TableCell align="right">{formatSeconds(c.local_execution_s)}</TableCell>
                    <TableCell align="right">{formatBytes(c.peak_memory_bytes)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Panel>
        </Grid>
      </Grid>
    </>
  );
}
