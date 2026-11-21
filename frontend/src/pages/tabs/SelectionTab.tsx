import {
  Grid, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField,
} from "@mui/material";
import { useState } from "react";

import { api } from "../../api/client";
import { ErrorState, Loading, Panel, StatusChip } from "../../components/common";
import { PALETTE, Plot } from "../../components/Plot";
import { formatNumber } from "../../lib/format";
import { jaccardMatrix, relevance } from "../../lib/transform";
import { useApi } from "../../lib/useApi";

export function SelectionTab({ id }: { id: string }) {
  const selection = useApi(() => api.selection(id), [id]);
  const results = useApi(() => api.results(id), [id]);
  const [strategy, setStrategy] = useState<string>();
  if (selection.error) return <ErrorState message={selection.error} />;
  if (!selection.data || !results.data) return <Loading />;
  const s = selection.data;
  const chosen = s.strategies.find((x) => x.name === strategy) ?? s.strategies[0];
  const selectedNames = new Set(chosen.selected.map((f) => f.feature));
  const rel = relevance(results.data);
  const jaccard = jaccardMatrix(s);
  return (
    <>
      <Panel title="Estratégias comparadas"
        subtitle={`${s.n_features_considered} atributos considerados · mesmas estatísticas federadas para todas`}>
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>Estratégia</TableCell><TableCell>Status</TableCell>
            <TableCell>Selecionados</TableCell><TableCell align="right">Elegíveis</TableCell>
            <TableCell align="right">Estabilidade sem um site (Nogueira)</TableCell>
            <TableCell align="right">Jaccard médio</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {s.strategies.map((x) => (
              <TableRow key={x.name} hover selected={x.name === chosen.name}
                sx={{ cursor: "pointer" }} onClick={() => setStrategy(x.name)}>
                <TableCell>{x.name}</TableCell>
                <TableCell><StatusChip status={x.completed ? "completed" : "failed"} /></TableCell>
                <TableCell>{x.selected.map((f) => f.feature).join(", ") || "—"}</TableCell>
                <TableCell align="right">{x.n_eligible}</TableCell>
                <TableCell align="right">
                  {formatNumber(x.leave_one_site_out?.nogueira_stability)}
                </TableCell>
                <TableCell align="right">
                  {formatNumber(x.leave_one_site_out?.mean_jaccard_to_full)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Panel>
      <Grid container spacing={2}>
        <Grid item xs={12} lg={7}>
          <Panel title="Atributos selecionados" actions={
            <TextField select size="small" label="Estratégia" value={chosen.name}
              onChange={(e) => setStrategy(e.target.value)} sx={{ minWidth: 220 }}>
              {s.strategies.map((x) => <MenuItem key={x.name} value={x.name}>{x.name}</MenuItem>)}
            </TextField>}>
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>#</TableCell><TableCell>Atributo</TableCell>
                <TableCell align="right">Confiabilidade</TableCell>
                <TableCell align="right">z combinado</TableCell>
                <TableCell align="right">p</TableCell>
                <TableCell align="right">Concordância</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {chosen.selected.map((f) => (
                  <TableRow key={f.feature}>
                    <TableCell>{f.rank}</TableCell>
                    <TableCell>{f.feature}</TableCell>
                    <TableCell align="right">{formatNumber(f.reliability)}</TableCell>
                    <TableCell align="right">{formatNumber(f.relevance_z, 2)}</TableCell>
                    <TableCell align="right">
                      {f.relevance_p === null || f.relevance_p === undefined ? "—"
                        : f.relevance_p.toExponential(1)}
                    </TableCell>
                    <TableCell align="right">{formatNumber(f.agreement)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            {Object.keys(chosen.excluded_counts ?? {}).length > 0 && (
              <Table size="small" sx={{ mt: 1 }}>
                <TableBody>
                  {Object.entries(chosen.excluded_counts ?? {}).map(([reason, count]) => (
                    <TableRow key={reason}>
                      <TableCell sx={{ color: "text.secondary" }}>excluídos: {reason}</TableCell>
                      <TableCell align="right">{count}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </Panel>
        </Grid>
        <Grid item xs={12} lg={5}>
          <Panel title="Sobreposição entre estratégias" subtitle="Jaccard">
            <Plot filename={`${id}-jaccard`} height={340}
              data={[{
                type: "heatmap", x: jaccard.names, y: jaccard.names, z: jaccard.z,
                zmin: 0, zmax: 1, colorscale: [[0, "#f4f6f9"], [1, PALETTE[0]]],
                hovertemplate: "%{y} × %{x}: %{z:.2f}<extra></extra>",
              }]}
              layout={{ margin: { l: 16, r: 16, t: 8, b: 16 }, xaxis: { tickangle: -35 } }} />
          </Panel>
        </Grid>
      </Grid>
      {rel && (
        <Panel title="Relevância e concordância de todos os atributos"
          subtitle="z do Cox estratificado por site × heterogeneidade entre sites (I²); destacados: selecionados">
          <Plot filename={`${id}-relevancia`} height={380}
            data={[false, true].map((isSelected) => {
              const idx = rel.features.map((_, i) => i)
                .filter((i) => selectedNames.has(rel.features[i]) === isSelected);
              // Label only what matters (selected, strong or heterogeneous); hover shows all.
              const notable = (i: number) => isSelected || Math.abs(rel.z[i] ?? 0) >= 3 ||
                (rel.iSquared[i] ?? 0) >= 0.5;
              return {
                type: "scatter", mode: "text+markers", name: isSelected ? "selecionados" : "outros",
                x: idx.map((i) => rel.z[i]), y: idx.map((i) => rel.iSquared[i]),
                text: idx.map((i) => (notable(i) ? rel.features[i] : "")),
                customdata: idx.map((i) => rel.features[i]), textposition: "top center",
                marker: { size: isSelected ? 11 : 7, color: isSelected ? PALETTE[1] : "#9aa8b6" },
                hovertemplate: "%{customdata}<br>z = %{x:.2f}<br>I² = %{y:.2f}<extra></extra>",
              };
            })}
            layout={{ xaxis: { title: { text: "z combinado (estratificado por site)" } },
                      yaxis: { title: { text: "I² (discordância entre sites)" },
                               range: [-0.05, 1.05] } }} />
        </Panel>
      )}
    </>
  );
}
