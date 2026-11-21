import {
  Grid, Table, TableBody, TableCell, TableHead, TableRow, Typography,
} from "@mui/material";

import { api } from "../../api/client";
import { ErrorState, Loading, Panel, StatCard } from "../../components/common";
import { Plot } from "../../components/Plot";
import { formatNumber } from "../../lib/format";
import { siteEffects } from "../../lib/transform";
import { useApi } from "../../lib/useApi";

export function HarmonizationTab({ id }: { id: string }) {
  const result = useApi(() => api.harmonization(id), [id]);
  const rounds = useApi(() => api.results(id), [id]);
  if (result.error) return <ErrorState message={result.error} />;
  if (!result.data || !rounds.data) return <Loading />;
  const h = result.data;
  const effects = siteEffects(rounds.data);
  return (
    <>
      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid item xs={6} md={3}><StatCard label="Método" value={h.method}
          hint={h.covariates.length ? `preserva: ${h.covariates.join(", ")}` : "sem covariáveis"} />
        </Grid>
        <Grid item xs={6} md={3}><StatCard label="η² de site (mediana) antes"
          value={formatNumber(h.site_effect_before?.median_eta_squared as number | undefined, 4)} />
        </Grid>
        <Grid item xs={6} md={3}><StatCard label="η² de site (mediana) depois"
          value={formatNumber(h.site_effect_after?.median_eta_squared as number | undefined, 4)} />
        </Grid>
        <Grid item xs={6} md={3}><StatCard label="Qui-quadrado da taxa de conversão"
          value={formatNumber(h.event_rate_chi2, 2)} hint="diferença bruta entre sites" />
        </Grid>
      </Grid>
      {effects && (
        <Panel title="Efeito de site por atributo"
          subtitle="fração da variância explicada pelo site (η²), antes e depois">
          <Plot filename={`${id}-efeito-de-site`} height={360}
            data={[
              { type: "bar", name: "antes", x: effects.features, y: effects.before },
              ...(effects.after ? [{ type: "bar" as const, name: "depois", x: effects.features,
                                      y: effects.after }] : []),
            ]}
            layout={{ barmode: "group", yaxis: { title: { text: "η²" }, rangemode: "tozero" },
                      xaxis: { tickangle: -35 } }} />
        </Panel>
      )}
      <Panel title="Os sites diferem clinicamente?"
        subtitle="diferenças que podem ser reais, e não técnicas">
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
          O ComBat remove tudo que difere entre sites e não é covariável listada. Diferenças em
          covariáveis, conversão ou seguimento indicam que parte do efeito de site pode ser
          biológica.
        </Typography>
        <Table size="small">
          <TableHead><TableRow>
            <TableCell>Covariável</TableCell><TableCell align="right">η² por site</TableCell>
          </TableRow></TableHead>
          <TableBody>
            {Object.entries(h.covariate_eta_squared).map(([name, value]) => (
              <TableRow key={name}>
                <TableCell>{name}</TableCell>
                <TableCell align="right">{formatNumber(value, 3)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {h.sites.length > 0 && (
          <Table size="small" sx={{ mt: 2 }}>
            <TableHead><TableRow>
              <TableCell>Site</TableCell><TableCell align="right">n (treino)</TableCell>
              <TableCell align="right">Taxa de conversão</TableCell>
              <TableCell align="right">Seguimento mediano (meses)</TableCell>
            </TableRow></TableHead>
            <TableBody>
              {h.sites.map((s) => (
                <TableRow key={s.client_id}>
                  <TableCell>{s.client_id}</TableCell>
                  <TableCell align="right">{s.n}</TableCell>
                  <TableCell align="right">{formatNumber(s.event_rate, 3)}</TableCell>
                  <TableCell align="right">{formatNumber(s.median_follow_up, 1)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Panel>
    </>
  );
}
