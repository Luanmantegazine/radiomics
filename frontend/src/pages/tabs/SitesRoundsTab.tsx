import {
  Table, TableBody, TableCell, TableContainer, TableHead, TablePagination, TableRow,
} from "@mui/material";
import { Fragment, useState } from "react";

import { api } from "../../api/client";
import { Empty, ErrorState, Loading, Panel, StatusChip } from "../../components/common";
import { formatDate, PHASE_LABELS } from "../../lib/format";
import { useApi } from "../../lib/useApi";

function RoundParticipation({ roundId }: { roundId: string }) {
  const { data } = useApi(() => api.round(roundId), [roundId]);
  if (!data) return null;
  return (
    <Table size="small">
      <TableBody>
        {data.participations.map((p) => (
          <TableRow key={p.client_id}>
            <TableCell sx={{ pl: 4 }}>{p.client_id}</TableCell>
            <TableCell><StatusChip status={p.status} /></TableCell>
            <TableCell>{p.reason_code ?? ""}</TableCell>
            <TableCell>{p.n_samples !== null ? `n = ${p.n_samples}` : ""}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

export function SitesRoundsTab({ id, refreshKey }: { id: string; refreshKey: number }) {
  const clients = useApi(() => api.clients(id), [id, refreshKey]);
  const rounds = useApi(() => api.rounds(id), [id, refreshKey]);
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<string>();
  const perPage = 25;
  return (
    <>
      <Panel title="Sites" subtitle="identificadores pseudonimizados">
        {clients.error && <ErrorState message={clients.error} />}
        {!clients.data ? <Loading /> : !clients.data.length ? <Empty>Nenhum site registrado.</Empty> : (
          <TableContainer>
            <Table size="small">
              <TableHead><TableRow>
                <TableCell>Site</TableCell><TableCell>Registro</TableCell>
                <TableCell>Motivo</TableCell><TableCell>Dados</TableCell>
                <TableCell>Software</TableCell><TableCell>Registrado em</TableCell>
              </TableRow></TableHead>
              <TableBody>
                {clients.data.map((c) => (
                  <TableRow key={c.client_id}>
                    <TableCell>{c.client_id}</TableCell>
                    <TableCell><StatusChip status={c.status} /></TableCell>
                    <TableCell>{c.reason_code ?? "—"}</TableCell>
                    <TableCell>{c.data_origin}</TableCell>
                    <TableCell sx={{ fontSize: 12 }}>
                      {Object.entries(c.software).map(([k, v]) => `${k} ${v}`).join(" · ")}
                    </TableCell>
                    <TableCell>{formatDate(c.registered_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Panel>
      <Panel title="Rodadas federadas" subtitle="clique numa rodada para ver a participação de cada site">
        {rounds.error && <ErrorState message={rounds.error} />}
        {!rounds.data ? <Loading /> : (
          <>
            <TableContainer>
              <Table size="small">
                <TableHead><TableRow>
                  <TableCell>#</TableCell><TableCell>Fase</TableCell><TableCell>Tarefa</TableCell>
                  <TableCell>Agregação</TableCell><TableCell>Status</TableCell>
                  <TableCell>Falha</TableCell>
                </TableRow></TableHead>
                <TableBody>
                  {rounds.data.slice(page * perPage, (page + 1) * perPage).map((r) => (
                    <Fragment key={r.id}>
                      <TableRow hover sx={{ cursor: "pointer" }}
                        onClick={() => setOpen(open === r.id ? undefined : r.id)}>
                        <TableCell>{r.round_number}</TableCell>
                        <TableCell>{PHASE_LABELS[r.phase] ?? r.phase}</TableCell>
                        <TableCell><code>{r.task}</code></TableCell>
                        <TableCell><code>{r.aggregator}</code></TableCell>
                        <TableCell><StatusChip status={r.status} /></TableCell>
                        <TableCell>{r.failure_reason ?? ""}</TableCell>
                      </TableRow>
                      {open === r.id && (
                        <TableRow>
                          <TableCell colSpan={6} sx={{ p: 0 }}>
                            <RoundParticipation roundId={r.id} />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination component="div" count={rounds.data.length} page={page}
              rowsPerPage={perPage} rowsPerPageOptions={[perPage]}
              onPageChange={(_, p) => setPage(p)} />
          </>
        )}
      </Panel>
    </>
  );
}
