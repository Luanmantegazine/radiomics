import {
  Box, Button, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography,
} from "@mui/material";
import { Link as RouterLink, useNavigate } from "react-router-dom";

import { api } from "../api/client";
import { Empty, ErrorState, Loading, OriginChip, Panel, StatusChip } from "../components/common";
import { formatDate } from "../lib/format";
import { useApi } from "../lib/useApi";

export function ExperimentsPage() {
  const navigate = useNavigate();
  const { data, error, loading } = useApi(() => api.experiments(), [], 5000);
  return (
    <>
      <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
        <Typography variant="h5" sx={{ flexGrow: 1 }}>Experimentos</Typography>
        <Button variant="contained" component={RouterLink} to="/new">Novo experimento</Button>
      </Box>
      {error && <ErrorState message={error} />}
      <Panel title="Execuções registradas" subtitle="atualiza a cada 5 s">
        {loading && !data ? <Loading /> : !data?.length ? (
          <Empty>Nenhum experimento ainda. Crie um a partir de um modelo de configuração.</Empty>
        ) : (
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Nome</TableCell>
                  <TableCell>Identificador</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Dados</TableCell>
                  <TableCell>Criado</TableCell>
                  <TableCell>Concluído</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((e) => (
                  <TableRow key={e.id} hover sx={{ cursor: "pointer" }}
                    onClick={() => navigate(`/experiments/${e.id}`)}>
                    <TableCell sx={{ fontWeight: 500 }}>{e.name}</TableCell>
                    <TableCell><code>{e.id}</code></TableCell>
                    <TableCell><StatusChip status={e.status} /></TableCell>
                    <TableCell><OriginChip origin={e.data_origin} /></TableCell>
                    <TableCell>{formatDate(e.created_at)}</TableCell>
                    <TableCell>{formatDate(e.finished_at)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Panel>
    </>
  );
}
