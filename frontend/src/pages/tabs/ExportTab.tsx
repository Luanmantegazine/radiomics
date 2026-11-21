import DownloadIcon from "@mui/icons-material/Download";
import { Alert, Button, Stack, Typography } from "@mui/material";
import { useState } from "react";

import { download, errorMessage } from "../../api/client";
import { Panel } from "../../components/common";

const TABLES = [
  ["evaluation", "Métricas de avaliação"],
  ["selected-features", "Atributos selecionados"],
  ["models", "Coeficientes dos modelos"],
  ["costs", "Custos por rodada"],
  ["rounds", "Rodadas"],
] as const;

export function ExportTab({ id }: { id: string }) {
  const [problem, setProblem] = useState<string>();
  const save = (path: string, filename: string) =>
    download(path, filename).catch((error) => setProblem(errorMessage(error)));
  return (
    <>
      {problem && <Alert severity="error" sx={{ mb: 2 }}>{problem}</Alert>}
      <Panel title="Relatório completo"
        subtitle="gerado pelo backend a partir dos resultados liberados; sem dados individuais">
        <Stack direction="row" spacing={1.5} useFlexGap flexWrap="wrap">
          <Button variant="contained" startIcon={<DownloadIcon />}
            onClick={() => save(`/reports/${id}?format=html&download=true`, `${id}.html`)}>
            Relatório HTML
          </Button>
          <Button variant="outlined" startIcon={<DownloadIcon />}
            onClick={() => save(`/reports/${id}?format=json&download=true`, `${id}.json`)}>
            Pacote JSON
          </Button>
        </Stack>
      </Panel>
      <Panel title="Tabelas (CSV)">
        <Stack direction="row" spacing={1.5} useFlexGap flexWrap="wrap">
          {TABLES.map(([table, label]) => (
            <Button key={table} variant="outlined" size="small" startIcon={<DownloadIcon />}
              onClick={() => save(`/reports/${id}/${table}.csv`, `${id}-${table}.csv`)}>
              {label}
            </Button>
          ))}
        </Stack>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
          Cada gráfico do dashboard também pode ser salvo em PNG pelo ícone de câmera.
          Experimentos com dados sintéticos levam o aviso no cabeçalho de cada arquivo.
        </Typography>
      </Panel>
    </>
  );
}
