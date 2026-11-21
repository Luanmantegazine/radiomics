import {
  Alert, Box, Button, Card, CardActionArea, CardContent, Chip, Grid, Stack, TextField,
  Typography,
} from "@mui/material";
import yaml from "js-yaml";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { api, errorMessage } from "../api/client";
import type { Template, Validation } from "../api/types";
import { ErrorState, Loading, Panel } from "../components/common";
import { PHASE_LABELS } from "../lib/format";
import { useApi } from "../lib/useApi";

/** Parse the editor text (YAML is a superset of JSON). */
export function parseConfig(text: string): { config?: unknown; error?: string } {
  try {
    const config = yaml.load(text);
    if (config === null || typeof config !== "object" || Array.isArray(config)) {
      return { error: "A configuração deve ser um objeto (mapa de chaves)." };
    }
    return { config };
  } catch (error) {
    return { error: `Erro de sintaxe: ${(error as Error).message}` };
  }
}

export function NewExperimentPage() {
  const navigate = useNavigate();
  const templates = useApi(() => api.templates(), []);
  const [selected, setSelected] = useState<string>();
  const [text, setText] = useState("");
  const [validation, setValidation] = useState<Validation>();
  const [problem, setProblem] = useState<string>();
  const [busy, setBusy] = useState(false);

  const choose = (template: Template) => {
    setSelected(template.id);
    setText(yaml.dump(template.config, { lineWidth: 100, noRefs: true }));
    setValidation(undefined);
    setProblem(undefined);
  };

  const check = async (): Promise<unknown | undefined> => {
    setProblem(undefined);
    const parsed = parseConfig(text);
    if (parsed.error) {
      setValidation(undefined);
      setProblem(parsed.error);
      return undefined;
    }
    try {
      const result = await api.validate(parsed.config);
      setValidation(result);
      return result.valid ? parsed.config : undefined;
    } catch (error) {
      setProblem(errorMessage(error));
      return undefined;
    }
  };

  const create = async (run: boolean) => {
    setBusy(true);
    try {
      const config = await check();
      if (!config) return;
      const experiment = await api.create(config);
      if (run) await api.run(experiment.id);
      navigate(`/experiments/${experiment.id}`);
    } catch (error) {
      setProblem(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Typography variant="h5" sx={{ mb: 2 }}>Novo experimento</Typography>
      <Panel title="1. Modelo de configuração"
        subtitle="configurações versionadas do repositório (experiments/configs)">
        {templates.error && <ErrorState message={templates.error} />}
        {templates.loading ? <Loading /> : (
          <Grid container spacing={1.5}>
            {templates.data?.map((t) => (
              <Grid item xs={12} sm={6} lg={3} key={t.id}>
                <Card variant="outlined"
                  sx={{ height: "100%", borderColor: selected === t.id ? "primary.main" : undefined,
                        borderWidth: selected === t.id ? 2 : 1 }}>
                  <CardActionArea onClick={() => choose(t)} sx={{ height: "100%" }}>
                    <CardContent>
                      <Typography variant="subtitle2">{t.name}</Typography>
                      <Typography variant="caption" color="text.secondary" component="p"
                        sx={{ my: 0.5 }}>{t.description}</Typography>
                      <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                        {t.phases.map((p) => <Chip key={p} size="small" label={PHASE_LABELS[p] ?? p} />)}
                        {t.data_origin === "synthetic" &&
                          <Chip size="small" color="warning" variant="outlined" label="sintético" />}
                      </Stack>
                    </CardContent>
                  </CardActionArea>
                </Card>
              </Grid>
            ))}
          </Grid>
        )}
      </Panel>
      <Panel title="2. Configuração" subtitle="YAML ou JSON; validada pelo backend antes de criar">
        <TextField multiline fullWidth minRows={16} maxRows={32} value={text}
          onChange={(e) => { setText(e.target.value); setValidation(undefined); }}
          placeholder="Escolha um modelo acima ou cole uma configuração."
          inputProps={{ "aria-label": "configuração", spellCheck: false,
                        style: { fontFamily: "ui-monospace, monospace", fontSize: 12 } }} />
        {problem && <Alert severity="error" sx={{ mt: 2 }}>{problem}</Alert>}
        {validation && !validation.valid && (
          <Alert severity="error" sx={{ mt: 2 }}>
            <strong>Configuração inválida</strong>
            <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
              {validation.errors?.map((e, i) => (
                <li key={i}><code>{e.location}</code>: {e.message}</li>
              ))}
            </ul>
          </Alert>
        )}
        {validation?.valid && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Configuração válida · hash <code>{validation.config_hash}</code> · fases:{" "}
            {validation.phases?.map((p) => PHASE_LABELS[p] ?? p).join(", ")} · sites:{" "}
            {validation.expected_clients?.join(", ")}
          </Alert>
        )}
        <Box sx={{ display: "flex", gap: 1, mt: 2 }}>
          <Button onClick={check} disabled={!text || busy}>Validar</Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button variant="outlined" onClick={() => create(false)} disabled={!text || busy}>
            Criar
          </Button>
          <Button variant="contained" onClick={() => create(true)} disabled={!text || busy}>
            Criar e executar
          </Button>
        </Box>
      </Panel>
    </>
  );
}
