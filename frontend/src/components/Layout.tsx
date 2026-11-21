import KeyIcon from "@mui/icons-material/Key";
import {
  AppBar, Box, Button, Container, Dialog, DialogActions, DialogContent, DialogTitle,
  IconButton, TextField, Toolbar, Tooltip, Typography,
} from "@mui/material";
import { type ReactNode, useState } from "react";
import { Link as RouterLink } from "react-router-dom";

import { getToken, setToken } from "../api/client";

function TokenDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [value, setValue] = useState(getToken());
  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle>Token de acesso à API</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Necessário apenas quando o backend define FEDRAD_API_TOKEN. Fica guardado neste
          navegador.
        </Typography>
        <TextField autoFocus fullWidth type="password" label="Token" value={value}
          onChange={(e) => setValue(e.target.value)} />
      </DialogContent>
      <DialogActions>
        <Button onClick={() => { setToken(""); setValue(""); onClose(); }}>Remover</Button>
        <Button variant="contained" onClick={() => { setToken(value.trim()); onClose(); }}>
          Salvar
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const [tokenOpen, setTokenOpen] = useState(false);
  return (
    <Box sx={{ minHeight: "100vh" }}>
      <AppBar position="sticky" color="inherit" elevation={0}
        sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Toolbar variant="dense" sx={{ gap: 1 }}>
          <Typography component={RouterLink} to="/" variant="subtitle1"
            sx={{ fontWeight: 700, color: "inherit", textDecoration: "none", mr: 2 }}>
            Radiômica federada · MCI → DA
          </Typography>
          <Button component={RouterLink} to="/">Experimentos</Button>
          <Button component={RouterLink} to="/new">Novo experimento</Button>
          <Box sx={{ flexGrow: 1 }} />
          <Tooltip title="Token de acesso">
            <IconButton aria-label="token de acesso" onClick={() => setTokenOpen(true)}>
              <KeyIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>
      <Container maxWidth="xl" sx={{ py: 3 }}>{children}</Container>
      {tokenOpen && <TokenDialog open onClose={() => setTokenOpen(false)} />}
    </Box>
  );
}
