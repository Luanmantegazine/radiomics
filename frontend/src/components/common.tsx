import { Alert, Box, Card, CardContent, Chip, CircularProgress, Typography } from "@mui/material";
import type { ReactNode } from "react";

export function SyntheticBanner({ origin }: { origin: string }) {
  if (origin !== "synthetic") return null;
  return (
    <Alert severity="warning" variant="outlined" sx={{ mb: 2 }} role="status">
      <strong>Dados sintéticos.</strong> Estes resultados validam apenas o software e não têm
      significado científico.
    </Alert>
  );
}

const STATUS_COLORS: Record<string, "default" | "info" | "success" | "error" | "warning"> = {
  created: "default",
  queued: "info",
  running: "info",
  completed: "success",
  failed: "error",
  registered: "success",
  rejected: "error",
  trained: "success",
  refused: "warning",
};

export function StatusChip({ status }: { status: string }) {
  return <Chip size="small" label={status} color={STATUS_COLORS[status] ?? "default"} />;
}

export function OriginChip({ origin }: { origin: string }) {
  return (
    <Chip size="small" variant="outlined" label={origin}
      color={origin === "synthetic" ? "warning" : "primary"} />
  );
}

export function StatCard({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <Card variant="outlined" sx={{ height: "100%" }}>
      <CardContent sx={{ py: 1.5, "&:last-child": { pb: 1.5 } }}>
        <Typography variant="caption" color="text.secondary">{label}</Typography>
        <Typography variant="h6" sx={{ fontVariantNumeric: "tabular-nums" }}>{value}</Typography>
        {hint && <Typography variant="caption" color="text.secondary">{hint}</Typography>}
      </CardContent>
    </Card>
  );
}

export function Panel({ title, subtitle, children, actions }: {
  title: string; subtitle?: string; children: ReactNode; actions?: ReactNode;
}) {
  return (
    <Card variant="outlined" sx={{ mb: 2 }}>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "baseline", gap: 2, mb: 1, flexWrap: "wrap" }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>{title}</Typography>
          {subtitle && <Typography variant="body2" color="text.secondary">{subtitle}</Typography>}
          <Box sx={{ flexGrow: 1 }} />
          {actions}
        </Box>
        {children}
      </CardContent>
    </Card>
  );
}

export function Loading() {
  return (
    <Box sx={{ display: "flex", justifyContent: "center", p: 4 }}>
      <CircularProgress size={28} aria-label="carregando" />
    </Box>
  );
}

export function ErrorState({ message }: { message: string }) {
  return <Alert severity="error" sx={{ my: 2 }}>{message}</Alert>;
}

export function Empty({ children }: { children: ReactNode }) {
  return <Typography color="text.secondary" sx={{ py: 2 }}>{children}</Typography>;
}
