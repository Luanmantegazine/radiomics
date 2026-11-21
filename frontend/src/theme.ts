import { createTheme } from "@mui/material";

export const theme = createTheme({
  palette: {
    primary: { main: "#2f6fb0" },
    secondary: { main: "#5f7487" },
    background: { default: "#f5f7fa" },
    text: { primary: "#1d2733" },
  },
  shape: { borderRadius: 8 },
  typography: { fontSize: 13 },
  components: {
    MuiTableCell: { styleOverrides: { root: { paddingTop: 6, paddingBottom: 6 } } },
    MuiCard: { defaultProps: { elevation: 0 } },
  },
});
