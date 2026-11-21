# Dashboard

React + TypeScript + Vite + Material UI + Plotly. Consome **apenas** a API do
backend (sob `/api`); não executa nenhum algoritmo científico no navegador.

```bash
npm ci
```

```bash
FEDRAD_API_URL=http://localhost:8000 npm run dev
```

```bash
npm test
```

```bash
npm run build
```

Os tipos de `src/api/schema.d.ts` são gerados do esquema OpenAPI do backend. Depois
de mudar a API:

```bash
cd .. && python -m backend.app.openapi frontend/openapi.json && cd frontend && npm run gen:api
```

Um teste do backend (`tests/federated/test_api.py`) falha se `openapi.json` estiver
desatualizado.
