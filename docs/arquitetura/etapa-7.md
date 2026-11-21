# Etapa 7 — Backend e frontend

Estado: **implementada, aguardando revisão**. O dashboard e a API foram
exercitados com dados sintéticos; nenhum resultado exibido tem valor
científico.

## 1. O que a etapa entrega

| Parte | Conteúdo |
|---|---|
| Persistência | migrações Alembic (`backend/migrations/`) aplicadas na inicialização; teste que falha se os modelos mudarem sem migração |
| API | respostas tipadas em todas as rotas de resultado; modelos de configuração, validação, resumo do experimento, resumo de custos, relatórios |
| Segurança | token Bearer opcional (`FEDRAD_API_TOKEN`) em todas as rotas exceto `/health` |
| Relatórios | pacote JSON completo, relatório HTML autocontido e tabelas CSV, gerados do que foi gravado |
| Dashboard | React + TypeScript + Vite + Material UI + Plotly + axios (`frontend/`) |
| Infraestrutura | imagem do dashboard (nginx, com proxy de `/api`), serviço no Docker Compose, correção da imagem do MLflow |

## 2. API consolidada

Recursos e rotas novas desta etapa:

| Rota | Função |
|---|---|
| `GET /experiments?status=&limit=&offset=` | lista com filtro e paginação |
| `GET /experiments/templates` | configurações versionadas do repositório (`experiments/configs/*.yaml`) |
| `POST /experiments/validate` | valida uma configuração sem criar nada; devolve erros por campo, hash, fases e sites |
| `GET /experiments/{id}/summary` | progresso (rodadas por fase), sites, seções de resultado disponíveis |
| `GET /results/{id}/costs/summary` | custos por fase, por site e por rodada |
| `GET /reports/{id}?format=json\|html` | relatório completo |
| `GET /reports/{id}/{tabela}.csv` | `evaluation`, `selected-features`, `models`, `costs`, `rounds` |

As rotas de seleção, harmonização e avaliação passaram a ter esquemas de
resposta tipados, os mesmos modelos que o coordenador produz. O esquema OpenAPI
completo fica em `frontend/openapi.json`. Os tipos TypeScript do dashboard são
gerados dele, e um teste falha se ele estiver desatualizado.

O backend continua **sem lógica científica**: resumos e relatórios só somam
custos e formatam o que o coordenador liberou.

### Migrações

```bash
python -m backend.app.migrations upgrade
```

```bash
python -m backend.app.migrations revision -m "descrição"
```

Um banco que já tem tabelas mas **não** tem histórico de migrações é
recusado, com mensagem explicando o que fazer. É o caso dos bancos de
desenvolvimento criados nas Etapas 1–6 com `create_all` (por exemplo,
`var/fedradiomics.db`). Para desenvolvimento, basta apagá-lo.

Verificado em SQLite (testes) e em PostgreSQL 16.

### Token de acesso

Com `FEDRAD_API_TOKEN` definido, toda rota exceto `/health` exige
`Authorization: Bearer <token>`; a comparação é em tempo constante. É uma
proteção mínima para uma instalação acessível por outras pessoas, **não** um
controle de usuários. No dashboard, o token é informado pelo ícone de chave e
guardado no navegador.

## 3. Dashboard

O dashboard consome **somente** a API do backend, sob `/api`, e não executa
nenhum cálculo científico: os gráficos apenas reorganizam números já
liberados (`src/lib/transform.ts`).

| Tela | Conteúdo |
|---|---|
| Experimentos | lista com status e origem dos dados; atualiza a cada 5 s |
| Novo experimento | escolher um modelo de configuração, editar em YAML/JSON, validar no backend, criar, executar |
| Visão geral | sites, rodadas, tráfego, duração, rodadas por fase, configuração versionada; acompanha a execução ao vivo |
| Sites e rodadas | sites pseudonimizados com versões de software; rodadas paginadas, com a participação de cada site |
| Harmonização | η² de site por atributo antes/depois; diferenças clínicas entre sites; perfil por site |
| Seleção | estratégias comparadas, estabilidade sem cada site, atributos e motivos de exclusão, Jaccard entre estratégias, relevância × heterogeneidade de todos os atributos |
| Modelos | coeficientes com IC 95% (Wald), convergência do Newton, risco basal liberado |
| Avaliação | métricas por horizonte, validação externa por site, tabela completa, calibração, curva de decisão, desempenho por site |
| Custos | tempo por fase, tráfego por rodada, por fase e por site, memória |
| Exportar | relatório HTML, pacote JSON, tabelas CSV |

Outros comportamentos:

- Todo experimento com dados sintéticos exibe um aviso no topo; os arquivos
  exportados levam o mesmo aviso.
- Cada gráfico pode ser salvo em PNG.
- As abas com gráficos carregam o Plotly (~4,7 MB) sob demanda; a lista abre
  com ~550 KB.

## 4. Verificação

| Verificação | Resultado |
|---|---|
| Backend | 241 testes, incluindo migrações, ausência de divergência entre modelos e migração, token, modelos de configuração, validação, resumo, consistência dos custos, relatórios e atualidade do OpenAPI |
| Frontend | 16 testes (transformações, cliente HTTP, páginas com respostas reais capturadas da API); `tsc` estrito e build de produção |
| PostgreSQL 16 | migração `0001` aplicada; experimento da Etapa 6 completo pela API real |
| Docker | as três imagens construídas e a pilha Compose (PostgreSQL, MLflow, backend, dashboard) executou um experimento completo pelo proxy nginx; o MLflow registrou métricas e artefatos |
| Navegador | Playwright em todas as abas, sem erros de console; as capturas levaram a quatro correções visuais |

**Defeito encontrado na infraestrutura**: a imagem do MLflow instalava o
`psycopg2`, mas o MLflow usa o `psycopg` 3 para URLs `postgresql://`, então o
servidor não iniciava no Compose. O defeito existia desde a Etapa 1 e só
apareceu agora, com o Docker disponível. A imagem agora fixa as versões do
`uv.lock` (`mlflow==3.18.0`, `psycopg[binary]==3.3.9`) e desliga a
telemetria.

Nota sobre o ambiente desta sessão, que não afeta o repositório: o Docker Hub
recusou por limite de requisições e o `ghcr.io` está bloqueado aqui. Para
construir, usei as mesmas imagens base de um espelho público e cópias dos
Dockerfiles com apenas o certificado e o proxy da sessão inseridos após cada
`FROM`. Os Dockerfiles do repositório não foram alterados por isso.

## 5. Como executar

Pilha completa:

```bash
cp infrastructure/.env.example infrastructure/.env
```

```bash
docker compose -f infrastructure/docker-compose.yml up --build
```

Dashboard em http://localhost:8080, API em http://localhost:8000/docs, MLflow
em http://localhost:5000.

Desenvolvimento local, sem Docker:

```bash
uv run uvicorn backend.app.main:app_factory --factory --port 8000
```

```bash
cd frontend && npm ci && npm run dev
```

## 6. Hipóteses e limitações

- **Sem controle de usuários**: um único token compartilhado; sem perfis nem
  registro de quem fez o quê.
- O acompanhamento ao vivo é por consulta periódica (2 s), sem WebSocket.
- O lançador ainda executa a simulação num subprocesso do backend; não há
  fila de execuções nem cancelamento.
- Os relatórios não incluem gráficos (o HTML é estático, sem JavaScript);
  os gráficos são exportados pelo dashboard em PNG.
- O dashboard não tem testes ponta a ponta automatizados no repositório; a
  verificação com navegador foi feita nesta sessão.

## 7. Decisões que pedem revisão

1. **Autenticação por token único.** Suficiente para uma instalação de
   laboratório; uma implantação entre instituições pediria usuários e
   perfis (por exemplo, OIDC institucional).
2. **Métricas por site no dashboard.** Aparecem quando o experimento as
   libera (`release_site_metrics`), o que segue a decisão da Etapa 6.
3. **Edição da configuração como texto** (YAML/JSON validado pelo backend), em
   vez de formulários. Formulários por seção são possíveis, mas duplicariam as
   regras de validação no frontend.
4. **Banco antigo recusado** em vez de migrado automaticamente.
5. **Integração contínua**: os testes não rodam em CI. Proponho adicionar um
   fluxo do GitHub Actions (pytest, testes do frontend, build das imagens) na
   Etapa 8.
