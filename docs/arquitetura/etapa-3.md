# Etapa 3 — Confiabilidade e seleção local

Estado: **implementada, aguardando revisão**. Tudo nesta etapa roda dentro de
um único site e grava arquivos locais; não há mensagem nova no protocolo
federado. A seleção federada é a Etapa 4.

> Os métodos foram verificados com respostas conhecidas da literatura, com
> dados simulados de estrutura plantada e com fantomas. **Nada foi executado
> sobre dados reais**; os números dos testes não são resultados científicos.

## 1. O que a etapa entrega

| Módulo | Função |
|---|---|
| `client/reliability/icc.py` | ICC(1,1), ICC(2,1) e ICC(3,1) com intervalos de confiança |
| `client/reliability/repeated.py` | Monta observações repetidas a partir de várias tabelas de atributos |
| `client/reliability/study.py` | Estudo de confiabilidade: gera as condições, estima ICC por família |
| `client/relevance.py` | Relevância prognóstica univariada com censura |
| `client/redundancy.py` | Correlação absoluta entre atributos e grupos redundantes |
| `client/selection/selectors.py` | Seletor modular e três métodos de referência |
| `client/selection/study.py` | Comparação entre seletores e estabilidade por bootstrap |
| `evaluation/stability.py` | Índice de estabilidade de Nogueira, Jaccard, frequência de seleção |

## 2. Confiabilidade

**A confiabilidade só é estimada a partir de observações repetidas ou
alternativas dos mesmos participantes.** Com uma única observação, ou com
menos participantes completos que `min_subjects`, as funções levantam
`ReliabilityUnavailable` e nenhum arquivo de confiabilidade é gravado.

Um estudo é uma lista de famílias. Cada família compara a extração de base com
condições alternativas de um mesmo tipo:

| Tipo de condição | O que varia | O que mede |
|---|---|---|
| `mask_morphology` | erosão ou dilatação da ROI em N voxels | robustez a variações de segmentação (aproximação controlada) |
| `settings_override` | parâmetros do PyRadiomics (por exemplo, largura de bin) | robustez a escolhas de processamento |
| `alternative_manifest` | outro manifesto para os mesmos participantes | concordância entre segmentações alternativas (segundo avaliador, outro algoritmo) |

Saídas locais: `reliability.csv` (atributo × família: ICC, intervalo, número de
participantes e de condições) e `reliability_summary.csv` (por atributo, o
menor ICC e o menor limite inferior entre as famílias).

O ICC foi conferido contra a tabela de Shrout e Fleiss (1979): as três formas e
seus intervalos de confiança são reproduzidos.

### Hipóteses e limitações

- **Robustez a perturbações não é confiabilidade teste-reteste.** Não captura
  variabilidade de equipamento, posicionamento ou biológica; só exames
  repetidos capturam. Por isso os resultados ficam separados por família e
  nunca são combinados em um único ICC.
- Erosão e dilatação são isotrópicas e alteram o volume de forma sistemática;
  a discordância real entre segmentações não é assim. Atributos de forma
  parecerão pouco confiáveis sob `icc2` por construção.
- O ICC supõe dados completos e balanceados, efeitos aditivos e resíduos
  aproximadamente normais. Atributos assimétricos violam isso e nenhuma
  transformação é aplicada.
- O ICC depende da variância entre participantes: não é transportável entre
  populações.
- As condições são escolhidas pelo analista, então o ICC descreve concordância
  entre essas condições específicas.
- Participantes em que alguma condição falha saem da família (casos completos).
- Atributo sem variação tem ICC indefinido (NaN).

## 3. Relevância prognóstica

Os dois métodos usam tempo de seguimento e indicador de evento. Participantes
censurados contribuem enquanto estão em risco e **nunca são tratados como não
conversores**.

| Método | Definição | Pontuação |
|---|---|---|
| `cox_score` | teste de escore de um modelo de Cox univariado em β = 0 (forma fechada, sem ajuste iterativo, invariante à escala) | \|z\| |
| `concordance` | índice C de Harrell do atributo isolado | 2·\|C − 0,5\| |

Com menos eventos que `min_events`, a relevância não é calculada.

Verificações: o teste de escore coincide com a definição calculada por força
bruta (inclusive com empates) e com o teste log-rank para um atributo binário.

### Hipóteses e limitações

- Univariada: sem ajuste por idade, sexo, site ou outros atributos.
- `cox_score` supõe riscos proporcionais e efeito log-linear; empates pelo
  método de Breslow. O C de Harrell é enviesado sob censura intensa.
- Supõe censura não informativa.
- Os valores-p são por atributo, sem correção para multiplicidade.
- Calcular relevância nos mesmos participantes usados depois para avaliar um
  modelo vaza informação; deve ser feita só na partição de treino.

## 4. Redundância

Matriz de correlação absoluta (Spearman ou Pearson) entre atributos, e grupos
redundantes definidos como componentes conexos do grafo em que arestas ligam
atributos com correlação acima de um limiar.

Limitações: só relações par a par (não detecta combinação linear de vários
atributos); correlações ruidosas com poucos participantes; ligação simples faz
cadeias de atributos correlacionados se fundirem em grupos grandes.

## 5. Seleção local

### Seletor modular (`component`)

Aplica as três dimensões disponíveis localmente como etapas sucessivas, cada
uma configurável e desligável:

```
filtro de confiabilidade ──> ordenação por relevância ──> poda de redundância ──> k primeiros
```

As dimensões não são misturadas em uma pontuação ponderada, de modo que
**nenhum peso precisa ser inventado**. Todo limiar é campo obrigatório da
configuração, sem valor padrão no código. A quarta dimensão, concordância
entre participantes, só existe no cenário federado (Etapa 4).

Se o seletor está configurado para filtrar por confiabilidade e não há
estimativas, ele **falha com erro**; desligar o filtro exige `reliability:
null` explícito. Atributo sem estimativa é excluído como
`reliability_unknown`.

### Métodos de referência

| Método | Família | Descrição |
|---|---|---|
| `correlation_filter` | correlação | ordenação univariada com poda gulosa por correlação |
| `lasso_cox` | regularização | regressão de Cox com penalidade L1; usa a penalidade mais forte do caminho que mantém no máximo `n_features` atributos |
| `graph_representatives` | grafo | um representante (o mais relevante) por componente conexo do grafo de correlação |

### Comparação

`client/selection/study.py` executa os seletores configurados e grava:

- `selection.json`: seleção ordenada de cada método, contagem de exclusões por
  motivo, sobreposição de Jaccard entre métodos, estabilidade por bootstrap;
- `selected_features.csv`: por método, posição, atributo, pontuação e
  frequência de seleção no bootstrap.

A comparação diz **quais atributos** cada método escolhe e **quão estável** é a
escolha. Não diz nada sobre desempenho preditivo, que é da Etapa 6.

### Hipóteses e limitações

- A seleção usa o desfecho; selecionar e avaliar na mesma coorte é otimista.
- `n_features` é entrada, não é estimado dos dados.
- Limiares rígidos tornam o resultado sensível a pequenas diferenças entre
  atributos correlacionados.
- No `lasso_cox` a penalidade não é escolhida por validação cruzada, para que
  a comparação seja em tamanho igual; entre atributos correlacionados o LASSO
  escolhe um de forma quase arbitrária.
- O índice de Nogueira trata atributos como rótulos: trocar um atributo por
  outro altamente correlacionado conta como discordância total.
- O bootstrap com reposição cria duplicatas e subestima a instabilidade em
  coortes pequenas; reamostras com poucos eventos são puladas e contadas.

## 6. Configurações

- `experiments/configs/reliability/perturbation_v1.yaml`: ICC(2,1), intervalo
  de 95%, mínimo de 10 participantes, famílias de morfologia (erosão e
  dilatação de 1 voxel) e de discretização (largura de bin 20 e 30).
- `experiments/configs/selection/local_v1.yaml`: seletor modular e os três
  métodos de referência, 10 atributos, mínimo de 20 eventos, 200 reamostras.

**Todos esses números são pontos de partida a justificar experimentalmente**,
não recomendações: limiar de ICC 0,75, correlação máxima 0,8, 10 atributos e
20 eventos mínimos precisam de análise de sensibilidade em dados reais.

## 7. Como executar

```bash
uv sync --extra backend --extra dev --extra imaging --extra science
```

```bash
uv run python -m client.reliability --manifest site/manifest.csv --extraction-config experiments/configs/extraction/t1_original_v1.yaml --study-config experiments/configs/reliability/perturbation_v1.yaml --output-dir var/sites/client-01/reliability
```

```bash
uv run python -m client.selection --features var/sites/client-01/features.csv --manifest site/manifest.csv --data-origin real --config experiments/configs/selection/local_v1.yaml --reliability var/sites/client-01/reliability/reliability_summary.csv --output-dir var/sites/client-01/selection
```

## 8. Decisões que pedem revisão

1. **Fonte de confiabilidade.** Sem exames repetidos nem segunda segmentação
   informados, a configuração entregue usa só perturbações controladas. Se
   existirem segmentações alternativas ou reteste, entram como uma família
   `alternative_manifest`.
2. **Forma do ICC.** A configuração usa ICC(2,1), concordância absoluta. Para
   perturbações que deslocam valores de forma sistemática, ICC(3,1)
   (consistência) responde a outra pergunta; a escolha deve vir do protocolo.
3. **Resumo entre famílias.** O resumo usa o pior ICC entre as famílias, o que
   é conservador e elimina atributos de forma sob erosão/dilatação.
4. **Etapas em vez de pesos.** O seletor modular filtra e ordena em sequência.
   Uma pontuação ponderada entre dimensões pode ser acrescentada como outro
   seletor, para comparação, quando houver base experimental para os pesos.
5. **Relevância univariada sem covariáveis.** Ajuste por idade e sexo, se
   desejado, entra como novo método de relevância.
6. **Limiares.** Ver seção 6.
