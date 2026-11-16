# Etapa 2 — Pipeline radiômico local

Estado: **implementada, aguardando revisão**. Confiabilidade, relevância,
redundância, seleção, harmonização e modelagem continuam sem implementação.

> O pipeline foi testado apenas com **fantomas sintéticos** (texturas aleatórias
> com máscaras geométricas). Ele ainda não foi executado sobre ressonâncias
> reais; nenhum número produzido até aqui tem significado científico.

## 1. O que a etapa entrega

```
manifesto do site ──> validação do par imagem/máscara ──> PyRadiomics ──> arquivos locais
   (CSV)                 (erros bloqueiam)                 (config versionada)
```

Tudo roda dentro do cliente (`client/imaging/`). Nenhum módulo desta etapa
transmite nada: a única ponte para a federação é
`client/feature_table.py`, que carrega a tabela extraída como `LocalDataset`,
e esse dado continua saindo do site só pelos payloads autorizados da Etapa 1.

| Módulo | Função |
|---|---|
| `client/imaging/manifest.py` | Associa participante, imagem, máscara, ROI e seguimento |
| `client/imaging/validation.py` | Verifica geometria, orientação, valores inválidos e a ROI |
| `client/imaging/extraction.py` | Configuração versionada e extrator PyRadiomics |
| `client/imaging/pipeline.py` | Executa o manifesto e grava resultados e proveniência |
| `client/imaging/__main__.py` | CLI `check` e `extract` |
| `client/feature_table.py` | Fonte de dados local para as rodadas federadas |

## 2. Manifesto

Uma linha por (participante, ROI):

```
participant_id,roi,image_path,mask_path,mask_label,time_months,event
```

- Caminhos relativos são resolvidos a partir da pasta do manifesto.
- `mask_label` (padrão 1) permite usar uma segmentação com vários rótulos
  (por exemplo `aparc+aseg`) para várias ROIs.
- `time_months` e `event` (1 = converteu; 0 = censurado em `time_months`) são
  opcionais, mas vêm juntos. **Seguimento em branco significa "desconhecido",
  nunca "não converteu"**: o participante é contado como sem seguimento e
  excluído do conjunto prognóstico.
- É erro: par (participante, ROI) duplicado, mais de uma imagem por
  participante, seguimento inconsistente entre linhas, tempo não positivo,
  coluna desconhecida.

Hipótese: uma imagem por participante (a de linha de base).

## 3. Validação

| Tipo | Verificação |
|---|---|
| Erro | arquivo ausente, ilegível ou não NIfTI; volume que não é 3D escalar |
| Erro | tamanho, espaçamento, origem ou direção diferentes entre imagem e máscara (acima de `geometry_tolerance`) |
| Erro | NaN ou Inf na imagem ou na máscara |
| Erro | máscara com valores não inteiros; rótulo ausente; ROI menor que `min_roi_voxels`; intensidade constante na ROI |
| Aviso | orientação diferente da esperada; espaçamento anisotrópico; ROI com mais de um componente; ROI tocando a borda; qform e sform divergentes |

Decisões e limitações:

- **Divergência de geometria é erro; a máscara nunca é reamostrada** de forma
  silenciosa (`correctMask` fica desligado no PyRadiomics).
- **O leitor NIfTI do ITK troca NaN/Inf por zero sem avisar.** Como o
  PyRadiomics lê pelo SimpleITK, esses voxels entrariam na extração como
  intensidade 0. Por isso a verificação é feita no arquivo bruto, com nibabel.
- A geometria considerada é a que o ITK interpreta, que é a usada na extração.
- A validação não sabe se a máscara delimita a anatomia correta nem se imagem
  e máscara são da mesma pessoa; isso depende do manifesto.
- Os limiares são configuração, não constantes validadas.

## 4. Extração

A configuração (`experiments/configs/extraction/t1_original_v1.yaml`) declara
**todos** os parâmetros do PyRadiomics que afetam valores; faltar um deles ou
incluir um desconhecido é erro de validação. Ela é identificada por
`config_id` e por um hash do conteúdo, gravados com cada resultado.

Configuração `t1-original-v1`:

| Item | Valor |
|---|---|
| Tipos de imagem | Original (LoG e Wavelet são configuráveis, com `sigma` explícito) |
| Classes | forma, primeira ordem, GLCM, GLRLM, GLSZM, GLDM, NGTDM — 107 atributos por ROI |
| Normalização | z-score da imagem inteira, escala 100 |
| Discretização | largura de bin fixa, 25 |
| Reamostragem | nenhuma |
| Texturas | 3D, distância 1, GLCM simétrica |

Esses valores repetem o protocolo OASIS-3 já existente no repositório
(`OASIS3-AD-RADIOMICS-v1.0`). São um ponto de partida, não uma escolha
validada.

Proteções: o número de atributos por ROI é conferido contra
`expected_features_per_roi`; todos os casos precisam produzir o mesmo esquema;
valor não finito faz o caso falhar.

### IBSI

O PyRadiomics segue as definições IBSI com desvios documentados, que este
módulo não corrige: bordas dos bins em múltiplos de `binWidth` (a IBSI ancora
no mínimo da ROI), alinhamento da reamostragem pelo voxel de origem, curtose
não excedente, e atributos marcados como obsoletos não são extraídos.

### Hipóteses e limitações

- T1 não tem escala absoluta de intensidade. O z-score é calculado sobre a
  imagem inteira, então depende do campo de visão e da remoção de crânio.
- O pipeline **não faz pré-processamento** (correção de viés, registro,
  remoção de crânio, reamostragem). O campo `preprocessing` descreve o que se
  espera das entradas, e isso precisa ser igual em todos os sites.
- Os valores dependem das versões das bibliotecas, gravadas na proveniência.

## 5. Saídas locais

| Arquivo | Conteúdo |
|---|---|
| `features.csv` | Uma linha por participante com todas as ROIs extraídas; colunas `<roi>__<atributo>` |
| `cases.csv` | Uma linha por (participante, ROI): status, erros, avisos, tamanho da ROI, geometria, hashes |
| `provenance.json` | Configuração, versões, hash do manifesto, esquema, contagens e SHA-256 de `features.csv` |
| `run_info.json` | Tempos de validação e extração, memória, plataforma, data |

Os três primeiros são determinísticos: duas execuções com as mesmas entradas,
configuração e ambiente produzem arquivos idênticos byte a byte (testado).
Participante com alguma ROI inválida fica fora de `features.csv`; não há
imputação. Esses arquivos contêm dados individuais e devem permanecer no site.

Ao carregar a tabela para a federação, o hash é conferido contra a
proveniência, para que uma tabela alterada não seja usada sem aviso.

## 6. Versão do PyRadiomics

O PyRadiomics 3.0.1, usado pelo pipeline OASIS-3 existente, não compila em
Python 3.12, e o pacote 3.1.0 do PyPI está incompleto. O framework usa o
commit exato do tag `v3.1.0` do repositório oficial, fixado em `pyproject.toml`
e `uv.lock`.

Comparação feita: 12 ROIs de fantomas, configuração `t1-original-v1`,
PyRadiomics 3.0.1 em Python 3.11 contra o commit fixado em Python 3.12 (mesmo
NumPy e SimpleITK). Os 107 atributos saíram **idênticos**. Isso não substitui
uma conferência em imagens reais antes de combinar tabelas dos dois pipelines.

## 7. Dependências novas (grupo `imaging`)

| Pacote | Função |
|---|---|
| pyradiomics (commit do tag v3.1.0) | extração de atributos |
| SimpleITK | leitura de imagens e geometria, como o PyRadiomics as vê |
| nibabel | leitura do arquivo bruto (valores não finitos, qform/sform) |
| pandas | tabelas locais |

## 8. Como executar

```bash
uv sync --extra backend --extra dev --extra imaging
```

```bash
uv run python -m client.imaging check --manifest site/manifest.csv --config experiments/configs/extraction/t1_original_v1.yaml
```

```bash
uv run python -m client.imaging extract --manifest site/manifest.csv --config experiments/configs/extraction/t1_original_v1.yaml --output-dir var/sites/client-01
```

Para usar a tabela em uma simulação federada, a fonte de dados do cliente na
configuração do experimento passa a ser:

```yaml
data_source:
  kind: feature_table
  features_path: var/sites/client-01/features.csv
  manifest_path: site/manifest.csv
  data_origin: real   # ou synthetic, para fantomas
```

Fontes `real` e `synthetic` não podem ser misturadas no mesmo experimento.

## 9. Decisões que pedem revisão

1. **Parâmetros de extração.** `t1-original-v1` copia o protocolo OASIS-3
   existente. Confirmar se é o protocolo desejado para o estudo MCI → DA, e se
   LoG/Wavelet entram.
2. **Pré-processamento fora do pipeline.** Hoje as entradas devem chegar já
   pré-processadas (por exemplo, T1 conformada do FreeSurfer). Se o framework
   deve fazer correção de viés ou reamostragem, isso vira um módulo próprio.
3. **Casos completos.** Participante sem alguma ROI fica fora da tabela.
4. **Uma imagem por participante.** Imagens longitudinais exigiriam uma chave
   de sessão no manifesto.
5. **Formato do manifesto.** Os dados OASIS-3 existentes estão em `.mgz` e em
   tabelas próprias; falta um conversor para o manifesto e para NIfTI, e a
   regra que define tempo e evento de conversão a partir dos dados clínicos.
6. **Consistência entre sites.** O hash da configuração de extração ainda não
   é verificado pelo coordenador; proponho exigir o mesmo hash de todos os
   clientes na Etapa 4.
