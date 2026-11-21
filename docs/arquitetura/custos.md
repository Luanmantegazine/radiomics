# Custos computacionais e de comunicação

> Gerado por `python -m experiments.benchmark_costs`. **Dados sintéticos**: medem
> como os custos do software crescem, não são resultados científicos. Tempos de uma
> única máquina (Linux-6.18.44-fc-v726-x86_64-with-glibc2.39, Python 3.12.3).

Cenário base: 3 sites, 15 atributos de imagem (+ 1 covariável clínica), 300 participantes por site.
Pipeline completo: ComBat federado, seleção federada (2 estratégias, estabilidade sem
cada site), Cox e logística por horizonte (12/24/36 meses, intercepto por site),
avaliação interna dos modelos federados e locais.

## Variando: sites

| Sites | Rodadas | Tráfego total (KB) | Harmonização (KB) | Seleção (KB) | Treinamento (KB) | Avaliação (KB) | Maior mensagem (KB) | Tempo (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 40 | 158.9 | 15.6 | 11.0 | 95.9 | 36.3 | 2.4 | 0.32 |
| 3 | 39 | 232.9 | 23.5 | 16.5 | 138.4 | 54.6 | 2.5 | 0.51 |
| 5 | 39 | 388.5 | 39.1 | 27.4 | 230.2 | 91.8 | 2.6 | 0.88 |
| 8 | 39 | 622.5 | 62.6 | 43.9 | 369.4 | 146.6 | 2.6 | 1.31 |

## Variando: atributos

| Atributos | Rodadas | Tráfego total (KB) | Harmonização (KB) | Seleção (KB) | Treinamento (KB) | Avaliação (KB) | Maior mensagem (KB) | Tempo (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 39 | 232.9 | 23.5 | 16.5 | 138.4 | 54.6 | 2.5 | 0.51 |
| 50 | 39 | 273.3 | 51.1 | 29.2 | 138.6 | 54.4 | 3.9 | 0.42 |
| 100 | 39 | 325.8 | 90.6 | 41.8 | 138.8 | 54.6 | 7.1 | 0.50 |

## Variando: participantes/site

| Participantes/site | Rodadas | Tráfego total (KB) | Harmonização (KB) | Seleção (KB) | Treinamento (KB) | Avaliação (KB) | Maior mensagem (KB) | Tempo (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 40 | 229.7 | 23.4 | 16.0 | 143.9 | 46.5 | 1.7 | 0.20 |
| 300 | 39 | 232.9 | 23.5 | 16.5 | 138.4 | 54.6 | 2.5 | 0.51 |
| 1000 | 39 | 242.4 | 23.5 | 17.1 | 139.0 | 62.8 | 3.1 | 1.10 |

## Transporte HTTP (rede local da máquina)

| Modo | Rodadas | Tráfego (KB) | Tempo total (s) | Treinamento (s) | Avaliação (s) |
|---|---:|---:|---:|---:|---:|
| Em processo | 39 | 232.9 | 0.51 | 0.26 | 0.18 |
| HTTP, sites como agentes | 39 | 232.9 | 1.37 | 0.55 | 0.26 |

Sobrecarga de transporte medida: ~22 ms por rodada. Entre instituições, cada rodada custa também pelo menos um tempo de ida e volta da rede;
com 39 rodadas e 100 ms de latência, isso acrescenta ~4 s.

## Leitura

- **O número de rodadas quase não depende de sites, atributos nem participantes**:
  é fixado pelo protocolo (4 de harmonização, até 4 de seleção, uma por iteração de
  Newton e por risco basal/intercepto, uma por modelo e variante na avaliação); só o
  número de iterações de Newton varia um pouco com os dados. O treinamento domina.
- **O tráfego cresce linearmente com o número de sites** (cada site envia as
  mesmas mensagens).
- **Com o número de atributos**, cresce a harmonização e a relevância (vetores de
  tamanho p) e a covariância dos candidatos (q(q+1)/2, com q limitado a 200 pela
  configuração); a maior mensagem continua pequena.
- **O número de participantes quase não muda o tráfego**: só estatísticas agregadas
  trafegam. A avaliação cresce um pouco porque, com mais participantes, menos faixas
  de calibração precisam ser fundidas. Ele aumenta o tempo de cálculo em cada site.
- Pico de memória do processo de simulação: 206 MB
  (todos os sites no mesmo processo; num site real é só o dele).
- A validação externa deixando um site de fora (desligada aqui) multiplica as
  rodadas de seleção e de treinamento pelo número de sites.
