"""Computational and communication cost study (synthetic sites).

    python -m experiments.benchmark_costs --output docs/arquitetura/custos-dados.json

Varies one factor at a time around a base scenario - number of sites, number
of features, participants per site - and runs the full pipeline (ComBat,
federated selection, Cox and horizon-logistic training, internal evaluation).
Measures rounds, bytes per phase, the largest message, wall-clock time per
phase and peak memory. The base scenario is also run over loopback HTTP to
show the transport overhead.

SYNTHETIC DATA: these are properties of the software (how costs scale), not
scientific results. Times are from one machine; in a real deployment each
round also costs at least one network round trip.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

from coordinator.transport import Direction, InProcessTransport
from experiments.local_federation import run_over_http
from experiments.simulation import run_simulation
from shared.config import ExperimentConfig
from shared.logging import configure_logging

BASE = {"sites": 3, "features": 15, "participants": 300}
GRID = {"sites": [2, 3, 5, 8], "features": [15, 50, 100], "participants": [100, 300, 1000]}


def scenario(sites: int, features: int, participants: int) -> ExperimentConfig:
    names = ["clin_age"] + [f"feat_{i:03d}" for i in range(features)]
    clients = []
    for k in range(sites):
        clients.append({"client_id": f"client-{k + 1:02d}", "data_source": {
            "kind": "synthetic", "n_participants": participants, "feature_names": names,
            "log_hazard_effects": {"feat_000": 0.8, "feat_001": 0.5, "clin_age": 0.3},
            "measurement_noise_sd": {"feat_002": 2.0},
            "derived_features": {"feat_003": {"source": "feat_000", "noise_sd": 0.15}},
            "scanner_scale": [1.0, 1.6, 0.7, 1.3, 0.8, 1.2, 0.9, 1.5][k % 8],
            "biological_features": {"clin_age": 0.2 * k},
        }})
    optimizer = {"ridge": 1.0, "max_iterations": 25, "tolerance": 1e-10}
    strategy = {"reliability": {"aggregation": "minimum", "min_value": 0.75},
                "relevance": {"aggregation": "stratified_score"},
                "agreement": {"method": "i_squared", "min_value": 0.5},
                "ranking": {"kind": "sequential"}, "max_abs_correlation": 0.8, "n_features": 5}
    return ExperimentConfig.model_validate({
        "name": f"cost-k{sites}-p{features}-n{participants}", "seed": 7,
        "data_origin": "synthetic",
        "privacy": {"min_cohort_size": 10, "min_clients_per_aggregate": 2, "min_events": 5},
        "expected_clients": [c["client_id"] for c in clients],
        "data_split": {"test_fraction": 0.3, "seed": 7},
        "harmonization": {"method": "combat", "covariates": ["clin_age"],
                          "empirical_bayes": True, "eb_tolerance": 1e-4,
                          "release_site_summaries": False},
        "feature_selection": {
            "reliability_column": "min_icc",
            "redundancy": {"correlation": "spearman", "max_candidates": 20},
            "leave_one_site_out": True,
            "strategies": [{"name": "sequential", **strategy},
                           {"name": "relevance_only", **strategy, "reliability": None,
                            "agreement": None, "max_abs_correlation": None}],
        },
        "training": {"horizons_months": [12, 24, 36], "models": [
            {"model": {"kind": "cox", "name": "cox", "optimizer": optimizer},
             "features": {"from_strategy": "sequential"}},
            {"model": {"kind": "horizon_logistic", "name": "logistic", "censoring": "ipcw",
                       "intercept": "site_specific", "optimizer": optimizer},
             "features": {"from_strategy": "sequential"}},
        ]},
        "evaluation": {"partition": "test", "calibration_bins": 10,
                       "decision_thresholds": [0.1, 0.2, 0.3], "site_baseline_variant": False,
                       "local_models": True, "release_site_metrics": False,
                       "external_validation": "none"},
        "simulation": {"clients": clients},
    })


def measure(outcome, records, wall_s: float) -> dict:
    by_phase = defaultdict(lambda: {"rounds": 0, "bytes": 0})
    for round_ in outcome.rounds:
        phase = by_phase[round_.phase.value]
        phase["rounds"] += 1
        phase["bytes"] += sum(c.bytes_sent + c.bytes_received for c in round_.clients)
    upward = [r.n_bytes for r in records if r.direction is Direction.TO_COORDINATOR]
    return {
        "completed": outcome.completed,
        "rounds": len(outcome.rounds),
        "total_bytes": outcome.total_bytes,
        "largest_message_bytes": max(upward) if upward else 0,
        "phases": dict(by_phase),
        "time_s": {"total": round(wall_s, 3),
                   "harmonization": round(outcome.harmonization_s, 3),
                   "training": round(outcome.training_s, 3),
                   "evaluation": round(outcome.evaluation_s, 3)},
    }


def run_grid() -> dict:
    rows = []
    seen = set()
    warmup = scenario(2, 15, 100)  # imports and first-call costs are not measured
    run_simulation(warmup.name, warmup)
    for factor, values in GRID.items():
        for value in values:
            params = {**BASE, factor: value}
            key = tuple(params.values())
            if key in seen:
                continue
            seen.add(key)
            config = scenario(**params)
            transport = InProcessTransport()
            started = time.perf_counter()
            outcome = run_simulation(config.name, config, transport=transport)
            row = {"factor": factor, **params,
                   **measure(outcome, transport.log.records, time.perf_counter() - started)}
            rows.append(row)
            print(f"{config.name}: {row['rounds']} rounds, {row['total_bytes']} bytes, "
                  f"{row['time_s']['total']} s", file=sys.stderr)
    base = scenario(**BASE)
    started = time.perf_counter()
    outcome, hub, _ = run_over_http(base.name, base)
    http = {**BASE, **measure(outcome, hub.log.records, time.perf_counter() - started)}
    return {
        "notice": "SYNTHETIC DATA - software cost properties, not scientific results.",
        "machine": {"platform": platform.platform(), "python": platform.python_version()},
        "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "base": BASE, "simulation": rows, "loopback_http": http,
    }


FACTOR_LABELS = {"sites": "Sites", "features": "Atributos", "participants": "Participantes/site"}
PHASES = ("harmonization", "feature_selection", "training", "evaluation")


def _kb(value: float) -> str:
    return f"{value / 1024:,.1f}".replace(",", " ")


def render_markdown(result: dict) -> str:
    lines = [
        "# Custos computacionais e de comunicação",
        "",
        "> Gerado por `python -m experiments.benchmark_costs`. **Dados sintéticos**: medem",
        "> como os custos do software crescem, não são resultados científicos. Tempos de uma",
        f"> única máquina ({result['machine']['platform']}, Python {result['machine']['python']}).",
        "",
        "Cenário base: {sites} sites, {features} atributos de imagem (+ 1 covariável clínica),"
        " {participants} participantes por site.".format(**result["base"]),
        "Pipeline completo: ComBat federado, seleção federada (2 estratégias, estabilidade sem",
        "cada site), Cox e logística por horizonte (12/24/36 meses, intercepto por site),",
        "avaliação interna dos modelos federados e locais.",
        "",
    ]
    for factor in ("sites", "features", "participants"):
        rows = sorted((r for r in result["simulation"]
                       if all(r[k] == result["base"][k] for k in result["base"] if k != factor)),
                      key=lambda r: r[factor])
        lines += [
            f"## Variando: {FACTOR_LABELS[factor].lower()}",
            "",
            f"| {FACTOR_LABELS[factor]} | Rodadas | Tráfego total (KB) | Harmonização (KB) | "
            "Seleção (KB) | Treinamento (KB) | Avaliação (KB) | Maior mensagem (KB) | Tempo (s) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            phase_kb = [_kb(r["phases"].get(p, {}).get("bytes", 0)) for p in PHASES]
            lines.append(
                f"| {r[factor]} | {r['rounds']} | {_kb(r['total_bytes'])} | "
                + " | ".join(phase_kb)
                + f" | {_kb(r['largest_message_bytes'])} | {r['time_s']['total']:.2f} |")
        lines.append("")
    base = next(r for r in result["simulation"]
                if all(r[k] == result["base"][k] for k in result["base"]))
    http = result["loopback_http"]
    per_round = (http["time_s"]["total"] - base["time_s"]["total"]) / max(http["rounds"], 1)
    lines += [
        "## Transporte HTTP (rede local da máquina)",
        "",
        "| Modo | Rodadas | Tráfego (KB) | Tempo total (s) | Treinamento (s) | Avaliação (s) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Em processo | {base['rounds']} | {_kb(base['total_bytes'])} | "
        f"{base['time_s']['total']:.2f} | {base['time_s']['training']:.2f} | "
        f"{base['time_s']['evaluation']:.2f} |",
        f"| HTTP, sites como agentes | {http['rounds']} | {_kb(http['total_bytes'])} | "
        f"{http['time_s']['total']:.2f} | {http['time_s']['training']:.2f} | "
        f"{http['time_s']['evaluation']:.2f} |",
        "",
        f"Sobrecarga de transporte medida: ~{per_round * 1000:.0f} ms por rodada. Entre "
        "instituições, cada rodada custa também pelo menos um tempo de ida e volta da rede;",
        f"com {http['rounds']} rodadas e 100 ms de latência, isso acrescenta "
        f"~{http['rounds'] * 0.1:.0f} s.",
        "",
        "## Leitura",
        "",
        "- **O número de rodadas quase não depende de sites, atributos nem participantes**:",
        "  é fixado pelo protocolo (4 de harmonização, até 4 de seleção, uma por iteração de",
        "  Newton e por risco basal/intercepto, uma por modelo e variante na avaliação); só o",
        "  número de iterações de Newton varia um pouco com os dados. O treinamento domina.",
        "- **O tráfego cresce linearmente com o número de sites** (cada site envia as",
        "  mesmas mensagens).",
        "- **Com o número de atributos**, cresce a harmonização e a relevância (vetores de",
        "  tamanho p) e a covariância dos candidatos (q(q+1)/2, com q limitado a 200 pela",
        "  configuração); a maior mensagem continua pequena.",
        "- **O número de participantes quase não muda o tráfego**: só estatísticas agregadas",
        "  trafegam. A avaliação cresce um pouco porque, com mais participantes, menos faixas",
        "  de calibração precisam ser fundidas. Ele aumenta o tempo de cálculo em cada site.",
        f"- Pico de memória do processo de simulação: {result['peak_memory_bytes'] / 1e6:.0f} MB",
        "  (todos os sites no mesmo processo; num site real é só o dele).",
        "- A validação externa deixando um site de fora (desligada aqui) multiplica as",
        "  rodadas de seleção e de treinamento pelo número de sites.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m experiments.benchmark_costs")
    parser.add_argument("--output", required=True, help="raw results (JSON)")
    parser.add_argument("--markdown", help="also write the report (Markdown)")
    parser.add_argument("--from-json", help="only render the report from earlier results")
    args = parser.parse_args(argv)
    configure_logging("ERROR")
    if args.from_json:
        result = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    else:
        result = run_grid()
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(render_markdown(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
