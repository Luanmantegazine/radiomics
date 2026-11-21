"""Cost study tooling (the study itself runs as a command, not as a test)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.benchmark_costs import render_markdown, scenario
from experiments.local_federation import run_over_http
from experiments.simulation import run_simulation

DATA = Path(__file__).resolve().parents[2] / "docs" / "arquitetura" / "custos-dados.json"


def test_scenarios_are_valid_and_scale_as_requested():
    config = scenario(sites=5, features=40, participants=120)
    assert len(config.expected_clients) == 5
    source = config.simulation.clients[0].data_source
    assert len(source.feature_names) == 41 and source.n_participants == 120


def test_report_renders_from_recorded_results():
    text = render_markdown(json.loads(DATA.read_text()))
    assert "Dados sintéticos" in text and "| Sites |" in text and "Transporte HTTP" in text


def test_loopback_federation_matches_the_simulation():
    config = scenario(sites=2, features=6, participants=80)
    outcome, _, results = run_over_http("exp-loopback", config)
    assert outcome.completed and all(v is True for v in results.values())
    assert outcome.training == run_simulation("exp-loopback", config).training
