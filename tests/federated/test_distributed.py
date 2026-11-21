"""Distributed mode: sites connect over HTTP to the coordinator (real sockets)."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from client.agent import AgentError, SiteAgent
from client.site_config import SiteConfig
from coordinator.network.api import create_site_api
from coordinator.network.hub import SiteHub
from coordinator.network.tokens import hash_token, identify, new_token
from coordinator.transport import Direction
from experiments.run_distributed import run_distributed
from experiments.simulation import build_clients, run_simulation
from shared.config import ExperimentConfig, load_experiment_config
from shared.protocol import PROTOCOL_VERSION, TaskKind, UpdateStatus
from tests.federated.conftest import CONFIG_PATH
from tests.federated.test_harmonization import STAGE6

EXPERIMENT_ID = "exp-distributed"


def _sites(config: ExperimentConfig, tmp_path, experiment_id=EXPERIMENT_ID):
    """Site configurations (local to each site) and the coordinator's token hashes."""
    sites, hashes = [], {}
    for spec in config.simulation.clients:
        token = new_token()
        token_file = tmp_path / f"{spec.client_id}.token"
        token_file.write_text(token)
        hashes[spec.client_id] = hash_token(token)
        sites.append(SiteConfig(
            client_id=spec.client_id, experiment_id=experiment_id,
            coordinator_url="http://127.0.0.1:0", token_file=str(token_file),
            data_source=spec.data_source, seed=config.seed, poll_wait_s=1.0,
            idle_timeout_s=60.0,
        ))
    return sites, hashes


def _distributed(config, tmp_path, *, site_overrides=None, round_timeout_s=60.0,
                 experiment_id=EXPERIMENT_ID):
    """Run the coordinator here and one agent thread per site, over real HTTP."""
    sites, hashes = _sites(config, tmp_path, experiment_id=EXPERIMENT_ID)
    coordinator_config = ExperimentConfig.model_validate(
        {**config.model_dump(mode="json"), "simulation": None})
    agents, threads, results = [], [], {}

    def start_agents(port: int) -> None:
        for site in sites:
            site = site.model_copy(update={"coordinator_url": f"http://127.0.0.1:{port}"})
            behaviour = (site_overrides or {}).get(site.client_id)
            agent = SiteAgent(site)
            agents.append(agent)

            def serve(agent=agent, behaviour=behaviour):
                try:
                    results[agent.config.client_id] = (behaviour or SiteAgent.run)(agent)
                except AgentError as exc:
                    results[agent.config.client_id] = exc

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            threads.append(thread)

    outcome, hub = run_distributed(
        experiment_id, coordinator_config, hashes, host="127.0.0.1", port=0,
        registration_timeout_s=30, round_timeout_s=round_timeout_s, capture=True,
        on_listening=start_agents,
    )
    for thread in threads:
        thread.join(timeout=30)
    return outcome, hub, results


@pytest.fixture(scope="module")
def stage6_both(tmp_path_factory):
    config = load_experiment_config(STAGE6)
    distributed = _distributed(config, tmp_path_factory.mktemp("dist"))
    simulated = run_simulation(EXPERIMENT_ID, config)
    return config, distributed, simulated


def test_distributed_run_equals_the_simulation(stage6_both):
    _, (outcome, _, results), simulated = stage6_both
    assert outcome.completed and results == {c: True for c in outcome.registered_clients}
    assert len(outcome.rounds) == len(simulated.rounds)
    for name in ("harmonization", "feature_selection", "training", "evaluation"):
        assert getattr(outcome, name) == getattr(simulated, name), name
    assert [r.aggregate for r in outcome.rounds] == [r.aggregate for r in simulated.rounds]
    from coordinator.provenance import outcome_digest

    assert outcome_digest(outcome) == outcome_digest(simulated)


def test_nothing_individual_crosses_the_network(stage6_both):
    config, (_, hub, _), _ = stage6_both
    datasets = [c._data_source.load() for c in build_clients(EXPERIMENT_ID, config)]
    wire = b"\n".join(r.raw for r in hub.log.records)
    assert not any(pid.encode() in wire for d in datasets for pid in d.participant_ids)
    sent = [v for r in hub.log.records if r.direction is Direction.TO_COORDINATOR
            for v in _numbers(json.loads(r.raw))]
    individual = np.concatenate([np.r_[d.features.ravel(), d.time_months] for d in datasets])
    assert np.abs(np.array(sent)[:, None] - individual[None, :]).min() > 1e-9


def _numbers(node):
    if isinstance(node, bool) or node is None:
        return
    if isinstance(node, int | float):
        yield float(node)
    elif isinstance(node, dict):
        for v in node.values():
            yield from _numbers(v)
    elif isinstance(node, list):
        for v in node:
            yield from _numbers(v)


def test_a_silent_site_does_not_block_the_experiment(tmp_path):
    """A site that registers but never answers is reported as not responding."""
    config = load_experiment_config(CONFIG_PATH)  # stage 1: three sites, two needed

    def register_then_vanish(agent: SiteAgent):
        agent.client.initialize()
        agent.http.post("/v1/registration", content=agent.client.build_registration())
        return "vanished"

    outcome, _, results = _distributed(
        config, tmp_path, site_overrides={"client-03": register_then_vanish},
        round_timeout_s=3.0,
    )
    assert outcome.completed and results["client-03"] == "vanished"
    for round_ in outcome.rounds:
        silent = next(c for c in round_.clients if c.client_id == "client-03")
        assert silent.status is UpdateStatus.FAILED
        assert silent.reason_code.value == "no_response"
    assert outcome.rounds[1].aggregate.n_clients == 2


def test_a_site_serves_only_the_experiment_it_agreed_to(tmp_path):
    config = load_experiment_config(CONFIG_PATH)

    def other_experiment(agent: SiteAgent):
        agent.client._experiment_id = "exp-something-else"
        return SiteAgent.run(agent)

    outcome, _, results = _distributed(
        config, tmp_path, site_overrides={"client-02": other_experiment})
    assert isinstance(results["client-02"], AgentError)
    assert "experiment_mismatch" in str(results["client-02"])
    assert outcome.registered_clients == ("client-01", "client-03")
    assert all(r.task is not TaskKind.MODEL_UPDATE for r in outcome.rounds)


# --------------------------------------------------------------------------- #
# Site endpoints: authentication and abuse
# --------------------------------------------------------------------------- #
@pytest.fixture
def api():
    tokens = {"client-01": new_token(), "client-02": new_token()}
    hub = SiteHub("exp-x", set(tokens), round_timeout_s=1.0)
    app = create_site_api(hub, {c: hash_token(t) for c, t in tokens.items()})
    return TestClient(app), tokens, hub


def _registration(client_id: str, experiment_id: str = "exp-x") -> bytes:
    return json.dumps({
        "message_type": "client_registration", "protocol_version": PROTOCOL_VERSION,
        "experiment_id": experiment_id, "client_id": client_id, "supported_tasks": [],
        "data_origin": "synthetic", "software": {},
    }).encode()


def test_site_endpoints_require_the_site_token(api):
    client, tokens, _ = api
    assert client.get("/v1/messages", params={"wait": 0}).status_code == 401
    assert client.get("/v1/messages", params={"wait": 0},
                      headers={"Authorization": "Bearer guess"}).status_code == 401
    ok = client.get("/v1/messages", params={"wait": 0},
                    headers={"Authorization": f"Bearer {tokens['client-01']}"})
    assert ok.status_code == 204


def test_a_site_cannot_act_as_another(api):
    client, tokens, hub = api
    as_one = {"Authorization": f"Bearer {tokens['client-01']}"}
    response = client.post("/v1/registration", content=_registration("client-02"),
                           headers=as_one)
    assert response.status_code == 403 and hub.client_ids == ()
    assert client.post("/v1/registration", content=_registration("client-01"),
                       headers=as_one).status_code == 202
    assert client.post("/v1/registration", content=_registration("client-01"),
                       headers=as_one).status_code == 409  # no double registration
    update = json.dumps({"client_id": "client-02", "round_id": "r1"}).encode()
    assert client.post("/v1/updates", content=update, headers=as_one).status_code == 403


def test_unsolicited_late_or_oversized_updates_are_refused(api):
    client, tokens, hub = api
    headers = {"Authorization": f"Bearer {tokens['client-01']}"}
    update = json.dumps({"client_id": "client-01", "round_id": "exp-x-r001"}).encode()
    assert client.post("/v1/updates", content=update, headers=headers).status_code == 409
    # The round times out without an answer: a later answer is refused, not used.
    reply = hub.exchange(("client-01",), json.dumps({"round_id": "exp-x-r001"}).encode())
    assert reply["client-01"].raw is None
    assert client.post("/v1/updates", content=update, headers=headers).status_code == 409
    too_big = b"x" * (1_048_576 + 1)
    assert client.post("/v1/updates", content=too_big, headers=headers).status_code == 413


def test_tokens_are_stored_hashed():
    token = new_token()
    hashes = {"client-01": hash_token(token)}
    assert identify(token, hashes) == "client-01"
    assert identify(token + "x", hashes) is None
    assert token not in json.dumps(hashes)
