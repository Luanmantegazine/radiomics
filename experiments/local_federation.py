"""Run a whole federation on this machine over real HTTP (loopback).

Coordinator endpoints and one site agent per simulated site, each agent in its
own thread with its own token: the same code paths as a distributed
deployment, without other machines. Used by the cost benchmark and handy for
trying the distributed mode.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from client.agent import AgentError, SiteAgent
from client.site_config import SiteConfig
from coordinator.network.tokens import hash_token, new_token
from experiments.run_distributed import run_distributed
from shared.config import ExperimentConfig


def run_over_http(experiment_id: str, config: ExperimentConfig, *, recorder=None,
                  round_timeout_s: float = 120.0, capture: bool = False):
    """Returns (outcome, hub, agent results)."""
    if config.simulation is None:
        raise ValueError("the simulation section describes the local sites")
    coordinator_config = ExperimentConfig.model_validate(
        {**config.model_dump(mode="json"), "simulation": None})
    threads, results = [], {}
    with tempfile.TemporaryDirectory() as secrets:
        sites, hashes = [], {}
        for spec in config.simulation.clients:
            token = new_token()
            (Path(secrets) / spec.client_id).write_text(token)
            hashes[spec.client_id] = hash_token(token)
            sites.append(SiteConfig(
                client_id=spec.client_id, experiment_id=experiment_id,
                coordinator_url="http://127.0.0.1:0",
                token_file=str(Path(secrets) / spec.client_id),
                data_source=spec.data_source, seed=config.seed, poll_wait_s=5.0,
            ))

        def start(port: int) -> None:
            for site in sites:
                agent = SiteAgent(site.model_copy(
                    update={"coordinator_url": f"http://127.0.0.1:{port}"}))

                def serve(agent=agent):
                    try:
                        results[agent.config.client_id] = agent.run()
                    except AgentError as exc:
                        results[agent.config.client_id] = exc

                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                threads.append(thread)

        outcome, hub = run_distributed(
            experiment_id, coordinator_config, hashes, host="127.0.0.1", port=0,
            registration_timeout_s=60, round_timeout_s=round_timeout_s, recorder=recorder,
            capture=capture, on_listening=start,
        )
        for thread in threads:
            thread.join(timeout=60)
    return outcome, hub, results
