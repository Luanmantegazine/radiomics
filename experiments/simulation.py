"""Single-machine simulation of a federated experiment.

Builds one :class:`client.node.FederatedClient` per simulated site, connects
them to a :class:`coordinator.engine.FederatedCoordinator` through the
serializing in-process transport, and runs the configured rounds.
"""

from __future__ import annotations

from client.node import FederatedClient
from client.sources import build_data_source
from coordinator.engine import FederatedCoordinator
from coordinator.outcomes import ExperimentOutcome
from coordinator.ports import RunRecorder
from coordinator.transport import InProcessTransport
from shared.config import ExperimentConfig


def build_clients(experiment_id: str, config: ExperimentConfig) -> list[FederatedClient]:
    """Instantiate the simulated sites, each with its own seeded local data."""
    if config.simulation is None:
        raise ValueError("configuration has no 'simulation' section")
    return [
        FederatedClient(
            client_id=spec.client_id,
            experiment_id=experiment_id,
            data_source=build_data_source(spec, config.seed),
        )
        for spec in config.simulation.clients
    ]


def run_simulation(
    experiment_id: str,
    config: ExperimentConfig,
    *,
    recorder: RunRecorder | None = None,
    transport: InProcessTransport | None = None,
) -> ExperimentOutcome:
    """Run the whole flow: initialise clients, register, run rounds, aggregate."""
    transport = transport or InProcessTransport()
    for client in build_clients(experiment_id, config):
        client.initialize()
        transport.attach(client)
    return FederatedCoordinator(experiment_id, config, transport, recorder).run()
