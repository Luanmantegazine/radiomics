"""Federated coordinator: registration, round orchestration and aggregation.

Three scientific phases are kept in separate sub-packages because they need
different aggregation strategies (FedAvg is *not* assumed anywhere):

* :mod:`coordinator.selection`  - federated feature selection (Stage 4)
* :mod:`coordinator.training`   - federated model training (Stage 5)
* :mod:`coordinator.evaluation` - distributed evaluation (Stage 6)

This package must not import ``client`` or ``backend``. It reaches clients
through :class:`coordinator.transport.ClientEndpoint` and reports progress
through :class:`coordinator.ports.RunRecorder`.
"""
