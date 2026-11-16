"""Central backend: REST API, persistence (PostgreSQL) and tracking (MLflow).

Contains no scientific algorithm. It stores configurations, run state,
aggregate results and costs, and triggers federated runs through a launcher.
This package must not import ``client``.
"""
