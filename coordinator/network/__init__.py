"""Network binding of the coordinator (Stage 8).

Sites connect *out* to the coordinator over HTTPS and poll for messages, so no
institution has to accept inbound connections. The bytes exchanged are exactly
the ``shared.protocol`` messages of the in-process simulation; the egress
guard (client) and ingress validation (coordinator) apply unchanged.

* :mod:`coordinator.network.hub`    - message queues; the engine's transport
* :mod:`coordinator.network.api`    - HTTP endpoints the sites call
* :mod:`coordinator.network.tokens` - per-site credentials (stored hashed)
"""
