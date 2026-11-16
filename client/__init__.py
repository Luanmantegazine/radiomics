"""Federated client: one institution / independent data partition.

Individual-level data (images, masks, labels, follow-up times, per-participant
feature vectors) lives only in :mod:`client.data` objects and never crosses the
boundary: every outgoing message is built from :mod:`shared.protocol` models and
passes through :class:`client.egress.EgressGuard`.

This package must not import ``coordinator`` or ``backend``.
"""
