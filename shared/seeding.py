"""Deterministic seed derivation (independent of ``PYTHONHASHSEED``)."""

from __future__ import annotations

import hashlib


def derive_seed(base_seed: int, *labels: str | int) -> int:
    """Derive an independent 32-bit seed for a named sub-stream.

    Example: ``derive_seed(seed, "client", "client-01")``.
    """
    material = "|".join([str(base_seed), *map(str, labels)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
