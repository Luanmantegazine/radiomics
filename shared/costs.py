"""Computational cost measurement helpers."""

from __future__ import annotations

import resource
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from shared.protocol import CostReport


def peak_memory_bytes() -> int:
    """Peak resident set size of the current process.

    Limitation: this is process-wide. In the single-machine simulation all
    clients share one process, so per-client memory is an upper bound, not an
    attribution. Distributed clients report their own process.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class CostTracker:
    """Accumulates wall-clock time per named stage."""

    def __init__(self) -> None:
        self._durations: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._durations[name] = self._durations.get(name, 0.0) + time.perf_counter() - start

    @property
    def durations(self) -> dict[str, float]:
        return dict(self._durations)

    def report(self) -> CostReport:
        return CostReport(durations_s=self.durations, peak_memory_bytes=peak_memory_bytes())
