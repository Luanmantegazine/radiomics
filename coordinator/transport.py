"""Transport between coordinator and clients.

The coordinator only knows :class:`ClientEndpoint`, a bytes-level interface.
:class:`InProcessTransport` implements it for single-machine simulation while
still forcing every message through serialization, so no Python object is ever
shared by reference across the boundary and every byte is accounted for.

Distributed deployment replaces this class (HTTP/gRPC, or a Flower adapter)
without changing the coordinator engine or the client node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Direction(StrEnum):
    TO_COORDINATOR = "client_to_coordinator"
    TO_CLIENT = "coordinator_to_client"


class ClientEndpoint(Protocol):
    """What the coordinator can ask of a remote site."""

    client_id: str

    def build_registration(self) -> bytes: ...

    def handle_registration_ack(self, raw: bytes) -> None: ...

    def handle_round(self, raw: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class WireRecord:
    """Metadata of one transmitted message (``raw`` kept only when capturing)."""

    direction: Direction
    client_id: str
    n_bytes: int
    raw: bytes | None = None


@dataclass
class WireLog:
    """Accounting of everything that crossed the boundary."""

    capture: bool = False
    records: list[WireRecord] = field(default_factory=list)

    def record(self, direction: Direction, client_id: str, raw: bytes) -> None:
        self.records.append(
            WireRecord(direction, client_id, len(raw), raw if self.capture else None)
        )

    def total_bytes(self, direction: Direction | None = None, client_id: str | None = None) -> int:
        return sum(
            r.n_bytes
            for r in self.records
            if (direction is None or r.direction is direction)
            and (client_id is None or r.client_id == client_id)
        )


class InProcessTransport:
    """Routes serialized messages to in-process client nodes.

    Each client is reachable only from the coordinator; there is no
    client-to-client route.
    """

    def __init__(self, *, capture: bool = False) -> None:
        self.log = WireLog(capture=capture)
        self._endpoints: dict[str, ClientEndpoint] = {}

    def attach(self, endpoint: ClientEndpoint) -> None:
        if endpoint.client_id in self._endpoints:
            raise ValueError(f"client {endpoint.client_id} already attached")
        self._endpoints[endpoint.client_id] = endpoint

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(self._endpoints)

    def fetch_registration(self, client_id: str) -> bytes:
        raw = bytes(self._endpoints[client_id].build_registration())
        self.log.record(Direction.TO_COORDINATOR, client_id, raw)
        return raw

    def send_registration_ack(self, client_id: str, raw: bytes) -> None:
        self.log.record(Direction.TO_CLIENT, client_id, raw)
        self._endpoints[client_id].handle_registration_ack(bytes(raw))

    def send_round(self, client_id: str, raw: bytes) -> bytes:
        self.log.record(Direction.TO_CLIENT, client_id, raw)
        reply = bytes(self._endpoints[client_id].handle_round(bytes(raw)))
        self.log.record(Direction.TO_COORDINATOR, client_id, reply)
        return reply
