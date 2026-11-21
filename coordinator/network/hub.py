"""Message hub between the HTTP endpoints (sites) and the engine (transport).

Thread-safe. Every message is stored and forwarded as the exact bytes the
sender produced; the hub only reads the routing fields (``round_id``).

Limitations: state lives in memory for the duration of one experiment; if the
coordinator process stops, the experiment must be restarted (there is no
resumption of a partially completed run).
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

from coordinator.transport import ClientUnavailable, Direction, Reply, WireLog
from shared.protocol import PROTOCOL_VERSION, SessionEnd, encode


class HubRejected(Exception):
    """A site request that the hub cannot accept (reason in ``code``)."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SiteHub:
    def __init__(
        self, experiment_id: str, expected_clients: set[str], *, round_timeout_s: float,
        capture: bool = False,
    ) -> None:
        self.experiment_id = experiment_id
        self.expected = set(expected_clients)
        self.round_timeout_s = round_timeout_s
        self.log = WireLog(capture=capture)
        self._cond = threading.Condition()
        self._registrations: dict[str, bytes] = {}
        self._outbox: dict[str, deque[bytes]] = {c: deque() for c in self.expected}
        self._awaited: dict[str, str] = {}  # client -> round id it must answer
        self._replies: dict[str, bytes] = {}
        self._last_seen: dict[str, float] = {}
        self.closed = False

    # -- site side (called by the HTTP endpoints) ---------------------------- #
    def register(self, client_id: str, raw: bytes) -> None:
        with self._cond:
            if client_id not in self.expected:
                raise HubRejected("unknown_client")
            if client_id in self._registrations:
                raise HubRejected("already_registered")
            self._registrations[client_id] = raw
            self._last_seen[client_id] = time.monotonic()
            self.log.record(Direction.TO_COORDINATOR, client_id, raw)
            self._cond.notify_all()

    def next_message(self, client_id: str, wait_s: float) -> bytes | None:
        deadline = time.monotonic() + wait_s
        with self._cond:
            self._last_seen[client_id] = time.monotonic()
            while not self._outbox[client_id]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._outbox[client_id].popleft()

    def submit_update(self, client_id: str, raw: bytes) -> None:
        try:
            round_id = json.loads(raw)["round_id"]
        except (ValueError, KeyError, TypeError):
            raise HubRejected("malformed") from None
        with self._cond:
            if self._awaited.get(client_id) != round_id:
                raise HubRejected("not_awaited")  # late, replayed or unsolicited
            del self._awaited[client_id]
            self._replies[client_id] = raw
            self.log.record(Direction.TO_COORDINATOR, client_id, raw)
            self._cond.notify_all()

    # -- engine side (transport protocol) ------------------------------------ #
    def wait_for_registrations(self, timeout_s: float) -> tuple[str, ...]:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while set(self._registrations) != self.expected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self.client_ids

    @property
    def client_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._registrations))

    def fetch_registration(self, client_id: str) -> bytes:
        return self._registrations[client_id]

    def _send(self, client_id: str, raw: bytes) -> None:
        self._outbox[client_id].append(raw)
        self.log.record(Direction.TO_CLIENT, client_id, raw)

    def send_registration_ack(self, client_id: str, raw: bytes) -> None:
        with self._cond:
            self._send(client_id, raw)
            self._cond.notify_all()

    def send_round(self, client_id: str, raw: bytes) -> bytes:
        reply = self.exchange((client_id,), raw)[client_id]
        if reply.raw is None:
            raise ClientUnavailable(client_id)
        return reply.raw

    def exchange(self, client_ids: tuple[str, ...], raw: bytes) -> dict[str, Reply]:
        """Queue the round for every site, then wait for all answers (or the deadline)."""
        round_id = json.loads(raw)["round_id"]
        started = time.monotonic()
        deadline = started + self.round_timeout_s
        answered: dict[str, Reply] = {}
        with self._cond:
            for client_id in client_ids:
                self._replies.pop(client_id, None)
                self._awaited[client_id] = round_id
                self._send(client_id, raw)
            self._cond.notify_all()
            while len(answered) < len(client_ids):
                for client_id in client_ids:
                    if client_id not in answered and client_id in self._replies:
                        answered[client_id] = Reply(self._replies.pop(client_id),
                                                    time.monotonic() - started)
                remaining = deadline - time.monotonic()
                if len(answered) == len(client_ids) or remaining <= 0:
                    break
                self._cond.wait(remaining)
            for client_id in client_ids:
                if client_id not in answered:
                    self._awaited.pop(client_id, None)  # a late answer will be refused
                    answered[client_id] = Reply(None, time.monotonic() - started)
        return answered

    def end_session(self, completed: bool) -> None:
        message = encode(SessionEnd(protocol_version=PROTOCOL_VERSION,
                                    experiment_id=self.experiment_id, completed=completed))
        with self._cond:
            for client_id in self._registrations:
                self._send(client_id, message)
            self.closed = True
            self._cond.notify_all()

    def drained(self) -> bool:
        with self._cond:
            return all(not queue for queue in self._outbox.values())
