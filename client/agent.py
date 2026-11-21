"""Site agent: connects out to the coordinator and serves one experiment.

    python -m client.agent --site-config /etc/fedradiomics/site.yaml

The agent registers, then polls for messages (registration acknowledgement,
round configurations, end of session). Every answer is produced by the same
:class:`client.node.FederatedClient` as in the simulation, so the egress guard
applies unchanged; the agent itself never reads or logs message contents
beyond routing fields.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from client.node import FederatedClient
from client.site_config import SiteConfig, load_site_config
from client.sources import build_data_source
from shared.config import SimulatedClientSpec
from shared.logging import configure_logging, get_logger

RETRY_BACKOFF_S = (1, 2, 5, 10, 30)


class AgentError(RuntimeError):
    pass


def build_client(config: SiteConfig) -> FederatedClient:
    source = build_data_source(
        SimulatedClientSpec(client_id=config.client_id, data_source=config.data_source),
        config.seed,
    )
    return FederatedClient(config.client_id, config.experiment_id, source)


class SiteAgent:
    def __init__(self, config: SiteConfig, client: FederatedClient | None = None,
                 http: httpx.Client | None = None) -> None:
        self.config = config
        self.client = client or build_client(config)
        token = Path(config.token_file).read_text("utf-8").strip()
        self.http = http or httpx.Client(
            base_url=config.coordinator_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(config.poll_wait_s + 30),
            verify=config.ca_bundle or True,
        )
        self.log = get_logger("client.agent").bind(client_id=config.client_id,
                                                    experiment_id=config.experiment_id)
        self.completed: bool | None = None

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt, pause in enumerate((*RETRY_BACKOFF_S, None)):
            try:
                response = self.http.request(method, url, **kwargs)
            except httpx.TransportError as exc:
                if pause is None:
                    raise AgentError(f"coordinator unreachable: {type(exc).__name__}") from exc
                self.log.warning("coordinator_unreachable", attempt=attempt + 1, retry_in_s=pause)
                time.sleep(pause)
                continue
            if response.status_code in (401, 403):
                raise AgentError(f"refused by the coordinator ({response.status_code})")
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def run(self) -> bool:
        """Serve the experiment until the coordinator ends the session."""
        self.client.initialize()
        response = self._request("POST", "/v1/registration",
                                 content=self.client.build_registration())
        if response.status_code != 202:
            raise AgentError(f"registration failed ({response.status_code})")
        self.log.info("registered")
        idle_since = time.monotonic()
        while True:
            response = self._request("GET", "/v1/messages",
                                     params={"wait": self.config.poll_wait_s})
            if response.status_code == 204:
                if time.monotonic() - idle_since > self.config.idle_timeout_s:
                    raise AgentError("no message from the coordinator before the idle timeout")
                continue
            response.raise_for_status()
            idle_since = time.monotonic()
            raw = response.content
            kind = json.loads(raw).get("message_type")
            if kind == "registration_ack":
                self.client.handle_registration_ack(raw)
                if not json.loads(raw).get("accepted"):
                    raise AgentError(f"registration rejected: {json.loads(raw).get('reason_code')}")
            elif kind == "round_config":
                answer = self.client.handle_round(raw)
                posted = self._request("POST", "/v1/updates", content=answer)
                if posted.status_code != 202:
                    self.log.warning("update_not_accepted", status=posted.status_code)
            elif kind == "session_end":
                self.completed = bool(json.loads(raw).get("completed"))
                self.log.info("session_end", completed=self.completed)
                return self.completed
            else:
                self.log.warning("unknown_message_ignored")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m client.agent")
    parser.add_argument("--site-config", required=True)
    parser.add_argument("--experiment-id",
                        help="the experiment this site agrees to serve (overrides the file)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        config = load_site_config(args.site_config)
        if args.experiment_id:
            config = config.model_copy(update={"experiment_id": args.experiment_id})
        completed = SiteAgent(config).run()
    except AgentError as exc:
        print(f"agent stopped: {exc}", file=sys.stderr)
        return 2
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
