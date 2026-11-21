"""HTTP endpoints the sites call (FastAPI).

Every request carries the site's own bearer token; the token identifies the
site, and a site can only act as itself. Bodies are forwarded as raw bytes
(at most ``MAX_MESSAGE_BYTES``); validation happens in the engine's ingress.

TLS: run behind a TLS-terminating reverse proxy or pass certificates to the
server (see ``experiments.run_distributed``); tokens must never travel in
clear text between institutions.
"""

from __future__ import annotations

import json

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status

from coordinator.network.hub import HubRejected, SiteHub
from coordinator.network.tokens import identify
from shared.protocol import MAX_MESSAGE_BYTES

JSON = "application/json"


def create_site_api(hub: SiteHub, token_hashes: dict[str, str]) -> FastAPI:
    app = FastAPI(title="Federated coordinator - site endpoints", version="1.4",
                  docs_url=None, redoc_url=None, openapi_url=None)

    def site(request: Request) -> str:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        client_id = identify(token, token_hashes) if scheme.lower() == "bearer" else None
        if client_id is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid site token",
                                headers={"WWW-Authenticate": "Bearer"})
        return client_id

    async def body(request: Request) -> bytes:
        raw = await request.body()
        if len(raw) > MAX_MESSAGE_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "message too large")
        return raw

    def claimed(raw: bytes) -> str | None:
        try:
            return json.loads(raw).get("client_id")
        except (ValueError, AttributeError):
            return None

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "experiment_id": hub.experiment_id}

    @app.post("/v1/registration", status_code=status.HTTP_202_ACCEPTED)
    async def register(raw: bytes = Depends(body), client_id: str = Depends(site)):
        if claimed(raw) != client_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "a site can only register itself")
        try:
            hub.register(client_id, raw)
        except HubRejected as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, exc.code) from None
        return {"status": "registered"}

    @app.get("/v1/messages")
    def next_message(wait: float = Query(default=20.0, ge=0, le=60),
                     client_id: str = Depends(site)):
        message = hub.next_message(client_id, wait)
        if message is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return Response(content=message, media_type=JSON)

    @app.post("/v1/updates", status_code=status.HTTP_202_ACCEPTED)
    async def submit(raw: bytes = Depends(body), client_id: str = Depends(site)):
        if claimed(raw) != client_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "a site can only answer for itself")
        try:
            hub.submit_update(client_id, raw)
        except HubRejected as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, exc.code) from None
        return {"status": "accepted"}

    return app
