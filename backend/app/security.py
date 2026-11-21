"""Optional bearer-token protection of the API.

A single shared token (``FEDRAD_API_TOKEN``) is a minimal safeguard for a
deployment reachable by other people; it is not user management. When unset,
the API is open (local development).
"""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status


def require_token(request: Request) -> None:
    expected = request.app.state.settings.api_token
    if not expected:
        return
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "missing or invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
