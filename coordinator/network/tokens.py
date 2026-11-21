"""Per-site access tokens.

Each site has its own random token; the coordinator stores only its SHA-256.
A token authenticates *which site* is calling, so a site cannot register,
read messages or submit updates as another site.

    python -m coordinator.network.tokens client-01
    # prints the token (give it to the site) and the line for sites.yaml
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from shared.protocol import ClientId


class SiteCredential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    client_id: ClientId
    token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SitesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sites: tuple[SiteCredential, ...] = Field(min_length=1)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def load_sites(path: str | Path) -> dict[str, str]:
    """client_id -> token hash."""
    sites = SitesFile.model_validate(yaml.safe_load(Path(path).read_text("utf-8")))
    ids = [s.client_id for s in sites.sites]
    if len(set(ids)) != len(ids):
        raise ValueError("client ids in the sites file must be unique")
    return {s.client_id: s.token_sha256 for s in sites.sites}


def identify(token: str, hashes: dict[str, str]) -> str | None:
    """The site a token belongs to (constant-time comparison per site)."""
    digest = hash_token(token)
    match = None
    for client_id, expected in hashes.items():
        if hmac.compare_digest(digest, expected):
            match = client_id
    return match


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m coordinator.network.tokens <client-id>", file=sys.stderr)
        return 2
    SiteCredential(client_id=args[0], token_sha256="0" * 64)  # validates the id
    token = new_token()
    print(f"token for the site (show once, store securely): {token}")
    print("entry for sites.yaml:")
    print(f"  - client_id: {args[0]}\n    token_sha256: {hash_token(token)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
