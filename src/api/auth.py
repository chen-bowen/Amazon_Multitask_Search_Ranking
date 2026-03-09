"""Optional API key authentication for ESCI Reranker API."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

API_KEY_ENV = "API_KEY"


def verify_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> None:
    """
    If API_KEY env is set, require X-API-Key or Authorization: Bearer <key>.
    When API_KEY is not set, no auth is required.
    """
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return
    provided = None
    if x_api_key:
        provided = x_api_key.strip()
    elif authorization and authorization.startswith("Bearer "):
        provided = authorization[7:].strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")
