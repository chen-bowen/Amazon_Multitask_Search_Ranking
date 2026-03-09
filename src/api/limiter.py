"""Rate limiter for ESCI Reranker API (per-IP, default 100/minute)."""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

_default = os.getenv("RATE_LIMIT", "100/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_default])
