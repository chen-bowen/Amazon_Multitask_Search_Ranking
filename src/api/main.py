"""
FastAPI app for ESCI multi-task reranker.

Loads model at startup (MODEL_PATH env), exposes POST /rerank and GET /health.
Includes Prometheus metrics, rate limiting, request logging, and optional API key auth.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.constants import DEFAULT_RERANKER_MODEL
from src.models.multi_task_reranker import load_multi_task_reranker
from src.utils import resolve_device

from . import deps
from .limiter import limiter
from .metrics import API_REGISTRY, MODEL_LOADED
from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load multi-task reranker at startup; release on shutdown."""
    path = deps.get_model_path()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_RERANKER_MODEL)
    device_str = os.environ.get("INFERENCE_DEVICE")
    device = resolve_device(device_str if device_str else None)
    logger.info(
        "Loading reranker from path=%s (fallback model_name=%s) on device=%s",
        path,
        model_name,
        device,
    )
    try:
        deps.reranker_instance = load_multi_task_reranker(
            model_path=path,
            model_name=model_name,
            device=device,
        )
        MODEL_LOADED.set(1)
        logger.info("Reranker loaded successfully")
    except Exception as e:
        logger.exception("Failed to load reranker: %s", e)
        deps.reranker_instance = None
        MODEL_LOADED.set(0)
    yield
    deps.reranker_instance = None
    MODEL_LOADED.set(0)


app = FastAPI(
    title="ESCI Reranker API",
    description=(
        "Rerank product candidates for a query; returns score, ESCI class, "
        "and substitute flag per product."
    ),
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Structured logging: path, method, status, request_id, latency_ms."""
    start = time.time()
    req_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = req_id
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((time.time() - start) * 1000)
        logger.exception(
            "request_error path=%s method=%s request_id=%s latency_ms=%d",
            request.url.path,
            request.method,
            req_id,
            elapsed_ms,
        )
        raise
    elapsed_ms = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = req_id
    logger.info(
        "request path=%s method=%s status=%d request_id=%s latency_ms=%d",
        request.url.path,
        request.method,
        response.status_code,
        req_id,
        elapsed_ms,
    )
    return response


@app.get("/ready")
@limiter.exempt
def ready() -> dict[str, str]:
    """Readiness probe: ready if model is loaded (for k8s)."""
    return (
        {"status": "ready"}
        if deps.reranker_instance is not None
        else {"status": "not_ready"}
    )


@app.get("/metrics")
@limiter.exempt
def metrics() -> Response:
    """Prometheus scrape endpoint (rate-limit exempt)."""
    return Response(
        content=generate_latest(API_REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


app.include_router(router)
