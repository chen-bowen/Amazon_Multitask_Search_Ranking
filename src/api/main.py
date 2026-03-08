"""
FastAPI app for ESCI multi-task reranker.

Loads model at startup (MODEL_PATH env), exposes POST /rerank and GET /health.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.constants import DEFAULT_RERANKER_MODEL
from src.models.multi_task_reranker import load_multi_task_reranker

from . import deps
from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load multi-task reranker at startup; release on shutdown."""
    path = deps.get_model_path()
    model_name = os.environ.get("MODEL_NAME", DEFAULT_RERANKER_MODEL)
    logger.info(
        "Loading reranker from path=%s (fallback model_name=%s)", path, model_name
    )
    deps.reranker_instance = load_multi_task_reranker(
        model_path=path, model_name=model_name
    )
    yield
    deps.reranker_instance = None


app = FastAPI(
    title="ESCI Reranker API",
    description=(
        "Rerank product candidates for a query; returns score, ESCI class, "
        "and substitute flag per product."
    ),
    lifespan=lifespan,
)

app.include_router(router)
