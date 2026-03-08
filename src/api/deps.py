"""Dependencies for ESCI Reranker API (model instance, path resolution)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.constants import CHECKPOINTS_DIR

if TYPE_CHECKING:
    from src.models.multi_task_reranker import MultiTaskReranker

# Global model reference; set in lifespan, read by routes
reranker_instance: "MultiTaskReranker | None" = None


def get_model_path() -> str:
    """Resolve model path from env; default to checkpoints/multi_task_reranker."""
    return os.environ.get("MODEL_PATH", str(CHECKPOINTS_DIR / "multi_task_reranker"))
