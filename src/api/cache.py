"""In-process LRU cache for multi-task reranker predict() results."""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Hashable

from src.models.multi_task_reranker import MultiTaskReranker

_PREDICT_CACHE_MAX_SIZE = 256
_predict_cache: "OrderedDict[Hashable, tuple[list[float], list[str], list[float]]]" = (
    OrderedDict()
)


def _make_predict_key(query: str, candidates: list[str]) -> Hashable:
    """Build a hashable cache key from query and candidate texts."""
    return ("predict", query, tuple(candidates))


def _get_predict_from_cache(key: Hashable) -> tuple[list[float], list[str], list[float]] | None:
    """LRU get for predict cache."""
    if key not in _predict_cache:
        return None
    value = _predict_cache.pop(key)
    _predict_cache[key] = value
    return value


def _set_predict_cache(key: Hashable, value: tuple[list[float], list[str], list[float]]) -> None:
    """LRU set for predict cache."""
    if key in _predict_cache:
        _predict_cache.pop(key)
    _predict_cache[key] = value
    if len(_predict_cache) > _PREDICT_CACHE_MAX_SIZE:
        _predict_cache.popitem(last=False)


def predict_with_cache(
    model: MultiTaskReranker,
    query: str,
    candidate_texts: list[str],
    batch_size: int = 32,
) -> tuple[list[float], list[str], list[float], float]:
    """Shared predict() with LRU cache and timing."""
    key = _make_predict_key(query, candidate_texts)
    cached = _get_predict_from_cache(key)
    t0 = time.perf_counter()
    if cached is not None:
        scores, esci_classes, sub_probs = cached
    else:
        scores, esci_classes, sub_probs = model.predict(
            [[query, t] for t in candidate_texts],
            batch_size=batch_size,
        )
        _set_predict_cache(key, (scores, esci_classes, sub_probs))
    model_ms = (time.perf_counter() - t0) * 1000
    return scores, esci_classes, sub_probs, model_ms
