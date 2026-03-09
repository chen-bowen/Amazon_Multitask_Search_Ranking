"""API route handlers for ESCI Reranker (health, task-specific routes)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request

from . import deps
from .auth import verify_api_key
from .cache import predict_with_cache
from .limiter import limiter
from .metrics import (
    CLASSIFY_LATENCY_SECONDS,
    CLASSIFY_REQUESTS_TOTAL,
    PREDICT_LATENCY_SECONDS,
    PREDICT_REQUESTS_TOTAL,
    RERANK_LATENCY_SECONDS,
    RERANK_REQUESTS_TOTAL,
    SUBSTITUTE_LATENCY_SECONDS,
    SUBSTITUTE_REQUESTS_TOTAL,
)
from .schemas import (
    ClassifyItem,
    ClassifyResponse,
    HealthResponse,
    InferenceStatistics,
    RankedItem,
    RankedScoreItem,
    RerankRequest,
    RerankResponse,
    RerankScoresResponse,
    SubstituteItem,
    SubstituteResponse,
)

router = APIRouter()


def _require_model() -> None:
    if deps.reranker_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")


@router.get("/health", response_model=HealthResponse)
@limiter.exempt
def health() -> HealthResponse:
    """
    Health check for load balancers and Docker.
    Returns 200 with status and model_loaded flag.
    """
    return HealthResponse(status="ok", model_loaded=deps.reranker_instance is not None)


@router.post("/rerank", response_model=RerankScoresResponse)
def rerank(
    request: Request,
    body: RerankRequest,
    _: None = Depends(verify_api_key),
) -> RerankScoresResponse:
    """
    Task 1: Rank candidates by relevance score.
    Returns candidates sorted by score descending (score only).
    """
    start = time.perf_counter()
    req_id = getattr(request.state, "request_id", "")
    n = len(body.candidates)
    try:
        _require_model()
        dev = str(deps.reranker_instance.device)
        if not body.candidates:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RerankScoresResponse(
                request_id=req_id,
                ranked=[],
                stats=InferenceStatistics(
                    total_latency_ms=elapsed_ms,
                    model_forward_time_ms=0.0,
                    num_candidates=0,
                    num_recommendations=0,
                    device=dev,
                    top_score=None,
                    avg_score=None,
                    timestamp=time.time(),
                ),
            )
        candidates_tuples = [(c.product_id, c.text) for c in body.candidates]
        texts = [text for _pid, text in candidates_tuples]
        scores, _, _ , model_ms = predict_with_cache(
            deps.reranker_instance, body.query, texts, batch_size=32
        )
        ranked_tuples = [
            (pid, float(sc))
            for (pid, _), sc in zip(candidates_tuples, scores)
        ]
        ranked_tuples.sort(key=lambda x: x[1], reverse=True)
        ranked = [
            RankedScoreItem(product_id=pid, score=score)
            for pid, score in ranked_tuples
        ]
        scores = [s for _, s in ranked_tuples]
        top = max(scores) if scores else None
        avg = sum(scores) / len(scores) if scores else None
        RERANK_REQUESTS_TOTAL.labels(status="success").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RerankScoresResponse(
            request_id=req_id,
            ranked=ranked,
            stats=InferenceStatistics(
                total_latency_ms=elapsed_ms,
                model_forward_time_ms=model_ms,
                num_candidates=n,
                num_recommendations=len(ranked),
                device=dev,
                top_score=top,
                avg_score=avg,
                timestamp=time.time(),
            ),
        )
    except Exception:
        RERANK_REQUESTS_TOTAL.labels(status="error").inc()
        raise
    finally:
        RERANK_LATENCY_SECONDS.observe(time.perf_counter() - start)


@router.post("/classify", response_model=ClassifyResponse)
def classify(
    request: Request,
    body: RerankRequest,
    _: None = Depends(verify_api_key),
) -> ClassifyResponse:
    """
    Task 2: Predict ESCI class (E/S/C/I) for each candidate.
    Returns results in same order as request.
    """
    start = time.perf_counter()
    req_id = getattr(request.state, "request_id", "")
    n = len(body.candidates)
    try:
        _require_model()
        dev = str(deps.reranker_instance.device)
        if not body.candidates:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ClassifyResponse(
                request_id=req_id,
                results=[],
                stats=InferenceStatistics(
                    total_latency_ms=elapsed_ms,
                    model_forward_time_ms=0.0,
                    num_candidates=0,
                    num_recommendations=0,
                    device=dev,
                    top_score=None,
                    avg_score=None,
                    timestamp=time.time(),
                ),
            )
        texts = [c.text for c in body.candidates]
        _, esci_classes, _ , model_ms = predict_with_cache(
            deps.reranker_instance, body.query, texts, batch_size=32
        )
        results = [
            ClassifyItem(product_id=c.product_id, esci_class=esc)
            for c, esc in zip(body.candidates, esci_classes)
        ]
        CLASSIFY_REQUESTS_TOTAL.labels(status="success").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ClassifyResponse(
            request_id=req_id,
            results=results,
            stats=InferenceStatistics(
                total_latency_ms=elapsed_ms,
                model_forward_time_ms=model_ms,
                num_candidates=n,
                num_recommendations=len(results),
                device=dev,
                top_score=None,
                avg_score=None,
                timestamp=time.time(),
            ),
        )
    except Exception:
        CLASSIFY_REQUESTS_TOTAL.labels(status="error").inc()
        raise
    finally:
        CLASSIFY_LATENCY_SECONDS.observe(time.perf_counter() - start)


@router.post("/substitute", response_model=SubstituteResponse)
def substitute(
    request: Request,
    body: RerankRequest,
    _: None = Depends(verify_api_key),
) -> SubstituteResponse:
    """
    Task 3: Predict substitute label (Substitute or non-Substitute) for each candidate.
    Returns results in same order as request.
    """
    start = time.perf_counter()
    req_id = getattr(request.state, "request_id", "")
    n = len(body.candidates)
    try:
        _require_model()
        dev = str(deps.reranker_instance.device)
        if not body.candidates:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return SubstituteResponse(
                request_id=req_id,
                results=[],
                stats=InferenceStatistics(
                    total_latency_ms=elapsed_ms,
                    model_forward_time_ms=0.0,
                    num_candidates=0,
                    num_recommendations=0,
                    device=dev,
                    top_score=None,
                    avg_score=None,
                    timestamp=time.time(),
                ),
            )
        texts = [c.text for c in body.candidates]
        _, _ , sub_probs, model_ms = predict_with_cache(
            deps.reranker_instance, body.query, texts, batch_size=32
        )
        results = [
            SubstituteItem(product_id=c.product_id, is_substitute=sub > 0.5)
            for c, sub in zip(body.candidates, sub_probs)
        ]
        SUBSTITUTE_REQUESTS_TOTAL.labels(status="success").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SubstituteResponse(
            request_id=req_id,
            results=results,
            stats=InferenceStatistics(
                total_latency_ms=elapsed_ms,
                model_forward_time_ms=model_ms,
                num_candidates=n,
                num_recommendations=len(results),
                device=dev,
                top_score=None,
                avg_score=None,
                timestamp=time.time(),
            ),
        )
    except Exception:
        SUBSTITUTE_REQUESTS_TOTAL.labels(status="error").inc()
        raise
    finally:
        SUBSTITUTE_LATENCY_SECONDS.observe(time.perf_counter() - start)


@router.post("/predict", response_model=RerankResponse)
def predict(
    request: Request,
    body: RerankRequest,
    _: None = Depends(verify_api_key),
) -> RerankResponse:
    """
    Combined: all three tasks (ranking, ESCI class, substitute probability).
    Returns candidates sorted by relevance score descending.
    """
    start = time.perf_counter()
    req_id = getattr(request.state, "request_id", "")
    n = len(body.candidates)
    try:
        _require_model()
        dev = str(deps.reranker_instance.device)
        if not body.candidates:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RerankResponse(
                request_id=req_id,
                ranked=[],
                stats=InferenceStatistics(
                    total_latency_ms=elapsed_ms,
                    model_forward_time_ms=0.0,
                    num_candidates=0,
                    num_recommendations=0,
                    device=dev,
                    top_score=None,
                    avg_score=None,
                    timestamp=time.time(),
                ),
            )
        candidates_tuples = [(c.product_id, c.text) for c in body.candidates]
        texts = [text for _pid, text in candidates_tuples]
        scores, esci_classes, sub_probs, model_ms = predict_with_cache(
            deps.reranker_instance, body.query, texts, batch_size=32
        )
        ranked_tuples = [
            (pid, float(sc), esc, float(sub))
            for (pid, _), sc, esc, sub in zip(
                candidates_tuples, scores, esci_classes, sub_probs
            )
        ]
        ranked_tuples.sort(key=lambda x: x[1], reverse=True)
        ranked = [
            RankedItem(
                product_id=pid,
                score=score,
                esci_class=esci_class,
                is_substitute=is_sub > 0.5,
            )
            for pid, score, esci_class, is_sub in ranked_tuples
        ]
        scores = [s for _, s, _, _ in ranked_tuples]
        top = max(scores) if scores else None
        avg = sum(scores) / len(scores) if scores else None
        PREDICT_REQUESTS_TOTAL.labels(status="success").inc()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RerankResponse(
            request_id=req_id,
            ranked=ranked,
            stats=InferenceStatistics(
                total_latency_ms=elapsed_ms,
                model_forward_time_ms=model_ms,
                num_candidates=n,
                num_recommendations=len(ranked),
                device=dev,
                top_score=top,
                avg_score=avg,
                timestamp=time.time(),
            ),
        )
    except Exception:
        PREDICT_REQUESTS_TOTAL.labels(status="error").inc()
        raise
    finally:
        PREDICT_LATENCY_SECONDS.observe(time.perf_counter() - start)
