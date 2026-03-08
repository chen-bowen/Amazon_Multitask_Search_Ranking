"""API route handlers for ESCI Reranker (health, rerank)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .deps import reranker_instance
from .schemas import HealthResponse, RankedItem, RerankRequest, RerankResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check for load balancers and Docker.
    Returns 200 with status and model_loaded flag.
    """
    return HealthResponse(status="ok", model_loaded=reranker_instance is not None)


@router.post("/rerank", response_model=RerankResponse)
def rerank(body: RerankRequest) -> RerankResponse:
    """
    Rerank candidates for a single query.
    Request: query string and list of { product_id, text }.
    Response: same candidates sorted by relevance score with score,
    esci_class, and is_substitute per item.
    """
    if reranker_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not body.candidates:
        return RerankResponse(ranked=[])

    candidates_tuples = [(c.product_id, c.text) for c in body.candidates]
    ranked_tuples = reranker_instance.rerank(
        body.query,
        candidates_tuples,
        batch_size=32,
    )
    ranked = _to_ranked_items(ranked_tuples)
    return RerankResponse(ranked=ranked)


def _to_ranked_items(
    ranked_tuples: list[tuple[str, float, str, float]],
) -> list[RankedItem]:
    """Convert raw reranker outputs into RankedItem models."""
    return [
        RankedItem(
            product_id=pid,
            score=score,
            esci_class=esci_class,
            is_substitute=is_sub,
        )
        for pid, score, esci_class, is_sub in ranked_tuples
    ]
