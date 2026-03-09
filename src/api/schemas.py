"""Pydantic schemas for ESCI Reranker API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceStatistics(BaseModel):
    """Per-request inference stats (latency, model time, scores, timestamp)."""

    total_latency_ms: float = Field(
        ..., description="End-to-end request latency in milliseconds."
    )
    model_forward_time_ms: float = Field(
        ..., description="Time spent in model forward pass (rerank/predict) in milliseconds."
    )
    num_candidates: int = Field(..., description="Number of candidates processed.")
    num_recommendations: int = Field(
        ..., description="Number of results returned (same as num_candidates for this API)."
    )
    device: str = Field(
        ...,
        description="Device used for inference (e.g. 'cuda', 'mps', or 'cpu').",
    )
    top_score: float | None = Field(
        default=None,
        description="Highest relevance score in results (rerank/predict only).",
    )
    avg_score: float | None = Field(
        default=None,
        description="Mean relevance score in results (rerank/predict only).",
    )
    timestamp: float = Field(
        ..., description="Unix timestamp when the response was built."
    )


class CandidateItem(BaseModel):
    """One candidate product for reranking."""

    product_id: str = Field(..., description="Unique product identifier.")
    text: str = Field(
        ...,
        description="Product text (e.g. title + description) to score against the query.",
    )


class RerankRequest(BaseModel):
    """Request body for POST /rerank."""

    query: str = Field(..., description="User search query.")
    candidates: list[CandidateItem] = Field(
        ..., description="List of product_id and text to rerank."
    )


class RankedItem(BaseModel):
    """One item in the reranked list: product_id, score, ESCI class, substitute label."""

    product_id: str = Field(..., description="Product identifier.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")
    esci_class: str = Field(..., description="Predicted ESCI class: E, S, C, or I.")
    is_substitute: bool = Field(
        ...,
        description="True if product is a substitute (ESCI=S), False otherwise.",
    )


class RerankResponse(BaseModel):
    """Response for POST /predict: ranked products with score, ESCI, substitute, request_id, stats."""

    request_id: str = Field(..., description="Request ID for tracing (same as X-Request-ID).")
    ranked: list[RankedItem] = Field(
        ..., description="Candidates sorted by score descending."
    )
    stats: InferenceStatistics = Field(
        ..., description="Inference statistics (latency, num_candidates)."
    )


# --- Task-specific responses ---

class RankedScoreItem(BaseModel):
    """Task 1: product with relevance score only."""

    product_id: str = Field(..., description="Product identifier.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")


class RerankScoresResponse(BaseModel):
    """Response for POST /rerank (Task 1 only): ranked by score, request_id, stats."""

    request_id: str = Field(..., description="Request ID for tracing (same as X-Request-ID).")
    ranked: list[RankedScoreItem] = Field(
        ..., description="Candidates sorted by score descending."
    )
    stats: InferenceStatistics = Field(
        ..., description="Inference statistics (latency, num_candidates)."
    )


class ClassifyItem(BaseModel):
    """Task 2: product with predicted ESCI class."""

    product_id: str = Field(..., description="Product identifier.")
    esci_class: str = Field(..., description="Predicted ESCI class: E, S, C, or I.")


class ClassifyResponse(BaseModel):
    """Response for POST /classify (Task 2): ESCI class per candidate, request_id, stats."""

    request_id: str = Field(..., description="Request ID for tracing (same as X-Request-ID).")
    results: list[ClassifyItem] = Field(
        ..., description="ESCI class per candidate (same order as request)."
    )
    stats: InferenceStatistics = Field(
        ..., description="Inference statistics (latency, num_candidates)."
    )


class SubstituteItem(BaseModel):
    """Task 3: product with substitute label (Substitute or non-Substitute)."""

    product_id: str = Field(..., description="Product identifier.")
    is_substitute: bool = Field(
        ...,
        description="True if product is a substitute (ESCI=S), False otherwise.",
    )


class SubstituteResponse(BaseModel):
    """Response for POST /substitute (Task 3): substitute label per candidate, request_id, stats."""

    request_id: str = Field(..., description="Request ID for tracing (same as X-Request-ID).")
    results: list[SubstituteItem] = Field(
        ..., description="Substitute label per candidate (same order as request)."
    )
    stats: InferenceStatistics = Field(
        ..., description="Inference statistics (latency, num_candidates)."
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(default="ok", description="Service status.")
    model_loaded: bool = Field(..., description="Whether the reranker model is loaded.")
