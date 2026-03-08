"""Pydantic schemas for ESCI Reranker API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    """One item in the reranked list: product_id, score, ESCI class, substitute probability."""

    product_id: str = Field(..., description="Product identifier.")
    score: float = Field(..., description="Relevance score (higher = more relevant).")
    esci_class: str = Field(..., description="Predicted ESCI class: E, S, C, or I.")
    is_substitute: float = Field(
        ...,
        description=(
            "Probability that the product is a substitute (Task 3: substitute identification)."
        ),
    )


class RerankResponse(BaseModel):
    """Response for POST /rerank: list of ranked products with scores and ESCI outputs."""

    ranked: list[RankedItem] = Field(
        ..., description="Candidates sorted by score descending."
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = Field(default="ok", description="Service status.")
    model_loaded: bool = Field(..., description="Whether the reranker model is loaded.")
