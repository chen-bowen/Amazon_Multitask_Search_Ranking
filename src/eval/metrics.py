"""
nDCG and MRR for query-product ranking. Supports graded relevance (e.g. 1-4).
"""
from __future__ import annotations  # Enable postponed evaluation of type hints

import numpy as np  # For array operations
import pandas as pd  # For DataFrame operations


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.
    DCG = sum over positions: (2^relevance - 1) / log2(position + 1)
    """
    relevances = np.asarray(relevances, dtype=float)[
        :k
    ]  # Convert to array and truncate to top k
    if relevances.size == 0:  # If empty array
        return 0.0  # Return zero DCG
    gains = (
        2.0**relevances - 1.0
    )  # Compute gains: 2^relevance - 1 (exponential gain for higher relevance)
    discounts = np.log2(
        np.arange(2, len(gains) + 2)
    )  # Compute discounts: log2(position + 1) for positions 1..k
    return np.sum(gains / discounts)  # Sum of gains divided by discounts


def _idcg_at_k(relevances: np.ndarray, k: int) -> float:
    """
    Compute Ideal DCG at rank k (DCG of perfectly sorted relevances).
    """
    ideal = np.sort(np.asarray(relevances, dtype=float))[
        ::-1
    ]  # Sort relevances descending (highest first)
    return _dcg_at_k(ideal, k)  # Compute DCG of ideal ranking


def compute_ndcg(
    relevances: list[float] | np.ndarray,
    k: int | None = None,
) -> float:
    """nDCG@k. relevances: graded relevance in predicted order (higher = better). If k is None, use full list."""
    relevances = np.asarray(relevances, dtype=float)  # Convert to numpy array
    if k is None:  # If k not specified
        k = len(relevances)  # Use full list length
    idcg = _idcg_at_k(relevances, k)  # Compute ideal DCG (perfect ranking)
    if idcg <= 0:  # If ideal DCG is zero (no relevant items)
        return 0.0  # Return zero nDCG
    return (
        _dcg_at_k(relevances, k) / idcg
    )  # Normalized DCG: DCG / IDCG (range 0-1, higher is better)


def compute_mrr(
    relevances: list[float] | np.ndarray,
    relevant_threshold: float = 2.0,
) -> float:
    """MRR: 1 / rank of first item with relevance >= relevant_threshold (1-indexed)."""
    relevances = np.asarray(relevances, dtype=float)  # Convert to numpy array
    pos = np.argmax(
        relevances >= relevant_threshold
    )  # Find first position where relevance >= threshold
    if (
        relevances[pos] < relevant_threshold
    ):  # If no item meets threshold (argmax returns 0 if all False)
        return 0.0  # Return zero MRR (no relevant item found)
    return 1.0 / (
        pos + 1
    )  # Return reciprocal rank (1-indexed: pos 0 -> rank 1 -> MRR 1.0)
