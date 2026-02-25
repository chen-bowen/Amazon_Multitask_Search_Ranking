"""
Lightweight partial-query augmentation for two-tower training.

Generates 0–2 variants per query (head/tail truncations, min 2 tokens)
so the model sees shorter, user-like queries with the same relevance labels.
"""

from __future__ import annotations

import random


def get_partial_query_variants(
    query: str, max_variants: int = 2, min_tokens: int = 2
) -> list[str]:
    """
    Return up to max_variants partial-query strings (truncations) for training augmentation.

    - Head: first 2-3 tokens.
    - Tail: last 2-3 tokens.
    Only queries with at least 3 tokens get variants; each variant has at least min_tokens.

    Parameters
    ----------
    query : str
        Original query string.
    max_variants : int
        Maximum number of variants to return (default 2).
    min_tokens : int
        Minimum tokens per variant (default 2).

    Returns
    -------
    list[str]
        Zero or more variant strings; same labels apply as for the original query.
    """
    tokens = query.strip().split()
    if len(tokens) < 3:
        return []

    variants: list[str] = []

    # Always try a head fragment (first 2–3 tokens)
    head = " ".join(tokens[: min(3, len(tokens))])
    if len(head.split()) >= min_tokens:
        variants.append(head)

    # For longer queries, also try a simple middle/tail fragment (2 tokens)
    if len(tokens) >= 4:
        mid_start = max(1, min(len(tokens) - 2, len(tokens) // 2))
        middle = " ".join(tokens[mid_start : mid_start + 2])
        if middle != head and len(middle.split()) >= min_tokens:
            variants.append(middle)

    # Cap to max_variants and dedupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
        if len(out) >= max_variants:
            break
    return out


def augment_query(query: str, prob: float = 0.25, max_variants: int = 2) -> str:
    """
    With probability prob, replace query with a random partial variant if any exist.

    Parameters
    ----------
    query : str
        Original query.
    prob : float
        Probability of substituting a variant (default 0.25).
    max_variants : int
        Passed to get_partial_query_variants.

    Returns
    -------
    str
        Either the original query or a random partial variant.
    """
    if prob <= 0 or random.random() >= prob:
        return query
    variants = get_partial_query_variants(query, max_variants=max_variants)
    if not variants:
        return query
    return random.choice(variants)
