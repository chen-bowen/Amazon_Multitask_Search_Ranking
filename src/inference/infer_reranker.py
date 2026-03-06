#!/usr/bin/env python3
"""
Run inference with the trained ESCI reranker on a single query.

Loads a CrossEncoder reranker from disk (matching the training config) and
logs the top-k products for a query from the test set.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.constants import REPO_ROOT
from src.data.load_data import load_esci, prepare_train_test
from src.models.reranker import load_reranker

logger = logging.getLogger(__name__)


def _load_test_df(data_dir: Path, small_version: bool) -> pd.DataFrame:
    """Load ESCI test split, preferring pre-saved parquet if available."""
    base = data_dir
    test_path = base / "esci_test.parquet"
    if test_path.exists():
        return pd.read_parquet(test_path)

    df = load_esci(data_dir=base, small_version=small_version)
    _, test_df = prepare_train_test(df=df)
    return test_df


def _select_query(test_df: pd.DataFrame, query_index: int) -> tuple[str, int]:
    """
    Select a query and its query_id from the test set by index.

    query_index is an index over unique query_ids in the test set, sorted
    by appearance.
    """
    by_qid = test_df.groupby("query_id").first().reset_index()
    if len(by_qid) == 0:
        raise ValueError("No queries found in test set.")
    if query_index < 0 or query_index >= len(by_qid):
        raise IndexError(f"query_index {query_index} out of range 0..{len(by_qid) - 1}")
    row = by_qid.iloc[query_index]
    return str(row["query"]), int(row["query_id"])


def run_inference(configs: dict) -> int:
    """Run inference using a config dict from YAML."""
    # Core knobs (YAML is the single source of truth).
    model_path = configs["model_path"]
    data_dir = Path(configs["data_dir"])
    product_col = configs.get("product_col", "product_text")
    query_override = configs.get("query")
    small_version = bool(configs.get("small_version", False))
    batch_size = int(configs.get("batch_size", 16))
    top_k = int(configs.get("top_k", 5))
    query_index = int(configs.get("query_index", 0))

    # Load test data
    test_df = _load_test_df(data_dir, small_version)
    if test_df.empty:
        logger.error("No test data found.")
        return 1

    # Select query
    query, qid = _select_query(test_df, query_index)
    if query_override:
        query = str(query_override)
    logger.info("Using query_index=%d (query_id=%s)", query_index, qid)
    logger.info("Query: %s", query)

    rows = test_df[test_df["query_id"] == qid]
    if len(rows) == 0:
        logger.error("No products found for query_id %s", qid)
        return 1

    if product_col not in rows.columns:
        logger.error(
            "Column '%s' not in test data; available: %s",
            product_col,
            list(rows.columns),
        )
        return 1

    candidates = [
        (str(r["product_id"]), str(r[product_col])) for _, r in rows.iterrows()
    ]

    # Load reranker and score candidates
    reranker = load_reranker(model_path=model_path)
    ranked = reranker.rerank(query, candidates, batch_size=batch_size)

    # Map product_id -> ESCI label if available
    labels = rows.get("esci_label", ["?"] * len(rows))
    pid_to_label = dict(zip(rows["product_id"].astype(str), labels))

    logger.info("Top %d products for query (query_id=%s):", top_k, qid)
    for rank, (pid, score) in enumerate(ranked[:top_k], start=1):
        label = pid_to_label.get(pid, "?")
        text = next(t for p, t in candidates if p == pid)
        logger.info("%d. [label=%s] product_id=%s score=%.4f", rank, label, pid, score)
        logger.info("    %s...", text[:200])

    return 0


def main() -> int:
    """CLI entrypoint: run inference with the trained reranker on one query."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(
        description="Run reranker inference on a sample test query."
    )
    p.add_argument(
        "--config", default="configs/reranker.yaml", help="Path to YAML config."
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="Override query text directly (candidates still come from selected query_id).",
    )
    p.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Override query_index from config (index over unique query_id values).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k from config (number of results to log).",
    )
    args = p.parse_args()

    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides config values when provided
    if args.query_index is not None:
        cfg = cfg or {}
        cfg["query_index"] = args.query_index
    if args.top_k is not None:
        cfg = cfg or {}
        cfg["top_k"] = args.top_k
    if args.query is not None:
        cfg = cfg or {}
        cfg["query"] = args.query

    return run_inference(cfg or {})


if __name__ == "__main__":
    sys.exit(main())
