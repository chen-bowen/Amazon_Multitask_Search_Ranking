#!/usr/bin/env python3
"""
Run inference with the trained multi-task ESCI reranker on a single query.

Loads MultiTaskReranker from disk (or Hugging Face Hub) and logs the top-k
products with ranking score, predicted ESCI class, and substitute probability.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.constants import INFER_MULTI_TASK_DEFAULTS, REPO_ROOT
from src.data.load_data import ESCIDataLoader
from src.models.multi_task_reranker import load_multi_task_reranker
from src.utils import load_config

logger = logging.getLogger(__name__)


class MultiTaskRerankerInference:
    """
    Run multi-task reranker inference on a single query from the test set.

    Configure via constructor; call run() to execute. Returns 0 on success, 1 on error.
    """

    def __init__(self, configs: dict) -> None:
        """Initialize inference from config dict (YAML + CLI overrides).

        Args:
            configs: Dict with model_path, data_dir, product_col, query (optional),
                small_version, batch_size, top_k, query_index.
        """
        self.model_path = configs.get("model_path", INFER_MULTI_TASK_DEFAULTS["model_path"])
        self.data_dir = Path(configs.get("data_dir", INFER_MULTI_TASK_DEFAULTS["data_dir"]))
        self.product_col = configs.get("product_col", "product_text")
        self.query_override = configs.get("query")
        self.small_version = bool(configs.get("small_version", False))
        self.batch_size = int(configs.get("batch_size", 16))
        self.top_k = int(configs.get("top_k", 5))
        self.query_index = int(configs.get("query_index", 0))

    def run(self) -> int:
        """Run inference; return 0 on success, 1 on error."""
        test_df = self._load_test_df()
        if test_df.empty:
            logger.error("No test data found.")
            return 1

        query, qid = self._select_query(test_df)
        if self.query_override is not None:
            query = str(self.query_override)
        logger.info("Using query_index=%d (query_id=%s)", self.query_index, qid)
        logger.info("Query: %s", query)

        rows = test_df[test_df["query_id"] == qid]
        if len(rows) == 0:
            logger.error("No products found for query_id %s", qid)
            return 1

        candidates = self._prepare_candidates(rows)
        if not candidates:
            return 1

        reranker = load_multi_task_reranker(model_path=self.model_path)
        ranked = reranker.rerank(query, candidates, batch_size=self.batch_size)
        self._log_ranked_results(ranked, candidates, rows, qid)
        return 0

    def _load_test_df(self) -> pd.DataFrame:
        """Load ESCI test split, preferring pre-saved parquet if available."""
        test_path = self.data_dir / "esci_test.parquet"
        if test_path.exists():
            return pd.read_parquet(test_path)
        loader = ESCIDataLoader(
            data_dir=self.data_dir, small_version=self.small_version
        )
        _, test_df = loader.prepare_train_test()
        return test_df

    def _select_query(self, test_df: pd.DataFrame) -> tuple[str, int]:
        """Select query and query_id from test set by index."""
        by_qid = test_df.groupby("query_id").first().reset_index()
        if len(by_qid) == 0:
            raise ValueError("No queries found in test set.")
        if self.query_index < 0 or self.query_index >= len(by_qid):
            raise IndexError(
                f"query_index {self.query_index} out of range 0..{len(by_qid) - 1}"
            )
        row = by_qid.iloc[self.query_index]
        return str(row["query"]), int(row["query_id"])

    def _prepare_candidates(self, rows: pd.DataFrame) -> list[tuple[str, str]]:
        """Build candidate (product_id, product_text) tuples."""
        if self.product_col not in rows.columns:
            logger.error(
                "Column '%s' not in test data; available: %s",
                self.product_col,
                list(rows.columns),
            )
            return []
        return [
            (str(r["product_id"]), str(r[self.product_col])) for _, r in rows.iterrows()
        ]

    def _log_ranked_results(
        self,
        ranked: list[tuple[str, float, str, float]],
        candidates: list[tuple[str, str]],
        rows: pd.DataFrame,
        qid: int,
    ) -> None:
        """Log top-k ranked products with labels, ESCI class, and substitute prob."""
        labels = rows.get("esci_label", ["?"] * len(rows))
        pid_to_label = dict(zip(rows["product_id"].astype(str), labels))

        logger.info("Top %d products for query (query_id=%s):", self.top_k, qid)
        for rank, (pid, score, esci_class, sub_prob) in enumerate(
            ranked[: self.top_k], start=1
        ):
            label = pid_to_label.get(pid, "?")
            text = next(t for p, t in candidates if p == pid)
            logger.info(
                "%d. [true=%s pred=%s] score=%.4f sub_prob=%.2f",
                rank,
                label,
                esci_class,
                score,
                sub_prob,
            )
            logger.info("    %s...", text[:200])


def main() -> int:
    """CLI entrypoint: run multi-task reranker inference on one query."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Run multi-task reranker inference on a sample test query."
    )
    p.add_argument(
        "--config",
        default="configs/multi_task_reranker.yaml",
        help="Path to YAML config.",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="Override query text directly.",
    )
    p.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Override query_index (index over unique query_id values).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override top_k (number of results to log).",
    )
    args = p.parse_args()

    config_path = REPO_ROOT / args.config
    cfg = load_config(config_path, INFER_MULTI_TASK_DEFAULTS)

    if args.query_index is not None:
        cfg = cfg or {}
        cfg["query_index"] = args.query_index
    if args.top_k is not None:
        cfg = cfg or {}
        cfg["top_k"] = args.top_k
    if args.query is not None:
        cfg = cfg or {}
        cfg["query"] = args.query

    return MultiTaskRerankerInference(cfg or {}).run()


if __name__ == "__main__":
    sys.exit(main())
