"""
Evaluate two-tower model on ESCI test set using InformationRetrievalEvaluator (nDCG@k, MRR@k).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from src.data.load_data import load_esci, prepare_train_test
from src.models.two_tower import TwoTowerEncoder  # Two-tower model

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory


def build_ir_eval_data(test_df: pd.DataFrame, min_relevance: int = 2) -> tuple[dict, dict, dict]:
    """
    Build (queries, corpus, relevant_docs) for InformationRetrievalEvaluator from ESCI-style test_df.
    Corpus = unique product_id -> product_text; relevant_docs = query_id -> set of product_ids with relevance >= min_relevance.
    """
    df = test_df.copy()
    if "product_id" not in df.columns:
        raise ValueError("test_df must have 'product_id' for InformationRetrievalEvaluator (from load_esci).")
    queries = df.groupby("query_id")["query"].first().astype(str).to_dict()
    queries = {str(k): v for k, v in queries.items()}
    corpus_df = df[["product_id", "product_text"]].drop_duplicates("product_id")
    corpus = {str(pid): str(text) for pid, text in zip(corpus_df["product_id"], corpus_df["product_text"])}
    relevant_docs: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        if row["relevance"] >= min_relevance:
            qid = str(row["query_id"])
            pid = str(row["product_id"])
            relevant_docs.setdefault(qid, set()).add(pid)
    return queries, corpus, relevant_docs


def run_evaluation(
    model: TwoTowerEncoder | None = None,
    model_path: str | Path | None = None,
    test_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """
    Evaluate model on test set: InformationRetrievalEvaluator (nDCG@k, MRR@k) over full per-query ranking.
    """
    if test_df is None:
        base = Path(data_dir or DATA_DIR)
        test_path = base / "esci_test.parquet"
        if not test_path.exists():
            # Load raw ESCI parquets from base data directory (expects files directly under `data/`).
            _df = load_esci(data_dir=base)
            _, test_df = prepare_train_test(df=_df)
        else:
            test_df = pd.read_parquet(test_path)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if model is None:
        model = TwoTowerEncoder(model_name=model_name, shared=False, normalize=True)
        if model_path is not None:
            ckpt = torch.load(Path(model_path), map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device)

    queries_ir, corpus_ir, relevant_docs_ir = build_ir_eval_data(test_df)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries_ir,
        corpus=corpus_ir,
        relevant_docs=relevant_docs_ir,
        name="esci-eval",
        show_progress_bar=False,
        write_csv=False,
        mrr_at_k=[k],
        ndcg_at_k=[k],
    )
    results = ir_evaluator(model)
    ndcg_key = f"esci-eval_cosine_ndcg@{k}"
    mrr_key = f"esci-eval_cosine_mrr@{k}"
    return {
        "nDCG@k": float(results.get(ndcg_key, 0.0)),
        "MRR@k": float(results.get(mrr_key, 0.0)),
    }


def main() -> int:
    """Command-line entry point: parse args, load config, run evaluation."""
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Evaluate two-tower on ESCI test set")
    p.add_argument("--config", type=str, default="configs/eval.yaml")
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--k", type=int, default=None)
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)
    metrics = run_evaluation(
        model_path=args.model_path or cfg.get("model_path", "data/model.pt"),
        data_dir=data_dir,
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L6-v2"),
        k=args.k if args.k is not None else cfg.get("k", 10),
    )
    logger.info("Metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Evaluate two-tower model on ESCI test set: compute nDCG@k and MRR.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

from src.eval.metrics import evaluate_ranking  # Ranking evaluation functions
from src.models.two_tower import TwoTowerEncoder  # Two-tower model

REPO_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels to project root
DATA_DIR = REPO_ROOT / "data"  # Path to data directory
BATCH_SIZE = 128  # Batch size for encoding (not currently used, but available)


def run_evaluation(
    model: TwoTowerEncoder | None = None,
    model_path: str | Path | None = None,
    test_df: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    k: int = 10,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """
    Evaluate model on test set: compute embeddings, rank products, compute nDCG@k and MRR.
    """
    if test_df is None:  # If test DataFrame not provided
        base = Path(data_dir or DATA_DIR)  # Use provided data_dir or default
        test_path = base / "esci_test.parquet"  # Path to preprocessed test parquet
        if not test_path.exists():  # If preprocessed file doesn't exist
            from src.data.load_data import load_esci, prepare_train_test

            # Load raw ESCI data and split into train/test
            _, test_df = prepare_train_test(data_dir=base / "esci-data" / "shopping_queries_dataset")
        else:  # If preprocessed file exists
            test_df = pd.read_parquet(test_path)  # Load preprocessed test data
    device = torch.device(device if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
    if model is None:  # If model not provided
        model = TwoTowerEncoder(model_name=model_name, shared=False, normalize=True)  # Create new model
        if model_path is not None:  # If model checkpoint path provided
            ckpt = torch.load(Path(model_path), map_location=device, weights_only=True)  # Load checkpoint
            model.load_state_dict(ckpt["model_state"], strict=True)  # Load model weights
        model = model.to(device)  # Move model to device (GPU/CPU)
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    test_df = test_df.copy()  # Copy DataFrame to avoid modifying original
    test_df["_row_idx"] = np.arange(len(test_df))  # Add row index column to track original order
    scores_arr = np.full(len(test_df), np.nan, dtype=float)  # Initialize scores array with NaN
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        for _qid, grp in test_df.groupby("query_id"):  # Group by query_id (iterate over each query)
            query = grp["query"].iloc[0]  # Get query string (same for all rows in group)
            products = grp["product_text"].tolist()  # Get list of product texts for this query
            row_indices = grp["_row_idx"].values  # Get original row indices for this group
            # Encode query (repeat for each product) and products
            q_embs = model.encode_queries([query] * len(products), device=device)  # [N, D] query embeddings
            p_embs = model.encode_products(products, device=device)  # [N, D] product embeddings
            # Compute similarity: dot product (cosine for normalized embeddings)
            sim = (q_embs * p_embs).sum(dim=1).cpu().numpy()  # [N] similarity scores, move to CPU
            scores_arr[row_indices] = sim  # Store scores at original row positions
    test_df.drop(columns=["_row_idx"], inplace=True)  # Remove temporary row index column
    # Evaluate ranking: compute nDCG@k and MRR per query, return averages
    return evaluate_ranking(test_df, scores_arr, query_id_col="query_id", relevance_col="relevance", k=k)


def main() -> int:
    """Command-line entry point: parse args, load config, run evaluation."""
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Evaluate two-tower on ESCI test set")  # Create argument parser
    p.add_argument("--config", type=str, default="configs/eval.yaml")  # Config file path
    p.add_argument("--model-path", type=str, default=None)  # Override model checkpoint path
    p.add_argument("--data-dir", type=str, default=None)  # Override data directory
    p.add_argument("--model-name", type=str, default=None)  # Override model name
    p.add_argument("--k", type=int, default=None)  # Override k for nDCG@k
    args = p.parse_args()  # Parse command-line arguments
    cfg = {}  # Initialize config dict
    config_path = REPO_ROOT / args.config  # Full path to config file
    if config_path.exists():  # If config file exists
        with open(config_path) as f:  # Open config file
            cfg = yaml.safe_load(f) or {}  # Load YAML into dict (empty dict if None)
    data_dir = Path(args.data_dir or cfg.get("data_dir") or DATA_DIR)  # Use arg > config > default
    # Call evaluation function with args/config merged (CLI args override config)
    metrics = run_evaluation(
        model_path=args.model_path or cfg.get("model_path", "data/model.pt"),  # Model checkpoint path
        data_dir=data_dir,  # Data directory
        model_name=args.model_name or cfg.get("model_name", "all-MiniLM-L6-v2"),  # Model name
        k=args.k if args.k is not None else cfg.get("k", 10),  # k for nDCG@k
    )
    logger.info("Metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())  # Run main and exit with return code
