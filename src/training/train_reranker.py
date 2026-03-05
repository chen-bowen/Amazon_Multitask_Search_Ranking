"""
Train cross-encoder reranker for ESCI Task 1 (query-product ranking).

Training: MSE loss on (query, product) pairs with ESCI gains (E=1.0, S=0.1, C=0.01, I=0.0).
The model predicts a scalar score; we regress toward the gain value.
Uses product_text (expanded) or product_title (ESCI-exact) per config.

Evaluation: nDCG, MRR, MAP, Recall@k on test set. Paper baseline ~0.852 nDCG for US.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.constants import DATA_DIR, ESCI_LABEL2GAIN, MODEL_CACHE_DIR, REPO_ROOT, DEFAULT_RERANKER_MODEL
from src.utils import resolve_device
from src.data.load_data import load_esci, prepare_train_val_test
from src.eval.evaluator import ESCIMetricsEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Defaults for config + CLI (config overrides these; CLI overrides config)
DEFAULTS = {
    "data_dir": "data",
    "model_name": DEFAULT_RERANKER_MODEL,
    "product_col": "product_text",
    "save_path": "data/reranker",
    "epochs": 1,
    "batch_size": 16,
    "lr": 7e-6,
    "warmup_steps": 5000,
    "evaluation_steps": 15000,
    "early_stopping_patience": 0,
    "val_frac": 0.1,
}


def load_data(base: Path, *, small_version: bool, val_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test DataFrames via load_esci + prepare_train_val_test.
    Val is a held-out subset of train (by query_id) for mid-training eval; test is
    completely held out until final eval.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Train, validation, and test DataFrames.
    """
    df = load_esci(data_dir=base, small_version=small_version)
    return prepare_train_val_test(df=df, small_version=small_version, val_frac=val_frac)


def build_dataloader(
    train_df: pd.DataFrame,
    *,
    product_col: str,
    batch_size: int,
) -> DataLoader:
    """Build DataLoader of InputExamples for CrossEncoder.fit()."""
    samples = []
    for _, row in train_df.iterrows():
        gain = ESCI_LABEL2GAIN.get(str(row["esci_label"]), 0.0)
        samples.append(InputExample(texts=[str(row["query"]), str(row[product_col])], label=float(gain)))
    return DataLoader(samples, shuffle=True, batch_size=batch_size, drop_last=True)


def create_model(
    model_name: str,
    *,
    max_length: int = 512,
    device: str | None = None,
    cache_folder: Path | str | None = MODEL_CACHE_DIR,
) -> CrossEncoder:
    """
    Create CrossEncoder with regression head (num_labels=1, Identity activation).
    Uses cache_folder so the model is downloaded once and reused.

    Parameters
    ----------
    model_name : str
        Name of the model to use.
    max_length : int
        Max length of the input sequence.
    device : str | None
        Device to use.
    cache_folder : Path | str | None
        Where to cache downloaded models. Default: data/.model_cache.

    Returns
    -------
    CrossEncoder
        CrossEncoder model.
    """
    device = str(resolve_device(device))
    cache = str(cache_folder) if cache_folder else None
    return CrossEncoder(
        model_name,
        num_labels=1,
        max_length=max_length,
        activation_fn=torch.nn.Identity(),
        device=device,
        cache_folder=cache,
    )


def run_training(
    data_dir: Path | str | None = None,
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    product_col: str = "product_text",
    save_path: str | Path | None = "data/reranker",
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 7e-6,
    warmup_steps: int = 5000,
    max_length: int = 512,
    evaluation_steps: int = 5000,
    eval_max_queries: int | None = None,
    small_version: bool = False,
    device: str | None = None,
    early_stopping_patience: int = 0,
    val_frac: float = 0.1,
):
    """
    Train cross-encoder reranker on ESCI (ESCI baseline approach).

    Parameters
    ----------
    data_dir : Path | str | None
        Directory with ESCI parquets or raw data.
    model_name : str
        Pretrained cross-encoder (e.g. ms-marco-MiniLM-L-12-v2).
    product_col : str
        Column for product text: "product_text" (full) or "product_title" (ESCI exact).
    save_path : str | Path | None
        Where to save the trained model.
    epochs : int
        Number of epochs.
    batch_size : int
        Training batch size.
    lr : float
        Learning rate.
    warmup_steps : int
        Warmup steps.
    max_length : int
        Max sequence length for [query, product].
    evaluation_steps : int
        Evaluate every N steps (0 = no mid-training eval).
    eval_max_queries : int | None
        Subsample eval to this many queries (default: all).
    small_version : bool
        Use Task 1 reduced set (~48k queries) if loading from raw.
    """
    # Check if sentence-transformers is installed
    if CrossEncoder is None or InputExample is None:
        raise ImportError("sentence-transformers is required for train_reranker")
    base = Path(data_dir or DATA_DIR)

    # Load train, validation, and test data (val for mid-training eval; test held out)
    train_df, val_df, test_df = load_data(base, small_version=small_version, val_frac=val_frac)

    logger.info("Data:")
    logger.info("------")
    logger.info("data_dir=%s", base)
    logger.info("small_version=%s", small_version)
    logger.info("product_col=%s", product_col)
    logger.info("train_rows=%d val_rows=%d test_rows=%d", len(train_df), len(val_df), len(test_df))
    # Prefer MPS on Apple Silicon when device not set (faster than CPU)
    if device is None and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon GPU) for training.")
    logger.info("Training:")
    logger.info("------")
    logger.info("model=%s device=%s", model_name, device or "auto")
    logger.info("epochs=%d batch_size=%d lr=%g", epochs, batch_size, lr)
    logger.info("warmup_steps=%d max_length=%d", warmup_steps, max_length)
    logger.info("eval_steps=%d eval_max_queries=%s", evaluation_steps, eval_max_queries)
    logger.info("save_path=%s", save_path)

    if early_stopping_patience > 0:
        logger.info("early_stopping_patience=%d", early_stopping_patience)
    logger.info("val_frac=%g (val used for mid-training eval; test held out until end)", val_frac)

    # Check if train_df has the required columns
    if "esci_label" not in train_df.columns:
        raise ValueError("train_df must have 'esci_label' (from load_esci)")
    if product_col not in train_df.columns:
        raise ValueError(f"train_df must have '{product_col}'")

    model = create_model(model_name, max_length=max_length, device=device)
    train_dataloader = build_dataloader(train_df, product_col=product_col, batch_size=batch_size)
    output_path = str(save_path) if save_path else None

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    evaluator = None
    if len(val_df) > 0 and evaluation_steps > 0:
        evaluator = SequentialEvaluator(
            [ESCIMetricsEvaluator(val_df, product_col=product_col, max_queries=eval_max_queries, batch_size=batch_size)]
        )

    # MSE loss for ESCI gain regression; Identity activation (logits = scores)
    model.fit(
        train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        loss_fct=torch.nn.MSELoss(),
        activation_fct=torch.nn.Identity(),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        evaluation_steps=evaluation_steps,
        output_path=output_path,
        save_best_model=True,
    )

    if output_path:
        model.save(output_path)
        logger.info("Reranker saved to %s", output_path)

    # Final eval on held-out test set (only now, after training is complete)
    if len(test_df) > 0:
        logger.info("------")
        logger.info("Final eval on held-out test set:")
        test_evaluator = ESCIMetricsEvaluator(test_df, product_col=product_col, max_queries=eval_max_queries, batch_size=batch_size)
        test_evaluator(model, output_path=None, epoch=-1, steps=-1)
        m = test_evaluator._last_metrics
        logger.info("  nDCG=%.4f MRR=%.4f MAP=%.4f Recall@10=%.4f", m["ndcg"], m["mrr"], m["map"], m["recall"])

    return model


def main() -> int:
    """CLI entrypoint: load config from YAML and run training."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Train cross-encoder reranker (ESCI approach)")
    p.add_argument("--config", default="configs/reranker.yaml", help="Path to YAML config")
    args = p.parse_args()

    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    opts = DEFAULTS | (cfg or {})

    run_training(
        data_dir=opts.get("data_dir"),
        model_name=opts.get("model_name"),
        product_col=opts.get("product_col"),
        save_path=opts.get("save_path"),
        epochs=opts.get("epochs"),
        batch_size=opts.get("batch_size"),
        lr=opts.get("lr"),
        warmup_steps=opts.get("warmup_steps"),
        evaluation_steps=opts.get("evaluation_steps"),
        eval_max_queries=opts.get("eval_max_queries"),
        small_version=opts.get("small_version", False),
        device=opts.get("device"),
        early_stopping_patience=opts.get("early_stopping_patience"),
        val_frac=opts.get("val_frac"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
