"""
Train multi-task learning ESCI model: Task 1 (query-product ranking),
Task 2 (4-class E/S/C/I), Task 3 (substitute identification).
Shared encoder with three heads; combined loss with configurable task weights.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.constants import (
    DATA_DIR,
    ESCI_LABEL2GAIN,
    ESCI_LABEL2ID,
    MODEL_CACHE_DIR,
    REPO_ROOT,
    DEFAULT_RERANKER_MODEL,
)
from src.utils import clear_torch_cache
from src.data.load_data import load_esci, prepare_train_val_test
from src.eval.evaluator import ESCIMetricsEvaluator
from src.models.multi_task_reranker import MultiTaskReranker

logger = logging.getLogger(__name__)

DEFAULTS = {
    "data_dir": "data",
    "model_name": DEFAULT_RERANKER_MODEL,
    "product_col": "product_text",
    "save_path": "checkpoints/multi_task_reranker",
    "val_frac": 0.1,
    "epochs": 1,
    "batch_size": 16,
    "max_length": 512,
    "lr": 7e-6,
    "warmup_steps": 5000,
    "task_weight_ranking": 1.0,
    "task_weight_esci": 0.5,
    "task_weight_substitute": 0.5,
    "evaluation_steps": 15000,
    "eval_max_queries": None,
    "recall_at": 10,
}


class MultiTaskDataset(Dataset):
    """
    Dataset of (query, product) pairs with multi-task learning targets:
    gain (Task 1: query-product ranking),
    class_id (Task 2: 4-class E/S/C/I),
    is_substitute (Task 3: substitute identification).
    """

    def __init__(
        self,
        pairs: list,
        gains: list,
        class_ids: list,
        is_substitute: list,
        tokenizer,
        max_length: int = 512,
    ) -> None:
        self.pairs = pairs
        self.gains = gains
        self.class_ids = class_ids
        self.is_substitute = is_substitute
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        enc = self.tokenizer(
            [self.pairs[i]],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        out = {k: v.squeeze(0) for k, v in enc.items()}
        out["gain"] = torch.tensor(self.gains[i], dtype=torch.float)
        out["class_id"] = torch.tensor(self.class_ids[i], dtype=torch.long)
        out["is_substitute"] = torch.tensor(
            float(self.is_substitute[i]), dtype=torch.float
        )
        return out


def build_multi_task_dataloader(
    train_df: pd.DataFrame,
    tokenizer,
    *,
    product_col: str,
    batch_size: int,
    max_length: int = 512,
) -> DataLoader:
    """
    Build DataLoader with (query, product_text, gain, class_id, is_substitute)
    per sample for multi-task learning (Task 1, Task 2, Task 3).
    Tokenization runs in MultiTaskDataset.__getitem__.
    """
    pairs = []
    gains = []
    class_ids = []
    is_substitute = []
    for _, row in train_df.iterrows():
        label = str(row["esci_label"])
        pairs.append([str(row["query"]), str(row[product_col])])
        gains.append(ESCI_LABEL2GAIN.get(label, 0.0))
        class_ids.append(ESCI_LABEL2ID.get(label, 3))  # default I
        is_substitute.append(1.0 if label == "S" else 0.0)

    dataset = MultiTaskDataset(
        pairs, gains, class_ids, is_substitute, tokenizer, max_length=max_length
    )
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )


class MultiTaskEvalWrapper:
    """
    Thin wrapper so ESCIMetricsEvaluator (expects model.predict -> scores only)
    works with multi-task learning reranker.
    """

    def __init__(self, model: MultiTaskReranker) -> None:
        self.model = model

    @property
    def device(self) -> torch.device:
        return self.model.device

    def predict(
        self, texts: list, batch_size: int = 32, show_progress_bar: bool = False
    ) -> list:
        scores, _, _ = self.model.predict(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar
        )
        return scores


def run_training(
    data_dir: Path | str | None = None,
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    product_col: str = "product_text",
    save_path: str | Path | None = "checkpoints/multi_task_reranker",
    epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 512,
    lr: float = 7e-6,
    warmup_steps: int = 5000,
    task_weight_ranking: float = 1.0,
    task_weight_esci: float = 0.5,
    task_weight_substitute: float = 0.5,
    evaluation_steps: int = 15000,
    eval_max_queries: int | None = None,
    small_version: bool = False,
    device: str | None = None,
    val_frac: float = 0.1,
    recall_at: int = 10,
) -> MultiTaskReranker:
    """
    Train multi-task learning reranker on ESCI with combined loss
    (ranking + 4-class + substitute).

    Parameters
    ----------
    data_dir : Path | str | None
        Directory with ESCI parquets or raw data.
    model_name : str
        Pretrained encoder (e.g. cross-encoder/ms-marco-MiniLM-L-12-v2).
    product_col : str
        Column for product text.
    save_path : str | Path | None
        Where to save the trained multi-task learning checkpoint.
    epochs : int
        Number of epochs.
    batch_size : int
        Training batch size.
    max_length : int
        Max sequence length.
    lr : float
        Learning rate.
    warmup_steps : int
        Linear warmup steps.
    task_weight_ranking : float
        Weight for MSE for query-product ranking (Task 1) in combined loss.
    task_weight_esci : float
        Weight for CrossEntropy for 4-class ESCI (Task 2) in combined loss.
    task_weight_substitute : float
        Weight for BCE for substitute identification (Task 3) in combined loss.
    evaluation_steps : int
        Evaluate every N steps (0 = no mid-training eval).
    eval_max_queries : int | None
        Subsample val queries for eval (None = all).
    small_version : bool
        Use query-product ranking (Task 1) reduced set if loading from raw.
    device : str | None
        Device (e.g. mps, cuda, cpu).
    val_frac : float
        Fraction of train for validation.
    recall_at : int
        Recall@k for eval metrics.
    """
    base = Path(data_dir or DATA_DIR)
    df = load_esci(data_dir=base, small_version=small_version)
    train_df, val_df, test_df = prepare_train_val_test(
        df=df, small_version=small_version, val_frac=val_frac
    )

    logger.info(
        "Data: data_dir=%s train_rows=%d val_rows=%d test_rows=%d",
        base,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    if device is None and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using MPS (Apple Silicon GPU) for training.")
    model = MultiTaskReranker(
        model_name=model_name,
        max_length=max_length,
        device=device,
        cache_folder=MODEL_CACHE_DIR,
    )
    train_dl = build_multi_task_dataloader(
        train_df,
        model.tokenizer,
        product_col=product_col,
        batch_size=batch_size,
        max_length=max_length,
    )
    output_path = str(save_path) if save_path else None
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)

    evaluator = None
    if len(val_df) > 0 and evaluation_steps > 0:
        evaluator = ESCIMetricsEvaluator(
            val_df,
            product_col=product_col,
            max_queries=eval_max_queries,
            batch_size=batch_size,
            recall_at_k=recall_at,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_steps = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=warmup_steps, num_training_steps=num_steps
    )
    best_ndcg: float | None = None
    global_step = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        for batch in pbar:
            dev = model.device
            gain = batch["gain"].to(dev)
            class_id = batch["class_id"].to(dev)
            is_sub = batch["is_substitute"].to(dev)
            token_type_ids = batch.get("token_type_ids")
            scores, esci_logits, sub_logits = model(
                batch["input_ids"].to(dev),
                batch["attention_mask"].to(dev),
                token_type_ids.to(dev) if token_type_ids is not None else None,
            )
            l1 = F.mse_loss(scores, gain)
            l2 = F.cross_entropy(esci_logits, class_id)
            l3 = F.binary_cross_entropy_with_logits(sub_logits, is_sub)
            loss = (
                task_weight_ranking * l1
                + task_weight_esci * l2
                + task_weight_substitute * l3
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

            if (
                evaluator is not None
                and evaluation_steps > 0
                and global_step % evaluation_steps == 0
            ):
                model.eval()
                clear_torch_cache()
                ndcg = evaluator(
                    MultiTaskEvalWrapper(model),
                    output_path=None,
                    epoch=epoch,
                    steps=global_step,
                )
                model.train()
                if best_ndcg is None or ndcg > best_ndcg:
                    best_ndcg = ndcg
                    if output_path:
                        model.save(output_path)
                        logger.info("Save model to %s", output_path)

    if output_path:
        model.save(output_path)
        logger.info("Multi-task learning reranker saved to %s", output_path)

    if len(test_df) > 0:
        logger.info("------")
        logger.info("Final eval on held-out test set:")
        test_eval = ESCIMetricsEvaluator(
            test_df,
            product_col=product_col,
            max_queries=eval_max_queries,
            batch_size=batch_size,
            recall_at_k=recall_at,
        )
        test_eval(MultiTaskEvalWrapper(model), output_path=None, epoch=-1, steps=-1)
        m = test_eval.last_metrics
        logger.info(
            "  nDCG=%.4f MRR=%.4f MAP=%.4f Recall@%d=%.4f",
            m["ndcg"],
            m["mrr"],
            m["map"],
            recall_at,
            m["recall"],
        )

    return model


def main() -> int:
    """
    CLI entrypoint: load config from YAML and run multi-task learning training.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description="Train multi-task learning ESCI reranker (ranking + 4-class + substitute)"
    )
    p.add_argument(
        "--config",
        default="configs/multi_task_reranker.yaml",
        help="Path to YAML config",
    )
    args = p.parse_args()
    cfg: dict = {}
    config_path = REPO_ROOT / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
    configs = DEFAULTS | (cfg or {})

    run_training(
        data_dir=configs.get("data_dir"),
        model_name=configs.get("model_name"),
        product_col=configs.get("product_col"),
        save_path=configs.get("save_path"),
        epochs=configs.get("epochs"),
        batch_size=configs.get("batch_size"),
        max_length=configs.get("max_length"),
        lr=configs.get("lr"),
        warmup_steps=configs.get("warmup_steps"),
        task_weight_ranking=configs.get("task_weight_ranking"),
        task_weight_esci=configs.get("task_weight_esci"),
        task_weight_substitute=configs.get("task_weight_substitute"),
        evaluation_steps=configs.get("evaluation_steps"),
        eval_max_queries=configs.get("eval_max_queries"),
        small_version=configs.get("small_version", False),
        device=configs.get("device"),
        val_frac=configs.get("val_frac"),
        recall_at=configs.get("recall_at"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
