"""
Early stopping callback for CrossEncoder training.

Monitors eval metric (nDCG via eval_sequential_score) and stops when no improvement
for `early_stopping_patience` evaluations.
"""

from __future__ import annotations

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when the monitored metric stops improving."""

    def __init__(
        self,
        early_stopping_patience: int = 3,
        metric_name: str = "eval_sequential_score",
        greater_is_better: bool = True,
    ):
        """Initialize the early stopping callback.

        Args:
            early_stopping_patience: Stop after this many evals without improvement.
            metric_name: Key in metrics dict (e.g. eval_sequential_score for nDCG).
            greater_is_better: True if higher metric is better (e.g. nDCG).
        """
        # Number of evals without improvement before stopping
        self.early_stopping_patience = early_stopping_patience
        # Key in metrics dict (eval_sequential_score = nDCG from SequentialEvaluator)
        self.metric_name = metric_name
        # True = higher is better (nDCG); False = lower is better (loss)
        self.greater_is_better = greater_is_better
        # Consecutive evals with no improvement
        self.patience_counter = 0
        # Best metric seen so far
        self.best_metric: float | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ) -> TrainerControl:
        """Called after each evaluation; sets control.should_training_stop if patience exceeded."""
        # Skip if no metrics (e.g. first call)
        if metrics is None:
            return control
        value = metrics.get(self.metric_name)
        if value is None:
            return control
        # Handle string metrics from Trainer logs
        try:
            value = float(value)
        except (TypeError, ValueError):
            return control

        # First eval: record as best, reset patience
        if self.best_metric is None:
            self.best_metric = value
            self.patience_counter = 0
            return control

        # Check if metric improved
        improved = (
            value > self.best_metric
            if self.greater_is_better
            else value < self.best_metric
        )
        if improved:
            self.best_metric = value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            # Stop once we've seen patience evals with no improvement
            if self.patience_counter >= self.early_stopping_patience:
                control.should_training_stop = True
        return control
