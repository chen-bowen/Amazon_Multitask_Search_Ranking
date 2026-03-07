"""Training pipelines for ESCI rerankers.

Provides RerankerTrainer (single-task CrossEncoder) and MultiTaskTrainer
(shared encoder + three heads). Both support YAML config and CLI entrypoints.
"""

__all__ = ["RerankerTrainer", "MultiTaskTrainer"]


def __getattr__(name: str):
    if name == "RerankerTrainer":
        from .train_reranker import RerankerTrainer

        return RerankerTrainer
    if name == "MultiTaskTrainer":
        from .train_multi_task_reranker import MultiTaskTrainer

        return MultiTaskTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
