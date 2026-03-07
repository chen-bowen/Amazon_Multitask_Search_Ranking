"""Evaluation utilities for ESCI rerankers.

Provides ESCIMetricsEvaluator (nDCG, MRR, MAP, Recall@k) and
evaluate_classification_tasks for multi-task Task 2/3 metrics.
"""

from .evaluator import ESCIMetricsEvaluator, compute_query_metrics

__all__ = ["ESCIMetricsEvaluator", "compute_query_metrics"]
