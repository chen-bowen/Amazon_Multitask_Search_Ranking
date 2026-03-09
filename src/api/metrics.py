"""Prometheus metrics for ESCI Reranker API (custom registry, no process/GC metrics)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

API_REGISTRY = CollectorRegistry()

# ---------------------------------------------------------------------------
# Counters: request counts per endpoint (label: status = success | error)
# ---------------------------------------------------------------------------

RERANK_REQUESTS_TOTAL = Counter(
    "rerank_requests_total",
    "Total number of /rerank requests",
    ["status"],
    registry=API_REGISTRY,
)

CLASSIFY_REQUESTS_TOTAL = Counter(
    "classify_requests_total",
    "Total number of /classify requests",
    ["status"],
    registry=API_REGISTRY,
)

SUBSTITUTE_REQUESTS_TOTAL = Counter(
    "substitute_requests_total",
    "Total number of /substitute requests",
    ["status"],
    registry=API_REGISTRY,
)

PREDICT_REQUESTS_TOTAL = Counter(
    "predict_requests_total",
    "Total number of /predict requests",
    ["status"],
    registry=API_REGISTRY,
)

# ---------------------------------------------------------------------------
# Histograms: latency per endpoint (seconds)
# ---------------------------------------------------------------------------

RERANK_LATENCY_SECONDS = Histogram(
    "rerank_latency_seconds",
    "End-to-end latency for /rerank in seconds",
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0),
    registry=API_REGISTRY,
)

CLASSIFY_LATENCY_SECONDS = Histogram(
    "classify_latency_seconds",
    "End-to-end latency for /classify in seconds",
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0),
    registry=API_REGISTRY,
)

SUBSTITUTE_LATENCY_SECONDS = Histogram(
    "substitute_latency_seconds",
    "End-to-end latency for /substitute in seconds",
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0),
    registry=API_REGISTRY,
)

PREDICT_LATENCY_SECONDS = Histogram(
    "predict_latency_seconds",
    "End-to-end latency for /predict in seconds",
    buckets=(0.05, 0.1, 0.5, 1.0, 5.0),
    registry=API_REGISTRY,
)

# ---------------------------------------------------------------------------
# Gauge: model readiness
# ---------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the reranker model is loaded and ready, 0 otherwise",
    registry=API_REGISTRY,
)
