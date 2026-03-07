"""FastAPI service for ESCI multi-task reranker.

Exposes POST /rerank (rerank product candidates for a query) and GET /health.
Model loaded at startup from MODEL_PATH env; fallback to MODEL_NAME if path invalid.
"""
