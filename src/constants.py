from __future__ import annotations

from pathlib import Path

# Project root and common data/checkpoint paths.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
# Directory for trained model checkpoints (separate from raw/processed data).
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
# Cache directory for downloaded pretrained weights.
MODEL_CACHE_DIR = DATA_DIR / ".model_cache"

# Default model and eval settings.
DEFAULT_MODEL_NAME = "all-MiniLM-L12-v2"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_EVAL_K = 10

# ESCI label -> gain for nDCG (paper: E=1.0, S=0.1, C=0.01, I=0.0)
ESCI_LABEL2GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

# ESCI label -> class index for multi-task learning Task 2 (4-way E/S/C/I classification). Order: E, S, C, I.
ESCI_LABEL2ID = {"E": 0, "S": 1, "C": 2, "I": 3}
ESCI_ID2LABEL = ["E", "S", "C", "I"]
