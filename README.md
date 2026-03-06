# Amazon Search Queries Ranking

Cross-encoder reranker for [Amazon ESCI Task 1](https://github.com/amazon-science/esci-data) (query-product ranking). Trained on the [Shopping Queries Dataset](https://arxiv.org/abs/2206.06588) with MSE loss on graded ESCI relevance gains. Paper baseline: nDCG ~0.852 (US); this implementation reaches ~0.96 on validation. Includes data loading (train/val/test split by query_id), training with early stopping, evaluation (nDCG, MRR, MAP, Recall@k), and Python API for scoring and reranking.

**Contents:** [Prediction problem](#prediction-problem) · [Architecture](#overall-architecture) · [Requirements](#requirements) · [Setup](#setup) · [How to use each component](#how-to-use-each-component) · [Data](#data) · [Pipeline](#pipeline) · [Training](#training) · [Results](#results) · [Inference](#inference) · [Troubleshooting](#troubleshooting) · [Project structure](#project-structure)

---

## Prediction Problem

- **Task:** Rank products for a search query so that the most relevant items appear at the top (ESCI Task 1).
- **Input:** A (query, product) pair. The query is the user's search string; the product is text (title, description, or both).
- **Output:** A scalar relevance score. Higher scores mean more relevant. Used to rerank a candidate list of products.
- **Train vs serve:** For **training**, each example is (query, product) with a target gain (E=1.0, S=0.1, C=0.01, I=0.0). For **serve and evaluation**, the model scores arbitrary (query, product) pairs and we rank by score.

### ESCI labels and gains

| Label | Meaning     | Gain (nDCG) |
| ----- | ----------- | ----------- |
| E     | Exact match | 1.0         |
| S     | Substitute  | 0.1         |
| C     | Complement  | 0.01        |
| I     | Irrelevant  | 0.0         |

Training uses MSE loss: the model predicts a scalar and is trained to match these gain values.

---

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Preparation                                                         │
│    Raw parquets → load_esci → prepare_train_val_test → train/val/test DFs  │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Training                                                                 │
│    CrossEncoder (ms-marco-MiniLM) + MSE loss → val eval every N steps      │
│    Early stopping on nDCG → best checkpoint saved to data/reranker          │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Inference                                                                │
│    load_reranker() → predict(pairs) or rerank(query, candidates)            │
│    Returns scores or ranked (product_id, score) list                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## How this model could be used

- **Search reranking:** A retrieval stage (e.g. BM25, two-tower) returns candidates; the cross-encoder reranks them for final display.
- **Product search APIs:** Expose a `/rerank` endpoint: input = query + candidate product IDs/texts, output = top-k by score.
- **A/B testing:** Compare cross-encoder reranking vs no reranking or vs a simpler heuristic on click-through, add-to-cart, or conversion.
- **Batch precompute:** For popular queries, precompute top-k products and cache; refresh periodically as the model or catalog changes.

---

## Requirements

- **Python** 3.12+ (managed via `uv` or your environment).
- **Amazon ESCI dataset** from [amazon-science/esci-data](https://github.com/amazon-science/esci-data). Place parquets under `data/`.
- **Disk:** ~1–2 GB for raw data + processed splits; model checkpoints add ~100–200 MB.
- **Memory:** 8 GB RAM is enough for data prep and inference; training benefits from 16 GB+ and a GPU (CUDA or Apple MPS) for speed.

---

## Setup

1. **Clone or open the repo** and enter the project root.
2. **Install dependencies** (prefer `uv` for a locked environment):

   ```bash
   uv sync
   ```

   Or with pip: `pip install -e .` (see `pyproject.toml` for dependencies).

3. **Download the Amazon ESCI parquets** into `data/`:
   - `shopping_queries_dataset_products.parquet`
   - `shopping_queries_dataset_examples.parquet`

4. **Materialize train/test splits:**

   ```bash
   uv run python -m src.data.load_data --save-splits
   ```

   Writes `data/esci_train.parquet` and `data/esci_test.parquet`.

5. **Verify:** Run training (see [Pipeline](#pipeline)); it will fail with a clear error if any file is missing.

---

## How to use each component

| Component     | Command / Usage                                                                             | When to use                                         |
| ------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Data prep** | `uv run python -m src.data.load_data --save-splits`                                         | First step: write train/test parquets from raw.     |
| **Train**     | `uv run python -m src.training.train_reranker --config configs/reranker.yaml`               | Train the cross-encoder reranker.                   |
| **Eval**      | `uv run python -m src.eval.eval_reranker --config configs/reranker.yaml`                    | Standalone eval on test set (nDCG, MRR, MAP, R@10). |
| **Inference** | `from src.models.reranker import load_reranker`; `reranker.predict()` / `reranker.rerank()` | Score pairs or rerank candidates in Python.         |
| **Tests**     | `uv run pytest tests/ -v`                                                                   | Run unit tests.                                     |

**Typical workflow:** 1) Data prep → 2) Train → 3) Eval (or run eval at end of training) → 4) Use `load_reranker` for inference.

---

## Data

### Input files (under `data/`)

| File                                          | Key columns                                                    | Role                                                     |
| --------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------- |
| **shopping_queries_dataset_examples.parquet** | query_id, query, product_id, esci_label, split                 | Query–product pairs and ESCI labels; split = train/test. |
| **shopping_queries_dataset_products.parquet** | product_id, product_locale, product_title, product_description | Product metadata; merged with examples for full text.    |

### Data prep output

| Output                 | Description                        |
| ---------------------- | ---------------------------------- |
| **esci_train.parquet** | Training split (split == "train"). |
| **esci_test.parquet**  | Test split (split == "test").      |

`load_esci` adds `product_text` (title + description) for richer input. Use `product_title` for shorter, ESCI-exact style. **Task 1 (small_version):** Reduced set (~48k queries) for faster experiments.

### Train/val/test split (at training time)

- **Train:** 90% of training split, by query_id (no query in both train and val).
- **Validation:** 10% of training split; used for mid-training eval and early stopping.
- **Test:** Original test split; used only for final evaluation after training.

---

## Pipeline

### 1. Prepare

Reads the parquets, merges examples with products, and writes train/test parquets using the dataset's `split` column.

```bash
uv run python -m src.data.load_data --save-splits
```

### 2. Train

Loads train/val/test, builds a CrossEncoder with MSE loss, and trains with validation every N steps. Saves the best checkpoint (by val nDCG) to `data/reranker`. Final test-set metrics are logged at the end.

```bash
uv run python -m src.training.train_reranker --config configs/reranker.yaml
```

**Config:** Edit `configs/reranker.yaml` for all settings. Use `--config path/to/other.yaml` to load a different config.

### 3. Eval (standalone)

Run evaluation on the test set without training:

```bash
uv run python -m src.eval.eval_reranker --config configs/reranker.yaml
```

### Commands

```bash
# 1. Prepare
uv run python -m src.data.load_data --save-splits

# 2. Train (uses configs/reranker.yaml)
uv run python -m src.training.train_reranker --config configs/reranker.yaml

# 3. Eval
uv run python -m src.eval.eval_reranker --config configs/reranker.yaml

# Different config
uv run python -m src.training.train_reranker --config configs/reranker_fast.yaml
```

---

## Training

- **Runtime:** On Apple Silicon (MPS): ~3.3 it/s training, ~10 query/s for eval. One epoch (~78k steps) takes several hours. On CUDA, expect faster steps.
- **Hyperparameters:** Default `lr=7e-6`, `batch_size=16`, `warmup_steps=5000`. Config in `configs/reranker.yaml`. Early stopping (patience=3) stops if val nDCG doesn't improve for 3 evals.
- **Checkpoints:** Best by val nDCG saved to `data/reranker`; final `model.save()` at end of training.

---

## Results

### Evaluation metrics

Setup: default config (`configs/reranker.yaml`), 1 epoch, batch size 16, ~78k steps, validation every 15k steps. Evaluation runs over the validation set (~7.5k queries) during training; test set is held out until final eval.

| Step   | Val nDCG   | Val MRR    | Val MAP    | Val Recall@10 |
| ------ | ---------- | ---------- | ---------- | ------------- |
| 15,000 | 0.9565     | 0.9875     | 0.9689     | 0.6131        |
| 30,000 | 0.9580     | 0.9877     | 0.9699     | 0.6127        |
| 45,000 | 0.9587     | 0.9882     | 0.9705     | 0.6141        |
| 60,000 | **0.9594** | **0.9886** | **0.9709** | **0.6144**    |

**What the metrics mean:** nDCG = normalized discounted cumulative gain with graded gains (E/S/C/I). MRR = mean reciprocal rank of first relevant item. MAP = mean average precision. Recall@10 = fraction of relevant items in top-10. All computed per query and averaged; higher is better.

Best checkpoint (by val nDCG) saved to `data/reranker`. Final test-set metrics are logged at the end of training.

**Reproducibility:** Exact numbers depend on hardware, seed, and hyperparameters. Use the same config and flags to approximate these results.

---

## Inference

### Python API

```python
from src.models.reranker import load_reranker

reranker = load_reranker(model_path="data/reranker")

# Score (query, product) pairs
pairs = [
    ["wireless bluetooth headphones", "Sony WH-1000XM4 Wireless Noise Cancelling Headphones"],
    ["wireless bluetooth headphones", "USB-C Cable 6ft"],
]
scores = reranker.predict(pairs)

# Rerank candidates for a single query
candidates = [
    ("prod_1", "Sony WH-1000XM4 Wireless Noise Cancelling Headphones"),
    ("prod_2", "USB-C Cable 6ft"),
]
ranked = reranker.rerank("wireless bluetooth headphones", candidates)
# -> [(product_id, score), ...] sorted by score descending
```

### CLI helper: sample inference on ESCI test set

There is a small script to inspect rankings on the ESCI **test** split:

```bash
uv run python -m src.inference.infer_reranker \
  --config configs/reranker.yaml \
  --query-index 0 \
  --top-k 5
```

- **Candidate products** are always taken from the ESCI **test set**: for the chosen `query_id`, all rows with that `query_id` become candidates.
- `query_index` selects which `query_id` to use (index over unique `query_id`s in the test set).
- You can override the query text while keeping the same candidate pool:

  ```bash
  uv run python -m src.inference.infer_reranker \
    --config configs/reranker.yaml \
    --query-index 0 \
    --query "screen privacy fence without holes"
  ```

  This is useful for quick experiments, but if your query is unrelated to that `query_id` and you do **not** also change the candidates, the ranking can look odd because the products themselves are mismatched to the query.

For real serving, you should:

1. Use your own retrieval stage (BM25, ANN, etc.) to build a candidate list for a user query.
2. Call `reranker.rerank(query, candidates)` from Python with those candidates, instead of relying on the ESCI test-set helper script.

### Example output

```
Top-5 reranked for "wireless bluetooth headphones":
  1. prod_1 (score=0.89) Sony WH-1000XM4 Wireless Noise Cancelling Headphones
  2. prod_2 (score=0.12) USB-C Cable 6ft
  ...
```

---

## Multi-task learning model

A multi-task learning variant trains one shared encoder with three heads for all ESCI tasks:

- **Task 1 (ranking):** Regression to ESCI gain; same nDCG evaluation.
- **Task 2 (4-class):** E/S/C/I classification (CrossEntropy).
- **Task 3 (substitute):** Binary “is substitute?” (BCE; label = 1 when ESCI = S).

Train with:

```bash
uv run python -m src.training.train_multi_task_reranker --config configs/multi_task_reranker.yaml
```

Config: `configs/multi_task_reranker.yaml` (task weights, lr, `save_path`, etc.). Checkpoint is saved to `data/multi_task_reranker` by default. The API and Docker image use this multi-task learning model so that `/rerank` returns score, `esci_class`, and `is_substitute` per product.

---

## API

A FastAPI service exposes the multi-task learning reranker for HTTP calls.

### Run locally

```bash
# From repo root; ensure data/multi_task_reranker exists or set MODEL_NAME for fallback.
export MODEL_PATH=data/multi_task_reranker
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

- **GET /health** – Returns `{"status": "ok", "model_loaded": true/false}` for load balancers and Docker.
- **POST /rerank** – Request body: `{"query": "wireless headphones", "candidates": [{"product_id": "p1", "text": "Sony WH-1000XM4 ..."}]}`. Response: `{"ranked": [{"product_id": "p1", "score": 0.89, "esci_class": "E", "is_substitute": 0.02}]}` (sorted by score descending).

---

## Docker

Build and run the API in a container:

```bash
# Build (from repo root).
docker build -t esci-reranker-api .

# Run with model mounted from host (train multi-task learning first so data/multi_task_reranker exists).
docker run -p 8000:8000 -v "$(pwd)/data/multi_task_reranker:/app/data/multi_task_reranker:ro" -e MODEL_PATH=/app/data/multi_task_reranker esci-reranker-api
```

Or use docker-compose:

```bash
docker compose up --build
```

The container serves the app on port 8000. Set `MODEL_PATH` (and optionally `MODEL_NAME` for fallback) via env or compose; if the mounted path is empty, the app loads the pretrained model from Hugging Face.

---

## Troubleshooting

| Issue                    | Fix                                                                    |
| ------------------------ | ---------------------------------------------------------------------- |
| MPS OOM on Apple Silicon | Add `device: cpu` to config or reduce `batch_size` to 8.               |
| Slow eval                | Set `eval_max_queries: 1000` in config or increase `evaluation_steps`. |
| Out of memory            | Reduce `batch_size`, use `product_title` instead of `product_text`.    |

---

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover constants, ESCI evaluator, data utils, and load_data (prepare_train_test, prepare_train_val_test).

---

## Project structure

| Path                                   | Description                                                                                    |
| -------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **configs/reranker.yaml**              | Training config: model, batch_size, lr, evaluation_steps, early_stopping, val_frac, save_path. |
| **configs/multi_task_reranker.yaml**   | Multi-task learning training: task weights, save_path (data/multi_task_reranker), lr, batch_size. |
| **src/api/main.py**                    | FastAPI app: POST /rerank, GET /health; load multi-task learning model at startup.             |
| **src/constants.py**                   | ESCI gains, ESCI_LABEL2ID, DATA_DIR, MODEL_CACHE_DIR, DEFAULT_RERANKER_MODEL.                  |
| **src/data/load_data.py**              | load_esci, prepare_train_test, prepare_train_val_test (split by query_id).                     |
| **src/data/utils.py**                  | Product text expansion (get_product_expanded_text).                                            |
| **src/eval/evaluator.py**              | ESCIMetricsEvaluator, compute_query_metrics (nDCG, MRR, MAP, Recall@k).                        |
| **src/eval/eval_reranker.py**          | Standalone eval script: load model, run on test set, print metrics.                            |
| **src/models/reranker.py**             | CrossEncoderReranker, load_reranker, predict(), rerank().                                      |
| **src/models/multi_task_reranker.py**  | MultiTaskReranker (ranking + 4-class + substitute), load_multi_task_reranker(), save/load.     |
| **src/training/train_reranker.py**     | Training entrypoint: load data, fit CrossEncoder (Task 1), final test eval.                    |
| **src/training/train_multi_task_reranker.py** | Multi-task learning training: combined loss, Task 1/2/3, save to data/multi_task_reranker. |
| **src/training/early_stopping.py**     | EarlyStoppingCallback (patience on nDCG).                                                      |
| **tests/**                             | test_constants, test_evaluator, test_data_utils, test_load_data.                               |
| **notebooks/**                         | load_data, train_reranker, inference_reranker.                                                 |
| **Dockerfile**, **docker-compose.yml** | Container build and run for the API service.                                                   |
| **pyproject.toml**, **uv.lock**        | Project and dependency lock (uv).                                                              |

---

## License

MIT. Use of the Amazon ESCI dataset is subject to its own terms.
