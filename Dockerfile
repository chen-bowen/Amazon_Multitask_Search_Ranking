# ESCI reranker API: Python 3.12, install deps, run uvicorn.
# Set MODEL_PATH at runtime (e.g. volume mount) or bake a checkpoint into the image.
FROM python:3.12-slim

WORKDIR /app

# Install project and dependencies.
COPY pyproject.toml ./
COPY src ./src/
RUN pip install --no-cache-dir .

# Default: run API on port 8000.
ENV MODEL_PATH=/app/checkpoints/multi_task_reranker
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
