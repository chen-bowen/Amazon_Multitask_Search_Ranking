#!/usr/bin/env python3
"""
Upload the multi-task ESCI reranker checkpoint to Hugging Face Hub.

The checkpoint directory (encoder, tokenizer, multi_task_heads.pt) is uploaded
as-is so it can be loaded with MultiTaskReranker.from_pretrained() after
downloading via snapshot_download.

Usage:
    # Login first (one-time): uv run hf auth login
    uv run python scripts/upload_to_huggingface.py
    uv run python scripts/upload_to_huggingface.py --repo-id USERNAME/amazon-multitask-reranker --private
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from huggingface_hub import HfApi, whoami

# Add project root for imports
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload multi-task ESCI reranker checkpoint to Hugging Face Hub")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "checkpoints" / "multi_task_reranker",
        help="Path to checkpoint directory (default: checkpoints/multi_task_reranker)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="amazon-multitask-reranker",
        help="Hugging Face repo ID (default: {username}/amazon-multitask-reranker)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/upload to a private repository",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload multi-task ESCI reranker checkpoint",
        help="Commit message for the upload",
    )
    args = parser.parse_args()

    path = args.model_path.resolve()
    if not path.is_dir():
        print(f"Error: checkpoint directory not found: {path}")
        sys.exit(1)

    required_files = {"config.json", "multi_task_heads.pt"}
    present = {f.name for f in path.iterdir() if f.is_file()}
    missing = required_files - present
    if missing:
        print(f"Error: checkpoint missing required files: {missing}")
        sys.exit(1)

    repo_id = args.repo_id
    if "/" not in repo_id:
        info = whoami()
        username = info.get("name") or getattr(info, "name", None)
        if not username:
            print("Error: could not determine Hugging Face username. Pass --repo-id USERNAME/amazon-multitask-reranker")
            sys.exit(1)
        repo_id = f"{username}/amazon-multitask-reranker"
        print(f"Using repo_id: {repo_id}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
    print(f"Uploading {path} to {repo_id}...")
    api.upload_folder(
        folder_path=str(path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
    )
    print(f"Done. Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
