"""Prefetch model weights and datasets for offline use.

Run this before deploying to a GPU cluster to ensure no network
access is needed during training.

Usage:
    python -m scripts.prefetch
    python -m scripts.prefetch --cache-dir /shared/hf_cache
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Prefetch model and data")
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory (default: ~/.cache/huggingface)",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir

    from nanoCodeRL.config import Config
    cfg = Config()

    # 1. Prefetch training datasets
    print("[1/4] Downloading training datasets...")
    from nanoCodeRL.data import load_training_data, load_eval_data
    try:
        train_problems = load_training_data(cfg.train_benchmarks)
        print(f"  {len(train_problems)} training problems cached.\n")
    except RuntimeError as e:
        print(f"  Warning: {e}\n")

    # 2. Prefetch eval datasets
    print("[2/4] Downloading eval datasets...")
    eval_problems = load_eval_data(cfg.eval_benchmarks)
    print(f"  {len(eval_problems)} eval problems cached.\n")

    # 3. Prefetch tokenizer
    print("[3/4] Downloading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    print(f"  Tokenizer cached: {cfg.model_name}\n")

    # 4. Prefetch model weights
    print("[4/4] Downloading model weights (this may take a while)...")
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(
        cfg.model_name, trust_remote_code=True, torch_dtype="auto",
    )
    print(f"  Model cached: {cfg.model_name}\n")

    cache_dir = args.cache_dir or os.path.expanduser("~/.cache/huggingface")
    print(f"All assets cached in: {cache_dir}")
    print("You can now run training offline with HF_HUB_OFFLINE=1")


if __name__ == "__main__":
    main()
