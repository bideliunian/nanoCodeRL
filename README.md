# nanoCodeRL

A minimal, end-to-end RL pipeline for training a coding agent using [DAPO](https://arxiv.org/abs/2503.14476) on [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B) — from base model weights to a RL-tuned coding agent on a single GPU.

> **Scope:** This project demonstrates the full agentic RL workflow — data loading, sandboxed reward computation, GRPO training, and evaluation — rather than pushing SOTA results. Significant improvements would require more compute, larger models, and longer training runs.

Inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat).

---

## What This Does

- Trains Qwen2.5-Coder-7B with **DAPO** (GRPO + Clip-Higher, dynamic sampling, token-level PG, truncated completion masking)
- Fits on a single A100 80GB GPU in BF16 with QLoRA (r=16) and gradient checkpointing
- Reward = fraction of unit tests passed via sandboxed subprocess execution (no reward model needed)
- Evaluates on **HumanEval** and **MBPP** pass@1
- Includes a **multi-turn agent demo**: generate → execute → observe error → revise → repeat

---

## Project Structure

```
nanoCodeRL/
├── nanoCodeRL/                 # core package
│   ├── config.py               # all hyperparameters in one place
│   ├── data.py                 # dataset loading (mbpp_full train, HumanEval/MBPP eval)
│   └── sandbox.py              # subprocess code execution & reward
├── scripts/                    # entry points
│   ├── train.py                # DAPO training loop (GRPOTrainer)
│   ├── eval.py                 # pass@1 evaluation (HumanEval + MBPP)
│   ├── autorl.py               # time-budget autonomous training loop
│   ├── agent_demo.py           # multi-turn iterative coding agent
│   ├── prefetch.py             # pre-download models & datasets
│   └── smoke_test.py           # CPU pipeline validation
├── runs/
│   └── quickstart.sh           # end-to-end pipeline script
├── Dockerfile.sandbox          # optional Docker sandboxed execution
└── pyproject.toml              # dependencies (managed by uv)
```

---

## Setup & Reproduce

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), CUDA GPU with ≥40GB VRAM (A100 recommended).

```bash
git clone https://github.com/bideliunian/nanoCodeRL && cd nanoCodeRL

# Install dependencies
uv sync

# Validate pipeline (CPU-only, no GPU needed)
uv run python -m scripts.smoke_test

# Pre-download model & datasets
uv run python -m scripts.prefetch

# Baseline evaluation
uv run python -m scripts.eval --bench humaneval mbpp

# DAPO training (50 steps ~1.5h, 200 steps ~6h on A100)
uv run python -m scripts.train --steps 50 --train-data mbpp_full \
    --batch-size 8 --num-rollouts 8 --no-vllm

# Post-RL evaluation
uv run python -m scripts.eval --ckpt checkpoints/last --bench humaneval mbpp

# Multi-turn agent demo
uv run python -m scripts.agent_demo --ckpt checkpoints/last --num-problems 5
```

### Key config (`nanoCodeRL/config.py`)

```python
model_name            = "Qwen/Qwen2.5-Coder-7B"
load_in_4bit          = False      # BF16 on A100 80GB; set True for ≤40GB
num_rollouts          = 16         # completions per prompt
batch_size            = 16         # micro-batch size
learning_rate         = 1e-6
max_completion_length = 2048       # tokens
```

---

## Results

All results use the FIM-token-fixed eval harness (see [Notable Bugs Fixed](#notable-bugs-fixed)).
Model: Qwen2.5-Coder-7B, BF16, QLoRA r=16, single A100 80GB.

| Run | Steps | Batch / Rollouts | HumanEval pass@1 | MBPP pass@1 | Composite |
|-----|-------|-----------------|-----------------|-------------|-----------|
| Baseline | — | — | 42.7% | 44.0% | 43.3% |
| DAPO (old, corrupted rewards) | 500 | 4 / 4 | 45.1% (+2.4) | 43.6% (−0.4) | 44.4% |
| **DAPO (clean rewards)** | **50** | **8 / 8** | **46.3% (+3.6)** | **44.4% (+0.4)** | **45.4%** |
| DAPO (clean rewards) | 200 | 8 / 8 | 43.3% (+0.6) | 38.1% (−5.9) | 40.7% |

**Takeaways:**
- The model improves most in the first ~50 steps; longer training on a single narrow dataset (mbpp_full, 374 problems) leads to reward collapse and regression.
- Clean reward signal matters: fixing the FIM token bug (see below) recovered ~20pp of masked improvement in the baseline itself.
- **50 steps is the best checkpoint** for this config. Further gains would require a larger and more diverse training set (e.g. CodeContests) and more compute.

---

## Notable Bugs Fixed

### FIM Token Leakage
Qwen2.5-Coder was pre-trained with Fill-in-the-Middle (FIM) special tokens (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`, `<|file_sep|>`). During generation these tokens can appear mid-completion, truncating output. Because they are not in `tokenizer.all_special_ids`, `skip_special_tokens=True` does not strip them — they pass through as literal text and cause `SyntaxError` in the sandbox.

**Fix (`nanoCodeRL/sandbox.py`):** `extract_code()` now truncates the completion at the first FIM marker before any other processing. This is the single most impactful fix in the project — it revealed the true baseline was ~44% (not the 21% the buggy eval reported).

### DynamicSampler (DAPO dynamic sampling)
The original `DynamicSampler` materialized the filtered dataset once at startup from an empty `zero_var_counts` dict, so deprioritization never activated. Replaced with `DynamicDataset` (`IterableDataset`) whose `__iter__` reads live counts on every epoch.

---

## References

- [DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)
- [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [HumanEval](https://github.com/openai/human-eval) / [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [nanochat](https://github.com/karpathy/nanochat)
