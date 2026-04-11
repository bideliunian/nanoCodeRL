# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoCodeRL is a minimal RL pipeline for training a coding agent using the **DAPO** algorithm on Qwen3.5-4B. It trains a base LLM to solve coding problems through group relative policy optimization, using actual unit test execution as the reward signal (no reward model needed).

## Package Manager

This project uses **uv** for dependency management. Always use `uv run python` instead of `python` directly.

```bash
uv sync                   # Install all dependencies
```

## Common Commands

```bash
# Validate setup (CPU-only, no GPU needed)
uv run python -m scripts.smoke_test
uv run python -m scripts.smoke_test --with-model  # also test model loading (~10GB RAM)

# Reward function unit tests (loads HumanEval, requires network on first run)
uv run python -m scripts.test_reward

# Pre-download models and datasets
uv run python -m scripts.prefetch

# Baseline evaluation (before training)
uv run python -m scripts.eval --bench humaneval mbpp
uv run python -m scripts.eval --bench humaneval mbpp --batch-size 16  # override batch size

# Training
uv run python -m scripts.train --steps 200 --wandb
uv run python -m scripts.train --steps 200 --train-data mbpp_full     # lighter alternative to CodeContests

# Post-training evaluation
uv run python -m scripts.eval --ckpt checkpoints/last --bench humaneval mbpp

# Multi-turn agent demo
uv run python -m scripts.agent_demo --num-problems 5 --ckpt checkpoints/last
uv run python -m scripts.agent_demo --task-id HumanEval/0              # single problem

# Debug reward computation (prints per-completion reward details)
REWARD_DEBUG=1 uv run python -m scripts.train --steps 1

# Full end-to-end pipeline
bash runs/quickstart.sh
```

## Architecture

### Core Package (`nanoCodeRL/`)

- **`config.py`** — Single `Config` dataclass containing all hyperparameters. This is the canonical source of truth. Override via CLI args in training scripts.
- **`data.py`** — Dataset loading and prompt formatting. Handles CodeContests (training), HumanEval, and MBPP (eval). `build_messages()` constructs the chat-formatted prompt; `apply_chat_template()` tokenizes it. Training sources: `code_contests` (~13K problems, stdin/stdout) and `mbpp_full` (~374 problems, assertions). HumanEval and MBPP sanitized are eval-only.
- **`sandbox.py`** — Code execution and reward computation. `extract_code()` strips markdown fences and `<think>` tags. Reward = `tests_passed / total_tests`. Supports assertion-based tests (HumanEval/MBPP) and stdin/stdout tests (CodeContests). Runs in sandboxed subprocesses via `ProcessPoolExecutor`. Optional Docker sandboxing with `use_docker_sandbox=True`.

### Scripts (`scripts/`)

- **`train.py`** — Main DAPO training loop using TRL's `GRPOTrainer`. Loads Qwen3.5-4B in 4-bit QLoRA, samples rollouts, executes them in the sandbox for rewards, updates LoRA weights.
- **`eval.py`** — Evaluates pass@1 on HumanEval and/or MBPP using batched generation. Saves results to `results/`.
- **`agent_demo.py`** — Multi-turn agent: generate → execute → observe errors → append feedback → revise (up to 5 turns). Accumulates the full conversation via chat message history.

### Data Flow

```
Config → Data Loader → Chat Template → Model Generation → extract_code() → Sandbox Execute → Reward → GRPO Update
```

### DAPO Algorithm

DAPO = GRPO + Clip-Higher asymmetric clipping:
- `clip_eps_low=0.2` (standard PPO lower clip)
- `clip_eps_high=0.28` (higher clip encourages improvement)
- Token-level policy gradient loss (not sequence-level)
- `num_rollouts=8` completions sampled per problem for group relative advantage
- `mask_truncated_completions=True` — completions hitting `max_completion_length` are masked from loss

### Key Training Components in `train.py`

- **`build_reward_fn()`** — Constructs the reward closure passed to `GRPOTrainer`. Uses integer `problem_idx` (injected into the dataset) to look up test cases, avoiding string-matching failures caused by TRL's internal tokenize-then-decode pipeline. Set `REWARD_DEBUG=1` to print per-completion reward details.
- **`DynamicSampler`** — DAPO dynamic sampling: tracks consecutive zero-variance reward groups per problem (all pass or all fail). Problems exceeding `max_zero_var_streak=3` are deprioritized to a 5% sampling probability, keeping training batches focused on learnable examples.
- **`IntermediateEvalCallback`** — Runs pass@1 on a subset of held-out benchmarks every `eval_every_n_steps` steps. Logs to `logs/eval_during_training.jsonl` and optionally to W&B.

### Model & Training Setup

- **Model**: Qwen3.5-4B, loaded in 4-bit NF4 quantization via bitsandbytes
- **LoRA**: r=16, alpha=32, targeting `q/k/v/o_proj` and `gate/up/down_proj`
- **Loading order**: Unsloth (fast path) → HF + PEFT (fallback)
- **Generation**: temperature=0.7, top_p=0.95, max 2048 completion tokens

### Key Config Parameters

All in `nanoCodeRL/config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `cc_max_difficulty` | 7 | Filter CodeContests by difficulty (0=unknown, 1-6=easy, 7-11=medium/hard, 12+=very hard) |
| `cc_max_solution_chars` | 800 | Skip problems where shortest reference solution is long (~300 tokens) |
| `eval_subset_size` | 30 | Problems per benchmark during mid-training eval (0 = full) |
| `eval_every_n_steps` | 50 | Eval frequency during training |
| `sandbox_timeout` | 5 | Per-execution timeout in seconds |
| `sandbox_max_workers` | 8 | Parallel sandbox processes |
| `enable_thinking` | False | Enable Qwen3's chain-of-thought thinking mode (generates long CoT; disabled for RL) |
| `use_vllm` | False | Use vLLM with PagedAttention + sleep mode to free VRAM during backward pass |
| `use_wandb` | False | Enable W&B tracking |

## Known Issues

- `agent_demo.py` does not call `extract_code()` before passing completions to `compute_reward()` — markdown fences cause execution errors
- `prefetch.py` hardcodes `max_difficulty=3` instead of reading from config's `cc_max_difficulty=7`
- Intermediate eval in `IntermediateEvalCallback` runs sequentially (no batching), making it slow on large `eval_subset_size`