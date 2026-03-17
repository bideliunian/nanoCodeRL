# nanoCodeRL

A minimal, reproducible RL pipeline for training a coding agent using [DAPO](https://arxiv.org/abs/2503.14476) on [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) — from base model weights to an iterative coding agent on a single GPU.

Inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat).

---

## What This Does

- Trains Qwen3.5-4B-Base with **DAPO** (GRPO + Clip-Higher, dynamic sampling, token-level PG, overlong reward shaping)
- Uses **QLoRA 4-bit** to fit on a single 32GB GPU (RTX 5090)
- Reward = fraction of unit tests passed via sandboxed subprocess execution (deterministic, no reward model)
- Evaluates on **HumanEval** and **MBPP** pass@1
- Includes a **multi-turn agent demo**: generate → execute → observe error → revise → repeat

---

## Project Structure

```
nanoCodeRL/
├── nanoCodeRL/                 # core package
│   ├── config.py               # all hyperparameters in one place
│   ├── data.py                 # dataset loading (CodeContests train, HumanEval/MBPP eval)
│   └── sandbox.py              # subprocess code execution & reward
├── scripts/                    # entry points
│   ├── train.py                # DAPO training loop
│   ├── eval.py                 # pass@1 evaluation
│   ├── agent_demo.py           # multi-turn iterative coding agent
│   ├── prefetch.py             # pre-download models & datasets
│   └── smoke_test.py           # CPU pipeline validation
├── runs/
│   └── quickstart.sh           # end-to-end pipeline script
├── Dockerfile.sandbox          # optional sandboxed execution
└── pyproject.toml              # dependencies (managed by uv)
```

---

## Setup & Reproduce

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), CUDA GPU with ≥32GB VRAM (RTX 5090 recommended).

### One command

```bash
git clone https://github.com/bideliunian/nanoCodeRL && cd nanoCodeRL
bash runs/quickstart.sh          # or: bash runs/quickstart.sh --wandb
```

### Step by step

```bash
# 1. Install dependencies
uv sync

# 2. (Optional) Pre-download model & datasets for offline training
uv run python -m scripts.prefetch

# 3. Validate pipeline on CPU
uv run python -m scripts.smoke_test

# 4. Baseline evaluation (pre-RL)
uv run python -m scripts.eval --bench humaneval mbpp

# 5. DAPO training (~200 steps)
uv run python -m scripts.train --steps 200 --wandb

# 6. Post-RL evaluation
uv run python -m scripts.eval --ckpt checkpoints/last --bench humaneval mbpp

# 7. Multi-turn agent demo
uv run python -m scripts.agent_demo --ckpt checkpoints/last --num-problems 5
```

### Key config (see `nanoCodeRL/config.py`)

```python
model_name           = "Qwen/Qwen3.5-4B-Base"
load_in_4bit         = True        # QLoRA
num_rollouts         = 8           # rollouts per prompt
batch_size           = 8           # prompts per step
num_train_steps      = 200
learning_rate        = 1e-6
max_completion_length = 1024       # tokens
```

---

## References

- [DAPO (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)
- [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [HumanEval](https://github.com/openai/human-eval) / [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [nanochat](https://github.com/karpathy/nanochat)
