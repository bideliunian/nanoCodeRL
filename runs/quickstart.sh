#!/bin/bash
# nanoCodeRL quickstart — full pipeline from setup to agent demo
#
# Usage:
#   bash runs/quickstart.sh          # full pipeline
#   bash runs/quickstart.sh --wandb  # with W&B logging
set -e

WANDB_FLAG=""
if [[ "$1" == "--wandb" ]]; then
    WANDB_FLAG="--wandb"
fi

echo "============================================"
echo "  nanoCodeRL — Quickstart Pipeline"
echo "============================================"
echo ""

# 0. Environment setup
echo "[0/4] Setting up environment..."
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync
echo "Dependencies installed."
echo ""

# 1. Baseline evaluation (pre-RL)
echo "[1/4] Baseline evaluation..."
uv run python -m scripts.eval --bench humaneval mbpp
echo ""

# 2. DAPO training
echo "[2/4] DAPO training (200 steps)..."
uv run python -m scripts.train $WANDB_FLAG
echo ""

# 3. Post-RL evaluation
echo "[3/4] Post-RL evaluation..."
uv run python -m scripts.eval --ckpt checkpoints/last --bench humaneval mbpp
echo ""

# 4. Agent demo
echo "[4/4] Agent demo..."
uv run python -m scripts.agent_demo --ckpt checkpoints/last --num-problems 5
echo ""

echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
