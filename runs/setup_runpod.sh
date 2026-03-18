#!/bin/bash
# nanoCodeRL — RunPod environment setup script (RTX 5090)
#
# Sets up everything needed on a fresh RunPod pod:
#   1. Install uv + dependencies
#   2. Verify GPU
#   3. Prefetch model & datasets
#   4. Smoke test
#   5. Dry-run training (2 steps) to verify VRAM & disk are sufficient
#
# After setup completes, run the experiment manually (see printed instructions).
#
# Usage:
#   bash runs/setup_runpod.sh
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Detect working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# If running from a fresh pod (curl | bash), clone first
if [[ ! -f "$REPO_DIR/pyproject.toml" ]]; then
    echo "[setup] Cloning nanoCodeRL..."
    cd /workspace
    if [[ ! -d "nanoCodeRL" ]]; then
        git clone https://github.com/bideliunian/nanoCodeRL.git
    fi
    REPO_DIR="/workspace/nanoCodeRL"
fi

cd "$REPO_DIR"

echo "============================================"
echo "  nanoCodeRL — RunPod Setup (RTX 5090)"
echo "============================================"
echo ""
echo "  Repo: $REPO_DIR"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Install uv
# ---------------------------------------------------------------------------
echo "[1/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    # Also add to bashrc for future shells
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
echo "  uv $(uv --version) installed."
echo ""

# ---------------------------------------------------------------------------
# Step 2: Install Python dependencies + verify GPU
# ---------------------------------------------------------------------------
echo "[2/5] Installing dependencies (including vllm)..."
uv sync --extra vllm
echo "  Dependencies installed."
echo ""

echo "  Verifying GPU..."
# Enable persistence mode to ensure CUDA initializes correctly
nvidia-smi -pm 1 &>/dev/null || true
GPU_INFO=$(uv run python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU detected!')
    exit(1)
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'{name} — {vram:.0f} GB VRAM')
")
echo "  GPU: $GPU_INFO"

# Check VRAM is sufficient
VRAM_GB=$(uv run python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))")
if [[ "$VRAM_GB" -lt 24 ]]; then
    echo "  WARNING: Only ${VRAM_GB}GB VRAM detected. Minimum 24GB recommended."
    echo "  Training may OOM. Consider reducing: --batch-size 1 --num-rollouts 4 --max-length 768"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Prefetch model & datasets (store on volume disk, not container)
# ---------------------------------------------------------------------------
echo "[3/5] Prefetching model & datasets (this may take 5-10 min)..."
export HF_HOME=/workspace/.cache/huggingface
echo 'export HF_HOME=/workspace/.cache/huggingface' >> ~/.bashrc
uv run python -m scripts.prefetch --cache-dir /workspace/.cache/huggingface
echo "  Prefetch complete. Assets stored in /workspace/.cache/huggingface"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Smoke test
# ---------------------------------------------------------------------------
echo "[4/5] Running smoke test..."
uv run python -m scripts.smoke_test --with-model
echo "  Smoke test passed."
echo ""

# ---------------------------------------------------------------------------
# Step 5: Dry-run training (2 steps)
# ---------------------------------------------------------------------------
echo "[5/5] Dry-run training (2 steps) — verifying VRAM, disk, and full pipeline..."
DRY_RUN_DIR="$REPO_DIR/checkpoints/dry_run"
uv run python -m scripts.train --steps 2

# Show VRAM usage after dry run
uv run python -c "
import torch
if torch.cuda.is_available():
    used = torch.cuda.max_memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    pct = used / total * 100
    print(f'  Peak VRAM usage: {used:.1f} GB / {total:.0f} GB ({pct:.0f}%)')
"

# Show disk usage
DISK_AVAIL=$(df -h "$REPO_DIR" | awk 'NR==2 {print $4}')
echo "  Disk available: $DISK_AVAIL"

# Clean up dry-run checkpoint
rm -rf "$DRY_RUN_DIR"
echo "  Dry run passed — training pipeline works end to end."
echo ""

# ---------------------------------------------------------------------------
# Done — print next steps
# ---------------------------------------------------------------------------
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Run the experiment step by step:"
echo ""
echo "  # 1. Baseline evaluation (~30-45 min)"
echo "  uv run python -m scripts.eval --bench humaneval mbpp --output results/eval_baseline.json"
echo ""
echo "  # 2. DAPO training (~6-10 hrs) — run in tmux to survive SSH disconnects"
echo "  tmux new -s train"
echo "  uv run python -m scripts.train --steps 200 --wandb"
echo "  # Ctrl+B, D to detach | tmux attach -t train to reconnect"
echo ""
echo "  # 3. Post-RL evaluation (~30-45 min)"
echo "  uv run python -m scripts.eval --ckpt checkpoints/last --bench humaneval mbpp --output results/eval_post-rl.json"
echo ""
echo "  # 4. Agent demo (~15 min)"
echo "  uv run python -m scripts.agent_demo --ckpt checkpoints/last --num-problems 5"
echo ""
echo "  # 5. Download results (from your local machine)"
echo "  scp -P <port> -r root@<pod-ip>:/workspace/nanoCodeRL/results/ ./results/"
echo "  scp -P <port> -r root@<pod-ip>:/workspace/nanoCodeRL/checkpoints/last/ ./checkpoints/"
echo ""
echo "REMEMBER: Stop or terminate your RunPod pod after downloading to stop billing!"
echo ""
