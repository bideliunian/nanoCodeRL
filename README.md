# nanoCodeRL: Coding Agent RL on Qwen3.5

> Full DAPO coding agent RL — minimal compute, fast iteration, reproducible results

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Qwen3.5--4B--Base-orange)](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
[![Hardware](https://img.shields.io/badge/Hardware-1x%20L4%2024GB-green)](https://www.runpod.io)

A minimal, reproducible pipeline for coding agent RL using **DAPO on Qwen3.5-4B-Base** — from raw base model weights to a live iterative coding agent, under $100 on a single L4 GPU.

The design constraint is **compute budget, not code complexity**: scripts, metrics, and datasets can grow — GPU hours stay cheap so anyone can iterate fast and fork from a known-working baseline.

Inspired by Karpathy's [nanochat](https://github.com/karpathy/nanochat).

---

## Quickstart

```bash
git clone https://github.com/bideliunian/nanoCodeRL && cd nanoCodeRL

# One-command setup + full pipeline
bash runs/quickstart.sh

# Or step by step:
uv sync                                                # install deps
uv run python -m scripts.eval                          # baseline eval (pre-RL)
uv run python -m scripts.train                         # DAPO training, ~200 steps
uv run python -m scripts.eval --ckpt checkpoints/last  # post-RL eval
uv run python -m scripts.agent_demo                    # live coding agent
```

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), CUDA GPU ≥16GB VRAM, ~$25 budget (RunPod L4 @ $0.35/hr).

---

## Project Structure

```
nanoCodeRL/
├── pyproject.toml              # dependencies (managed by uv)
├── .python-version             # Python 3.10
├── nanoCodeRL/                 # core package
│   ├── config.py               # all hyperparameters in one place
│   ├── data.py                 # HumanEval + MBPP dataset loading
│   └── sandbox.py              # subprocess code execution & reward
├── scripts/                    # entry points (python -m scripts.xxx)
│   ├── train.py                # DAPO training loop
│   ├── eval.py                 # HumanEval/MBPP pass@1 evaluation
│   └── agent_demo.py           # multi-turn iterative coding agent
└── runs/                       # experiment scripts
    └── quickstart.sh           # end-to-end pipeline
```

---

## What This Is

**Model:** Qwen3.5-4B-Base — fits on L4 with QLoRA; strong code capability at the smallest Qwen3.5 size.

**Algorithm:** [DAPO](https://arxiv.org/abs/2503.14476) (extends GRPO with 4 stabilization techniques):
- **Clip-Higher** — asymmetric clipping (ε_high=0.28) prevents entropy collapse
- **Dynamic Sampling** — filters all-pass/all-fail batches; keeps training signal informative
- **Token-Level Policy Gradient** — per-token loss; avoids length bias
- **Overlong Reward Shaping** — penalizes truncated responses

**Reward:** Python subprocess executes generated code against hidden unit tests. No reward model needed.
```
reward = k / n      # fraction of unit tests passing (0.0–1.0)
reward = 0          # syntax error, runtime error, or timeout (5s)
```

**Benchmarks:** HumanEval pass@1, MBPP pass@1, and optionally LiveCodeBench.

**Agent loop:** `generate → execute → observe failure → revise → repeat`

---

## Why Coding RL (Not Math)

| | Coding RL | Math RL |
|---|---|---|
| Reward function | Python subprocess (deterministic) | Symbolic parser or reward model |
| Response length | 512–1024 tokens | 4K–8K tokens |
| Step time (L4) | ~5–10 min | ~1 hr |

Disabling thinking mode (`enable_thinking=False`) keeps completions short — the single biggest lever for fast iteration on budget hardware. Enable it later once the pipeline is stable.

---

## Training Config

```python
model                = "Qwen/Qwen3.5-4B-Base"
quantization         = "QLoRA 4-bit"          # ~14GB VRAM on L4
enable_thinking      = False
rollouts_per_prompt  = 8
max_response_length  = 1024                    # tokens
learning_rate        = 1e-6
target_steps         = 200                     # ~$10–25 on L4
```

---

## Cost Estimate (RunPod L4 @ $0.35/hr)

| Phase | Time | Cost |
|---|---|---|
| Setup + model download | 2–4 hrs | ~$1 |
| Debugging | 4–8 hrs | ~$2 |
| Baseline eval | 1–2 hrs | ~$0.50 |
| DAPO training (200 steps) | 20–36 hrs | ~$7–13 |
| Post-RL eval | 2–3 hrs | ~$1 |
| **Total** | | **~$14–25** |

Scale-up options: 1x A100 80GB for Qwen3.5-9B (~$150–200); 4x A100 for thinking mode + multi-seed runs (~$500+).

---

## Roadmap

- [ ] Phase 1: Baseline eval (HumanEval, MBPP)
- [ ] Phase 2: `train.py` — DAPO training loop with sandbox reward
- [ ] Phase 3: Post-RL eval — first published DAPO coding RL result on Qwen3.5-4B
- [ ] Phase 4: `agent_demo.py` — multi-turn iterative coding agent
- [ ] Phase 5: HuggingFace release (LoRA adapter + write-up)
- [ ] Stretch: Scale to Qwen3.5-9B on A100 for 4B→9B comparison

---

## Contributing

All contributions are welcome. A few priorities:

- Reproducibility fixes and hardware compatibility reports
- Additional benchmark support (LiveCodeBench, BigCodeBench)
- Tested configs for other Qwen3.5 sizes or cloud hardware
- Results — if you ran nanoCodeRL, open an issue with your numbers and config

The core design goal is keeping the compute cost low enough for fast iteration. Changes that increase GPU requirements without clear benefit will be carefully weighed.

---

## References

- [DAPO Paper (arXiv:2503.14476)](https://arxiv.org/abs/2503.14476)
- [DAPO GitHub](https://github.com/BytedTsinghua-SIA/DAPO)
- [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Unsloth GRPO Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/grpo_trainer)
- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp)
- [nanochat](https://github.com/karpathy/nanochat)
