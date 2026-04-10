# Improvement Plan — nanoCodeRL

Hardware target: **single A100 SXM 80GB** (adjustments for 40GB noted inline).

---

## Critical Issues (Current Results)

| Benchmark | Baseline | Step 50 | Step 100 |
|-----------|----------|---------|----------|
| HumanEval pass@1 | 75.0% | +1.8% | +1.8% |
| MBPP pass@1 | 53.7% | -1.6% | **-6.2%** |

Root causes:
1. **Train/eval domain mismatch** — training on CodeContests (stdin/stdout) hurts MBPP (assertion-based functions)
2. **Dynamic sampler is dead code** — `DynamicSampler` runs once at startup when all counts are zero; the feedback loop never activates during training
3. **200 steps is too few** — 12,800 total completions is vanishingly small for RL
4. **Eval subset too small** — 30 problems per benchmark gives ±8–10% variance; results are noise

---

## Phase 1 — Free Wins (code changes only, ~0 extra GPU hours)

### 1a. Fix dataset domain mismatch
Switch training data to align with eval format. MBPP `train` split uses the same assertion style as the `test` split used for eval.

```python
# config.py
train_benchmarks: list[str] = ["mbpp_full", "code_contests"]
```

Or train exclusively on `mbpp_full` (374 problems) for a clean ablation.

### 1b. Fix dynamic sampler
`DynamicSampler` needs to re-evaluate `zero_var_counts` between steps, not just at startup. Simplest fix: implement as a custom `IterableDataset` that reads `reward_fn.zero_var_counts` live on each `__next__` call, or hook into `on_step_end` to rebuild the filtered dataset.

### 1c. Use full eval sets
Replace `eval_subset_size=30` with the full HumanEval (164) + MBPP test (399):

```python
eval_subset_size: int = 0  # 0 = full benchmark
```

Adds ~20 min per eval run on A100 SXM.

---

## Phase 2 — Drop QLoRA, Run BF16 (same time, cleaner gradients)

80GB eliminates the need for 4-bit quantization on a 4B model. Full BF16 removes quantization noise from gradients and allows larger batches.

```python
load_in_4bit: bool = False   # BF16 on A100 SXM
batch_size: int = 16         # was 8; 80GB has headroom
num_rollouts: int = 16       # was 8; more diverse rollouts
```

VRAM budget (4B BF16 + GRPO, batch=16, rollouts=16, 2048 tokens):
- Weights: ~8 GB
- LoRA optimizer states: ~4 GB
- KV cache: ~18 GB
- Activations + overhead: ~10 GB
- **Total: ~40 GB** — comfortable on 80GB

> **40GB SXM**: Keep `load_in_4bit=True`, `batch_size=8`, `num_rollouts=8`.

---

## Phase 3 — Enable vLLM Colocate (already wired in)

The codebase already supports this (`config.py:78`, `train.py:409–415`). On 80GB it becomes practical:

```bash
uv pip install vllm==0.10.2
```

```python
use_vllm: bool = True
vllm_gpu_memory_utilization: float = 0.45   # 36 GB for vLLM generation
```

vLLM sleep mode releases the 36 GB during the backward pass. Generation throughput increases ~2–3× vs `model.generate()` via PagedAttention + continuous batching. Net effect: ~2× more steps in the same wall-clock time.

With vLLM on A100 SXM: **500 steps in ~6–8h** (same time as 200 steps on RTX 5090).

---

## Phase 4 — Upgrade to Qwen2.5-Coder-7B (recommended model)

Qwen2.5-Coder is purpose-trained on code corpora. RL finetuning has more signal to work with because the base distribution is already strongly aligned to code patterns.

```python
model_name: str = "Qwen/Qwen2.5-Coder-7B"
load_in_4bit: bool = False   # BF16 fits on 80GB
```

VRAM budget (7B BF16 + LoRA, batch=8, rollouts=8, 2048 tokens):
- Weights: ~14 GB
- LoRA optimizer states: ~6 GB
- KV cache: ~16 GB
- Activations + overhead: ~12 GB
- **Total: ~48 GB** — fits with ~30 GB to spare

| Config | Steps | Est. time (A100 SXM) |
|--------|-------|-----------------------|
| 4B BF16 + vLLM | 500 | ~6–8h |
| **7B BF16 + vLLM** | **500** | **~10–14h** |
| 7B BF16 + vLLM + thinking | 500 | ~18–24h |
| 14B QLoRA + vLLM | 500 | ~22–30h |

> **40GB SXM**: Use `load_in_4bit=True` for 7B. 14B QLoRA is tight (~38–42 GB); reduce `batch_size=4`.

---

## Phase 5 — Enable Thinking Mode (largest quality lever)

Qwen3.5/Qwen2.5 chain-of-thought reasoning demonstrably improves coding accuracy. Currently disabled (`enable_thinking=False`). With vLLM handling variable-length sequences via PagedAttention, 4096-token completions are efficient on 80GB.

```python
enable_thinking: bool = True
max_completion_length: int = 4096
vllm_gpu_memory_utilization: float = 0.50   # more room for long sequences
```

> **40GB SXM**: Requires `batch_size=4`, `num_rollouts=4` to fit 4096-token KV cache.

---

## Phase 6 — Multi-Turn RL Training (largest engineering effort)

Train the full generate → execute → observe error → revise loop instead of single-shot generation. The model learns to interpret execution feedback during training, not just at inference.

Architecture: extend `reward_fn` and dataset to pass conversation history across turns; use the final-turn reward (or max across turns) as the GRPO signal.

Recommended config: **3 turns** (diminishing returns after 3), 7B model:
- Per-step time: ~3–4× vs single-turn (more generation + sandbox execution)
- 500 steps on A100 SXM: **~30–45h**

---

## Priority Roadmap

| Phase | Change | Est. training time | Effort |
|-------|--------|--------------------|--------|
| 1 | Fix dataset, sampler, full eval | 2–3h | 2–3h code |
| 2 | Drop QLoRA → BF16, larger batch | 2–3h | Config only |
| 3 | Enable vLLM colocate | 4–6h (500 steps) | `pip install` + config |
| 4 | Upgrade to Qwen2.5-Coder-7B | 10–14h (500 steps) | Model name change |
| 5 | Enable thinking mode | 18–24h (500 steps) | Config + length |
| 6 | Multi-turn RL training | 30–45h (500 steps) | Significant rewrite |

**Recommended path for a solid single-card result**: Phases 1–4 together (~12–16h total), producing Qwen2.5-Coder-7B BF16 trained on MBPP-aligned data with vLLM generation for 500 steps.

---

## Framework Notes

- **TRL GRPOTrainer** (current): adequate for single-GPU; vLLM colocate mode (Phase 3) closes most of the throughput gap
- **rLLM**: worth considering if TRL's overhead becomes a bottleneck; ~30–50% more steps/hour on same hardware, moderate migration cost
- **veRL**: designed for multi-GPU Ray clusters; no meaningful benefit on a single card — skip
