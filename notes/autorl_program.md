# autorl — Autonomous RL Research Program

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: an AI agent modifies the training script, runs an experiment on a
fixed time budget, checks if the metric improved, keeps or discards the commit,
and repeats — indefinitely, overnight, without human intervention.

---

## Setup (do this once per session)

Work with the user to:

1. **Agree on a run tag** — propose a tag based on today's date, e.g. `apr12`.
   The branch `autorl/<tag>` must not already exist.

2. **Create the experiment branch**:
   ```
   git checkout -b autorl/<tag>
   ```

3. **Read the in-scope files** (do this before every session):
   - `notes/autorl_program.md` — this file. The rules of the game.
   - `notes/improvement_plan.md` — known issues and prioritised changes.
   - `scripts/autorl.py` — the ONLY file you edit. Read it fully.
   - `nanoCodeRL/config.py` — available hyperparameters and their meaning.
   - `nanoCodeRL/data.py` — datasets and prompt formats. Do NOT modify.
   - `nanoCodeRL/sandbox.py` — reward computation. Do NOT modify.
   - `results.tsv` — all previous experiments. Study the trend.
   - `metrics/live.json` — live state of the current (or last) run.

4. **Establish the baseline**: run the script as-is before making any changes:
   ```
   uv run python -m scripts.autorl > run.log 2>&1
   ```
   Extract the key metric:
   ```
   grep "^composite_pass@1:" run.log
   ```
   Record in `results.tsv` with status `keep` and description `baseline`.

5. **Confirm setup looks good**, then begin the experiment loop.

---

## The Single Metric

**composite_pass@1** = (humaneval_pass@1 + mbpp_pass@1) / 2

This is the one number that determines keep vs discard. Higher is better.

It plays the role that `val_bpb` plays in autoresearch — it is vocab-size-
independent, benchmark-balanced, and directly comparable across experiments.

Do NOT optimise for one benchmark at the expense of the other.

---

## What You Can and Cannot Modify

**You CAN modify:**
- `scripts/autorl.py` — the AGENT EDITABLE section at the top is where all
  hyperparameter constants live. Start there. You may also modify the training
  loop itself (below the frozen section marker) if you have a good reason.

**You CANNOT modify:**
- `nanoCodeRL/sandbox.py` — the execution sandbox and `compute_reward()`.
  This is the ground-truth eval harness, equivalent to `prepare.py` in
  autoresearch. It is frozen.
- `nanoCodeRL/data.py` — dataset loading and prompt formatting.
- `nanoCodeRL/sandbox.py`, `scripts/eval.py` — eval infrastructure.
- `_eval_pass_at_1()` inside `autorl.py` — the frozen final-eval function.
- `results.tsv` header row.
- Install new packages. Use only what is in `pyproject.toml`.

**The goal:** highest `composite_pass@1` within the `TIME_BUDGET_HOURS`
wall-clock window. Architecture, hyperparameters, training schedule, data
mix — all fair game inside `autorl.py`.

---

## Output Format

When a run completes, the script prints a parseable summary block:

```
---
composite_pass@1: 0.435000
humaneval_pass@1: 0.451000
mbpp_pass@1:      0.419000
training_hours:   7.83
total_steps:      487
peak_vram_gb:     42.3
commit:           a1b2c3d
status:           complete
---
```

Extract key metrics from the log:
```bash
grep "^composite_pass@1:\|^peak_vram_gb:\|^training_hours:" run.log
```

If the grep returns nothing, the run crashed. Investigate:
```bash
tail -n 80 run.log
```

---

## Live Metrics Monitoring

The script writes `metrics/live.json` continuously during training. Read it at
any time to check progress without waiting for the final summary:

```bash
cat metrics/live.json
```

Fields of interest:
- `status` — `initialising` | `loading_model` | `training` | `final_eval` | `complete`
- `training_step` — current step
- `elapsed_hours` / `remaining_hours` — time budget progress
- `last_eval_step` / `last_eval_scores` — most recent mid-training eval
- `composite_pass@1` — final metric (only populated after `complete`)

To reset live metrics before a new run (optional):
```bash
uv run python -m scripts.autorl --clear
```
Or pass `--clear` at the start of any run:
```bash
uv run python -m scripts.autorl --clear > run.log 2>&1
```

---

## Logging Results to results.tsv

After every run, log the result. The TSV has these columns (tab-separated):

```
commit  composite_pass@1  humaneval_pass@1  mbpp_pass@1  peak_vram_gb  training_hours  status  description
```

- `commit` — 7-char git hash (from the summary block or `git rev-parse --short HEAD`)
- `composite_pass@1` — the primary metric (6 decimal places)
- `humaneval_pass@1` — per-benchmark breakdown
- `mbpp_pass@1` — per-benchmark breakdown
- `peak_vram_gb` — peak GPU memory, rounded to .1f
- `training_hours` — wall-clock training time (excludes final eval)
- `status` — `keep`, `discard`, or `crash`
- `description` — brief description of what this experiment tried

The script auto-appends a `keep` row with `description = "autorl run"` after
each successful run. **You must update the status and description manually** by
editing the last line of `results.tsv`. Change `keep` → `discard` if
composite_pass@1 did not improve.

Do NOT commit `results.tsv` — leave it untracked.

Example:
```
commit	composite_pass@1	humaneval_pass@1	mbpp_pass@1	peak_vram_gb	training_hours	status	description
5011a83	0.000000	0.000000	0.000000	0.0	0.00	keep	header — no run yet
a441d19	0.435000	0.451000	0.419000	42.3	7.83	keep	baseline: Qwen2.5-Coder-7B BF16 MBPP-only
b1c2d3e	0.448000	0.463000	0.433000	42.5	7.91	keep	increase NUM_ROLLOUTS 16→24, LR 1e-6→2e-6
c2d3e4f	0.431000	0.445000	0.417000	42.3	7.87	discard	add code_contests to TRAIN_BENCHMARKS (regression)
```

---

## The Experiment Loop

The loop runs on the experiment branch (e.g. `autorl/apr12`).

**LOOP FOREVER:**

1. **Review state** — read `results.tsv` for trend, `metrics/live.json` for
   last run details, `notes/improvement_plan.md` for prioritised ideas.

2. **Propose one change** — modify the AGENT EDITABLE section of `autorl.py`.
   One hypothesis per experiment. Good starting points from the improvement plan:
   - Increase `NUM_ROLLOUTS` (more diverse GRPO signal)
   - Add `"code_contests"` to `TRAIN_BENCHMARKS` (generalisation)
   - Reduce `LEARNING_RATE` (if reward is unstable)
   - Set `USE_VLLM = True` (2-3x generation throughput)
   - Tune `CLIP_EPS_HIGH` (DAPO asymmetric clip)
   - Enable `ENABLE_THINKING = True` with longer `MAX_COMPLETION_LENGTH`

3. **Commit**:
   ```
   git add scripts/autorl.py
   git commit -m "experiment: <one-line description>"
   ```

4. **Run**:
   ```
   uv run python -m scripts.autorl --clear > run.log 2>&1
   ```
   Each run takes up to `TIME_BUDGET_HOURS` hours. Do not interrupt.

5. **Extract metric**:
   ```
   grep "^composite_pass@1:\|^peak_vram_gb:" run.log
   ```

6. **Evaluate**:
   - If run **crashed**: check `tail -n 80 run.log`. Fix trivial bugs and re-run.
     If the idea is fundamentally broken, log `crash` and revert.
   - If composite_pass@1 **improved**: log `keep`. Advance on this commit.
   - If composite_pass@1 **same or worse**: log `discard`.
     `git reset --hard HEAD~1` to revert to the previous good state.

7. **Update results.tsv** — edit the last auto-appended row to set the correct
   `status` and `description`.

8. **Repeat** — back to step 1. Never ask the human if you should continue.

---

## Keep / Discard Rules

| Condition | Action |
|---|---|
| composite_pass@1 strictly improved | `keep` — stay on this commit |
| composite_pass@1 same or worse | `discard` — `git reset --hard HEAD~1` |
| Crashed (OOM, bug) — trivially fixable | Fix and re-run same commit |
| Crashed (idea fundamentally broken) | `crash` — revert, move on |
| VRAM increased dramatically with no gain | `discard` — not worth it |
| Simplification with equal metric | `keep` — simpler is better |

---

## VRAM Budget

A100 SXM 80GB headroom for Qwen2.5-Coder-7B BF16:
- Weights: ~14 GB
- LoRA + optimizer states: ~6 GB
- KV cache (2048 tokens, batch 16, rollouts 16): ~22 GB
- Activations + overhead: ~12 GB
- **Total: ~54 GB** — 26 GB free headroom

Watch `peak_vram_gb` in results.tsv. Experiments blowing past 70 GB risk OOM.

---

## Simplicity Criterion

All else being equal, simpler is better. A 0.001 improvement that adds 50
lines of hacky code is not worth it. Removing something and getting equal or
better results is a great outcome. When evaluating whether to keep a change,
weigh complexity cost against improvement magnitude.

---

## NEVER STOP

Once the loop has begun, do NOT pause to ask the human if you should continue.
Do NOT ask "is this a good stopping point?" The human may be asleep. You are
autonomous. If you run out of ideas, read `notes/improvement_plan.md` again,
look at which experiments regressed and understand why, try combining two
near-misses, or try a more radical change. The loop runs until you are manually
interrupted. Period.

At `TIME_BUDGET_HOURS = 8`, each experiment takes roughly 8–10 hours end-to-end
(training + final eval). Plan accordingly — this is slower than autoresearch's
5-minute budget, but each experiment is a full RL training run.
