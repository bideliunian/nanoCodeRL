"""autorl.py — Autonomous RL training script for nanoCodeRL.

Philosophy (adapted from karpathy/autoresearch):
  - This is the ONE file the agent modifies.
  - The frozen eval harness (sandbox.py, data.py, eval functions below) is
    the ground truth metric. Never touch it.
  - Training runs for a fixed TIME_BUDGET_HOURS wall-clock. Steps are a
    consequence, not a target.
  - The single optimisation target is composite_pass@1 = mean(humaneval, mbpp).
  - Every run appends one row to results.tsv (commit, metric, memory, status).
  - Live progress is written to metrics/live.json throughout the run so the
    agent can inspect training without waiting for the final summary.

Usage:
    uv run python -m scripts.autorl             # standard run
    uv run python -m scripts.autorl --clear     # clear live metrics first
    uv run python -m scripts.autorl > run.log 2>&1   # redirect for agent

Extract key metric after the run:
    grep "^composite_pass@1:" run.log
"""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT EDITABLE SECTION
# Modify anything inside this block. All hyperparameters live here.
# The goal is a higher composite_pass@1 = (humaneval_pass@1 + mbpp_pass@1) / 2.
# Run: uv run python -m scripts.autorl > run.log 2>&1
# ═══════════════════════════════════════════════════════════════════════════════

# Wall-clock time budget (hours). Training stops after this, then final eval runs.
TIME_BUDGET_HOURS: float = 8.0

# --- Model ---
MODEL_NAME: str = "Qwen/Qwen2.5-Coder-7B"
LOAD_IN_4BIT: bool = False          # BF16 on A100-80GB; set True for ≤40GB VRAM

# --- LoRA ---
LORA_R: int = 16
LORA_ALPHA: int = 32
LORA_DROPOUT: float = 0.0
LORA_TARGET_MODULES: list = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# --- Generation ---
MAX_COMPLETION_LENGTH: int = 2048
MAX_PROMPT_LENGTH: int = 1024
TEMPERATURE: float = 0.7
TOP_P: float = 0.95
ENABLE_THINKING: bool = False       # Qwen3 chain-of-thought (long sequences, slow)

# --- DAPO / GRPO ---
NUM_ROLLOUTS: int = 16              # completions sampled per prompt per step
BATCH_SIZE: int = 16                # completions per micro-batch (not prompts)
GRADIENT_ACCUMULATION_STEPS: int = 1
LEARNING_RATE: float = 1e-6
LR_SCHEDULER: str = "cosine"
WARMUP_STEPS: int = 10
CLIP_EPS: float = 0.2               # standard PPO lower clip
CLIP_EPS_HIGH: float = 0.28         # DAPO asymmetric upper clip (> CLIP_EPS)

# --- Data ---
TRAIN_BENCHMARKS: list = ["mbpp_full"]   # options: "mbpp_full", "code_contests"
CC_MAX_DIFFICULTY: int = 7               # CodeContests difficulty ceiling
CC_MAX_SOLUTION_CHARS: int = 800         # skip problems with long reference solutions

# --- Sandbox ---
SANDBOX_TIMEOUT: int = 5
SANDBOX_MAX_WORKERS: int = 8

# --- vLLM (recommended on 80GB A100 for 2-3x generation throughput) ---
USE_VLLM: bool = False
VLLM_GPU_MEMORY_UTILIZATION: float = 0.45

# --- Intermediate evaluation (mid-training monitoring, not the final metric) ---
EVAL_EVERY_N_STEPS: int = 50
EVAL_SUBSET_SIZE: int = 20    # problems per benchmark; 0 = full (adds ~20 min per eval)

# ═══════════════════════════════════════════════════════════════════════════════
# END AGENT EDITABLE SECTION
# ═══════════════════════════════════════════════════════════════════════════════

import argparse
import json
import os
import random
import subprocess
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from nanoCodeRL.config import Config
from nanoCodeRL.data import (
    apply_chat_template,
    build_messages,
    load_eval_data,
    load_training_data,
)
from nanoCodeRL.sandbox import (
    compute_reward,
    compute_rewards_parallel,
    extract_code,
)

# ---------------------------------------------------------------------------
# Fixed paths — not configurable by agent
# ---------------------------------------------------------------------------
_METRICS_DIR = "metrics"
_LIVE_PATH = os.path.join(_METRICS_DIR, "live.json")
_RESULTS_TSV = "results.tsv"
_RESULTS_HEADER = (
    "commit\tcomposite_pass@1\thumaneval_pass@1\tmbpp_pass@1\t"
    "peak_vram_gb\ttraining_hours\tstatus\tdescription\n"
)


# ═══════════════════════════════════════════════════════════════════════════════
# FROZEN SECTION — DO NOT MODIFY
# Eval harness, metrics, results logging. These are ground truth.
# ═══════════════════════════════════════════════════════════════════════════════

def _write_live(data: dict) -> None:
    """Overwrite metrics/live.json with current training state.

    Call this on every interesting event (step, eval, start, end).
    The agent can `cat metrics/live.json` at any time to check progress.
    """
    os.makedirs(_METRICS_DIR, exist_ok=True)
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(_LIVE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _clear_live() -> None:
    """Clear live metrics file (--clear flag)."""
    os.makedirs(_METRICS_DIR, exist_ok=True)
    _write_live({"status": "cleared", "message": "waiting for next run to start"})


def _ensure_results_tsv() -> None:
    if not os.path.exists(_RESULTS_TSV):
        with open(_RESULTS_TSV, "w") as f:
            f.write(_RESULTS_HEADER)


def _append_results_tsv(
    commit: str,
    composite: float,
    humaneval: float,
    mbpp: float,
    peak_vram_gb: float,
    training_hours: float,
    status: str,
    description: str,
) -> None:
    _ensure_results_tsv()
    with open(_RESULTS_TSV, "a") as f:
        f.write(
            f"{commit}\t{composite:.6f}\t{humaneval:.6f}\t{mbpp:.6f}\t"
            f"{peak_vram_gb:.1f}\t{training_hours:.2f}\t{status}\t{description}\n"
        )


def _git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.getcwd(),
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _peak_vram_gb() -> float:
    try:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass
    return 0.0


def _eval_pass_at_1(model, tokenizer, cfg: Config, subset_size: int = 0) -> dict[str, float]:
    """FROZEN EVAL HARNESS — do not modify.

    Evaluate pass@1 on HumanEval and MBPP (or a fixed subset).
    This is the ground-truth metric used for keep/discard decisions.
    Returns {source: pass@1_rate}.
    """
    problems = load_eval_data(cfg.eval_benchmarks)
    if subset_size > 0:
        by_source: dict[str, list] = {}
        for p in problems:
            by_source.setdefault(p["source"], []).append(p)
        problems = []
        for probs in by_source.values():
            problems.extend(probs[:subset_size])

    model.eval()
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    fim_tokens = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|file_sep|>"]
    fim_ids = [
        tid for tok in fim_tokens
        if (tid := _tok.convert_tokens_to_ids(tok)) != _tok.unk_token_id
    ]
    eos_ids = list({_tok.eos_token_id} | set(fim_ids))

    passed_by_source: dict[str, int] = {}
    total_by_source: dict[str, int] = {}

    for problem in problems:
        source = problem["source"]
        messages = build_messages(problem["prompt"], source)
        text = apply_chat_template(tokenizer, messages, enable_thinking=cfg.enable_thinking)
        inputs = _tok(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=_tok.pad_token_id,
                eos_token_id=eos_ids,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion = _tok.decode(new_tokens, skip_special_tokens=True)
        clean = extract_code(completion)
        code = (problem["prompt"] + clean) if source == "humaneval" else clean

        result = compute_reward(
            code=code,
            test_cases=problem["test_cases"],
            timeout=cfg.sandbox_timeout,
            use_docker=cfg.use_docker_sandbox,
        )
        passed_by_source[source] = passed_by_source.get(source, 0) + (
            1 if result["reward"] == 1.0 else 0
        )
        total_by_source[source] = total_by_source.get(source, 0) + 1

    model.train()

    results = {}
    for source, total in total_by_source.items():
        passed = passed_by_source.get(source, 0)
        results[source] = passed / total if total > 0 else 0.0
        print(f"  {source}: {passed}/{total} = {results[source]:.1%}")
    return results


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TimeBudgetCallback(TrainerCallback):
    """Stop training when wall-clock budget is exhausted."""

    def __init__(self, budget_hours: float, live_state: dict) -> None:
        self.deadline = time.time() + budget_hours * 3600
        self.live_state = live_state

    def on_step_end(self, args, state, control, **kwargs) -> None:
        remaining = self.deadline - time.time()
        self.live_state["training_step"] = state.global_step
        self.live_state["remaining_hours"] = round(max(remaining, 0) / 3600, 2)
        self.live_state["elapsed_hours"] = round(
            (time.time() - (self.deadline - TIME_BUDGET_HOURS * 3600)) / 3600, 2
        )
        _write_live(self.live_state)
        if remaining <= 0:
            print(f"\nTime budget exhausted after {state.global_step} steps. Stopping.")
            control.should_training_stop = True


class IntermediateEvalCallback(TrainerCallback):
    """Evaluate pass@1 on a held-out subset every N steps; write to live.json."""

    def __init__(self, model, tokenizer, cfg: Config, live_state: dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.live_state = live_state
        self._log_path = os.path.join(cfg.log_dir, "eval_during_training.jsonl")

    def on_step_end(self, args, state, control, **kwargs) -> None:
        step = state.global_step
        if step == 0 or step % self.cfg.eval_every_n_steps != 0:
            return

        print(f"\n--- Mid-training eval at step {step} "
              f"(subset={self.cfg.eval_subset_size or 'full'}) ---")
        scores = _eval_pass_at_1(
            self.model, self.tokenizer, self.cfg,
            subset_size=self.cfg.eval_subset_size,
        )

        log_entry = {"step": step}
        for src, rate in scores.items():
            log_entry[f"{src}_pass@1"] = rate
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Update live.json with latest mid-training eval
        self.live_state["last_eval_step"] = step
        self.live_state["last_eval_scores"] = scores
        _write_live(self.live_state)

        if self.cfg.use_wandb:
            try:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in log_entry.items()}, step=step)
            except Exception:
                pass


class DynamicDataset(torch.utils.data.Dataset):
    """DAPO dynamic sampling — deprioritise zero-variance problems."""

    def __init__(
        self,
        full_dataset: list[dict],
        reward_fn,
        max_zero_var_streak: int = 3,
        deprioritize_prob: float = 0.05,
    ) -> None:
        self.full_dataset = full_dataset
        self.reward_fn = reward_fn
        self.max_zero_var_streak = max_zero_var_streak
        self.deprioritize_prob = deprioritize_prob
        self._active_indices = list(range(len(full_dataset)))

    def refresh(self) -> None:
        counts = self.reward_fn.zero_var_counts
        n = len(self.full_dataset)
        active = [i for i in range(n) if counts.get(i, 0) < self.max_zero_var_streak]
        inactive = [i for i in range(n) if counts.get(i, 0) >= self.max_zero_var_streak]
        order = active.copy()
        random.shuffle(order)
        for idx in inactive:
            if random.random() < self.deprioritize_prob:
                order.append(idx)
        self._active_indices = order
        if inactive:
            print(f"  DynamicDataset: {len(active)} active, {len(inactive)} deprioritised")

    def __len__(self) -> int:
        return len(self._active_indices)

    def __getitem__(self, idx) -> dict:
        return self.full_dataset[self._active_indices[idx]]


class DynamicSamplingCallback(TrainerCallback):
    def __init__(self, dynamic_dataset: DynamicDataset) -> None:
        self.dynamic_dataset = dynamic_dataset

    def on_epoch_begin(self, args, state, control, **kwargs) -> None:
        self.dynamic_dataset.refresh()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(cfg: Config):
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_completion_length + 512,
            load_in_4bit=cfg.load_in_4bit,
            dtype=None,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        print("Loaded model via Unsloth")
    except ImportError:
        print("Unsloth not available — falling back to HF + PEFT")
        from peft import LoraConfig, get_peft_model
        from transformers import BitsAndBytesConfig

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if cfg.load_in_4bit
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                task_type="CAUSAL_LM",
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _build_reward_fn(cfg: Config, problems: list[dict]):
    _problems = list(problems)
    _debug = os.environ.get("REWARD_DEBUG", "0") == "1"
    _step = [0]
    zero_var_counts: dict[int, int] = {}

    def reward_fn(completions: list[str], problem_idx: list[int], **kwargs) -> list[float]:
        tasks = []
        for completion, idx in zip(completions, problem_idx):
            problem = _problems[idx]
            clean = extract_code(completion)
            code = (problem["prompt"] + clean) if problem["source"] == "humaneval" else clean
            tasks.append((code, problem["test_cases"]))

        rewards = compute_rewards_parallel(
            tasks,
            timeout=cfg.sandbox_timeout,
            max_workers=cfg.sandbox_max_workers,
            use_docker=cfg.use_docker_sandbox,
        )

        grouped: dict[int, list[float]] = {}
        for idx, r in zip(problem_idx, rewards):
            grouped.setdefault(idx, []).append(r)
        for idx, group_rewards in grouped.items():
            if len(set(group_rewards)) <= 1:
                zero_var_counts[idx] = zero_var_counts.get(idx, 0) + 1
            else:
                zero_var_counts[idx] = 0

        if _debug:
            _step[0] += 1
            print(f"\n=== REWARD_DEBUG step {_step[0]} ===")
            for i, (completion, idx, reward) in enumerate(
                zip(completions, problem_idx, rewards)
            ):
                problem = _problems[idx]
                clean = extract_code(completion)
                print(
                    f"  [{i}] task={problem['task_id']} reward={reward:.2f} "
                    f"len={len(clean)}"
                )
        return rewards

    reward_fn.zero_var_counts = zero_var_counts
    return reward_fn


def _prepare_dataset(problems: list[dict], tokenizer, cfg: Config) -> list[dict]:
    dataset = []
    for i, p in enumerate(problems):
        messages = build_messages(p["prompt"], p["source"])
        prompt_text = apply_chat_template(
            tokenizer, messages, enable_thinking=cfg.enable_thinking,
        )
        dataset.append({"prompt": prompt_text, "problem_idx": i})
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_cfg() -> Config:
    cfg = Config()
    cfg.model_name = MODEL_NAME
    cfg.load_in_4bit = LOAD_IN_4BIT
    cfg.lora_r = LORA_R
    cfg.lora_alpha = LORA_ALPHA
    cfg.lora_dropout = LORA_DROPOUT
    cfg.lora_target_modules = LORA_TARGET_MODULES
    cfg.enable_thinking = ENABLE_THINKING
    cfg.max_completion_length = MAX_COMPLETION_LENGTH
    cfg.max_prompt_length = MAX_PROMPT_LENGTH
    cfg.temperature = TEMPERATURE
    cfg.top_p = TOP_P
    cfg.num_rollouts = NUM_ROLLOUTS
    cfg.batch_size = BATCH_SIZE
    cfg.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    cfg.num_train_steps = 999_999  # stopped by TimeBudgetCallback
    cfg.learning_rate = LEARNING_RATE
    cfg.lr_scheduler = LR_SCHEDULER
    cfg.warmup_steps = WARMUP_STEPS
    cfg.clip_eps = CLIP_EPS
    cfg.clip_eps_high = CLIP_EPS_HIGH
    cfg.sandbox_timeout = SANDBOX_TIMEOUT
    cfg.sandbox_max_workers = SANDBOX_MAX_WORKERS
    cfg.train_benchmarks = TRAIN_BENCHMARKS
    cfg.cc_max_difficulty = CC_MAX_DIFFICULTY
    cfg.cc_max_solution_chars = CC_MAX_SOLUTION_CHARS
    cfg.use_vllm = USE_VLLM
    cfg.vllm_gpu_memory_utilization = VLLM_GPU_MEMORY_UTILIZATION
    cfg.eval_every_n_steps = EVAL_EVERY_N_STEPS
    cfg.eval_subset_size = EVAL_SUBSET_SIZE
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="autorl — autoresearch-style RL training")
    parser.add_argument(
        "--clear", action="store_true",
        help="Clear metrics/live.json before starting (fresh monitoring slate)",
    )
    args = parser.parse_args()

    if args.clear:
        _clear_live()
        print("Live metrics cleared.")

    _ensure_results_tsv()
    cfg = _build_cfg()
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)

    commit = _git_commit_hash()

    # Shared mutable dict that all callbacks write into before _write_live()
    live_state: dict = {
        "status": "initialising",
        "commit": commit,
        "model": MODEL_NAME,
        "time_budget_hours": TIME_BUDGET_HOURS,
        "training_step": 0,
        "elapsed_hours": 0.0,
        "remaining_hours": TIME_BUDGET_HOURS,
        "last_eval_step": None,
        "last_eval_scores": {},
        "final_scores": {},
    }
    _write_live(live_state)

    print("=== autorl — nanoCodeRL autonomous training ===")
    print(f"Commit:       {commit}")
    print(f"Model:        {MODEL_NAME}")
    print(f"Budget:       {TIME_BUDGET_HOURS}h wall-clock")
    print(f"Train data:   {TRAIN_BENCHMARKS}")
    print(f"vLLM:         {USE_VLLM}")
    print()

    problems = load_training_data(
        cfg.train_benchmarks,
        max_difficulty=cfg.cc_max_difficulty,
        max_solution_chars=cfg.cc_max_solution_chars,
    )
    print(f"Training problems: {len(problems)}")

    live_state["status"] = "loading_model"
    _write_live(live_state)

    model, tokenizer = _load_model_and_tokenizer(cfg)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    reward_fn = _build_reward_fn(cfg, problems)
    dataset = _prepare_dataset(problems, tokenizer, cfg)

    training_config = GRPOConfig(
        output_dir=cfg.checkpoint_dir,
        num_train_epochs=1,
        max_steps=cfg.num_train_steps,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_generations=cfg.num_rollouts,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=cfg.max_prompt_length,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=cfg.warmup_steps,
        logging_steps=1,
        save_steps=cfg.eval_every_n_steps,
        bf16=True,
        gradient_checkpointing=True,
        loss_type="dapo",
        epsilon=cfg.clip_eps,
        epsilon_high=cfg.clip_eps_high,
        mask_truncated_completions=True,
        report_to="none",
    )

    if cfg.use_vllm:
        training_config.use_vllm = True
        training_config.vllm_mode = "colocate"
        training_config.vllm_gpu_memory_utilization = cfg.vllm_gpu_memory_utilization
        training_config.vllm_enable_sleep_mode = True

    dynamic_dataset = DynamicDataset(dataset, reward_fn)
    callbacks = [
        TimeBudgetCallback(TIME_BUDGET_HOURS, live_state),
        IntermediateEvalCallback(model, tokenizer, cfg, live_state),
        DynamicSamplingCallback(dynamic_dataset),
    ]

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dynamic_dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    live_state["status"] = "training"
    _write_live(live_state)

    print("Starting DAPO training (time-budget mode)...")
    train_start = time.time()
    trainer.train()
    training_hours = (time.time() - train_start) / 3600
    total_steps = trainer.state.global_step

    # Save checkpoint
    ckpt_dir = os.path.join(cfg.checkpoint_dir, "last")
    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # Final evaluation (frozen harness — full benchmark)
    live_state["status"] = "final_eval"
    _write_live(live_state)
    print(f"\n--- Final evaluation (full benchmarks) ---")
    final_scores = _eval_pass_at_1(model, tokenizer, cfg, subset_size=0)

    humaneval = final_scores.get("humaneval", 0.0)
    mbpp = final_scores.get("mbpp", 0.0)
    composite = (humaneval + mbpp) / 2 if final_scores else 0.0
    peak_vram = _peak_vram_gb()

    # Update live.json — final state
    live_state.update({
        "status": "complete",
        "training_step": total_steps,
        "elapsed_hours": round(training_hours, 2),
        "remaining_hours": 0.0,
        "final_scores": final_scores,
        "composite_pass@1": round(composite, 6),
        "peak_vram_gb": round(peak_vram, 1),
    })
    _write_live(live_state)

    # Append to results.tsv
    _append_results_tsv(
        commit=commit,
        composite=composite,
        humaneval=humaneval,
        mbpp=mbpp,
        peak_vram_gb=peak_vram,
        training_hours=training_hours,
        status="keep",          # agent overwrites this to "discard" if metric regressed
        description="autorl run",
    )

    # Parseable summary block — grep "^composite_pass@1:" run.log
    print(f"""
---
composite_pass@1: {composite:.6f}
humaneval_pass@1: {humaneval:.6f}
mbpp_pass@1:      {mbpp:.6f}
training_hours:   {training_hours:.2f}
total_steps:      {total_steps}
peak_vram_gb:     {peak_vram:.1f}
commit:           {commit}
status:           complete
---""")


if __name__ == "__main__":
    main()
