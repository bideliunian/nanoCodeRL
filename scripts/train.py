"""nanoCodeRL training script — DAPO coding RL on Qwen3.5-4B.

Usage:
    python -m scripts.train
    python -m scripts.train --steps 500 --wandb
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from nanoCodeRL.config import Config
from nanoCodeRL.data import load_training_data, load_eval_data, build_messages, apply_chat_template
from nanoCodeRL.sandbox import compute_reward, compute_rewards_parallel, extract_code


def load_model_and_tokenizer(cfg: Config):
    """Load Qwen3.5 with QLoRA via Unsloth (or fallback to standard HF)."""
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
        print("Loaded model via Unsloth (QLoRA)")

    except ImportError:
        print("Unsloth not available, falling back to standard HF + PEFT")
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if cfg.load_in_4bit else None

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_reward_fn(cfg: Config, problems: list[dict]):
    """Build the reward function for GRPO training with parallel sandbox execution.

    Uses integer problem indices passed through the dataset to look up test cases,
    avoiding string-matching issues caused by TRL's tokenize-then-decode pipeline.
    """
    # Index problems by their integer position
    _problems = list(problems)

    _debug = os.environ.get("REWARD_DEBUG", "0") == "1"
    _step = [0]

    def reward_fn(completions: list[str], problem_idx: list[int], **kwargs) -> list[float]:
        tasks = []
        for completion, idx in zip(completions, problem_idx):
            problem = _problems[idx]
            clean = extract_code(completion)

            if problem["source"] == "humaneval":
                code = problem["prompt"] + clean
            else:
                code = clean

            tasks.append((code, problem["test_cases"]))

        rewards = compute_rewards_parallel(
            tasks,
            timeout=cfg.sandbox_timeout,
            max_workers=cfg.sandbox_max_workers,
            use_docker=cfg.use_docker_sandbox,
        )

        if _debug:
            _step[0] += 1
            print(f"\n=== REWARD_DEBUG step {_step[0]} ===")
            for i, (completion, idx, reward) in enumerate(zip(completions, problem_idx, rewards)):
                problem = _problems[idx]
                clean = extract_code(completion)
                print(f"  [{i}] task={problem['task_id']} reward={reward:.2f} "
                      f"raw_len={len(completion)} clean_len={len(clean)}")
                print(f"       raw[:120]:   {repr(completion[:120])}")
                print(f"       clean[:120]: {repr(clean[:120])}")

        return rewards

    return reward_fn


def prepare_dataset(problems: list[dict], tokenizer, cfg: Config) -> list[dict]:
    """Convert problems into the format expected by GRPOTrainer.

    Includes a ``problem_idx`` column that TRL passes through to the reward
    function via kwargs, avoiding prompt string-matching issues.
    """
    dataset = []
    for i, p in enumerate(problems):
        messages = build_messages(p["prompt"], p["source"])
        prompt_text = apply_chat_template(
            tokenizer, messages, enable_thinking=cfg.enable_thinking,
        )
        dataset.append({"prompt": prompt_text, "problem_idx": i})
    return dataset


# ---------------------------------------------------------------------------
# Intermediate evaluation callback
# ---------------------------------------------------------------------------

class IntermediateEvalCallback(TrainerCallback):
    """Run eval on a subset of held-out benchmarks every N steps."""

    def __init__(self, model, tokenizer, cfg: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.eval_problems = None  # lazy load
        self.eval_log_path = os.path.join(cfg.log_dir, "eval_during_training.jsonl")

    def _load_eval_problems(self):
        """Load a subset of eval problems for quick intermediate evaluation."""
        all_problems = load_eval_data(self.cfg.eval_benchmarks)
        if self.cfg.eval_subset_size > 0:
            # Take first N from each benchmark for deterministic subset
            by_source = {}
            for p in all_problems:
                by_source.setdefault(p["source"], []).append(p)
            subset = []
            for source, problems in by_source.items():
                subset.extend(problems[:self.cfg.eval_subset_size])
            return subset
        return all_problems

    def _generate_solution(self, prompt: str, source: str) -> str:
        messages = build_messages(prompt, source)
        text = apply_chat_template(
            self.tokenizer, messages, enable_thinking=self.cfg.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_completion_length,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step == 0 or step % self.cfg.eval_every_n_steps != 0:
            return

        if self.eval_problems is None:
            self.eval_problems = self._load_eval_problems()

        print(f"\n--- Intermediate eval at step {step} ({len(self.eval_problems)} problems) ---")
        self.model.eval()

        passed_by_source = {}
        total_by_source = {}

        for problem in self.eval_problems:
            source = problem["source"]
            completion = self._generate_solution(problem["prompt"], source)

            clean = extract_code(completion)
            if source == "humaneval":
                code = problem["prompt"] + clean
            else:
                code = clean

            result = compute_reward(
                code=code,
                test_cases=problem["test_cases"],
                timeout=self.cfg.sandbox_timeout,
                use_docker=self.cfg.use_docker_sandbox,
            )

            passed_by_source[source] = passed_by_source.get(source, 0) + (1 if result["reward"] == 1.0 else 0)
            total_by_source[source] = total_by_source.get(source, 0) + 1

        self.model.train()

        # Log results
        log_entry = {"step": step}
        for source in sorted(total_by_source):
            p = passed_by_source.get(source, 0)
            t = total_by_source[source]
            rate = p / t if t > 0 else 0.0
            log_entry[f"{source}_pass@1"] = rate
            log_entry[f"{source}_passed"] = p
            log_entry[f"{source}_total"] = t
            print(f"  {source}: {p}/{t} = {rate:.1%}")

        with open(self.eval_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Log to wandb if enabled
        if self.cfg.use_wandb:
            try:
                import wandb
                wandb.log({f"eval/{k}": v for k, v in log_entry.items()}, step=step)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="nanoCodeRL DAPO training")
    parser.add_argument("--steps", type=int, default=None, help="Override num_train_steps")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--train-data", type=str, nargs="+", default=None,
        help="Training data sources (code_contests, mbpp_full). Default: code_contests",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--num-rollouts", type=int, default=None, help="Override num_rollouts")
    parser.add_argument("--max-length", type=int, default=None, help="Override max_completion_length")
    args = parser.parse_args()

    cfg = Config()
    if args.steps:
        cfg.num_train_steps = args.steps
    if args.wandb:
        cfg.use_wandb = True
    if args.lr:
        cfg.learning_rate = args.lr
    if args.train_data:
        cfg.train_benchmarks = args.train_data
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.num_rollouts:
        cfg.num_rollouts = args.num_rollouts
    if args.max_length:
        cfg.max_completion_length = args.max_length

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print(f"=== nanoCodeRL Training ===")
    print(f"Model: {cfg.model_name}")
    print(f"Steps: {cfg.num_train_steps}")
    print(f"Rollouts/prompt: {cfg.num_rollouts}")
    print(f"Max completion: {cfg.max_completion_length} tokens")
    print(f"Sandbox workers: {cfg.sandbox_max_workers}")
    print()

    # Load data (training split only — eval uses separate held-out data)
    problems = load_training_data(cfg.train_benchmarks)

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Fix for trl/peft compatibility: Add warnings_issued attribute
    if not hasattr(model, 'warnings_issued'):
        model.warnings_issued = {}

    # Build reward function (uses problem indices, not string matching)
    reward_fn = build_reward_fn(cfg, problems)

    # Prepare dataset (includes problem_idx for reward lookup)
    dataset = prepare_dataset(problems, tokenizer, cfg)

    # Configure GRPO with DAPO settings
    training_config = GRPOConfig(
        output_dir=cfg.checkpoint_dir,
        num_train_epochs=1,
        max_steps=cfg.num_train_steps,
        per_device_train_batch_size=cfg.batch_size,
        num_generations=cfg.num_rollouts,
        max_completion_length=cfg.max_completion_length,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=cfg.warmup_steps,
        logging_steps=1,
        save_steps=cfg.eval_every_n_steps,
        bf16=True,
        loss_type="dapo",
        report_to="wandb" if cfg.use_wandb else "none",
        run_name=f"nanoCodeRL-{cfg.model_name.split('/')[-1]}",
    )

    # Build callbacks
    eval_callback = IntermediateEvalCallback(model, tokenizer, cfg)

    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )

    # Train
    print("Starting DAPO training...")
    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    # Save final checkpoint
    final_dir = os.path.join(cfg.checkpoint_dir, "last")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    print(f"\nTraining complete in {elapsed / 3600:.1f} hours")
    print(f"Final checkpoint saved to {final_dir}")
    print(f"Next: python -m scripts.eval --ckpt {final_dir}")


if __name__ == "__main__":
    main()
