"""nanoCodeRL evaluation — HumanEval and MBPP pass@1.

Usage:
    python -m scripts.eval                             # baseline (pre-RL)
    python -m scripts.eval --ckpt checkpoints/last     # post-RL
    python -m scripts.eval --bench humaneval           # single benchmark
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanoCodeRL.config import Config
from nanoCodeRL.data import load_eval_data, build_messages, apply_chat_template
from nanoCodeRL.sandbox import compute_reward, extract_code


def load_model_for_eval(model_name: str, ckpt: str | None, cfg: Config):
    """Load model for evaluation, optionally from a LoRA checkpoint."""
    if ckpt:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=ckpt,
                max_seq_length=cfg.max_completion_length + 512,
                load_in_4bit=cfg.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
            print(f"Loaded checkpoint from {ckpt} via Unsloth")
        except ImportError:
            from peft import PeftModel
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, ckpt)
            model = model.merge_and_unload()
            tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
            print(f"Loaded checkpoint from {ckpt} via PEFT")
    else:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=cfg.max_completion_length + 512,
                load_in_4bit=cfg.load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            if torch.cuda.is_available() and cfg.load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                bnb_config = None
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        print(f"Loaded base model: {model_name}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_solution(model, tokenizer, prompt: str, cfg: Config,
                      source: str = "humaneval") -> str:
    """Generate a code solution for a given prompt."""
    messages = build_messages(prompt, source)
    text = apply_chat_template(tokenizer, messages, enable_thinking=cfg.enable_thinking)
    # Qwen3.5 is a VLM: Unsloth patches the processor __call__ to handle images,
    # which breaks text-only tokenization. Use the underlying text tokenizer directly.
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = _tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_completion_length,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return completion


def evaluate_benchmark(
    model, tokenizer, problems: list[dict], cfg: Config, benchmark_name: str,
) -> dict:
    """Evaluate pass@1 on a set of problems."""
    passed = 0
    total = len(problems)
    results = []

    print(f"\nEvaluating {benchmark_name} ({total} problems)...")

    for i, problem in enumerate(problems):
        completion = generate_solution(
            model, tokenizer, problem["prompt"], cfg, source=problem["source"]
        )

        clean = extract_code(completion)
        if problem["source"] == "humaneval":
            full_code = problem["prompt"] + clean
        else:
            full_code = clean

        result = compute_reward(
            code=full_code,
            test_cases=problem["test_cases"],
            timeout=cfg.sandbox_timeout,
        )

        is_pass = result["reward"] == 1.0
        if is_pass:
            passed += 1

        results.append({
            "task_id": problem["task_id"],
            "passed": is_pass,
            "reward": result["reward"],
            "errors": result["errors"][:1] if result["errors"] else [],
        })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] pass@1 so far: {passed}/{i+1} = {passed/(i+1):.1%}")

    pass_at_1 = passed / total if total > 0 else 0.0
    print(f"\n{benchmark_name} pass@1: {passed}/{total} = {pass_at_1:.1%}")

    return {
        "benchmark": benchmark_name,
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="nanoCodeRL evaluation")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument(
        "--bench", type=str, nargs="+", default=["humaneval", "mbpp"],
        help="Benchmarks to evaluate (humaneval, mbpp)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    tag = "post-rl" if args.ckpt else "baseline"
    print(f"=== nanoCodeRL Eval ({tag}) ===")

    model, tokenizer = load_model_for_eval(cfg.model_name, args.ckpt, cfg)

    all_results = {}
    for bench in args.bench:
        problems = load_eval_data([bench])
        result = evaluate_benchmark(model, tokenizer, problems, cfg, bench)
        all_results[bench] = result

    # Save results
    output_path = args.output or os.path.join(
        cfg.results_dir, f"eval_{tag}.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n=== Summary ===")
    for bench, result in all_results.items():
        print(f"  {bench} pass@1: {result['pass_at_1']:.1%} ({result['passed']}/{result['total']})")


if __name__ == "__main__":
    main()
