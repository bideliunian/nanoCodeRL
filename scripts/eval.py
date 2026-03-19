"""nanoCodeRL evaluation — HumanEval and MBPP pass@1.

Usage:
    python -m scripts.eval                             # baseline (pre-RL)
    python -m scripts.eval --ckpt checkpoints/last     # post-RL
    python -m scripts.eval --bench humaneval           # single benchmark
"""

import argparse
import json
import os

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
        except (ImportError, NotImplementedError):
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
        except (ImportError, NotImplementedError):
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


def generate_solutions_batch(
    model, tokenizer, problems: list[dict], cfg: Config, batch_size: int = 8,
) -> list[str]:
    """Generate code solutions for a list of problems using batched inference.

    Left-pads prompts so shorter sequences don't waste compute. Returns one
    completion string per problem, in the same order as the input list.
    """
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    # Ensure left-padding for batched generation (decoder-only models)
    original_side = _tok.padding_side
    _tok.padding_side = "left"
    if _tok.pad_token_id is None:
        _tok.pad_token_id = _tok.eos_token_id

    # Build all prompt texts
    prompt_texts = []
    for p in problems:
        messages = build_messages(p["prompt"], p["source"])
        text = apply_chat_template(tokenizer, messages, enable_thinking=cfg.enable_thinking)
        prompt_texts.append(text)

    completions = []
    for start in range(0, len(prompt_texts), batch_size):
        batch_texts = prompt_texts[start : start + batch_size]
        inputs = _tok(
            batch_texts, return_tensors="pt", padding=True, truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg.max_completion_length,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                do_sample=True,
                pad_token_id=_tok.pad_token_id,
            )

        # Decode only the newly generated tokens for each sequence
        for j, output_ids in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[prompt_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            completions.append(completion)

    _tok.padding_side = original_side
    return completions


def evaluate_benchmark(
    model, tokenizer, problems: list[dict], cfg: Config, benchmark_name: str,
    batch_size: int = 8,
) -> dict:
    """Evaluate pass@1 on a set of problems with batched generation."""
    total = len(problems)
    print(f"\nEvaluating {benchmark_name} ({total} problems, batch_size={batch_size})...")

    # Batched generation for all problems at once
    completions = generate_solutions_batch(
        model, tokenizer, problems, cfg, batch_size=batch_size,
    )

    # Score completions (sandbox execution)
    passed = 0
    results = []
    for i, (problem, completion) in enumerate(zip(problems, completions)):
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
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for generation")
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    tag = "post-rl" if args.ckpt else "baseline"
    print(f"=== nanoCodeRL Eval ({tag}) ===")

    model, tokenizer = load_model_for_eval(cfg.model_name, args.ckpt, cfg)

    all_results = {}
    for bench in args.bench:
        problems = load_eval_data([bench])
        result = evaluate_benchmark(model, tokenizer, problems, cfg, bench,
                                    batch_size=args.batch_size)
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
