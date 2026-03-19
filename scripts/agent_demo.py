"""nanoCodeRL agent demo — multi-turn iterative coding agent.

Usage:
    python -m scripts.agent_demo
    python -m scripts.agent_demo --ckpt checkpoints/last
    python -m scripts.agent_demo --task-id HumanEval/0
    python -m scripts.agent_demo --max-turns 5
"""

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nanoCodeRL.config import Config
from nanoCodeRL.data import load_humaneval, SYSTEM_PROMPT, build_messages, apply_chat_template
from nanoCodeRL.sandbox import compute_reward


def load_agent_model(cfg: Config, ckpt: str | None):
    """Load model for the agent demo."""
    model_name = ckpt or cfg.model_name
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=cfg.max_completion_length + 2048,
            load_in_4bit=cfg.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], cfg: Config) -> str:
    """Generate a response given chat messages."""
    prompt_text = apply_chat_template(tokenizer, messages, enable_thinking=cfg.enable_thinking)
    _tok = getattr(tokenizer, "tokenizer", tokenizer)
    inputs = _tok(prompt_text, return_tensors="pt").to(model.device)

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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def agent_loop(
    model, tokenizer, problem: dict, cfg: Config, max_turns: int = 5,
) -> dict:
    """Run the multi-turn agent loop on a single problem."""
    print(f"\n{'='*60}")
    print(f"Problem: {problem['task_id']}")
    print(f"{'='*60}")
    print(problem["prompt"])

    # Build initial messages for chat template
    messages = build_messages(problem["prompt"], problem["source"])

    history = []

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn}/{max_turns} ---")

        completion = generate(model, tokenizer, messages, cfg)
        print(f"Generated code:\n{completion[:500]}{'...' if len(completion) > 500 else ''}")

        if problem["source"] == "humaneval":
            full_code = problem["prompt"] + completion
        else:
            full_code = completion

        result = compute_reward(
            code=full_code,
            test_cases=problem["test_cases"],
            timeout=cfg.sandbox_timeout,
        )

        print(f"Result: {result['passed']}/{result['total']} tests passed (reward={result['reward']:.2f})")

        history.append({
            "turn": turn,
            "completion": completion,
            "reward": result["reward"],
            "passed": result["passed"],
            "total": result["total"],
        })

        if result["reward"] == 1.0:
            print("\nAll tests passed!")
            return {"success": True, "turns": turn, "history": history}

        if turn < max_turns:
            error_feedback = result["errors"][0] if result["errors"] else "Tests failed."
            if len(error_feedback) > 300:
                error_feedback = error_feedback[:300] + "..."

            revision_prompt = (
                f"The code failed with the following error:\n\n"
                f"```\n{error_feedback}\n```\n\n"
                f"Please fix the code and try again. Output only the corrected function."
            )

            # Accumulate multi-turn conversation via chat messages
            messages.append({"role": "assistant", "content": completion})
            messages.append({"role": "user", "content": revision_prompt})
            print(f"Error: {error_feedback[:200]}")

    print(f"\nMax turns ({max_turns}) reached without passing all tests.")
    return {"success": False, "turns": max_turns, "history": history}


def main():
    parser = argparse.ArgumentParser(description="nanoCodeRL agent demo")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--task-id", type=str, default=None, help="Specific HumanEval task ID")
    parser.add_argument("--max-turns", type=int, default=5, help="Max revision turns")
    parser.add_argument("--num-problems", type=int, default=5, help="Number of problems to demo")
    args = parser.parse_args()

    cfg = Config()
    model, tokenizer = load_agent_model(cfg, args.ckpt)

    problems = load_humaneval()

    if args.task_id:
        problems = [p for p in problems if p["task_id"] == args.task_id]
        if not problems:
            print(f"Task {args.task_id} not found. Available: HumanEval/0 .. HumanEval/163")
            return
    else:
        import random
        random.seed(42)
        problems = random.sample(problems, min(args.num_problems, len(problems)))

    print(f"=== nanoCodeRL Agent Demo ===")
    print(f"Model: {args.ckpt or cfg.model_name}")
    print(f"Problems: {len(problems)}")
    print(f"Max turns: {args.max_turns}")

    successes = 0
    total_turns = 0

    for problem in problems:
        result = agent_loop(model, tokenizer, problem, cfg, args.max_turns)
        if result["success"]:
            successes += 1
        total_turns += result["turns"]

    print(f"\n{'='*60}")
    print(f"=== Agent Demo Summary ===")
    print(f"Solved: {successes}/{len(problems)} ({successes/len(problems):.0%})")
    print(f"Avg turns: {total_turns/len(problems):.1f}")


if __name__ == "__main__":
    main()
