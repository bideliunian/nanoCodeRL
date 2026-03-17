"""Dataset loading for training and evaluation.

Training:  CodeContests (stdin/stdout) — completely separate source from eval.
Eval:      HumanEval + MBPP sanitized test — held-out benchmarks, never trained on.
"""

import json

from datasets import load_dataset


SYSTEM_PROMPT = (
    "You are a Python coding assistant. Write a correct Python function "
    "that solves the given problem. Output only the function code, no explanation."
)

SYSTEM_PROMPT_IO = (
    "You are a Python coding assistant. Write a complete Python program "
    "that reads from stdin and writes to stdout. Output only the code, no explanation."
)


def build_messages(user_prompt: str, source: str) -> list[dict]:
    """Build chat messages for the instruct model.

    Returns a list of message dicts suitable for ``tokenizer.apply_chat_template``.
    """
    sys = SYSTEM_PROMPT_IO if source == "code_contests" else SYSTEM_PROMPT
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_prompt},
    ]


def apply_chat_template(tokenizer, messages: list[dict], enable_thinking: bool = False) -> str:
    """Apply the tokenizer's chat template to messages, returning a string prompt.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        messages: List of message dicts (role/content).
        enable_thinking: Whether to enable Qwen3.5's thinking mode (generates
            very long chains-of-thought; disabled by default for RL training).
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Fallback for tokenizers that don't support enable_thinking kwarg
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _extract_mbpp_func_name(problem: dict) -> str | None:
    """Extract the expected function name from MBPP test assertions."""
    import re
    tests = problem.get("test_list", [])
    if not tests:
        return None
    match = re.search(r'assert (\w+)\(', tests[0])
    return match.group(1) if match else None


def format_prompt(problem: dict, source: str) -> str:
    """Format a coding problem into a prompt string."""
    if source == "humaneval":
        return problem["prompt"]
    elif source == "mbpp":
        desc = problem['text'] if 'text' in problem else problem['prompt']
        # Extract expected function name from test cases so the model uses the right name
        func_name = _extract_mbpp_func_name(problem)
        if func_name:
            return f"# {desc}\n# Function signature: def {func_name}(...)\n\n"
        return f"# {desc}\n\n"
    elif source == "code_contests":
        return problem["description"]
    else:
        raise ValueError(f"Unknown source: {source}")


def get_test_cases(problem: dict, source: str) -> list[str] | list[dict]:
    """Extract test cases from a problem.

    Returns:
      - list[str] for assertion-based tests (HumanEval/MBPP)
      - list[dict] with {"input": ..., "output": ...} for stdin/stdout tests
    """
    if source == "humaneval":
        # HumanEval test field defines check(candidate) but doesn't call it
        test_code = problem["test"] + f"\ncheck({problem['entry_point']})\n"
        return [test_code]
    elif source == "mbpp":
        return problem.get("test_list", [])
    elif source == "code_contests":
        inputs = problem["public_tests"]["input"] + problem["private_tests"]["input"]
        outputs = problem["public_tests"]["output"] + problem["private_tests"]["output"]
        # Also include generated tests (up to 20 to keep execution fast)
        gen_inputs = problem["generated_tests"]["input"][:20]
        gen_outputs = problem["generated_tests"]["output"][:20]
        inputs += gen_inputs
        outputs += gen_outputs
        return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
    else:
        raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# Eval-only datasets
# ---------------------------------------------------------------------------

def load_humaneval() -> list[dict]:
    """Load HumanEval dataset (eval-only, single 'test' split)."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for item in ds:
        problems.append({
            "task_id": item["task_id"],
            "prompt": format_prompt(item, "humaneval"),
            "test_cases": get_test_cases(item, "humaneval"),
            "entry_point": item["entry_point"],
            "source": "humaneval",
        })
    return problems


def load_mbpp(split: str = "train") -> list[dict]:
    """Load MBPP sanitized dataset for the given split (train/validation/test)."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
    problems = []
    for item in ds:
        problems.append({
            "task_id": f"mbpp_{item['task_id']}",
            "prompt": format_prompt(item, "mbpp"),
            "test_cases": get_test_cases(item, "mbpp"),
            "source": "mbpp",
        })
    return problems


# ---------------------------------------------------------------------------
# Training datasets
# ---------------------------------------------------------------------------

def load_code_contests(split: str = "train", max_problems: int | None = None,
                       min_tests: int = 2, max_difficulty: int | None = 3,
                       max_solution_chars: int | None = 800) -> list[dict]:
    """Load CodeContests dataset for training (stdin/stdout format).

    Args:
        split: Dataset split to load.
        max_problems: Cap the number of problems (None = all).
        min_tests: Skip problems with fewer than this many test cases.
        max_difficulty: Skip problems harder than this difficulty level.
            The difficulty field is an internal enum. Distribution:
            0=unknown(33%), 1-6=easy(7%), 7-11=medium/hard(52%), 12+=very hard.
            Default 7 keeps easy problems only; set None to disable filtering.
        max_solution_chars: Skip problems where the shortest accepted solution
            exceeds this character count (proxy for required completion length).
            Default 800 chars ≈ ~300 tokens. Set None to disable.
    """
    try:
        ds = load_dataset("deepmind/code_contests", split=split)
    except (FileNotFoundError, Exception) as e:
        raise RuntimeError(
            f"CodeContests dataset not available: {e}\n"
            f"Download it first: python -m scripts.prefetch\n"
            f"Or use --train-benchmarks mbpp_full as a lighter alternative."
        ) from e

    problems = []
    for item in ds:
        # Filter by difficulty (skip hard competitive programming problems)
        if max_difficulty is not None:
            diff = item.get("difficulty", 0)
            if diff > max_difficulty:
                continue

        # Filter by solution length (proxy for required completion tokens)
        if max_solution_chars is not None:
            sols = item.get("solutions", {})
            sol_codes = sols.get("solution", []) if isinstance(sols, dict) else []
            if sol_codes:
                shortest = min(len(c) for c in sol_codes)
                if shortest > max_solution_chars:
                    continue

        # Only keep problems with sufficient tests
        test_cases = get_test_cases(item, "code_contests")
        if len(test_cases) < min_tests:
            continue

        problems.append({
            "task_id": f"cc_{item['name']}",
            "prompt": format_prompt(item, "code_contests"),
            "test_cases": test_cases,
            "source": "code_contests",
        })

        if max_problems and len(problems) >= max_problems:
            break

    return problems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_training_data(benchmarks: list[str] | None = None, **kwargs) -> list[dict]:
    """Load training data from dedicated training sources.

    Supported sources:
      - "code_contests": CodeContests train split (stdin/stdout, ~13K problems)
      - "mbpp_full": MBPP full train split (assertions, ~374 problems)

    HumanEval and MBPP sanitized are eval-only and cannot be used for training.
    """
    if benchmarks is None:
        benchmarks = ["code_contests"]

    problems = []
    for bench in benchmarks:
        if bench == "code_contests":
            problems.extend(load_code_contests(
                split="train",
                max_difficulty=kwargs.get("max_difficulty", 3),
                max_solution_chars=kwargs.get("max_solution_chars", 800),
            ))
        elif bench == "mbpp_full":
            ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
            for item in ds:
                problems.append({
                    "task_id": f"mbpp_full_{item['task_id']}",
                    "prompt": format_prompt(item, "mbpp"),
                    "test_cases": item.get("test_list", []),
                    "source": "mbpp",
                })
        else:
            raise ValueError(
                f"Unknown training source: {bench}. "
                f"Available: code_contests, mbpp_full. "
                f"(HumanEval and MBPP sanitized are eval-only.)"
            )

    print(f"Loaded {len(problems)} training problems from {benchmarks}")
    return problems


def load_eval_data(benchmarks: list[str] | None = None) -> list[dict]:
    """Load eval data. HumanEval + MBPP sanitized test (no overlap with training)."""
    if benchmarks is None:
        benchmarks = ["humaneval", "mbpp"]

    problems = []
    for bench in benchmarks:
        if bench == "humaneval":
            problems.extend(load_humaneval())
        elif bench == "mbpp":
            problems.extend(load_mbpp(split="test"))
        else:
            raise ValueError(f"Unknown eval benchmark: {bench}")

    print(f"Loaded {len(problems)} eval problems from {benchmarks}")
    return problems
