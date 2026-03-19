"""Test that the reward function correctly looks up problems and scores code.

Verifies the fix for the zero-reward bug where TRL's tokenize-then-decode
pipeline altered prompt strings, causing cache lookups to fail.

Usage:
    python -m scripts.test_reward
"""

import sys


def test_reward_lookup():
    """Test that reward_fn receives problem_idx and returns correct rewards."""
    from nanoCodeRL.config import Config
    from nanoCodeRL.data import load_humaneval
    from scripts.train import build_reward_fn

    cfg = Config()
    problems = load_humaneval()[:5]  # just a few

    reward_fn = build_reward_fn(cfg, problems)

    # HumanEval/0: has_close_elements — use a known-correct solution.
    # Real model completions are markdown-fenced complete functions. The reward
    # function calls extract_code() to strip fences, then does prompt + clean.
    # The second def overrides the prompt's stub, so the test calls the model's version.
    correct_completion = (
        "```python\n"
        "def has_close_elements(numbers, threshold):\n"
        "    for i, n1 in enumerate(numbers):\n"
        "        for j, n2 in enumerate(numbers):\n"
        "            if i != j and abs(n1 - n2) < threshold:\n"
        "                return True\n"
        "    return False\n"
        "```"
    )

    # Test 1: Correct solution should get reward > 0
    rewards = reward_fn(
        completions=[correct_completion],
        problem_idx=[0],
    )
    print(f"  Test 1 — correct solution: reward={rewards[0]:.2f} (expected 1.0)")
    assert rewards[0] == 1.0, f"Expected reward 1.0, got {rewards[0]}"

    # Test 2: Broken solution should get reward 0
    broken_completion = "```python\ndef has_close_elements(numbers, threshold):\n    return 'wrong'\n```"
    rewards = reward_fn(
        completions=[broken_completion],
        problem_idx=[0],
    )
    print(f"  Test 2 — broken solution:  reward={rewards[0]:.2f} (expected 0.0)")
    assert rewards[0] == 0.0, f"Expected reward 0.0, got {rewards[0]}"

    # Test 3: Multiple completions with different indices
    rewards = reward_fn(
        completions=[correct_completion, broken_completion],
        problem_idx=[0, 0],
    )
    print(f"  Test 3 — batch rewards:    {rewards} (expected [1.0, 0.0])")
    assert rewards[0] == 1.0 and rewards[1] == 0.0

    print("  All reward lookup tests passed!")


def test_markdown_extraction():
    """Test that markdown-wrapped completions get correct rewards (not always 0)."""
    from nanoCodeRL.config import Config
    from nanoCodeRL.data import load_humaneval
    from scripts.train import build_reward_fn

    cfg = Config()
    problems = load_humaneval()[:1]
    reward_fn = build_reward_fn(cfg, problems)

    # Same correct solution but wrapped in markdown fences (as models actually output)
    markdown_completion = (
        "```python\n"
        "def has_close_elements(numbers, threshold):\n"
        "    for i, n1 in enumerate(numbers):\n"
        "        for j, n2 in enumerate(numbers):\n"
        "            if i != j and abs(n1 - n2) < threshold:\n"
        "                return True\n"
        "    return False\n"
        "```"
    )

    rewards = reward_fn(completions=[markdown_completion], problem_idx=[0])
    print(f"  Test — markdown-wrapped correct solution: reward={rewards[0]:.2f} (expected 1.0)")
    assert rewards[0] == 1.0, (
        f"Expected reward 1.0 for markdown-wrapped solution, got {rewards[0]}. "
        "This indicates extract_code() is not stripping fences before sandbox execution."
    )
    print("  Markdown extraction test passed!")


def test_dataset_has_problem_idx():
    """Test that prepare_dataset includes problem_idx column."""
    from transformers import AutoTokenizer
    from nanoCodeRL.config import Config
    from nanoCodeRL.data import load_humaneval
    from scripts.train import prepare_dataset

    cfg = Config()
    problems = load_humaneval()[:3]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    dataset = prepare_dataset(problems, tokenizer, cfg)

    print(f"  Dataset has {len(dataset)} entries")
    for i, entry in enumerate(dataset):
        assert "prompt" in entry, f"Entry {i} missing 'prompt'"
        assert "problem_idx" in entry, f"Entry {i} missing 'problem_idx'"
        assert entry["problem_idx"] == i, f"Entry {i}: expected idx={i}, got {entry['problem_idx']}"
        # Verify prompt contains chat template markers
        assert "<|im_start|>" in entry["prompt"] or "<|start|>" in entry["prompt"] or len(entry["prompt"]) > len(problems[i]["prompt"]), \
            f"Entry {i}: prompt doesn't look like a chat template was applied"

    print(f"  All {len(dataset)} entries have prompt + problem_idx, chat template applied!")


def main():
    print("=" * 50)
    print("  Reward Function Tests")
    print("=" * 50)
    print()

    try:
        print("[1/3] Testing reward lookup via problem_idx...")
        test_reward_lookup()
        print()

        print("[2/3] Testing markdown code fence extraction...")
        test_markdown_extraction()
        print()

        print("[3/3] Testing dataset includes problem_idx...")
        test_dataset_has_problem_idx()
        print()

        print("=" * 50)
        print("  All tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
