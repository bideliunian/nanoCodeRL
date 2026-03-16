"""CPU smoke test — validate the full pipeline without GPU.

Runs a minimal end-to-end check: data loading, sandbox execution,
and (optionally) a tiny model forward pass. Use this to verify
everything works before deploying to a GPU cluster.

Usage:
    python -m scripts.smoke_test
    python -m scripts.smoke_test --with-model   # also test model loading (slow, ~10GB RAM)
"""

import argparse
import sys
import time


def test_imports():
    """Verify all required packages are importable."""
    print("[1/5] Testing imports...")
    import torch
    import transformers
    import trl
    import datasets
    import accelerate
    import peft
    print(f"  torch={torch.__version__}, transformers={transformers.__version__}, trl={trl.__version__}")
    print("  OK\n")


def test_data_loading():
    """Verify datasets load correctly."""
    print("[2/5] Testing data loading...")
    from nanoCodeRL.data import load_eval_data
    problems = load_eval_data(["humaneval"])
    assert len(problems) > 0, "No problems loaded"

    # Check schema
    p = problems[0]
    for key in ["task_id", "prompt", "test_cases", "source"]:
        assert key in p, f"Missing key: {key}"
    print(f"  Loaded {len(problems)} problems, schema OK\n")
    return problems


def test_sandbox(problems):
    """Verify sandbox execution works."""
    print("[3/5] Testing sandbox execution...")
    from nanoCodeRL.sandbox import compute_reward

    # Test with a known-correct solution
    code = "def has_close_elements(numbers, threshold):\n    for i, n1 in enumerate(numbers):\n        for j, n2 in enumerate(numbers):\n            if i != j and abs(n1 - n2) < threshold:\n                return True\n    return False\n"

    # Find the HumanEval/0 problem
    he0 = [p for p in problems if p["task_id"] == "HumanEval/0"]
    if he0:
        result = compute_reward(code, he0[0]["test_cases"], timeout=5)
        print(f"  HumanEval/0 correct solution: reward={result['reward']:.1f} (expected 1.0)")
        assert result["reward"] == 1.0, f"Expected reward 1.0, got {result['reward']}"

    # Test with a broken solution
    result = compute_reward("def foo(): raise Exception('broken')", ["foo()"], timeout=5)
    print(f"  Broken solution: reward={result['reward']:.1f} (expected 0.0)")
    assert result["reward"] == 0.0

    # Test timeout
    code_hang = "import time; time.sleep(100)"
    t0 = time.time()
    result = compute_reward(code_hang, [code_hang], timeout=2)
    elapsed = time.time() - t0
    print(f"  Timeout test: {elapsed:.1f}s (expected ~2s), reward={result['reward']:.1f}")
    assert elapsed < 5, "Timeout took too long"
    print("  OK\n")


def test_config():
    """Verify config is valid."""
    print("[4/5] Testing config...")
    from nanoCodeRL.config import Config
    cfg = Config()
    assert cfg.model_name == "Qwen/Qwen3.5-4B-Base"
    assert cfg.num_rollouts > 0
    assert cfg.batch_size > 0
    assert 0 < cfg.clip_eps < cfg.clip_eps_high
    print(f"  model={cfg.model_name}, steps={cfg.num_train_steps}, rollouts={cfg.num_rollouts}")
    print("  OK\n")


def test_model_loading():
    """Test model loading on CPU (slow, requires ~10GB RAM)."""
    print("[5/5] Testing model loading (CPU, this is slow)...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from nanoCodeRL.config import Config

    cfg = Config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Tiny forward pass
    inputs = tokenizer("def hello():", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"  Forward pass OK, logits shape: {outputs.logits.shape}")
    print("  OK\n")


def main():
    parser = argparse.ArgumentParser(description="nanoCodeRL CPU smoke test")
    parser.add_argument("--with-model", action="store_true", help="Also test model loading (slow)")
    args = parser.parse_args()

    print("=" * 50)
    print("  nanoCodeRL Smoke Test")
    print("=" * 50)
    print()

    try:
        test_imports()
        problems = test_data_loading()
        test_sandbox(problems)
        test_config()

        if args.with_model:
            test_model_loading()
        else:
            print("[5/5] Skipping model loading (use --with-model to enable)\n")

        print("=" * 50)
        print("  All tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
