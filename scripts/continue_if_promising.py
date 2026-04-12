"""Post-training pipeline: eval 200-step checkpoint, continue 500 steps if promising.

Usage (launched automatically after 200-step run):
    python scripts/continue_if_promising.py --train-pid <pid> --ckpt checkpoints/checkpoint-200
"""
import argparse
import json
import os
import subprocess
import sys
import time

# Pass condition: composite pass@1 (mean of humaneval + mbpp) must beat this.
# Baselines: humaneval=42.68%, mbpp=43.97% → composite=43.3%
# Threshold = baseline + 1pp to confirm real improvement.
COMPOSITE_THRESHOLD = 0.443

EVAL_CMD_TEMPLATE = [
    ".venv/bin/python", "-m", "scripts.eval",
    "--bench", "mbpp", "humaneval",
    "--batch-size", "4",
    "--ckpt", "{ckpt}",
    "--output", "{output}",
]

TRAIN_500_CMD = [
    ".venv/bin/python", "-m", "scripts.train",
    "--steps", "500",
    "--train-data", "mbpp_full",
    "--no-vllm",
    "--batch-size", "8",
    "--num-rollouts", "8",
    "--eval-subset-size", "5",
    "--save-steps", "100",
]


def wait_for_pid(pid: int) -> None:
    print(f"Waiting for training process {pid} to complete...", flush=True)
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"Process {pid} finished.", flush=True)
            return
        time.sleep(30)


def run_eval(ckpt: str, output: str) -> dict:
    cmd = [
        c.format(ckpt=ckpt, output=output) if "{" in c else c
        for c in EVAL_CMD_TEMPLATE
    ]
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    print(f"\nRunning eval: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)
    with open(output) as f:
        return json.load(f)


def composite(results: dict) -> float:
    scores = [v["pass_at_1"] for v in results.values()]
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pid", type=int, required=True,
                        help="PID of the running 200-step training process")
    parser.add_argument("--ckpt", type=str, default="checkpoints/checkpoint-200",
                        help="Checkpoint to evaluate after training")
    args = parser.parse_args()

    # 1. Wait for training
    wait_for_pid(args.train_pid)

    # 2. Eval
    output_path = "results/eval_200steps_b8r8.json"
    results = run_eval(args.ckpt, output_path)

    he = results.get("humaneval", {})
    mb = results.get("mbpp", {})
    comp = composite(results)

    print(f"\n=== 200-step eval results ===")
    print(f"  HumanEval: {he.get('pass_at_1', 0):.1%} ({he.get('passed')}/{he.get('total')})")
    print(f"  MBPP:      {mb.get('pass_at_1', 0):.1%} ({mb.get('passed')}/{mb.get('total')})")
    print(f"  Composite: {comp:.1%}  (threshold: {COMPOSITE_THRESHOLD:.1%})")

    # 3. Decide
    if comp >= COMPOSITE_THRESHOLD:
        print(f"\nResults promising ({comp:.1%} >= {COMPOSITE_THRESHOLD:.1%}). "
              f"Launching 500-step training run...\n", flush=True)
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        subprocess.run(TRAIN_500_CMD, check=True, env=env)
        print("\n500-step training complete.", flush=True)
    else:
        print(f"\nResults below threshold ({comp:.1%} < {COMPOSITE_THRESHOLD:.1%}). "
              f"Stopping.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
