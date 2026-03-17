"""Sandboxed code execution for reward computation.

Executes model-generated code against unit tests in a subprocess
with a timeout. Supports both assertion-based tests (HumanEval/MBPP)
and stdin/stdout tests (CodeContests). Optionally runs inside Docker.
"""

import re
import subprocess
import tempfile
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(completion: str) -> str:
    """Strip markdown fences and thinking tags from a model completion.

    Models almost always wrap output in ```python...``` even when instructed
    not to, which causes SyntaxError when executed directly.
    """
    # Remove <think>...</think> blocks (Qwen3 thinking mode leaking into output)
    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()

    # Extract first python code block
    match = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences — return as-is
    return completion.strip()


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

_DOCKER_IMAGE = "nanocodrl-sandbox"
_docker_available = None


def _check_docker() -> bool:
    """Check if Docker is available and the sandbox image exists."""
    global _docker_available
    if _docker_available is not None:
        return _docker_available
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", _DOCKER_IMAGE],
            capture_output=True, timeout=5,
        )
        _docker_available = r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        _docker_available = False
    return _docker_available


def _run_in_docker(tmp_path: str, timeout: int, stdin_input: str | None = None):
    """Run a script inside the Docker sandbox container."""
    cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "--memory=256m",
        "--cpus=1",
        "-v", f"{tmp_path}:/code/run.py:ro",
        _DOCKER_IMAGE,
        "python", "/code/run.py",
    ]
    return subprocess.run(
        cmd,
        input=stdin_input,
        capture_output=True,
        text=True,
        timeout=timeout + 5,  # extra margin for container startup
    )


def _run_subprocess(tmp_path: str, timeout: int, stdin_input: str | None = None):
    """Run a script in a local subprocess."""
    return subprocess.run(
        ["python", tmp_path],
        input=stdin_input,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


# ---------------------------------------------------------------------------
# Core execution
# ---------------------------------------------------------------------------

def execute_code(code: str, test_code: str, timeout: int = 5,
                 use_docker: bool = False) -> dict:
    """Run generated code + assertion-based test code."""
    full_script = code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        runner = _run_in_docker if (use_docker and _check_docker()) else _run_subprocess
        result = runner(tmp_path, timeout)

        if result.returncode == 0:
            return {"passed": 1, "total": 1, "reward": 1.0, "error": None}
        else:
            error_msg = result.stderr.strip()[-500:] if result.stderr else "Unknown error"
            return {"passed": 0, "total": 1, "reward": 0.0, "error": error_msg}

    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": 1, "reward": 0.0, "error": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"passed": 0, "total": 1, "reward": 0.0, "error": str(e)}
    finally:
        os.unlink(tmp_path)


def execute_code_io(code: str, stdin_input: str, expected_output: str,
                    timeout: int = 5, use_docker: bool = False) -> dict:
    """Run generated code with stdin input, compare stdout to expected output."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        runner = _run_in_docker if (use_docker and _check_docker()) else _run_subprocess
        result = runner(tmp_path, timeout, stdin_input=stdin_input)

        if result.returncode != 0:
            error_msg = result.stderr.strip()[-500:] if result.stderr else "Runtime error"
            return {"passed": 0, "total": 1, "reward": 0.0, "error": error_msg}

        actual = result.stdout.strip()
        expected = expected_output.strip()
        if actual == expected:
            return {"passed": 1, "total": 1, "reward": 1.0, "error": None}
        else:
            return {
                "passed": 0, "total": 1, "reward": 0.0,
                "error": f"Expected: {expected[:200]}\nGot: {actual[:200]}",
            }

    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": 1, "reward": 0.0, "error": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"passed": 0, "total": 1, "reward": 0.0, "error": str(e)}
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    code: str,
    test_cases: list[str] | list[dict],
    timeout: int = 5,
    use_docker: bool = False,
) -> dict:
    """Compute fractional reward over multiple test cases.

    test_cases can be:
      - list[str]: assertion-based test code (HumanEval/MBPP)
      - list[dict]: stdin/stdout pairs {"input": ..., "output": ...} (CodeContests)
    """
    if not test_cases:
        return {"reward": 0.0, "passed": 0, "total": 0, "errors": ["No test cases"]}

    passed = 0
    errors = []

    for test in test_cases:
        if isinstance(test, dict):
            result = execute_code_io(
                code, test["input"], test["output"],
                timeout=timeout, use_docker=use_docker,
            )
        else:
            result = execute_code(code, test, timeout=timeout, use_docker=use_docker)

        if result["reward"] > 0:
            passed += 1
        if result["error"]:
            errors.append(result["error"])

    total = len(test_cases)
    reward = passed / total if total > 0 else 0.0

    return {
        "reward": reward,
        "passed": passed,
        "total": total,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Parallel batch reward computation
# ---------------------------------------------------------------------------

def _compute_single(args: tuple) -> float:
    """Worker function for parallel reward computation."""
    code, test_cases, timeout, use_docker = args
    result = compute_reward(code, test_cases, timeout=timeout, use_docker=use_docker)
    return result["reward"]


def compute_rewards_parallel(
    tasks: list[tuple[str, list]],
    timeout: int = 5,
    max_workers: int = 8,
    use_docker: bool = False,
) -> list[float]:
    """Compute rewards for a batch of (code, test_cases) pairs in parallel."""
    work = [(code, tests, timeout, use_docker) for code, tests in tasks]

    if max_workers <= 1:
        return [_compute_single(w) for w in work]

    rewards = [0.0] * len(work)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_compute_single, w): i for i, w in enumerate(work)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                rewards[idx] = future.result()
            except Exception:
                rewards[idx] = 0.0

    return rewards
