"""Reward scoring for LiveCodeBench (code_generation_lite).

The ground_truth stored in the parquet is encoded as:
    base64 -> zlib -> pickle -> JSON string
The decoded JSON has the shape:
    {"inputs": [str, ...], "outputs": [str, ...], "fn_name": str | None}

All LCB problems in this dataset are stdin/stdout style (fn_name=None).
We extract the python code from the model response, execute it in a sandboxed
subprocess for each test case, and compare stdout against expected output.
"""

import base64
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import traceback
import zlib

# Global timeout for the entire compute_score call (seconds).
# Prevents indefinite hangs even if individual subprocess timeouts fail.
# With max_tests=5 and per-test timeout=5s, worst case ~25s per sample.
GLOBAL_TIMEOUT = 30


def _decode_test_cases(ground_truth: str) -> dict:
    """Decode the compressed ground-truth blob into an in_outs dict."""
    raw = base64.b64decode(ground_truth)
    decompressed = zlib.decompress(raw, 15)
    s = pickle.loads(decompressed)
    return json.loads(s)


def _extract_code(solution_str: str) -> str:
    """Extract python code from markdown code block, or return as-is."""
    # Try ```python ... ``` first
    parts = solution_str.split("```python")
    if len(parts) > 1:
        code = parts[-1].split("```")[0]
        return code.strip()
    # Try generic ``` ... ```
    parts = solution_str.split("```")
    if len(parts) >= 3:
        return parts[-2].strip()
    return solution_str.strip()


def _run_single_test(code: str, test_input: str, expected_output: str, timeout: int = 10) -> bool:
    """Run code in a subprocess with the given stdin and check stdout."""
    try:
        # Use start_new_session to create a new process group so we can kill the
        # entire tree (prevents zombie child processes from blocking).
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(input=test_input, timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the entire process group
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
            return False

        actual = stdout.strip()
        expected = expected_output.strip()
        return actual == expected
    except Exception:
        # Ensure process is dead if anything goes wrong
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        return False


def compute_score(solution_str: str, ground_truth: str, max_tests: int = 5, timeout: int = 5) -> dict:
    """Compute reward for a LiveCodeBench code-generation problem.

    Args:
        solution_str: The model's full response (should contain ```python ... ```).
        ground_truth: The encoded test-case blob from the parquet.
        max_tests: Maximum number of test cases to run (for efficiency).
        timeout: Per-test timeout in seconds.

    Returns:
        Dict with score, acc, and pred.
    """
    try:
        test_cases = _decode_test_cases(ground_truth)
    except Exception:
        traceback.print_exc(5)
        return {"score": -1.0, "acc": False, "pred": "[DECODE_ERROR]"}

    code = _extract_code(solution_str)
    if not code:
        return {"score": -1.0, "acc": False, "pred": "[NO_CODE]"}

    inputs = test_cases["inputs"]
    outputs = test_cases["outputs"]
    total = min(len(inputs), max_tests)

    if total == 0:
        return {"score": -1.0, "acc": False, "pred": "[NO_TESTS]"}

    passed = 0
    start_time = time.time()

    for i in range(total):
        # Global timeout check: bail out if we've exceeded the budget
        elapsed = time.time() - start_time
        if elapsed > GLOBAL_TIMEOUT:
            # Treat remaining tests as failed
            return {
                "score": 2.0 * (passed / total) - 1.0,
                "acc": passed / total,
                "pred": f"pass_rate={passed}/{total}[GLOBAL_TIMEOUT@{i}]",
            }

        if _run_single_test(code, inputs[i], outputs[i], timeout=timeout):
            passed += 1

    acc = passed / total
    # Map [0, 1] -> [-1, 1]
    score = 2.0 * acc - 1.0

    return {
        "score": score,
        "acc": acc,
        "pred": f"pass_rate={passed}/{total}",
    }
