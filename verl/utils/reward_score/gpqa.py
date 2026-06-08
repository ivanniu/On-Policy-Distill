"""Reward scoring for GPQA (Graduate-Level Google-Proof Q&A) multiple-choice questions.

The ground truth is a single letter (A/B/C/D).
The model is expected to output "Answer: X" where X is one of A, B, C, D.
"""

import re


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """Compute reward for GPQA multiple-choice answers.

    Args:
        solution_str: The model's full response string.
        ground_truth: The correct answer letter (e.g. "A", "B", "C", "D").

    Returns:
        Dict with score, acc, and pred.
    """
    # Normalize ground truth
    gt = ground_truth.strip().upper()

    # Try to extract "Answer: X" pattern (case-insensitive), take the last match
    matches = re.findall(r"(?i)(?:the\s+)?answer\s*(?:is)?\s*[:：]\s*\(?([A-D])\)?", solution_str)
    if not matches:
        # Fallback: look for a standalone letter at the very end
        matches = re.findall(r"\b([A-D])\b\s*$", solution_str.strip())

    pred = matches[-1].upper() if matches else "[NO_ANSWER]"
    correct = pred == gt

    return {
        "score": 1.0 if correct else -1.0,
        "acc": correct,
        "pred": pred,
    }
