"""Verifier feedback for bounded repair (cell D).

Deterministic text built from observable signals only: execution outcome,
error message, and gold-free features (design §6, objective §12 — the repair
model never sees gold SQL).
"""
from __future__ import annotations


def build_feedback(features: dict, error_message: str | None = None) -> str:
    """Critique of one candidate. Failed execution is reported with the raw
    SQLite message; otherwise the observable weaknesses are listed."""
    if not features.get("executed"):
        msg = (error_message or "").strip()
        msg = msg[:300] if msg else features.get("error", "unknown error")
        return f"The query does not run: {msg}"

    parts: list[str] = []
    if features.get("pool_disagrees"):
        n = int(features.get("pool_majority_n", 0))
        k = int(features.get("pool_k", 0))
        parts.append(f"Most other sampled attempts ({n} of {k}) produced a "
                     "different result set; re-check how the question maps to "
                     "joins and filters.")
    if features.get("empty"):
        parts.append("The query returns no rows; check the filters and joins "
                     "against the question.")
    if features.get("consistency", 1.0) < 0.5:
        parts.append("The result disagrees with the majority of equivalent "
                     "query attempts, suggesting a wrong interpretation.")
    if features.get("expects_count") and not features.get("scalar"):
        parts.append("The question asks for a count, but the query does not "
                     "return a single number.")
    if not parts:
        parts.append("The query runs, but the answer could not be verified "
                     "against the question's requirements.")
    return " ".join(parts)