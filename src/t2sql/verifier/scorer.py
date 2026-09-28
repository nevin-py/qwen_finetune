"""Rule-based verifier scoring (design §6).

Rules first: a learned scorer is only introduced if it measurably beats this
on validation. Scores rank candidates for the same question — absolute
values are meaningless, ordering is the contract.
"""
from __future__ import annotations

DEFAULT_WEIGHTS = {
    "executed": 1.0,
    "consistency": 2.0,
    "answer_in_question": 0.4,
    "count_shape_match": 0.3,
    "empty": -0.4,
    "truncated": -0.3,
    "slow": -0.2,
    # schema_link (checklist 4.5): 0.0 until tuned — tuned weights are passed
    # in as an explicit dict (scripts/tune_verifier.py writes the JSON).
    "schema_link": 0.0,
}

_COMPLEXITY_PENALTY = 0.1   # per table beyond 4
_COMPLEXITY_FREE = 4.0


def score(features: dict, weights: dict | None = None) -> float:
    """Higher = more trustworthy. Failed execution is never trustworthy."""
    if not features.get("executed"):
        return 0.0
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(features.get(k, 0.0) * v for k, v in w.items())
    excess_tables = max(0.0, features.get("n_tables", 0.0) - _COMPLEXITY_FREE)
    return total - _COMPLEXITY_PENALTY * excess_tables