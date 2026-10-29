"""Rule-based verifier scoring (design §6).

Rules first: a learned scorer is only introduced if it measurably beats this
on validation. Scores rank candidates for the same question — ordering is the
contract.

Two invariants make the ranking safe for *any* weight vector, which the
original floor-at-zero rule did not:

1. A candidate whose SQL failed to execute always ranks below every candidate
   that executed. Achieved by giving failures a fixed sentinel (:data:`FAILED_SCORE`)
   far outside the reachable executed range instead of clamping them to ``0.0``.
   With penalty weights (e.g. ``schema_link = -1``) an executed candidate can
   legitimately score negative, so a zero clamp lets crashing SQL win argmax.
2. Any absolute threshold is checked against :func:`max_reachable_score` for
   the weights actually in use. A threshold above the ceiling flags the whole
   dataset and silently degrades "bounded repair" to "repair everything", so
   callers that need a flag threshold calibrate one (see ``tune_verifier``)
   instead of hardcoding a value from a different weight scale.
"""
from __future__ import annotations

# Sentinel score for a candidate that failed to execute. Chosen far below any
# reachable executed score so failures can never outrank a working query.
FAILED_SCORE = -1e6

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

# Every feature is bounded in [0, 1] by construction (see features.py), so the
# executed score lies in [sum of negative weights, sum of positive weights]
# minus at most _COMPLEXITY_PENALTY * (n_tables - _COMPLEXITY_FREE).
FEATURE_MIN, FEATURE_MAX = 0.0, 1.0
_COMPLEXITY_PENALTY = 0.1   # per table beyond 4
_COMPLEXITY_FREE = 4.0
# No realistic schema references more than this many tables; it only bounds the
# complexity penalty so max_reachable_score stays finite.
_MAX_TABLES = 64


def resolve_weights(weights: dict | None = None) -> dict:
    """Effective weights: tuned values layered over the rule defaults."""
    return {**DEFAULT_WEIGHTS, **(weights or {})}


def _executed_score(features: dict, w: dict) -> float:
    total = sum(features.get(k, FEATURE_MIN) * v for k, v in w.items())
    excess = max(0.0, features.get("n_tables", 0.0) - _COMPLEXITY_FREE)
    return total - _COMPLEXITY_PENALTY * excess


def score(features: dict, weights: dict | None = None) -> float:
    """Higher = more trustworthy. Failed execution is never trustworthy.

    Ranking is the contract, so the only thing that matters is that a failed
    candidate sorts below every executed one for the same weight vector.
    """
    if not features.get("executed"):
        return FAILED_SCORE
    return _executed_score(features, resolve_weights(weights))


def max_reachable_score(weights: dict | None = None) -> float:
    """Largest score any executing candidate can reach under ``weights``.

    Achieved when every positively-weighted feature is 1.0 and every
    negatively-weighted feature is 0.0 (a real candidate needs fewer than
    :data:`_MAX_TABLES` tables to avoid the complexity penalty). Callers use
    this to reject a threshold that could never fire.
    """
    w = resolve_weights(weights)
    return sum(v for v in w.values() if v > 0)