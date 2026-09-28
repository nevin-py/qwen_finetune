"""Verifier features — observable signals only, never gold SQL (design §6, §12).

Every feature is computable at inference time from the question, the
candidate SQL, and its sandbox execution result. ``consistency`` (self-
agreement over k sampled candidates) is supplied by the caller because it
requires multiple generations, not because it is privileged information.
"""
from __future__ import annotations

import re

_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", re.IGNORECASE)
_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_COUNT_HINT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", re.IGNORECASE)
_VALUE_HINT = re.compile(r"'([^']{2,})'")


def n_tables_referenced(sql: str) -> int:
    return len({m for m in _TABLE.findall(sql)})


def expects_count(question: str) -> bool:
    return bool(_COUNT_HINT.search(question))


def scalar_value_in_question(question: str, columns: tuple, rows: tuple) -> bool:
    """Does the (single-cell) answer copy a value that appears in the question?"""
    if len(rows) != 1 or len(rows[0]) != 1:
        return False
    value = str(rows[0][0]).strip()
    if len(value) < 2:
        return False
    return value.lower() in question.lower()


def extract_features(
    question: str,
    sql: str,
    *,
    executed: bool,
    error: str | None,
    columns: tuple = (),
    rows: tuple = (),
    truncated: bool = False,
    duration_ms: float = 0.0,
    consistency: float = 1.0,
) -> dict:
    """Numeric feature vector for one candidate execution."""
    if not executed:
        return {
            "executed": 0.0,
            f"error_{error or 'unknown'}": 1.0,
            "consistency": consistency,
            "n_rows": 0.0,
            "scalar": 0.0,
            "answer_in_question": 0.0,
            "expects_count": 1.0 if expects_count(question) else 0.0,
            "count_shape_match": 0.0,
            "n_tables": float(n_tables_referenced(sql)),
            "has_agg": 1.0 if _AGG.search(sql) else 0.0,
            "empty": 1.0,
            "truncated": 1.0 if truncated else 0.0,
            "slow": 1.0 if duration_ms > 1000 else 0.0,
        }

    scalar = len(rows) == 1 and len(columns) == 1
    empty = len(rows) == 0
    exp_count = expects_count(question)
    return {
        "executed": 1.0,
        "consistency": consistency,
        "n_rows": float(len(rows)),
        "scalar": 1.0 if scalar else 0.0,
        "answer_in_question": 1.0 if scalar_value_in_question(question, columns, rows) else 0.0,
        "expects_count": 1.0 if exp_count else 0.0,
        "count_shape_match": 1.0 if (exp_count == scalar or not exp_count) else 0.0,
        "n_tables": float(n_tables_referenced(sql)),
        "has_agg": 1.0 if _AGG.search(sql) else 0.0,
        "empty": 1.0 if empty else 0.0,
        "truncated": 1.0 if truncated else 0.0,
        "slow": 1.0 if duration_ms > 1000 else 0.0,
    }