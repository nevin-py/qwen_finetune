"""Verifier features — observable signals only, never gold SQL (design §6, §12).

Every feature is computable at inference time from the question, the
candidate SQL, and its sandbox execution result. ``consistency`` (self-
agreement over k sampled candidates) is supplied by the caller because it
requires multiple generations, not because it is privileged information.
"""
from __future__ import annotations

import hashlib
import re

_AGG = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", re.IGNORECASE)
_TABLE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
_COUNT_HINT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b|\btotal\b", re.IGNORECASE)
_VALUE_HINT = re.compile(r"'([^']{2,})'")

 
def pool_signature_counts(signatures) -> dict[str, int]:
    """Tally executed-result signatures across a candidate pool.

    Only successful executions contribute a signature, so failed candidates
    raise the denominator without supporting any answer.
    """
    counts: dict[str, int] = {}
    for sig in signatures:
        if sig:
            counts[sig] = counts.get(sig, 0) + 1
    return counts

 
def pool_consistency(sig: str | None, sig_counts: dict[str, int], pool_size: int) -> float:
    """Self-consistency of one candidate against the sampled pool.

    One definition shared by every cell that scores a candidate against a pool
    (verifier selection and repair), because the feature is compared across
    cells: a repair acceptance gate that subtracts a margin between a
    pool-derived score and a freshly scored candidate is only meaningful if
    both use the same denominator. ``pool_size`` is the number of candidates
    sampled for the question, identical for every candidate in the pool,
    including one produced later by repair.
    """
    if not sig or pool_size <= 0:
        return 0.0
    return sig_counts.get(sig, 0) / pool_size


def result_signature(columns, rows) -> str:
    """Order-insensitive signature of an executed result (gold-free)."""
    blob = repr((tuple(map(str, columns)), sorted(map(repr, rows))))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def n_tables_referenced(sql: str) -> int:
    return len({m for m in _TABLE.findall(sql)})


def tables_referenced(sql: str) -> list[str]:
    """Distinct table names after FROM/JOIN, lower-cased."""
    return sorted({m.lower() for m in _TABLE.findall(sql)})


def schema_link(question: str, sql: str) -> float:
    """Fraction of referenced tables named in the question (schema-linking
    signal, checklist 4.5). 0 when the query reads nothing."""
    tables = tables_referenced(sql)
    if not tables:
        return 0.0
    q = question.lower()
    hits = sum(1 for t in tables if t in q or t.replace("_", " ") in q)
    return hits / len(tables)


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
            "schema_link": float(schema_link(question, sql)),
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
        "schema_link": float(schema_link(question, sql)),
        "has_agg": 1.0 if _AGG.search(sql) else 0.0,
        "empty": 1.0 if empty else 0.0,
        "truncated": 1.0 if truncated else 0.0,
        "slow": 1.0 if duration_ms > 1000 else 0.0,
    }