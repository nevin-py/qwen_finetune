"""Failure-mode taxonomy for incorrect predictions (design §13, objective §13).

Assigns ONE best-fitting category to each wrong prediction:
    execution failures keep their sandbox taxonomy class (syntax, no_such_table,
    no_such_column, timeout, ...); predictions that execute but return the wrong
    result are compared component-wise against gold and mapped to a single
    category — join_condition, filter, aggregation, group_by, having, ordering,
    limit, subquery, date_function, null_handling, function, multiple_clauses,
    semantic_mismatch (valid but different).

Gold lives here — eval-only module (objective §12). Component extraction is a
documented approximation: keyword-anchored spans, not a full SQL parse.
"""
from __future__ import annotations

import re

_TAILS_WHERE = [r"group\s+by", r"having", r"order\s+by", r"limit"]
_TAILS_GROUP = [r"having", r"order\s+by", r"limit"]
_TAILS_HAVING = [r"order\s+by", r"limit"]
_TAILS_ORDER = [r"limit"]

_DATE_FNS = {"year", "month", "day", "date", "time", "datetime", "julianday",
             "strftime", "unixepoch"}
_AGG_FNS = {"count", "sum", "avg", "min", "max", "total", "group_concat"}
_NULL_OPS = re.compile(r"\bis\s+not\s+null\b|\bis\s+null\b|\bcoalesce\s*\(|"
                       r"\bifnull\s*\(|\bnullif\s*\(")
_NULL_FNS = {"coalesce", "ifnull", "nullif"}
_SQL_KEYWORDS = {"select", "from", "where", "group", "order", "having", "limit",
                 "join", "on", "in", "exists", "when", "values", "case", "and",
                 "or", "not", "between", "like", "all", "distinct", "union",
                 "except", "intersect", "using", "as"}
_FUNC = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(")


def _whiten_null(text: str) -> str:
    """Null-operator spellings don't change the structural span they sit in
    (IS NULL vs IS NOT NULL stays a null_handling difference, not a filter one)."""
    text = re.sub(r"\b(?:coalesce|ifnull|nullif)\s*\(", " (", text)
    return re.sub(r"\bis\s+not\s+null\b|\bis\s+null\b", " ", text)


def _norm(sql: str) -> str:
    return " ".join(sql.lower().split()).rstrip(";").strip()


def _clause(lc: str, head: str, tails: list[str]) -> str | None:
    if not tails:  # no following clause: capture to end of statement
        m = re.search(rf"\b{head}\b(.*)", lc, re.S)
    else:
        m = re.search(rf"\b{head}\b(.*?)(?:\b(?:{'|'.join(tails)})\b|$)", lc, re.S)
    return m.group(1).strip() if m else None


def components(sql: str) -> dict:
    lc = _norm(sql)
    all_funcs = set(_FUNC.findall(lc))
    span = lambda head, tails: (lambda s: _whiten_null(s) if s else s)(
        _clause(lc, head, tails))
    return {
        "tables": set(re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)", lc)),
        "agg": {f for f in all_funcs if f in _AGG_FNS},
        "group_by": span(r"group\s+by", _TAILS_GROUP),
        "having": span(r"having", _TAILS_HAVING),
        "ordering": span(r"order\s+by", _TAILS_ORDER),
        "limit": span(r"limit", []),
        "filter": span(r"where", _TAILS_WHERE),
        "join_condition": _whiten_null(" ".join(re.findall(
            r"\bon\s+.*?(?=\bjoin\b|\bwhere\b|\bgroup\b|\border\b|\blimit\b|$)",
            lc, re.S))),
        "subquery": len(re.findall(r"\(\s*select", lc)),
        "date_function": all_funcs & _DATE_FNS,
        "null_handling": set(_NULL_OPS.findall(lc)),
        "function": all_funcs - _DATE_FNS - _AGG_FNS - _NULL_FNS - _SQL_KEYWORDS,
    }


def classify(pred_sql: str, gold_sql: str, *, pred_error: str | None = None,
             correct: bool = False) -> tuple[str, list[str]]:
    """Return (category, differing components) for one prediction."""
    if correct:
        return "correct", []
    if pred_error:
        return pred_error, []
    p, g = components(pred_sql), components(gold_sql)
    diffs = sorted(k for k in p if p[k] != g[k])
    if not diffs:
        return "semantic_mismatch", []
    if len(diffs) > 1:
        return "multiple_clauses", diffs
    return diffs[0], diffs