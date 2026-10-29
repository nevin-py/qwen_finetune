"""Schema serialization: SQLite DDL -> prompt text (design §6).

Verbatim ``CREATE TABLE`` statements from ``sqlite_master`` are used rather
than a re-serialized model: they are lossless (types, PK, FK, CHECK all
preserved) and need no transformation logic to maintain.

Optionally appends value-level context (checklist 3.3): per-table row counts
and up to ``n_samples`` non-null example values per column, read from the
database content only — never from any gold answer. All output is
deterministic (row-count exact; samples ordered by value).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_EXCLUDE = "name NOT LIKE 'sqlite_%' AND type = 'table'"
_MAX_VALUE_CHARS = 40
SAMPLE_LINE_BUDGET = 30   # sample lines per schema (keeps prompts in --max-len)


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 2}
 
 
def relevant_values(con: sqlite3.Connection, table: str, col: str,
                    question: str, *, limit: int = 3,
                    max_scan: int = 4000) -> list[str]:
    """Column values that look like the question, most-relevant first.

    Showing the alphabetically first values of every column recovers almost
    nothing: measured over BIRD dev gold, `ORDER BY 1 LIMIT 3` surfaces just
    8.2% of the literals the queries actually filter on, because the value
    named in a question is rarely the lexicographically smallest one. Scoring
    stored values by token overlap with the question raises the hit rate
    without any gold knowledge: this reads database content only.

    Values are truncated and de-duplicated so the budget in
    ``schema_text`` still bounds prompt size.
    """
    ident, qcol = _quote_ident(table), _quote_ident(col)
    qt = _tokens(question)
    if not qt:
        return []
    try:
        rows = con.execute(
            f"SELECT DISTINCT {qcol} FROM {ident} WHERE {qcol} IS NOT NULL "
            f"LIMIT {int(max_scan)}").fetchall()
    except sqlite3.Error:
        return []
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for (v,) in rows:
        if v is None:
            continue
        s = str(v)
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        toks = _tokens(s)
        if not toks:
            continue
        overlap = len(toks & qt)
        if overlap:
            # Prefer a value the question names outright, then partial hits.
            exact = 2 if key in qt else 0
            scored.append((overlap + exact, s))
    scored.sort(key=lambda p: (-p[0], len(p[1])))
    return [s for _, s in scored[:limit]]



def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fmt_value(v: object) -> str:
    s = str(v).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= _MAX_VALUE_CHARS else s[: _MAX_VALUE_CHARS - 1] + "…"


def schema_text(db_path: str | Path, *, samples: bool = False,
                n_samples: int = 3,
                sample_line_budget: int = SAMPLE_LINE_BUDGET,
                question: str | None = None) -> str:
    """Return DDL for all user tables, ordered by name for determinism.

    With ``samples=True`` each table's DDL is followed by its row count and
    per-column example values. Sample lines are capped globally by
    ``sample_line_budget`` so prompt size stays bounded regardless of schema
    width; because the budget is spent in table-name order, wide schemas lose
    their later tables first, so raise the budget when prompt length allows.

    Passing ``question`` switches value selection from "first values
    alphabetically" to "values the question actually names", which is what
    makes the hint useful for linking a question to a stored value. With no
    question the original deterministic behaviour is kept.
    """
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"SELECT name, sql FROM sqlite_master WHERE {_EXCLUDE} ORDER BY name"
        ).fetchall()
        blocks: list[str] = []
        budget = sample_line_budget
        for name, ddl in rows:
            if not ddl:
                continue
            block = ddl.rstrip(";")
            if samples and budget > 0:
                section, used = _sample_section(con, name, n_samples, budget,
                                                question)
                budget -= used
                block += section
            blocks.append(block)
        return "\n".join(blocks)
    finally:
        con.close()


def _sample_section(con: sqlite3.Connection, table: str, n_samples: int,
                    budget: int, question: str | None = None) -> tuple[str, int]:
    ident = _quote_ident(table)
    parts: list[str] = []
    try:
        (n,) = con.execute(f"SELECT COUNT(*) FROM {ident}").fetchone()
        parts.append(f"-- rows: {n}")
    except sqlite3.Error:
        return "", 0
    if len(parts) >= budget:
        return "\n" + "\n".join(parts), len(parts)
    try:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({ident})")]
    except sqlite3.Error:
        cols = []
    for col in cols:
        if len(parts) >= budget:
            break
        vals: list[object] = []
        if question:
            # Prefer values the question names, but never at the cost of
            # showing no value domain at all: an unrelated column still needs
            # its representative values for the schema to be usable.
            vals = relevant_values(con, table, col, question, limit=n_samples)
        if not vals:
            qcol = _quote_ident(col)
            try:
                rows_ = con.execute(
                    f"SELECT DISTINCT {qcol} FROM {ident} "
                    f"WHERE {qcol} IS NOT NULL ORDER BY 1 "
                    f"LIMIT {int(n_samples)}").fetchall()
            except sqlite3.Error:
                continue
            vals = [v for (v,) in rows_]
        if vals:
            shown = ", ".join(_fmt_value(v) for v in vals)
            parts.append(f"-- {col} examples: {shown}")
    return "\n" + "\n".join(parts), len(parts)
 
 
class SchemaCache:
    """Schema text memoised per (database, question).

    The key includes the question because question-conditioned value sampling
    makes the rendered text depend on it: a cache keyed by database alone
    would silently hand one question's example values to another, which is
    worse than not caching at all. Callers get one object instead of an
    open-coded dict so every runner keys it the same way.
    """
 
    def __init__(self, *, samples: bool = False,
                 question_values: bool = False, n_samples: int = 3,
                 sample_line_budget: int = SAMPLE_LINE_BUDGET) -> None:
        self.samples = samples
        self.question_values = question_values
        self.n_samples = n_samples
        self.sample_line_budget = sample_line_budget
        self._cache: dict[tuple[str, str | None], str] = {}
 
    def get(self, db_path: str | Path, question: str | None = None) -> str:
        q = question if self.question_values else None
        key = (str(Path(db_path).resolve()), q)
        if key not in self._cache:
            self._cache[key] = schema_text(
                db_path, samples=self.samples, n_samples=self.n_samples,
                sample_line_budget=self.sample_line_budget, question=q)
        return self._cache[key]
