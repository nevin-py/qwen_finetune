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

import sqlite3
from pathlib import Path

_EXCLUDE = "name NOT LIKE 'sqlite_%' AND type = 'table'"
_MAX_VALUE_CHARS = 40
SAMPLE_LINE_BUDGET = 30   # sample lines per schema (keeps prompts in --max-len)


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _fmt_value(v: object) -> str:
    s = str(v).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= _MAX_VALUE_CHARS else s[: _MAX_VALUE_CHARS - 1] + "…"


def schema_text(db_path: str | Path, *, samples: bool = False,
                n_samples: int = 3,
                sample_line_budget: int = SAMPLE_LINE_BUDGET) -> str:
    """Return DDL for all user tables, ordered by name for determinism.

    With ``samples=True`` each table's DDL is followed by its row count and
    per-column example values (``ORDER BY value`` for determinism). Sample
    lines are capped globally by ``sample_line_budget`` so prompt size stays
    bounded regardless of schema width.
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
                section, used = _sample_section(con, name, n_samples, budget)
                budget -= used
                block += section
            blocks.append(block)
        return "\n".join(blocks)
    finally:
        con.close()


def _sample_section(con: sqlite3.Connection, table: str, n_samples: int,
                    budget: int) -> tuple[str, int]:
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
        cols = [r[1] for r in con.execute(
            f"PRAGMA table_info({ident})")]
    except sqlite3.Error:
        cols = []
    for col in cols:
        if len(parts) >= budget:
            break
        qcol = _quote_ident(col)
        try:
            vals = con.execute(
                f"SELECT DISTINCT {qcol} FROM {ident} WHERE {qcol} IS NOT NULL "
                f"ORDER BY 1 LIMIT {int(n_samples)}").fetchall()
        except sqlite3.Error:
            continue
        if vals:
            shown = ", ".join(_fmt_value(v) for (v,) in vals)
            parts.append(f"-- {col} examples: {shown}")
    return "\n" + "\n".join(parts), len(parts)
