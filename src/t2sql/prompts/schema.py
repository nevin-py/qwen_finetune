"""Schema serialization: SQLite DDL -> prompt text (design §6).

Verbatim ``CREATE TABLE`` statements from ``sqlite_master`` are used rather
than a re-serialized model: they are lossless (types, PK, FK, CHECK all
preserved) and need no transformation logic to maintain.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

_EXCLUDE = "name NOT LIKE 'sqlite_%' AND type = 'table'"


def schema_text(db_path: str | Path) -> str:
    """Return DDL for all user tables, ordered by name for determinism."""
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        rows = con.execute(
            f"SELECT sql FROM sqlite_master WHERE {_EXCLUDE} ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return "\n".join(ddl.rstrip(";") for (ddl,) in rows if ddl)