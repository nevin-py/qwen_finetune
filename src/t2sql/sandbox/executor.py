"""Read-only execution sandbox for candidate SQL (design §6, objective §17).

Security boundary: every candidate query runs against a read-only connection
(``mode=ro``) guarded by an allowlist authorizer — only SELECT/READ/FUNCTION
actions pass, so DDL/DML/PRAGMA/ATTACH are denied even if the file were
writable. A progress-handler deadline hard-interrupts runaway queries, and
materialized output is capped so Cartesian bombs cannot exhaust memory.

The returned :class:`ExecResult` carries a taxonomy class in ``error``:
    syntax | no_such_table | no_such_column | ambiguous_column |
    no_such_function | forbidden | timeout | operational
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# Only read-shaped actions may touch the database; everything else (CREATE,
# DROP, INSERT, UPDATE, ATTACH, PRAGMA, TRANSACTION, ...) is denied.
_ALLOWED_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, "SQLITE_SELECT", None),
        getattr(sqlite3, "SQLITE_READ", None),
        getattr(sqlite3, "SQLITE_FUNCTION", None),
        getattr(sqlite3, "SQLITE_RECURSIVE", None),
    )
    if action is not None
)

_ERROR_PATTERNS = (
    ("not authorized", "forbidden"),
    ("attempt to write", "forbidden"),
    ("read-only", "forbidden"),
    ("no such table", "no_such_table"),
    ("no such column", "no_such_column"),
    ("ambiguous column name", "ambiguous_column"),
    ("no such function", "no_such_function"),
    ("only execute one statement", "syntax"),
    ("unrecognized token", "syntax"),
    ("incomplete input", "syntax"),
    ("syntax error", "syntax"),
)


@dataclass(frozen=True)
class ExecResult:
    ok: bool
    columns: tuple[str, ...] = ()
    rows: tuple[tuple, ...] = ()
    truncated: bool = False
    duration_ms: float = 0.0
    error: str | None = None          # taxonomy class, None when ok
    error_message: str | None = None  # raw SQLite message (debugging only)


def classify(message: str) -> str:
    msg = message.lower()
    for needle, kind in _ERROR_PATTERNS:
        if needle in msg:
            return kind
    return "operational"


class Sandbox:
    """Executes SQL against one SQLite file, read-only, with limits."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout_s: float = 10.0,
        max_rows: int = 10_000,
    ) -> None:
        path = Path(db_path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        self._con.execute("PRAGMA query_only = ON")
        self._con.set_authorizer(self._authorize)

    @staticmethod
    def _authorize(action: int, _arg1, _arg2, _db, _source) -> int:
        return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY

    def execute(
        self,
        sql: str,
        *,
        timeout_s: float | None = None,
        max_rows: int | None = None,
    ) -> ExecResult:
        """Run one statement and return a capped, classified result."""
        deadline = time.monotonic() + (timeout_s if timeout_s is not None else self.timeout_s)
        row_cap = max_rows if max_rows is not None else self.max_rows
        timed_out = False

        def _progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1  # interrupt the VM
            return 0

        started = time.perf_counter()
        self._con.set_progress_handler(_progress, 2_000)
        try:
            cur = self._con.execute(sql)
            columns = tuple(d[0] for d in cur.description or ())
            rows: list[tuple] = []
            truncated = False
            while True:
                batch = cur.fetchmany(row_cap + 1 - len(rows))
                if not batch:
                    break
                rows.extend(batch)
                if len(rows) > row_cap:
                    truncated = True
                    rows = rows[:row_cap]
                    break
            return ExecResult(
                ok=True,
                columns=columns,
                rows=tuple(rows),
                truncated=truncated,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except sqlite3.Error as exc:
            message = str(exc)
            if timed_out:
                kind = "timeout"
            else:
                kind = classify(message)
            return ExecResult(
                ok=False,
                duration_ms=(time.perf_counter() - started) * 1000,
                error=kind,
                error_message=message,
            )
        finally:
            self._con.set_progress_handler(None, 0)

    def close(self) -> None:
        self._con.close()