"""Phase 3 gate: sandbox executes correctly and survives adversarial SQL."""

import sqlite3
import time
from pathlib import Path

import pytest

from t2sql.sandbox.executor import Sandbox, classify

SPIDER_DB = (
    Path(__file__).resolve().parents[1]
    / "data" / "spider" / "database" / "concert_singer" / "concert_singer.sqlite"
)


@pytest.fixture()
def sandbox(tmp_path):
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", [(i, f"n{i}") for i in range(500)])
    con.execute("CREATE TABLE big (id INTEGER, b INTEGER)")
    con.execute(
        "INSERT INTO big SELECT t.id, t2.id FROM t, t t2 WHERE t.id < 40 AND t2.id < 40"
    )
    con.commit()
    con.close()
    sb = Sandbox(db, timeout_s=2.0, max_rows=100)
    yield sb
    sb.close()


def test_happy_path(sandbox):
    r = sandbox.execute("SELECT id, name FROM t WHERE id <= 3 ORDER BY id")
    assert r.ok and r.error is None
    assert r.columns == ("id", "name")
    assert r.rows == ((0, "n0"), (1, "n1"), (2, "n2"), (3, "n3"))
    assert not r.truncated and r.duration_ms >= 0


def test_taxonomy_no_such_table(sandbox):
    r = sandbox.execute("SELECT * FROM does_not_exist")
    assert not r.ok and r.error == "no_such_table"


def test_taxonomy_no_such_column(sandbox):
    r = sandbox.execute("SELECT nope FROM t")
    assert not r.ok and r.error == "no_such_column"


def test_taxonomy_syntax(sandbox):
    assert sandbox.execute("SELEC 1").error == "syntax"
    assert sandbox.execute("SELECT * FROM").error == "syntax"


def test_ambiguous_column(sandbox):
    r = sandbox.execute("SELECT id FROM t, big")  # id exists in both tables
    assert not r.ok and r.error == "ambiguous_column"


def test_writes_denied_and_db_untouched(sandbox):
    for sql in (
        "DROP TABLE t",
        "DELETE FROM t",
        "INSERT INTO t VALUES (999, 'x')",
        "UPDATE t SET name = 'hax'",
        "CREATE TABLE evil (x INTEGER)",
        "ALTER TABLE t ADD COLUMN x INTEGER",
    ):
        r = sandbox.execute(sql)
        assert not r.ok and r.error == "forbidden", sql
    check = sandbox.execute("SELECT COUNT(*) FROM t")
    assert check.rows == ((500,),)
    names = {row[0] for row in sandbox.execute("SELECT name FROM t").rows}
    assert "hax" not in names


def test_attach_and_pragma_denied(sandbox):
    assert sandbox.execute("ATTACH DATABASE '/tmp/x.db' AS x").error == "forbidden"
    assert sandbox.execute("PRAGMA journal_mode = DELETE").error == "forbidden"
    assert sandbox.execute("DETACH DATABASE x").error == "forbidden"


def test_multi_statement_rejected(sandbox):
    r = sandbox.execute("SELECT 1; DROP TABLE t")
    assert not r.ok and r.error == "syntax"  # only first statement is even parsed
    assert sandbox.execute("SELECT COUNT(*) FROM t").rows == ((500,),)


def test_timeout_interrupts_runaway_query(sandbox):
    t0 = time.monotonic()
    r = sandbox.execute(
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
        "SELECT sum(x) FROM c",
        timeout_s=1.0,
    )
    assert not r.ok and r.error == "timeout"
    assert time.monotonic() - t0 < 5  # hard interrupt, not full scan


def test_row_cap_truncates(sandbox):
    r = sandbox.execute("SELECT * FROM big", max_rows=10)
    assert r.ok and r.truncated and len(r.rows) == 10
    full = sandbox.execute("SELECT COUNT(*) FROM big")
    assert full.rows[0][0] > 10 and not full.truncated


def test_real_spider_database():
    assert SPIDER_DB.exists(), "run scripts/download_data.py first"
    sb = Sandbox(SPIDER_DB)
    r = sb.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    assert r.ok and "singer" in {row[0] for row in r.rows}
    sb.close()


def test_missing_db_raises():
    with pytest.raises(FileNotFoundError):
        Sandbox("/nonexistent/db.sqlite")


def test_classify_helper():
    assert classify("near \"foo\": syntax error") == "syntax"
    assert classify("no such column: z") == "no_such_column"
    assert classify("something new") == "operational"