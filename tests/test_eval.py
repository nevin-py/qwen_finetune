"""Phase 5 gate: execution comparison semantics + harness behavior."""

import json
import sqlite3
from pathlib import Path

import pytest

from t2sql.eval.execution import compare, postprocess, prepare, remove_distinct
from t2sql.eval.harness import evaluate
from t2sql.sandbox.executor import Sandbox

ROOT = Path(__file__).resolve().parents[1]


# --- comparison semantics (ported official behavior) ---

def test_empty_results_match():
    assert compare("SELECT a FROM t", [], "SELECT b FROM t", [])


def test_row_order_ignored_without_order_by():
    assert compare("SELECT a FROM t", [(1,), (2,)], "SELECT a FROM t", [(2,), (1,)])
    assert not compare("SELECT a FROM t ORDER BY a", [(1,), (2,)],
                       "SELECT a FROM t ORDER BY a", [(2,), (1,)])


def test_column_order_tolerated_even_with_order_by():
    # official result_eq searches column permutations regardless of ORDER BY
    assert compare("SELECT a, b FROM t ORDER BY a", [(1, "x"), (2, "y")],
                   "SELECT b, a FROM t ORDER BY a", [("x", 1), ("y", 2)])
    # but row order is enforced when gold has ORDER BY
    assert not compare("SELECT a, b FROM t ORDER BY a", [(1, "x"), (2, "y")],
                       "SELECT b, a FROM t ORDER BY a", [("y", 2), ("x", 1)])


def test_length_and_width_mismatch():
    assert not compare("SELECT a FROM t", [(1,)], "SELECT a FROM t", [(1,), (2,)])
    assert not compare("SELECT a FROM t", [(1,)], "SELECT a, b FROM t", [(1, 2)])


def test_prepare_is_idempotent_and_official_equivalent():
    prepared = prepare("SELECT  > = 1")
    assert "> =" not in prepared and prepared == "SELECT  >= 1"
    assert prepare(prepared) == prepared
    assert "distinct" not in remove_distinct("SELECT DISTINCT a FROM t").lower()
    assert postprocess("a ! = b") == "a != b"


# --- harness end-to-end on a temp database ---

@pytest.fixture()
def world(tmp_path):
    db = tmp_path / "w.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (id INTEGER, a INTEGER, b TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)",
                    [(1, 1, "x"), (2, 1, "y"), (3, 2, "z")])
    con.commit()
    con.close()
    items = [
        {"db_id": "w", "question": "q0", "query": "SELECT a FROM t"},
        {"db_id": "w", "question": "q1", "query": "SELECT id FROM t ORDER BY id"},
        {"db_id": "w", "question": "q2", "query": "SELECT * FROM missing"},
        {"db_id": "w", "question": "q3", "query": "SELECT b FROM t",
         "difficulty": "simple"},
    ]
    preds = [
        {"index": 0, "db_id": "w", "question": "q0",
         "prediction": "SELECT DISTINCT a FROM t"},   # distinct stripped officially -> match
        {"index": 1, "db_id": "w", "question": "q1",
         "prediction": "SELECT id FROM t ORDER BY id DESC"},  # wrong order
        {"index": 2, "db_id": "w", "question": "q2", "prediction": "SELECT 1"},  # gold bad
        {"index": 3, "db_id": "w", "question": "q3", "prediction": "SEL"},       # syntax err
    ]
    db_dir = tmp_path  # Sandbox path pattern: db_dir/<db_id>/<db_id>.sqlite
    (tmp_path / "w").mkdir()
    db.rename(tmp_path / "w" / "w.sqlite")
    return items, preds, db_dir


def test_harness_scoring(world):
    items, preds, db_dir = world
    rep = evaluate(items, preds, db_dir)
    assert rep.n == 4
    assert rep.n_skipped == 1                      # broken gold excluded
    scored = {r.index: r for r in rep.scored}
    assert scored[0].correct is True               # distinct stripped -> match
    assert scored[1].correct is False              # wrong ORDER BY
    assert scored[3].pred_error == "syntax"        # taxonomy propagated
    assert rep.accuracy == pytest.approx(1 / 3)
    assert rep.error_counts == {"syntax": 1}


def test_harness_rejects_misaligned_predictions(world):
    items, preds, db_dir = world
    preds[0] = {**preds[0], "question": "someone else's question"}
    with pytest.raises(ValueError, match="misaligned"):
        evaluate(items, preds, db_dir)


def test_report_difficulty_breakdown(world):
    items, preds, db_dir = world
    subset = [items[0], items[3]]                       # q0 and q3
    preds_subset = [preds[0], {**preds[3], "index": 1}]  # renumber to subset
    rep = evaluate(subset, preds_subset, db_dir)
    assert rep.by_difficulty == {"simple": 0.0}  # only q3 carries difficulty


def test_real_eval_report_shape():
    rep = json.loads((ROOT / "reports" / "eval_baseline_val_n20.json").read_text()) \
        if (ROOT / "reports" / "eval_baseline_val_n20.json").exists() else None
    if rep is None:
        pytest.skip("report not generated yet")
    assert set(rep) == {"n", "scored", "skipped", "accuracy", "by_difficulty", "pred_errors"}