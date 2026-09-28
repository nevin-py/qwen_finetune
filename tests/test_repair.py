"""Phase 8: repair feedback, repair prompt, gold isolation."""

from pathlib import Path

from t2sql.prompts.template import build_messages, build_repair_messages
from t2sql.repair.feedback import build_feedback

REPAIR_DIR = Path(__file__).resolve().parents[1] / "src" / "t2sql" / "repair"


def test_failed_execution_feedback_has_raw_message():
    f = {"executed": 0.0, "error": "no_such_column"}
    fb = build_feedback(f, "no such column: xyz")
    assert "does not run" in fb and "no such column: xyz" in fb


def test_failed_execution_feedback_falls_back_to_class():
    fb = build_feedback({"executed": 0.0, "error": "timeout"}, None)
    assert "timeout" in fb


def test_executed_feedback_warnings():
    assert "returns no rows" in build_feedback(
        {"executed": 1.0, "empty": 1.0, "consistency": 1.0})
    assert "disagrees" in build_feedback(
        {"executed": 1.0, "empty": 0.0, "consistency": 0.2})
    assert "count" in build_feedback(
        {"executed": 1.0, "empty": 0.0, "consistency": 1.0,
         "expects_count": 1.0, "scalar": 0.0})
    assert "could not be verified" in build_feedback(
        {"executed": 1.0, "empty": 0.0, "consistency": 1.0,
         "expects_count": 0.0, "scalar": 0.0})


def test_repair_prompt_shares_system_with_generation():
    gen = build_messages("q", "SCHEMA")
    rep = build_repair_messages("q", "SCHEMA", "SELECT 1", "It fails")
    assert rep[0] == gen[0], "system prompt must stay on-distribution"
    assert rep[1]["role"] == "user"
    for token in ("q", "SELECT 1", "It fails"):
        assert token in rep[1]["content"]


def test_repair_modules_never_import_gold():
    for path in REPAIR_DIR.glob("*.py"):
        src = path.read_text()
        assert "t2sql.eval" not in src, path.name


class _FakeLLM:
    def __init__(self, replies):
        self.replies = iter(replies)

    def generate(self, messages, **kwargs):
        return next(self.replies)


def _record():
    return {"index": 0, "db_id": "t", "question": "list all rows",
            "prediction": "SELECT b FROM t", "verifier_score": 0.0,
            "candidates": [{"sql": "SELECT b FROM t", "score": 0.0,
                            "result_sig": None, "consistency": 0.0}]}


def _sandbox(tmp_path):
    import sqlite3
    from t2sql.sandbox.executor import Sandbox
    db = tmp_path / "t.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t(a)")
    con.execute("INSERT INTO t VALUES (1),(2)")
    con.close()
    return Sandbox(db)


def test_repair_loop_replaces_and_stops_on_no_progress(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_repair import repair_item
    rec = _record()
    llm = _FakeLLM(["SELECT a FROM t", "SELECT a FROM t"])
    attempts, repl = repair_item(llm, _sandbox(tmp_path), "CREATE TABLE t(a)", rec,
                                 max_iterations=2)
    assert attempts == 1 and repl == 1          # 2nd reply == current -> loop guard
    assert rec["prediction"] == "SELECT a FROM t"
    assert rec["repairs"][0]["replaced"] is True
    assert rec["repairs"][0]["executed"] is True


def test_repair_loop_keeps_current_when_repair_scores_worse(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_repair import repair_item
    rec = _record()
    rec["verifier_score"] = 2.5               # current best is already strong
    llm = _FakeLLM(["SELECT zzz FROM t"] * 2)  # repair fails execution
    attempts, repl = repair_item(llm, _sandbox(tmp_path), "CREATE TABLE t(a)", rec,
                                 max_iterations=2)
    assert attempts == 2 and repl == 0
    assert rec["prediction"] == "SELECT b FROM t"   # unchanged (score 0 <= 2.5)
    assert all(not x["replaced"] for x in rec["repairs"])


def test_repair_loop_respects_iteration_budget(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_repair import repair_item
    rec = _record()
    rec["verifier_score"] = -99                # everything scores better
    llm = _FakeLLM(["SELECT a FROM t", "SELECT 1 FROM t", "SELECT 2 FROM t"])
    attempts, repl = repair_item(llm, _sandbox(tmp_path), "CREATE TABLE t(a)", rec,
                                 max_iterations=2)
    assert attempts == 2 and len(rec["repairs"]) == 2   # hard budget: no 3rd attempt
    assert repl <= 2