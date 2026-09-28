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