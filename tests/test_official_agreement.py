"""M2 gate: our harness must agree with the official Spider evaluator
(test-suite-sql-eval, taoyds) item-for-item on real predictions."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "test-suite-sql-eval"

pytestmark = pytest.mark.skipif(
    not (TOOLS / "exec_eval.py").exists(),
    reason="official evaluator not cloned (tools/test-suite-sql-eval)",
)
sys.path.insert(0, str(TOOLS))

from exec_eval import eval_exec_match           # noqa: E402  (official code)
from t2sql.eval.execution import prepare        # noqa: E402
from t2sql.eval.harness import evaluate         # noqa: E402


def _official_correct(items, preds, db_dir):
    """Run the official comparison on each pair (original database only)."""
    out = {}
    for pred in preds:
        gold = items[pred["index"]]
        db_file = db_dir / gold["db_id"] / f"{gold['db_id']}.sqlite"
        out[pred["index"]] = bool(eval_exec_match(
            str(db_file),
            prepare(pred["prediction"]),
            prepare(gold["query"]),
            False,   # plug_value: we predict values ourselves
            False,   # keep_distinct
            False,   # progress bar
        ))
    return out


def test_agreement_with_official_evaluator():
    """Runs on a frozen fixture (split-independent): 20 real baseline
    predictions with their gold, captured from the v1 val carve."""
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "official_agreement.json").read_text())
    items = [{"db_id": x["db_id"], "question": x["question"],
              "query": x["query"]} for x in fixture]
    predictions = [{"index": x["index"], "db_id": x["db_id"],
                    "question": x["question"], "prediction": x["prediction"]}
                   for x in fixture]
    db_dir = ROOT / "data" / "spider" / "database"

    ours = evaluate(items, predictions, db_dir)
    official = _official_correct(items, predictions, db_dir)

    disagreements = [
        (r.index, r.correct, official[r.index])
        for r in ours.results if r.correct != official[r.index]
    ]
    assert not disagreements, f"disagreements (index, ours, official): {disagreements}"