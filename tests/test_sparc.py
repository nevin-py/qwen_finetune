"""Phase 9: SParC dialogue prompts, metrics, gold isolation."""

import json
from pathlib import Path

import pytest

from t2sql.eval.sparc import sparc_metrics
from t2sql.prompts.template import build_dialogue_messages

ROOT = Path(__file__).resolve().parents[1]


class _R:  # minimal ItemResult stand-in
    def __init__(self, index, correct):
        self.index, self.correct = index, correct


def test_sparc_metrics_basics():
    items = [
        {"index": 0, "dialogue": 0, "turn": 1},
        {"index": 1, "dialogue": 0, "turn": 2},
        {"index": 2, "dialogue": 1, "turn": 1},
        {"index": 3, "dialogue": 1, "turn": 5},
    ]
    results = [_R(0, True), _R(1, False), _R(2, True), _R(3, False)]
    m = sparc_metrics(results, items)
    assert m["turns_scored"] == 4 and m["turn_accuracy"] == 0.5
    assert m["interaction_accuracy"] == 0.0        # neither dialogue fully correct
    assert m["average_progress"] == 0.5            # (1/2 + 1/2) / 2
    assert m["per_turn_accuracy"] == {"1": 1.0, "2": 0.0, ">4": 0.0}


def test_dialogue_prompt_uses_history_and_current_turn_only():
    hist = [{"utterance": "u1", "query": "SELECT a FROM t"}]
    msgs = build_dialogue_messages("u2", "SCHEMA", hist)
    user = msgs[1]["content"]
    assert "Dialogue so far" in user and "SELECT a FROM t" in user
    assert user.rstrip().endswith("Q: u2")
    assert msgs[0] == build_dialogue_messages("u2", "SCHEMA", [])[0]  # shared system


def test_dialogue_prompt_without_history_is_plain_question():
    msgs = build_dialogue_messages("just me", "SCHEMA", [])
    assert msgs[1]["content"] == "just me"


def test_run_sparc_never_reads_current_gold():
    src = (ROOT / "scripts" / "run_sparc.py").read_text()
    assert "dev_gold" not in src


@pytest.mark.skipif(not (ROOT / "data" / "sparc" / "dev.json").exists(),
                    reason="SParC data not downloaded")
def test_builder_splits_gold_from_prompts():
    from scripts.build_sparc_eval import main as build  # noqa: F401
    import scripts.build_sparc_eval as mod
    mod.main()
    prompts = json.loads((ROOT / "data" / "sparc" / "dev_prompts.json").read_text())
    gold = json.loads((ROOT / "data" / "sparc" / "dev_gold.json").read_text())
    assert len(prompts) == len(gold)
    for p, g in zip(prompts, gold):
        assert p["index"] == g["index"] and p["question"] == g["question"]
        assert "query" not in p and "gold" not in p          # prompt rows gold-free
        assert len(p["history"]) == g["turn"] - 1            # history = strictly previous
        if g["turn"] > 1:  # current gold must not appear as its own history
            assert g["query"] not in [h["query"] for h in p["history"]]