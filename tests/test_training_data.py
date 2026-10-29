"""Phase 6: training-set construction integrity."""

import json
from pathlib import Path

import pytest

from t2sql.training.data import build_training_examples

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "spider" / "database"


def test_build_examples_structure():
    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())[:3]
    examples = build_training_examples(items, DB_DIR)
    assert len(examples) == 3
    for item, ex in zip(items, examples):
        msgs = ex["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[1]["content"] == item["question"]
        assert msgs[2]["content"] == item["query"]          # gold = assistant turn
        assert item["query"].upper().lstrip().startswith(("SELECT", "WITH"))
        assert "Schema:" in msgs[0]["content"]
        assert "CREATE TABLE" in msgs[0]["content"]         # schema actually inlined


def test_build_examples_is_deterministic():
    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())[:5]
    assert build_training_examples(items, DB_DIR) == build_training_examples(items, DB_DIR)


def test_prebuilt_training_set():
    path = ROOT / "data" / "splits" / "train_sft.json"
    if not path.exists():
        pytest.skip("run scripts/build_training_set.py first")
    examples = json.loads(path.read_text())
    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    # Dedup, gold quarantine, few-shot exclusion and the database holdout may
    # each drop rows; every surviving example must trace back to a raw item.
    assert len(items) - 60 <= len(examples) <= len(items)
    pairs = {(x["question"], x["query"]) for x in items}
    for ex in examples:
        q, a = ex["messages"][1]["content"], ex["messages"][2]["content"]
        assert (q, a) in pairs