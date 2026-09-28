"""Phase 10 plumbing: dataset locations, full-split sampling, BIRD evidence."""

from pathlib import Path

import pytest

from t2sql.datasets import db_file, dataset_items, sample_indices
from t2sql.prompts.template import full_question

ROOT = Path(__file__).resolve().parents[1]
HAS_SPIDER = (ROOT / "data" / "splits" / "val.json").exists()
HAS_BIRD = (ROOT / "data" / "bird" / "dev.json").exists()


def test_db_file_shape():
    assert db_file("spider", "aircraft").name == "aircraft.sqlite"
    assert str(db_file("bird", "financial")).endswith(
        "data/bird/databases/financial/financial.sqlite")


@pytest.mark.skipif(not HAS_SPIDER, reason="spider data missing")
def test_spider_items_and_full_split_sampling():
    assert dataset_items("spider", "val")[0]["db_id"]
    total = len(dataset_items("spider", "val"))
    assert sample_indices("spider", "val", 0, 0) == list(range(total))
    assert sample_indices("spider", "val", -1, 0) == list(range(total))
    s1 = sample_indices("spider", "val", 7, 3)
    assert s1 == sample_indices("spider", "val", 7, 3) and len(s1) == 7


@pytest.mark.skipif(not HAS_BIRD, reason="bird data missing")
def test_bird_items_and_db():
    items = dataset_items("bird", "dev")
    assert items[0]["difficulty"] in {"simple", "moderate", "challenging"}
    assert sample_indices("bird", "dev", 0, 0) == list(range(len(items)))
    first = items[0]
    assert db_file("bird", first["db_id"]).exists()


def test_bird_gold_normalized_for_harness():
    """BIRD spells gold as 'SQL'; the harness contract is 'query' (regression)."""
    if not HAS_BIRD:
        pytest.skip("bird data missing")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_eval import resolve_split
    items, db_dir = resolve_split("bird", "dev")
    assert len(items) == 1534
    assert all(it["query"] and it["difficulty"] for it in items)
    assert db_dir.name == "databases"


def test_full_question_appends_evidence_only_when_present():
    assert full_question({"question": "q"}) == "q"
    assert full_question({"question": "q", "evidence": "hint"}) == "q\nEvidence: hint"
    assert full_question({"question": "q", "evidence": ""}) == "q"