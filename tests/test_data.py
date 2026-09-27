"""Phase 2 gate: datasets present, counts intact, every db_id resolves."""

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

EXPECTED = {
    "spider/dev.json": 1034,          # locked single-turn test
    "spider/train_spider.json": 7000,  # training pool (val carved from here)
    "sparc/dev.json": 422,            # multi-turn, eval-only
    "sparc/train.json": 3034,
    "bird/dev.json": 1534,            # locked held-out benchmark
}


def _db_id(q: dict) -> str:
    """Spider/BIRD use `db_id`, official SParC uses `database_id`."""
    return q.get("db_id") or q["database_id"]


def test_counts():
    for rel, expected in EXPECTED.items():
        assert len(json.loads((DATA / rel).read_text())) == expected, rel


def test_gold_files_present():
    assert (DATA / "spider/dev_gold.sql").stat().st_size > 100_000
    assert (DATA / "bird/dev.sql").stat().st_size > 100_000


def test_every_db_id_resolves():
    checks = [
        ("spider/dev.json", "spider/database"),
        ("spider/train_spider.json", "spider/database"),
        ("sparc/dev.json", "spider/database"),
        ("sparc/train.json", "spider/database"),
        ("bird/dev.json", "bird/databases"),
    ]
    for questions_rel, db_root in checks:
        ids = {_db_id(q) for q in json.loads((DATA / questions_rel).read_text())}
        for db_id in ids:
            assert (DATA / db_root / db_id / f"{db_id}.sqlite").exists(), \
                f"{db_id} missing for {questions_rel}"


def test_bird_has_difficulty_and_evidence():
    dev = json.loads((DATA / "bird/dev.json").read_text())
    assert all(q.get("difficulty") for q in dev)
    assert {q["difficulty"] for q in dev} <= {"simple", "moderate", "challenging"}