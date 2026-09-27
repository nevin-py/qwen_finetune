"""Phase 4 gate: schema serialization, prompt template, SQL extraction, split."""

import json
import random
from pathlib import Path

from t2sql.inference.engine import extract_sql
from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import build_messages

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "spider" / "database" / "concert_singer" / "concert_singer.sqlite"


def test_schema_text_contains_ddl_and_is_deterministic():
    a, b = schema_text(DB), schema_text(DB)
    assert a == b
    assert 'CREATE TABLE "singer"' in a
    assert 'CREATE TABLE "concert"' in a
    assert "REFERENCES" in a            # foreign keys preserved verbatim
    assert "sqlite_" not in a           # internal tables excluded
    assert a.splitlines()[0].startswith("CREATE TABLE")


def test_build_messages_structure():
    schema = "CREATE TABLE t (id INTEGER PRIMARY KEY)"
    msgs = build_messages("how many t rows?", schema)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "how many t rows?"
    assert schema in msgs[0]["content"]
    assert "SQLite" in msgs[0]["content"]
    assert "Examples:" not in msgs[0]["content"]  # no few-shot by default


def test_build_messages_few_shot_is_content():
    msgs = build_messages("q", "s", few_shot="Q: a\nA: SELECT 1")
    assert "Examples:\nQ: a\nA: SELECT 1" in msgs[0]["content"]


def test_extract_sql_fenced():
    assert extract_sql("Here you go:\n```sql\nSELECT id FROM t\n```\n done") == \
        "SELECT id FROM t"


def test_extract_sql_with_prose_prefix():
    assert extract_sql("The answer is SELECT COUNT(*) FROM singer") == \
        "SELECT COUNT(*) FROM singer"


def test_extract_sql_single_statement():
    assert extract_sql("SELECT 1;\nSELECT 2") == "SELECT 1"


def test_extract_sql_with_query():
    assert extract_sql("WITH x AS (SELECT 1) SELECT * FROM x") == \
        "WITH x AS (SELECT 1) SELECT * FROM x"


def test_split_counts_and_disjointness():
    train = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    val = json.loads((ROOT / "data" / "splits" / "val.json").read_text())
    assert (len(train), len(val)) == (6300, 700)
    key = lambda x: (x["db_id"], x["question"])
    assert not ({key(x) for x in train} & {key(x) for x in val})


def test_split_is_reproducible():
    items = json.loads((ROOT / "data" / "spider" / "train_spider.json").read_text())
    rng = random.Random(13)
    order = list(range(len(items)))
    rng.shuffle(order)
    val = json.loads((ROOT / "data" / "splits" / "val.json").read_text())
    # same subset regardless of iteration order (script preserves source order)
    assert {key(x) for x in val} == {key(items[i]) for i in order[:700]}


def key(x):
    return (x["db_id"], x["question"])