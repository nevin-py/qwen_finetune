"""v2 checklist tests: instruction variants, schema samples, features,
pool feedback, training-set dedup/few-shot (Stages 0/3/4 code)."""

import json
from pathlib import Path

from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import (INSTRUCTIONS, INSTRUCTION_VARIANTS,
                                    build_messages, build_repair_messages,
                                    variant_for)
from t2sql.repair.feedback import build_feedback
from t2sql.verifier.features import schema_link

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "spider" / "database" / "concert_singer" / "concert_singer.sqlite"


def test_variants_are_format_safe_and_v0_is_original():
    assert INSTRUCTION_VARIANTS[0] is INSTRUCTIONS
    assert len(INSTRUCTION_VARIANTS) >= 4
    for text in INSTRUCTION_VARIANTS:
        assert text.count("{examples}") == 1
        assert "{" not in text.replace("{examples}", "")
        assert "SQLite" in text and "Rules:" in text and "SQL statement" in text


def test_variant_for_is_deterministic_and_balanced():
    assert variant_for("anything", 1) == 0
    assert variant_for("q", 4) == variant_for("q", 4)
    picks = {variant_for(f"question number {i}", 4) for i in range(50)}
    assert picks == {0, 1, 2, 3}


def test_variant_zero_is_byte_identical_to_v1():
    a = build_messages("q", "SCHEMA")
    b = build_messages("q", "SCHEMA", variant=0, few_shot="")
    assert a == b
    rep = build_repair_messages("q", "SCHEMA", "SELECT 1", "fix it")
    assert rep[0] == a[0]  # repair shares the generation system prompt


def test_variants_change_system_prompt_only():
    v0 = build_messages("q", "SCHEMA")
    v1 = build_messages("q", "SCHEMA", variant=1)
    assert v0[1] == v1[1] == {"role": "user", "content": "q"}
    assert v0[0]["content"] != v1[0]["content"]
    assert "SCHEMA" in v1[0]["content"]


def test_schema_samples_add_gold_free_context():
    plain = schema_text(DB)
    sampled = schema_text(DB, samples=True)
    # every table's DDL preserved verbatim, appended context only
    assert sampled.count("CREATE TABLE") == plain.count("CREATE TABLE")
    assert all(p in sampled for p in plain.split("CREATE TABLE")[1:])
    assert "-- rows:" in sampled
    assert "examples:" in sampled
    assert schema_text(DB, samples=True) == sampled   # deterministic


def test_schema_link_feature():
    assert schema_link("how many singers are there?", "SELECT COUNT(*) FROM singer") == 1.0
    assert schema_link("no names here", "SELECT * FROM singer") == 0.0
    assert schema_link("q", "SELECT 1") == 0.0
    assert 0 < schema_link("singer table", "SELECT * FROM singer JOIN concert") < 1


def test_feedback_mentions_pool_disagreement():
    f = {"executed": 1.0, "pool_disagrees": 1.0, "pool_majority_n": 3, "pool_k": 5}
    msg = build_feedback(f)
    assert "3 of 5" in msg and "different result" in msg
    # agreement -> no such warning, falls back to the generic critique
    f2 = {"executed": 1.0, "pool_disagrees": 0.0}
    assert "3 of 5" not in build_feedback(f2)


def test_dedup_drops_duplicates_and_val_overlap():
    from scripts.build_training_set import dedup

    items = [
        {"db_id": "a", "question": "One?", "query": "SELECT 1"},
        {"db_id": "a", "question": "  one? ", "query": "SELECT 2"},   # dup
        {"db_id": "b", "question": "Two?", "query": "SELECT 3"},      # in val
        {"db_id": "b", "question": "Three?", "query": "SELECT 4"},
    ]
    val_keys = {("b", "two?")}
    kept, n_dup, n_val, n_excluded = dedup(items, val_keys)
    assert (n_dup, n_val, n_excluded) == (1, 1, 0)
    assert [x["question"] for x in kept] == ["One?", "Three?"]


def test_fewshot_block_is_deterministic():
    from scripts.build_training_set import few_shot_examples

    items = [{"question": f"q{i}", "query": f"SELECT {i}"} for i in range(30)]
    a = few_shot_examples(items, 2)
    b = few_shot_examples(items, 2)
    assert a == b and len(a) == 2


def test_training_examples_honor_variants_and_fewshot():
    from t2sql.training.data import build_training_examples

    item = {"db_id": "concert_singer", "question": "how many singers?",
            "query": "SELECT COUNT(*) FROM singer"}
    base = build_training_examples([item], DB.parents[1])
    varied = build_training_examples([item], DB.parents[1], prompt_variants=4,
                                     few_shot="Q: x\nA: SELECT 1")
    assert base[0]["messages"][0] != varied[0]["messages"][0]
    assert "Examples:" in varied[0]["messages"][0]["content"]
    assert base[0]["messages"][1]["content"] == "how many singers?"  # user turn fixed
    assert base[0]["messages"][-1] == {"role": "assistant",
                                       "content": "SELECT COUNT(*) FROM singer"}
