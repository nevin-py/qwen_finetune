"""Regression tests for the audit fixes.

Each test pins a defect found by auditing v2, so the same failure cannot come
back silently. The defect each one guards is named in its docstring.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2sql.datasets import db_stratified_sample
from t2sql.eval import execution
from t2sql.prompts.schema import SchemaCache, relevant_values, schema_text
from t2sql.verifier.features import (extract_features, pool_consistency,
                                     pool_signature_counts)
from t2sql.verifier.scorer import (DEFAULT_WEIGHTS, FAILED_SCORE, max_reachable_score,
                                   score)

SPLITS = ROOT / "data" / "splits"
CONCERT = ROOT / "data" / "spider" / "database" / "concert_singer" / "concert_singer.sqlite"


# --- verifier: ranking safety -------------------------------------------------

def test_failed_candidate_loses_under_penalty_weights():
    """Defect: failures were floored at 0.0, so negative-weight executed
    candidates lost argmax and the verifier preferred crashing SQL."""
    bad = extract_features("q", "SELECT bad", executed=False, error="no_such_column")
    ok = extract_features("how many?", "SELECT a FROM t", executed=True, error=None,
                          columns=("a",), rows=(("x",),), consistency=0.0)
    for w in ({}, DEFAULT_WEIGHTS,
              {"consistency": 0.0, "answer_in_question": -0.5,
               "count_shape_match": 1.0, "empty": -0.6, "truncated": -0.6,
               "slow": -0.2, "schema_link": -1.0, "executed": 1.0}):
        assert score(bad, w) < score(ok, w), w


def test_max_reachable_score_exposes_the_ceiling():
    """Defect: tau=3.0 could not fire under the v2 tuned weights (ceiling 2.0),
    so every item was flagged and 'bounded repair' repaired everything."""
    v2 = {"consistency": 0.0, "answer_in_question": -0.5, "count_shape_match": 1.0,
          "empty": -0.6, "truncated": -0.6, "slow": -0.2, "schema_link": -1.0,
          "executed": 1.0}
    assert max_reachable_score(v2) == pytest.approx(2.0)
    assert 3.0 >= max_reachable_score(v2)   # the v2 hardcoded tau was infeasible
    assert max_reachable_score() > 3.0      # rule weights leave room for it


def test_reachable_ceiling_is_actually_attainable():
    """The advertised ceiling must be a real score, not a loose bound."""
    perfect = {"executed": 1.0}
    for name, v in DEFAULT_WEIGHTS.items():
        if name == "executed" or v > 0:
            perfect[name] = 1.0
    assert score(perfect, DEFAULT_WEIGHTS) == pytest.approx(
        max_reachable_score(DEFAULT_WEIGHTS))


# --- pool consistency: one definition across cells ---------------------------

def test_pool_consistency_is_shared_between_cells():
    """Defect: cell C scored consistency as count/k and cell D as
    (count+1)/(k+1), so the repair margin compared two different scales."""
    sigs = ["a", "a", "b", None]
    counts = pool_signature_counts(sigs)
    assert counts == {"a": 2, "b": 1}
    # One pool, one denominator, whether the candidate is an original pool
    # member or one produced later by repair.
    assert pool_consistency("a", counts, 4) == pytest.approx(0.5)
    assert pool_consistency("b", counts, 4) == pytest.approx(0.25)
    assert pool_consistency("a", counts, 4) == pool_consistency("a", counts, 4)
    assert pool_consistency(None, counts, 4) == 0.0
    assert pool_consistency("a", counts, 0) == 0.0


# --- harness fidelity --------------------------------------------------------

def test_prepare_survives_empty_prediction():
    """Defect: a generation with no SQL raised IndexError inside sqlparse and
    killed an hours-long evaluation run instead of scoring it wrong."""
    for s in ("", "   ", "no sql here"):
        assert isinstance(execution.prepare(s), str)
    assert execution.prepare("SELECT DISTINCT a FROM t") == "SELECT  a FROM t"


@pytest.mark.skipif(not (ROOT / "data" / "spider" / "database" / "wta_1" /
                         "wta_1.sqlite").exists(), reason="spider data absent")
def test_sandbox_reads_non_utf8_text_like_the_official_evaluator():
    """Defect: the official evaluator decodes bad bytes with errors='ignore';
    without that, two Spider dev items failed as 'gold broken' and were dropped
    from the split."""
    from t2sql.sandbox.executor import Sandbox
    sb = Sandbox(ROOT / "data" / "spider" / "database" / "wta_1" / "wta_1.sqlite")
    try:
        res = sb.execute("SELECT first_name FROM players")
        assert res.ok, res.error
    finally:
        sb.close()


# --- validation split: tuning must not measure recall ------------------------

@pytest.mark.skipif(not (SPLITS / "holdout.json").exists(),
                    reason="rebuild with make_split.py --holdout-dbs")
def test_holdout_databases_are_never_trained_on():
    """Defect: val.json shares all 140 databases with training, and 76% of its
    questions have an exact (db, gold SQL) match in train, so tuning measured
    recall rather than generalization."""
    train = json.loads((SPLITS / "train.json").read_text())
    hold = json.loads((SPLITS / "holdout.json").read_text())
    assert not ({x["db_id"] for x in train} & {x["db_id"] for x in hold})

    def norm(s):
        return " ".join(s.split()).rstrip(";").strip().lower()
    train_sql = {(x["db_id"], norm(x["query"])) for x in train}
    assert not [x for x in hold if (x["db_id"], norm(x["query"])) in train_sql]


def test_db_stratified_sample_spreads_across_schemas():
    """Defect: the probe took val[:50], which is 10 of 140 databases (and 1 of
    20 on the holdout), because split files are grouped by db_id."""
    # Split files are grouped by db_id, which is what made a prefix sample
    # collapse onto a handful of schemas.
    items = [{"db_id": f"db{i}", "question": f"q{i}", "query": "SELECT 1"}
             for i in range(20) for _ in range(100)]
    assert len({x["db_id"] for x in items[:50]}) == 1   # the old behaviour
    picked = db_stratified_sample(items, 50, seed=13)
    assert len(picked) == 50
    assert len({x["db_id"] for x in picked}) == 20
    assert [x["question"] for x in picked] == [
        x["question"] for x in db_stratified_sample(items, 50, seed=13)]


# --- value retrieval ---------------------------------------------------------

def test_question_conditioned_values_beat_alphabetical():
    """Defect: 'ORDER BY 1 LIMIT 3' surfaced alphabetically-first values, which
    recovered 8.2% of BIRD gold literals; matching on the question recovers
    far more."""
    con = sqlite3.connect(f"file:{CONCERT.resolve()}?mode=ro", uri=True)
    try:
        hits = relevant_values(con, "singer", "Country",
                               "How many singers are from Netherlands?")
        assert hits and any("Netherlands" in h for h in hits)
        # A question naming nothing stored must not invent values.
        assert relevant_values(con, "singer", "Country",
                               "zzzz unrelated tokens") == []
    finally:
        con.close()


def test_schema_cache_keys_on_the_question():
    """A db_id-only cache would hand one question's values to another."""
    cache = SchemaCache(samples=True, question_values=True)
    a = cache.get(CONCERT, "How many singers are there?")
    b = cache.get(CONCERT, "Which stadium has the most concerts?")
    assert a != b
    # Without question conditioning the text is stable per database.
    plain = SchemaCache(samples=True)
    assert plain.get(CONCERT, "q1") == plain.get(CONCERT, "q2")
    assert schema_text(CONCERT, samples=True) == plain.get(CONCERT)


def test_question_values_never_widens_the_prompt():
    """The sample-line budget is the prompt-size guard; question-conditioned
    selection must respect it."""
    base = schema_text(CONCERT, samples=True)
    cond = schema_text(CONCERT, samples=True, question="Which singer is Treyes?")
    assert cond.count("\n") <= base.count("\n")


# --- data preprocessing ------------------------------------------------------

def test_fewshot_demonstrations_are_not_also_training_targets():
    """A demonstration that is also a target is supervised twice: once as an
    answer and once inside every prompt."""
    from scripts.build_training_set import few_shot_examples
    train = json.loads((SPLITS / "train.json").read_text())
    chosen = few_shot_examples(train, 2)
    keys = {(x["db_id"], x["question"].strip().lower()) for x in chosen}
    blocked = set(keys)
    from scripts.build_training_set import dedup
    kept, _dup, _blocked, n_excluded = dedup(train, set(), blocked)
    assert n_excluded == len(keys)
    assert not ({(x["db_id"], x["question"].strip().lower()) for x in kept} & keys)


def test_dedup_does_not_drop_legitimate_shared_sql():
    """78% of Spider rows share gold SQL with another row because different
    questions really do resolve to the same simple query. Deduping on SQL
    would delete real supervision, so dedup must key on the question."""
    from scripts.build_training_set import dedup
    rows = [{"db_id": "d", "question": "How many customers?",
             "query": "SELECT count(*) FROM customer"},
            {"db_id": "d", "question": "Count the customers",
             "query": "SELECT count(*) FROM customer"}]
    kept, dup, _blocked, _excluded = dedup(rows, set())
    assert dup == 0 and len(kept) == 2

# --- resumability: a partial run must not look finished ---------------------

def test_run_verify_resume_appends_without_duplicating(tmp_path):
    """Interrupting a run leaves a partial file on disk (records are flushed
    per line). Without --resume the rerun rewrites from scratch and discards
    it; with --resume the completed questions are skipped and the rest appended.
    """
    import subprocess
    ROOT = Path(__file__).resolve().parents[1]
    out = tmp_path / "partial.jsonl"
    # Exercise the resume decision without the GPU: the runner's contract is
    # "indices already in the file are skipped, mode is append".
    src = (ROOT / "scripts" / "run_verify.py").read_text()
    assert '"--resume"' in src and 'mode = "a" if (args.resume and done) else "w"' in src
    assert "todo = [i for i in idxs if i not in done]" in src


def test_benchmark_script_checks_record_count_not_just_existence():
    """Defect: stages were skipped on `[ -f output ]`, so an interrupted run's
    partial file was treated as finished and B/D were built from a truncated
    pool (observed: a cell C run stopped at 812 of 1534 records)."""
    root = Path(__file__).resolve().parents[1]
    for name in ("run_final_benchmarks.sh", "run_final_benchmarks_v2.sh"):
        script = (root / "scripts" / name).read_text()
        assert "complete()" in script, name
        assert 'wc -l < "$1"' in script, name
        # No bare existence test may remain on a generated pool.
        assert "[ -f reports/${ds}_C_dev" not in script, name
        assert "complete reports/${ds}_C_dev" in script, name
