"""Phase 7 prep: verifier features, rule scorer, gold isolation."""

from pathlib import Path

import pytest

from t2sql.verifier.features import (
    extract_features,
    expects_count,
    n_tables_referenced,
    scalar_value_in_question,
)
from t2sql.verifier.scorer import FAILED_SCORE, score

VERIFIER_DIR = Path(__file__).resolve().parents[1] / "src" / "t2sql" / "verifier"


def test_failed_execution_ranks_below_every_executed_candidate():
    """A failed query must never outrank a working one, whatever the weights.

    The original rule floored failures at 0.0, so with negative penalty
    weights an executing candidate could score below zero and argmax would
    prefer SQL that crashes.
    """
    f = extract_features("q", "SELECT bad", executed=False, error="no_such_column")
    assert f["executed"] == 0.0 and f["error_no_such_column"] == 1.0
    ok = extract_features("q", "SELECT a FROM t", executed=True, error=None,
                          columns=("a",), rows=(("x",),))
    penalties = {"consistency": 0.0, "answer_in_question": -0.5,
                 "count_shape_match": 1.0, "empty": -0.6, "truncated": -0.6,
                 "slow": -0.2, "schema_link": -1.0, "executed": 1.0}
    assert score(f) < score(ok)
    assert score(f, penalties) < score(ok, penalties)
    assert score(f, penalties) == FAILED_SCORE


def test_features_of_successful_execution():
    f = extract_features(
        "how many singers are there?", "SELECT COUNT(*) FROM singer",
        executed=True, error=None, columns=("COUNT(*)",), rows=((5,),),
        duration_ms=12.0,
    )
    assert f["executed"] == 1.0 and f["scalar"] == 1.0
    assert f["expects_count"] == 1.0 and f["count_shape_match"] == 1.0
    assert f["n_tables"] == 1.0 and f["empty"] == 0.0


def test_count_question_with_row_set_is_shape_mismatch():
    f = extract_features(
        "how many singers?", "SELECT name FROM singer",
        executed=True, error=None, columns=("name",), rows=[("a",), ("b",)],
    )
    assert f["expects_count"] == 1.0 and f["count_shape_match"] == 0.0


def test_scalar_value_in_question():
    q = "Who is the singer named Andrea?"
    assert scalar_value_in_question(q, ("name",), (("Andrea",),))
    assert not scalar_value_in_question(q, ("name",), (("Bob",),))
    assert not scalar_value_in_question(q, ("name",), ())          # empty
    assert not scalar_value_in_question(q, ("n", "m"), ((1, 2),))  # not scalar


def test_expects_count_and_table_refs():
    assert expects_count("how many films?")
    assert expects_count("total runtime of films")
    assert not expects_count("which films are long")
    assert n_tables_referenced("SELECT * FROM a JOIN b ON a.i=b.i") == 2
    assert n_tables_referenced("SELECT 1") == 0


def test_score_ranking_prefers_consistency_and_copy_match():
    base = extract_features("q", "SELECT a FROM t", executed=True, error=None,
                            columns=("a",), rows=(("x",),))
    noisy = {**base, "consistency": 0.34}
    copy = {**base, "answer_in_question": 1.0}
    assert score(base) > score(noisy)
    assert score(copy) > score(base)
    assert score({**base, "empty": 1.0, "n_rows": 0.0,
                  "rows": ()}) < score(base)   # empty-result penalty applies


def test_verifier_never_imports_gold_module():
    """Structural gold isolation (objective §12): verifier code must not
    reference t2sql.eval — the only module that may read gold SQL."""
    for path in VERIFIER_DIR.glob("*.py"):
        src = path.read_text()
        assert "t2sql.eval" not in src and "from .eval" not in src, path.name


def test_extractor_handles_error_kinds_from_taxonomy():
    for kind in ("syntax", "no_such_table", "timeout", "forbidden"):
        f = extract_features("q", "x", executed=False, error=kind)
        assert f[f"error_{kind}"] == 1.0