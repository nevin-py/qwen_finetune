"""Phase 11: failure-mode taxonomy classification."""

from t2sql.eval.error_taxonomy import classify, components

GOLD = "SELECT a FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3"


def test_correct_and_exec_error_passthrough():
    assert classify("s", GOLD, correct=True) == ("correct", [])
    assert classify("s", GOLD, pred_error="syntax")[0] == "syntax"
    assert classify("s", GOLD, pred_error="no_such_column")[0] == "no_such_column"


def test_single_component_categories():
    cases = [
        # (pred, gold, expected category)
        ("SELECT a FROM t WHERE x = 2 GROUP BY a ORDER BY a LIMIT 3", GOLD, "filter"),
        ("SELECT a FROM t GROUP BY a ORDER BY a LIMIT 3", GOLD, "filter"),
        ("SELECT a FROM t WHERE x = 1 ORDER BY a LIMIT 3", GOLD, "group_by"),
        ("SELECT a FROM t WHERE x = 1 GROUP BY a LIMIT 3", GOLD, "ordering"),
        ("SELECT a FROM t WHERE x = 1 GROUP BY a ORDER BY a", GOLD, "limit"),
        ("SELECT a FROM u WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3", GOLD, "tables"),
        ("SELECT COUNT(a) FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3", GOLD, "agg"),
        ("SELECT a FROM t WHERE x = 1 GROUP BY a HAVING COUNT(a) > 2 ORDER BY a LIMIT 3",
         "SELECT COUNT(a) FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3", "having"),
        ("SELECT (SELECT 1) FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3",
         GOLD, "subquery"),
        ("SELECT DATE(a) FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3",
         GOLD, "date_function"),
        ("SELECT a FROM t WHERE x IS NOT NULL GROUP BY a ORDER BY a LIMIT 3",
         "SELECT a FROM t WHERE x IS NULL GROUP BY a ORDER BY a LIMIT 3", "null_handling"),
        ("SELECT ROUND(a) FROM t WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3",
         GOLD, "function"),
        ("SELECT tt.a FROM t AS tt WHERE x = 1 GROUP BY a ORDER BY a LIMIT 3",
         GOLD, "semantic_mismatch"),
    ]
    for pred, gold, expected in cases:
        cat, diffs = classify(pred, gold)
        assert cat == expected, f"{pred!r} -> {cat} (diffs={diffs}), expected {expected}"


def test_multiple_clauses():
    pred = "SELECT b FROM u WHERE x = 9 GROUP BY a ORDER BY a LIMIT 3"
    cat, diffs = classify(pred, GOLD)
    assert cat == "multiple_clauses" and len(diffs) >= 2


def test_join_condition_category():
    gold = "SELECT a.x FROM a JOIN b ON a.id = b.id"
    pred = "SELECT a.x FROM a JOIN b ON a.k = b.k"
    assert classify(pred, gold)[0] == "join_condition"


def test_components_extraction_sanity():
    c = components("SELECT COUNT(*) FROM t JOIN u ON t.id = u.id "
                   "WHERE x IN (SELECT y FROM z) GROUP BY t.id HAVING COUNT(*) > 1 "
                   "ORDER BY x DESC LIMIT 5")
    assert c["tables"] == {"t", "u", "z"}
    assert c["agg"] == {"count"} and c["subquery"] == 1
    assert c["group_by"] and c["having"] and c["ordering"] and c["limit"]
    assert c["join_condition"]


def test_null_spelling_does_not_pollute_filter():
    a = components("SELECT a FROM t WHERE IFNULL(x, 0) = 1")
    b = components("SELECT a FROM t WHERE COALESCE(x, 0) = 1")
    assert a["filter"] == b["filter"]            # structural span identical
    assert a["null_handling"] != b["null_handling"]