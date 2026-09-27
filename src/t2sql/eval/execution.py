"""Execution-accuracy comparison, ported from the official Spider evaluator.

Faithful port of ``test-suite-sql-eval/exec_eval.py`` semantics (original
database only — the test-suite neighbor DBs are the stricter leaderboard
variant and are not required for pairwise gold/pred agreement):

    postprocess  ``> =`` -> ``>=`` family repairs
    remove_distinct  DISTINCT stripped from gold and pred (sqlparse tokens)
    order matters iff gold contains ``order by``
    result_eq    empty/len/col checks, in-row unordering, column-permutation
                 search, multiset equality (bag semantics)

The official code samples rows without a seed when pruning column
permutations; we seed the sampler for reproducibility. Pruning only removes
permutations inconsistent with observed values, so a truly matching
permutation always survives — deterministic and official-equivalent.
"""
from __future__ import annotations

import random
import re
from collections import defaultdict
from itertools import product

import sqlparse

_ORDER_BY = "order by"


def postprocess(query: str) -> str:
    return query.replace("> =", ">=").replace("< =", "<=").replace("! =", "!=")


def remove_distinct(sql: str) -> str:
    """Identical to the official implementation (sqlparse token filter)."""
    toks = [t.value for t in sqlparse.parse(sql)[0].flatten()]
    return "".join(t for t in toks if t.lower() != "distinct")


def replace_cur_year(query: str) -> str:
    return re.sub(r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020",
                  query, flags=re.IGNORECASE)


def _unorder_row(row: tuple) -> tuple:
    return tuple(sorted(row, key=lambda x: str(x) + str(type(x))))


def _quick_rej(r1: list, r2: list, order_matters: bool) -> bool:
    s1, s2 = [_unorder_row(r) for r in r1], [_unorder_row(r) for r in r2]
    return s1 == s2 if order_matters else set(s1) == set(s2)


def _multiset_eq(l1: list, l2: list) -> bool:
    if len(l1) != len(l2):
        return False
    counts: dict = defaultdict(int)
    for e in l1:
        counts[e] += 1
    for e in l2:
        counts[e] -= 1
        if counts[e] < 0:
            return False
    return True


def _candidate_perms(tab1_sets_by_columns: list, result2: list, rng: random.Random):
    num_cols = len(result2[0])
    perm_constraints = [{i for i in range(num_cols)} for _ in range(num_cols)]
    if num_cols <= 3:
        return product(*perm_constraints)
    for _ in range(20):  # official constraint sampling, seeded here
        random_row = rng.choice(result2)
        for tab1_col in range(num_cols):
            for tab2_col in set(perm_constraints[tab1_col]):
                if random_row[tab2_col] not in tab1_sets_by_columns[tab1_col]:
                    perm_constraints[tab1_col].discard(tab2_col)
    return product(*perm_constraints)


def result_eq(r1: list, r2: list, *, order_matters: bool, seed: int = 0) -> bool:
    if not r1 and not r2:
        return True
    if len(r1) != len(r2):
        return False
    num_cols = len(r1[0])
    if len(r2[0]) != num_cols:
        return False
    if not _quick_rej(r1, r2, order_matters):
        return False
    rng = random.Random(seed)
    tab1_sets = [{row[i] for row in r1} for i in range(num_cols)]
    for perm in _candidate_perms(tab1_sets, r2, rng):
        if len(perm) != len(set(perm)):
            continue
        r2_perm = r2 if num_cols == 1 else [tuple(p[i] for i in perm) for p in r2]
        if order_matters:
            if r1 == r2_perm:
                return True
        elif set(r1) == set(r2_perm) and _multiset_eq(r1, r2_perm):
            return True
    return False


def prepare(sql: str) -> str:
    """What the official evaluator actually executes (idempotent)."""
    return replace_cur_year(remove_distinct(postprocess(sql)))


def compare(gold_sql: str, gold_rows: list, pred_sql: str, pred_rows: list) -> bool:
    """Official-protocol denotation comparison for one executed pair."""
    gold = prepare(gold_sql)
    order_matters = _ORDER_BY in gold.lower()
    return result_eq(gold_rows, pred_rows, order_matters=order_matters)