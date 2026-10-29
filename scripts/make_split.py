#!/usr/bin/env python3
"""Deterministic 90/10 carve of Spider's official train split -> data/splits/.

Stratified by database (checklist 0.4, opt-in via --stratify): every db_id
contributes to the val carve in proportion to its size (largest-remainder
allocation to exactly VAL_N), so the val distribution of databases cannot
drift by chance. The default mode reproduces the v1 carve (seeded shuffle)
byte-for-byte, so existing val artifacts stay valid until the split is
explicitly rebuilt at a training kickoff. The official dev split stays
untouched (locked test). All tuning uses only ``data/splits/val.json``.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VAL_N = 700
 
 
SEED = 13
 
 
def norm_sql(sql: str) -> str:
    """Case/whitespace-folded SQL, for leak detection across splits."""
    return " ".join(sql.split()).rstrip(";").strip().lower()


def stratified_val_indices(items: list[dict], val_n: int = VAL_N,
                           seed: int = SEED) -> set[int]:
    """Val indices: proportional per-db quota (largest remainder), shuffled
    in place with a seeded RNG. Duplicate (db_id, question) clusters are
    assigned atomically so train/val can never share a key. Deterministic
    for a fixed item order; always returns exactly ``val_n`` indices."""
    rng = random.Random(seed)
    key_of = lambda x: (x["db_id"], x["question"].strip().lower())

    # cluster indices by key, per db, first-seen order
    clusters: dict[str, list[list[int]]] = {}
    by_key: dict[tuple[str, str], list[int]] = {}
    for i, x in enumerate(items):
        k = key_of(x)
        if k not in by_key:
            by_key[k] = []
            clusters.setdefault(x["db_id"], []).append(by_key[k])
        by_key[k].append(i)

    total = len(items)
    quota = {d: sum(len(c) for c in g) * val_n / total
             for d, g in clusters.items()}
    base = {d: int(q) for d, q in quota.items()}
    # largest-remainder top-up to exactly val_n
    need = val_n - sum(base.values())
    for d in sorted(clusters, key=lambda d: (-(quota[d] - base[d]), d)):
        if need <= 0:
            break
        if quota[d] > base[d]:
            base[d] += 1
            need -= 1

    val_idx: set[int] = set()
    for d, g in clusters.items():
        g = list(g)
        rng.shuffle(g)
        remaining = base[d]
        for c in g:
            if len(c) <= remaining:
                val_idx.update(c)
                remaining -= len(c)
    # exact top-up from the leftover pool (size-1 clusters make this exact)
    if len(val_idx) < val_n:
        pool = [c for g in clusters.values() for c in g if not set(c) & val_idx]
        rng.shuffle(pool)
        for c in pool:
            if len(val_idx) >= val_n:
                break
            if len(val_idx) + len(c) <= val_n:
                val_idx.update(c)
    assert len(val_idx) == val_n, (len(val_idx), val_n)
    return val_idx


def canonical_val_indices(items: list[dict], val_n: int = VAL_N,
                          seed: int = SEED) -> set[int]:
    """v1 carve: seeded shuffle, first VAL_N (kept bit-compatible)."""
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    return set(order[:val_n])
 
def holdout_db_indices(items: list[dict], n_dbs: int,
                       seed: int = SEED) -> set[int]:
    """Indices belonging to ``n_dbs`` whole databases, removed from training.

    A validation split drawn from databases the model trained on cannot
    measure generalization: every one of those schemas is already memorized.
    The locked Spider and BIRD dev sets both consist entirely of unseen
    databases, so tuning has to happen on unseen databases too. Databases are
    chosen by seeded shuffle (reproducible) and taken whole, so no question
    from a held-out database can reach the training set.
    """
    rng = random.Random(seed)
    dbs = sorted({x["db_id"] for x in items})
    chosen = set(rng.sample(dbs, min(n_dbs, len(dbs))))
    return {i for i, x in enumerate(items) if x["db_id"] in chosen}



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stratify", action="store_true",
                    help="v2: proportional per-db val carve (changes membership)")
    ap.add_argument("--holdout-dbs", type=int, default=0,
                    help="also carve these whole databases out of training "
                        "into holdout.json, so tuning can be measured on "
                        "unseen schemas like the locked benchmarks")
    args = ap.parse_args()
    items = json.loads((DATA / "spider" / "train_spider.json").read_text())
    val_idx = (stratified_val_indices(items) if args.stratify
               else canonical_val_indices(items))
    hold_idx: set[int] = set()
    if args.holdout_dbs > 0:
        # Carved from the non-val pool so no holdout question is also a val
        # question. The stratified val carve still takes a quota from every
        # database, so holdout databases do appear in val; that is harmless
        # because val drives the eval-loss diagnostic, while the guarantee
        # that matters for tuning is that no holdout database is trained on.
        free = [i for i in range(len(items)) if i not in val_idx]
        local = {j: i for j, i in enumerate(free)}
        picked = holdout_db_indices([items[i] for i in free], args.holdout_dbs)
        hold_idx = {local[j] for j in picked}

    train = [x for i, x in enumerate(items) if i not in val_idx | hold_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    out = DATA / "splits"
    out.mkdir(exist_ok=True)
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps(val))
    print(f"train={len(train)} val={len(val)} seed={SEED} "
          f"({'stratified by db_id' if args.stratify else 'canonical'} , "
          f"{len({x['db_id'] for x in val})} dbs in val)")
    if hold_idx:
        hold = [x for i, x in enumerate(items) if i in hold_idx]
        (out / "holdout.json").write_text(json.dumps(hold))
        # Assert the properties the probe depends on rather than reporting a
        # weaker one: no holdout database is trainable, and no holdout gold
        # query appears verbatim in the training set.
        train_dbs = {x["db_id"] for x in train}
        overlap = train_dbs & {x["db_id"] for x in hold}
        assert not overlap, f"holdout databases leaked into training: {overlap}"
        train_sql = {(x["db_id"], norm_sql(x["query"])) for x in train}
        leaked = [x for x in hold
                  if (x["db_id"], norm_sql(x["query"])) in train_sql]
        assert not leaked, (f"{len(leaked)} holdout gold queries also occur "
                            f"in training, e.g. {leaked[:1]}")
        print(f"holdout={len(hold)} over {len({x['db_id'] for x in hold})} "
              f"databases; 0 databases and 0 gold queries shared with train")


if __name__ == "__main__":
    main()
