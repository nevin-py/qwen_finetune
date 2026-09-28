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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stratify", action="store_true",
                    help="v2: proportional per-db val carve (changes membership)")
    args = ap.parse_args()
    items = json.loads((DATA / "spider" / "train_spider.json").read_text())
    val_idx = (stratified_val_indices(items) if args.stratify
               else canonical_val_indices(items))
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    out = DATA / "splits"
    out.mkdir(exist_ok=True)
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps(val))
    print(f"train={len(train)} val={len(val)} seed={SEED} "
          f"({'stratified by db_id' if args.stratify else 'canonical'} , "
          f"{len({x['db_id'] for x in val})} dbs in val)")


if __name__ == "__main__":
    main()
