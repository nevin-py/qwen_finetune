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
    in place with a seeded RNG. Deterministic for a fixed item order."""
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, x in enumerate(items):
        groups.setdefault(x["db_id"], []).append(i)
    total = len(items)
    quota = {d: len(g) * val_n / total for d, g in groups.items()}
    base = {d: int(q) for d, q in quota.items()}
    order = sorted(groups, key=lambda d: (-(quota[d] - base[d]), d))
    for d in order[: val_n - sum(base.values())]:
        base[d] += 1
    val_idx: set[int] = set()
    for d, g in groups.items():
        rng.shuffle(g)
        val_idx.update(g[: base[d]])
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
