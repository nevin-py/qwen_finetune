#!/usr/bin/env python3
"""Deterministic 90/10 carve of Spider's official train split -> data/splits/.

The official dev split stays untouched (locked test). All tuning — including
prompt iteration, verifier thresholding and repair decisions — uses only
``data/splits/val.json``.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VAL_N = 700
SEED = 13


def main() -> None:
    items = json.loads((DATA / "spider" / "train_spider.json").read_text())
    rng = random.Random(SEED)
    order = list(range(len(items)))
    rng.shuffle(order)
    val_idx = set(order[:VAL_N])
    train = [x for i, x in enumerate(items) if i not in val_idx]
    val = [x for i, x in enumerate(items) if i in val_idx]
    out = DATA / "splits"
    out.mkdir(exist_ok=True)
    (out / "train.json").write_text(json.dumps(train))
    (out / "val.json").write_text(json.dumps(val))
    print(f"train={len(train)} val={len(val)} seed={SEED}")


if __name__ == "__main__":
    main()