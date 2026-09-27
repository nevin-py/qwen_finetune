#!/usr/bin/env python3
"""Build data/splits/train_sft.json from the training split (idempotent-ish)."""
from __future__ import annotations

import json
from pathlib import Path

from t2sql.training.data import build_training_examples, save_examples

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    examples = build_training_examples(items, ROOT / "data" / "spider" / "database")
    out = ROOT / "data" / "splits" / "train_sft.json"
    save_examples(examples, out)
    print(f"wrote {out.relative_to(ROOT)}: {len(examples)} examples")


if __name__ == "__main__":
    main()