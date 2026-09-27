#!/usr/bin/env python3
"""Score a predictions file against its gold split by execution accuracy.

Usage:
    python scripts/run_eval.py --preds reports/baseline_val_n20.jsonl
    python scripts/run_eval.py --dataset bird --preds reports/bird_d.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from t2sql.eval.harness import evaluate

ROOT = Path(__file__).resolve().parents[1]


def resolve_split(dataset: str, split: str) -> tuple[list[dict], Path]:
    if dataset == "bird":
        return (json.loads((ROOT / "data" / "bird" / "dev.json").read_text()),
                ROOT / "data" / "bird" / "databases")
    path = ROOT / "data" / "splits" / f"{split}.json"
    if not path.exists():
        path = ROOT / "data" / "spider" / f"{split}.json"
    return json.loads(path.read_text()), ROOT / "data" / "spider" / "database"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--dataset", default="spider", choices=["spider", "bird"])
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    preds_path = Path(args.preds)
    predictions = [json.loads(line) for line in preds_path.open()]
    items, db_dir = resolve_split(args.dataset, args.split)

    report = evaluate(items, predictions, db_dir)
    out_path = ROOT / "reports" / f"eval_{preds_path.stem}.json"
    out_path.write_text(json.dumps(report.to_json(), indent=2))
    print(json.dumps(report.to_json(), indent=2))
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()