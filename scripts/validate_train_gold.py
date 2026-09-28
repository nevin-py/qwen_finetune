#!/usr/bin/env python3
"""Stage 0.1: execution-validate the training gold (checklist 0.1).

Runs every (question, gold SQL) pair from ``data/splits/train.json`` in the
read-only sandbox and writes ``data/splits/train_quarantine.json`` for the
failures, so training never supervises an unrunnable target. Touches only
the training split — never dev.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from t2sql.datasets import db_file
from t2sql.sandbox.executor import Sandbox

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    sandboxes: dict[str, Sandbox] = {}
    bad: list[dict] = []
    t0 = time.time()
    for pos, item in enumerate(items, 1):
        db_id = item["db_id"]
        if db_id not in sandboxes:
            sandboxes[db_id] = Sandbox(db_file("spider", db_id))
        res = sandboxes[db_id].execute(item["query"], timeout_s=15.0)
        if not res.ok:
            bad.append({
                "db_id": db_id,
                "question": item["question"],
                "query": item["query"],
                "error": res.error,
                "error_message": res.error_message,
            })
        if pos % 1500 == 0:
            print(f"[{pos}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    out = ROOT / "data" / "splits" / "train_quarantine.json"
    out.write_text(json.dumps(bad, indent=2))
    print(f"validated {len(items)} | failed {len(bad)} -> "
          f"{out.relative_to(ROOT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
