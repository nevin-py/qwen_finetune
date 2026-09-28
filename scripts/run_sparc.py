#!/usr/bin/env python3
"""SParC multi-turn evaluation runner (Phase 9, design §3/§4).

Reads ONLY the gold-free prompt contexts (dev_prompts.json): each turn's
utterance plus the gold history of previous turns (official SParC protocol).
Current-turn gold SQL lives in a separate eval-only file that this script
never opens. Predictions are written without gold; the harness joins by index.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from t2sql.inference.engine import LLM, extract_sql
from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import build_dialogue_messages

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=0, help="0 = all turns")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--schema-samples", action="store_true",
                    help="append row counts + example values to the schema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = json.loads((ROOT / "data" / "sparc" / "dev_prompts.json").read_text())
    if args.n:
        rows = rows[:args.n]

    llm = LLM(args.model, adapter=args.adapter)
    schema_cache: dict[str, str] = {}
    out_path = Path(args.out).resolve()
    t_start = time.time()

    with out_path.open("w") as fh:
        for pos, row in enumerate(rows, 1):
            db_id = row["db_id"]
            if db_id not in schema_cache:
                db_file = ROOT / "data" / "spider" / "database" / db_id / f"{db_id}.sqlite"
                schema_cache[db_id] = schema_text(db_file, samples=args.schema_samples)
            messages = build_dialogue_messages(
                row["question"], schema_cache[db_id], row["history"])
            pred = extract_sql(llm.generate(messages, max_new_tokens=args.max_new_tokens))
            fh.write(json.dumps({
                "index": row["index"], "db_id": db_id,
                "question": row["question"], "prediction": pred,
            }) + "\n")
            if pos % 100 == 0:
                print(f"[{pos}/{len(rows)}] {time.time()-t_start:.0f}s", flush=True)

    print(f"wrote {out_path} ({len(rows)} turns, {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()