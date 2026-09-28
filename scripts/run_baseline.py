#!/usr/bin/env python3
"""Experiment A — baseline: base model + prompt, greedy decoding (design §2).

Runs over a split sample and writes predictions WITHOUT gold SQL: the eval
harness joins predictions to the split by index, so generation code cannot
leak answers (design §4, objective §12).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from t2sql.datasets import db_file, dataset_items, sample_indices
from t2sql.inference.engine import LLM, extract_sql
from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import build_messages, full_question

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--split", default="val", choices=["val", "train", "dev"])
    ap.add_argument("--dataset", default="spider", choices=["spider", "bird"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--fp16", action="store_true",
                    help="load unquantized (needs VRAM headroom)")
    ap.add_argument("--adapter", default=None,
                    help="PEFT adapter dir (experiment B/C/D); default: base model")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = dataset_items(args.dataset, args.split)
    idxs = sample_indices(args.dataset, args.split, args.n, args.seed)
    sample = [(i, items[i]) for i in idxs]

    out_path = Path(args.out).resolve() if args.out else \
        ROOT / "reports" / f"baseline_{args.split}_n{len(sample)}.jsonl"
    llm = LLM(args.model, quantize=not args.fp16, adapter=args.adapter)
    schema_cache: dict[str, str] = {}
    t_start = time.time()
    with out_path.open("w") as fh:
        for i, (index, item) in enumerate(sample):
            db_id = item["db_id"]
            if db_id not in schema_cache:
                schema_cache[db_id] = schema_text(db_file(args.dataset, db_id))
            messages = build_messages(full_question(item), schema_cache[db_id])
            pred = extract_sql(llm.generate(messages, max_new_tokens=args.max_new_tokens))
            # no gold in the output file: eval joins on index and verifies the pair
            fh.write(json.dumps({"index": index, "db_id": db_id,
                                 "question": item["question"],
                                 "prediction": pred}) + "\n")
            print(f"[{i + 1}/{len(sample)}] {db_id}: {pred[:80]}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
    print(f"wrote {out_path.relative_to(ROOT)} in {time.time() - t_start:.0f}s, "
          f"peak VRAM {peak:.2f} GiB")

if __name__ == "__main__":
    main()