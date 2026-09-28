#!/usr/bin/env python3
"""Experiment C — verifier-driven candidate selection (design §6, §2).

Per sampled question: generate k candidates (1 greedy + k-1 sampled),
execute every candidate in the sandbox, score gold-free features, and let
the verifier pick the winner. The greedy candidate (identical to cell B's
output) is part of the pool, so the verifier can always fall back to it.

All candidates + scores are recorded so self-consistency majority and other
selection rules can be compared offline without rerunning the GPU (Phase 11).
Gold SQL never enters this script (objective §12).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from t2sql.inference.engine import LLM, extract_sql
from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import build_messages
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import extract_features, result_signature
from t2sql.verifier.scorer import score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--split", default="val", choices=["val", "train", "dev"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=5, help="candidates per question")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import random

    split_path = ROOT / "data" / "splits" / f"{args.split}.json"
    if not split_path.exists():
        split_path = ROOT / "data" / "spider" / f"{args.split}.json"
    items = json.loads(split_path.read_text())
    idxs = random.Random(args.seed).sample(range(len(items)), min(args.n, len(items)))

    llm = LLM(args.model, adapter=args.adapter)
    schema_cache: dict[str, str] = {}
    sandboxes: dict[str, Sandbox] = {}
    out_path = Path(args.out).resolve()
    t_start = time.time()

    with out_path.open("w") as fh:
        for pos, i in enumerate(idxs, 1):
            item = items[i]
            db_id = item["db_id"]
            if db_id not in schema_cache:
                db_file = ROOT / "data" / "spider" / "database" / db_id / f"{db_id}.sqlite"
                schema_cache[db_id] = schema_text(db_file)
                sandboxes[db_id] = Sandbox(db_file)
            messages = build_messages(item["question"], schema_cache[db_id])

            # candidate 0 = greedy (bit-identical to cell B); rest sampled
            candidates = [llm.generate(messages, max_new_tokens=args.max_new_tokens)]
            for _ in range(args.k - 1):
                candidates.append(llm.generate(
                    messages, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature))

            executed = []  # (sql, ExecResult, sig)
            for sql in candidates:
                res = sandboxes[db_id].execute(extract_sql(sql))
                sig = result_signature(res.columns, res.rows) if res.ok else None
                executed.append((extract_sql(sql), res, sig))

            sig_counts: dict[str, int] = {}
            for _, res, sig in executed:
                if res.ok:
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1

            records, feats = [], []
            for sql, res, sig in executed:
                consistency = sig_counts.get(sig, 0) / len(executed) if res.ok else 0.0
                f = extract_features(
                    item["question"], sql, executed=res.ok, error=res.error,
                    columns=res.columns, rows=res.rows, truncated=res.truncated,
                    duration_ms=res.duration_ms, consistency=consistency)
                feats.append(f)
                records.append({
                    "sql": sql, "score": round(score(f), 4),
                    "executed": res.ok, "error": res.error,
                    "result_sig": sig, "consistency": round(consistency, 3),
                })

            best = max(range(len(records)), key=lambda j: records[j]["score"])
            fh.write(json.dumps({
                "index": i, "db_id": db_id, "question": item["question"],
                "prediction": executed[best][0],
                "verifier_score": records[best]["score"],
                "selected": best, "candidates": records,
            }) + "\n")
            fh.flush()
            if pos % 10 == 0:
                print(f"[{pos}/{len(idxs)}] {time.time()-t_start:.0f}s", flush=True)

    print(f"wrote {out_path} ({len(idxs)} questions, k={args.k}, "
          f"{time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()