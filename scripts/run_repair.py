#!/usr/bin/env python3
"""Experiment D — bounded, verifier-gated repair (design §6, §12).

Loads the verifier cell's candidates (C), flags items whose best score is
below --tau, and runs at most --max-iterations repair rounds per flagged item.
The same rule scorer is the sole judge: a repair replaces the current best
only with a strictly higher score. Unflagged items keep their C prediction.

Gold SQL never enters this script (objective §12).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from t2sql.datasets import db_file
from t2sql.inference.engine import LLM, extract_sql
from t2sql.prompts.schema import schema_text
from t2sql.prompts.template import build_repair_messages, full_question
from t2sql.repair.feedback import build_feedback
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import extract_features, result_signature
from t2sql.verifier.scorer import score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--cfile", required=True, help="verifier cell output (C)")
    ap.add_argument("--dataset", default="spider", choices=["spider", "bird"])
    ap.add_argument("--tau", type=float, default=3.0,
                    help="flag items whose best score is below this")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.cfile).open()]
    llm = LLM(args.model, adapter=args.adapter)
    schema_cache: dict[str, str] = {}
    sandboxes: dict[str, Sandbox] = {}
    t_start = time.time()
    n_flagged = n_replaced = n_attempts_total = 0

    with Path(args.out).resolve().open("w") as fh:
        for rec in records:
            db_id = rec["db_id"]
            if db_id not in schema_cache:
                f = db_file(args.dataset, db_id)
                schema_cache[db_id] = schema_text(f)
                sandboxes[db_id] = Sandbox(f)

            best = max(rec["candidates"], key=lambda c: c["score"])
            rec["flagged"] = rec["verifier_score"] < args.tau
            rec["repairs"] = []
            if rec["flagged"]:
                n_flagged += 1
                k = len(rec["candidates"])
                sig_counts: dict[str, int] = {}
                for c in rec["candidates"]:
                    if c["result_sig"]:
                        sig_counts[c["result_sig"]] = sig_counts.get(c["result_sig"], 0) + 1

                cur = {"sql": rec["prediction"], "score": rec["verifier_score"],
                       "consistency": best["consistency"]}
                for _ in range(args.max_iterations):
                    res = sandboxes[db_id].execute(cur["sql"])
                    feats_cur = extract_features(
                        rec["question"], cur["sql"], executed=res.ok, error=res.error,
                        columns=res.columns, rows=res.rows,
                        duration_ms=res.duration_ms, consistency=cur["consistency"])
                    feedback = build_feedback(feats_cur, res.error_message)
                    new_sql = extract_sql(llm.generate(
                        build_repair_messages(full_question(rec), schema_cache[db_id],
                                              cur["sql"], feedback),
                        max_new_tokens=args.max_new_tokens))
                    if new_sql == cur["sql"]:
                        break  # loop guard: no progress
                    res_new = sandboxes[db_id].execute(new_sql)
                    sig_new = result_signature(res_new.columns, res_new.rows) if res_new.ok else None
                    feats_new = extract_features(
                        rec["question"], new_sql, executed=res_new.ok, error=res_new.error,
                        columns=res_new.columns, rows=res_new.rows,
                        duration_ms=res_new.duration_ms,
                        consistency=((sig_counts.get(sig_new, 0) + 1) / (k + 1)) if res_new.ok else 0.0)
                    new_score = score(feats_new)
                    replaced = new_score > cur["score"]
                    rec["repairs"].append({
                        "sql": new_sql, "score": round(new_score, 4),
                        "executed": res_new.ok, "error": res_new.error,
                        "feedback": feedback, "replaced": replaced,
                    })
                    n_attempts_total += 1
                    if replaced:
                        cur = {"sql": new_sql, "score": new_score,
                               "consistency": feats_new["consistency"]}
                        n_replaced += 1
                rec["prediction"] = cur["sql"]
                rec["verifier_score"] = round(cur["score"], 4)

            fh.write(json.dumps(rec) + "\n")

    print(f"wrote {Path(args.out).resolve()} | flagged {n_flagged}/{len(records)} | "
          f"{n_attempts_total} repair attempts, {n_replaced} replacements | "
          f"{time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()