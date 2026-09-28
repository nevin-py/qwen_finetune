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
from t2sql.prompts.template import build_repair_messages, full_question, variant_for
from t2sql.repair.feedback import build_feedback
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import extract_features, result_signature
from t2sql.verifier.scorer import score


def repair_item(llm, sandbox, schema: str, rec: dict, *, max_iterations: int,
                max_new_tokens: int = 512, weights: dict | None = None,
                few_shot: str = "", variant: int = 0) -> tuple[int, int]:
    """Bounded repair loop for one flagged record (cell D).

    Mutates ``rec`` in place (``repairs`` log, and on replacement
    ``prediction``/``verifier_score``). The verifier score is the sole judge:
    a repair replaces the current best only with a strictly higher score;
    an identical regeneration stops the loop. Returns (attempts, replacements).
    """
    best = max(rec["candidates"], key=lambda c: c["score"])
    rec["repairs"] = []
    k = len(rec["candidates"])
    sig_counts: dict[str, int] = {}
    for c in rec["candidates"]:
        if c["result_sig"]:
            sig_counts[c["result_sig"]] = sig_counts.get(c["result_sig"], 0) + 1
    # pool-level signal for feedback (gold-free, from the C candidate set)
    majority_sig, majority_n = (max(sig_counts.items(), key=lambda kv: kv[1])
                                if sig_counts else (None, 0))

    cur = {"sql": rec["prediction"], "score": rec["verifier_score"],
           "consistency": best["consistency"]}
    attempts = replaced = 0
    for _ in range(max_iterations):
        res = sandbox.execute(cur["sql"])
        feats_cur = extract_features(
            full_question(rec), cur["sql"], executed=res.ok, error=res.error,
            columns=res.columns, rows=res.rows,
            duration_ms=res.duration_ms, consistency=cur["consistency"])
        cur_sig = result_signature(res.columns, res.rows) if res.ok else None
        feats_cur["pool_disagrees"] = float(bool(
            cur_sig and majority_n >= (k + 1) // 2 and majority_sig != cur_sig))
        feats_cur["pool_majority_n"], feats_cur["pool_k"] = majority_n, k
        feedback = build_feedback(feats_cur, res.error_message)
        new_sql = extract_sql(llm.generate(
            build_repair_messages(full_question(rec), schema, cur["sql"], feedback,
                                  few_shot=few_shot, variant=variant),
            max_new_tokens=max_new_tokens))
        if new_sql == cur["sql"]:
            break  # loop guard: no progress
        res_new = sandbox.execute(new_sql)
        sig_new = result_signature(res_new.columns, res_new.rows) if res_new.ok else None
        feats_new = extract_features(
            full_question(rec), new_sql, executed=res_new.ok, error=res_new.error,
            columns=res_new.columns, rows=res_new.rows,
            duration_ms=res_new.duration_ms,
            consistency=((sig_counts.get(sig_new, 0) + 1) / (k + 1)) if res_new.ok else 0.0)
        new_score = score(feats_new, weights)
        is_better = new_score > cur["score"]
        rec["repairs"].append({
            "sql": new_sql, "score": round(new_score, 4),
            "executed": res_new.ok, "error": res_new.error,
            "feedback": feedback, "replaced": is_better,
        })
        attempts += 1
        if is_better:
            cur = {"sql": new_sql, "score": new_score,
                   "consistency": feats_new["consistency"]}
            replaced += 1
    rec["prediction"] = cur["sql"]
    rec["verifier_score"] = round(cur["score"], 4)
    return attempts, replaced


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--cfile", required=True, help="verifier cell output (C)")
    ap.add_argument("--dataset", default="spider", choices=["spider", "bird"])
    ap.add_argument("--tau", type=float, default=3.0,
                    help="flag items whose best score is below this")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--weights", default=None,
                    help="tuned verifier weights JSON (same run's --weights)")
    ap.add_argument("--prompt-variants", type=int, default=1,
                    help="instruction variants in play (1 = original text)")
    ap.add_argument("--few-shot-file", default=None,
                    help="text file with the few-shot block")
    ap.add_argument("--schema-samples", action="store_true",
                    help="append row counts + example values to the schema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = [json.loads(line) for line in Path(args.cfile).open()]
    llm = LLM(args.model, adapter=args.adapter)
    weights = json.loads(Path(args.weights).read_text()) if args.weights else None
    few_shot = Path(args.few_shot_file).read_text() if args.few_shot_file else ""
    schema_cache: dict[str, str] = {}
    sandboxes: dict[str, Sandbox] = {}
    t_start = time.time()
    n_flagged = n_replaced = n_attempts_total = 0

    with Path(args.out).resolve().open("w") as fh:
        for rec in records:
            db_id = rec["db_id"]
            if db_id not in schema_cache:
                f = db_file(args.dataset, db_id)
                schema_cache[db_id] = schema_text(f, samples=args.schema_samples)
                sandboxes[db_id] = Sandbox(f)

            rec["flagged"] = rec["verifier_score"] < args.tau
            rec["repairs"] = []
            if rec["flagged"]:
                n_flagged += 1
                a, r = repair_item(llm, sandboxes[db_id], schema_cache[db_id], rec,
                                   max_iterations=args.max_iterations,
                                   max_new_tokens=args.max_new_tokens,
                                   weights=weights, few_shot=few_shot,
                                   variant=variant_for(rec["question"],
                                                       args.prompt_variants))
                n_attempts_total += a
                n_replaced += r

            fh.write(json.dumps(rec) + "\n")

    print(f"wrote {Path(args.out).resolve()} | flagged {n_flagged}/{len(records)} | "
          f"{n_attempts_total} repair attempts, {n_replaced} replacements | "
          f"{time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()