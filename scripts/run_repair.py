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
from t2sql.prompts.schema import SchemaCache
from t2sql.prompts.template import build_repair_messages, full_question, variant_for
from t2sql.repair.feedback import build_feedback
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import (extract_features, pool_consistency,
                                      pool_signature_counts, result_signature)
from t2sql.verifier.scorer import max_reachable_score, score

_MIN_MARGIN = 0.1   # val-tuned score margin for accepting a repair (checklist 4.6)


def repair_item(llm, sandbox, schema: str, rec: dict, *, max_iterations: int,
                max_new_tokens: int = 512, weights: dict | None = None,
                few_shot: str = "", variant: int = 0) -> tuple[int, int]:
    """Bounded repair loop for one flagged record (cell D).

    Mutates ``rec`` in place (``repairs`` log, and on replacement
    ``prediction``/``verifier_score``). A repair replaces the current best only
    if it clears the val-tuned score margin AND does not contradict the
    candidate-pool majority signature (checklist 4.6 — replay on val showed
    the bare score gate was net-negative); an identical regeneration stops
    the loop. Returns (attempts, replacements).
    """
    rec["repairs"] = []
    k = len(rec["candidates"])
    sig_counts = pool_signature_counts(c["result_sig"] for c in rec["candidates"])
    # pool-level signal for feedback (gold-free, from the C candidate set).
    # `pool_consistency` below reuses this same pool and denominator, so a
    # repair candidate is scored on exactly the scale the gate compares it to.
    majority_sig, majority_n = (max(sig_counts.items(), key=lambda kv: kv[1])
                                if sig_counts else (None, 0))

    # Recompute the incumbent's score here instead of trusting the
    # rec["verifier_score"] written by the verifier run. The stored value was
    # produced by whatever feature code that run used; scoring the incumbent
    # again through this same function guarantees the acceptance margin
    # compares two candidates computed by identical code, so the gate cannot
    # silently change meaning when the pool is re-scored.
    cur_sql = rec["prediction"]
    res0 = sandbox.execute(cur_sql)
    sig0 = result_signature(res0.columns, res0.rows) if res0.ok else None
    feats0 = extract_features(
        full_question(rec), cur_sql, executed=res0.ok, error=res0.error,
        columns=res0.columns, rows=res0.rows, duration_ms=res0.duration_ms,
        consistency=pool_consistency(sig0, sig_counts, k))
    cur = {"sql": cur_sql, "score": score(feats0, weights),
           "consistency": feats0["consistency"]}
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
            consistency=pool_consistency(sig_new, sig_counts, k))
        new_score = score(feats_new, weights)
        pool_disagrees = bool(
            sig_new and majority_n >= (k + 1) // 2 and majority_sig != sig_new)
        is_better = (new_score >= cur["score"] + _MIN_MARGIN
                     and not pool_disagrees)
        rec["repairs"].append({
            "sql": new_sql, "score": round(new_score, 4),
            "executed": res_new.ok, "error": res_new.error,
            "feedback": feedback, "replaced": is_better,
            "pool_veto": pool_disagrees,
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
    ap.add_argument("--tau", type=float, default=None,
                    help="flag items scoring below this; default is the "
                         "flag_threshold calibrated by tune_verifier.py for "
                         "these weights (required when --weights is given)")
    ap.add_argument("--flag-quantile", type=float, default=None,
                    help="override: calibrate --tau as this quantile of the "
                         "recorded verifier scores (0 = flag everything)")
    ap.add_argument("--max-iterations", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--weights", default=None,
                    help="tuned verifier weights JSON (same run's --weights)")
    ap.add_argument("--prompt-variants", type=int, default=1,
                    help="instruction variants in play (1 = original text)")
    ap.add_argument("--few-shot-file", default=None,
                    help="text file with the few-shot block")
    ap.add_argument("--question-values", action="store_true",
                    help="pick schema example values that match the question")
    ap.add_argument("--schema-samples", action="store_true",
                    help="append row counts + example values to the schema")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = [json.loads(line) for line in Path(args.cfile).open()]
    calibrated_tau = None
    if args.weights:
        raw = json.loads(Path(args.weights).read_text())
        weights = raw.get("tuned_weights", raw)
        calibrated_tau = raw.get("flag_threshold")
    else:
        weights = None

    # Resolve the flag threshold on the same weight scale the scores were
    # produced with. A threshold above the reachable ceiling flags every item,
    # which silently turns bounded repair into repair-everything.
    ceiling = max_reachable_score(weights)
    if args.flag_quantile is not None:
        recorded = sorted(r["verifier_score"] for r in records)
        tau = recorded[min(len(recorded) - 1,
                           max(0, round(args.flag_quantile * (len(recorded) - 1))))]
    elif args.tau is not None:
        tau = args.tau
    elif calibrated_tau is not None:
        tau = calibrated_tau
    else:
        raise SystemExit(
            "no --tau and no calibrated flag_threshold in the weights file; "
            "pass --tau, --flag-quantile, or --weights from tune_verifier.py")
    if tau >= ceiling:
        raise SystemExit(
            f"--tau {tau} is at or above the maximum score reachable under "
            f"these weights ({ceiling}); it would flag every item. Recalibrate "
            f"with tune_verifier.py or lower --tau.")
    print(f"tau={tau} (ceiling {ceiling}); flagged items score below it", flush=True)
    # Loaded only after the threshold check, so a doomed run costs no GPU time.
    llm = LLM(args.model, adapter=args.adapter)
    few_shot = Path(args.few_shot_file).read_text() if args.few_shot_file else ""
    schema_cache = SchemaCache(samples=args.schema_samples,
                               question_values=args.question_values)
    sandboxes: dict[str, Sandbox] = {}
    t_start = time.time()
    n_flagged = n_replaced = n_attempts_total = 0

    with Path(args.out).resolve().open("w") as fh:
        for rec in records:
            db_id = rec["db_id"]
            if db_id not in sandboxes:
                sandboxes[db_id] = Sandbox(db_file(args.dataset, db_id))


            rec["flagged"] = rec["verifier_score"] < tau
            rec["repairs"] = []
            if rec["flagged"]:
                n_flagged += 1
                a, r = repair_item(
                    llm, sandboxes[db_id],
                    schema_cache.get(db_file(args.dataset, db_id),
                                     full_question(rec)),
                    rec,
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