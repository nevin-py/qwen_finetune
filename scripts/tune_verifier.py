#!/usr/bin/env python3
"""Stage 4.1: tune the verifier weights offline (checklist 4.1).

Re-scores every candidate from the validation C run with full gold-free
features (re-executed in the sandbox) and labels each candidate by
execution correctness against val gold (tuning uses val — never dev). A
seeded random search over the weight grid maximises selection accuracy
(argmax-scored candidate is correct); the winner is written to
experiments/verifier_weights.json next to the v1 rule baseline. CPU only,
no model involved.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from t2sql.datasets import db_file
from t2sql.eval import execution
from t2sql.eval.harness import EVAL_MAX_ROWS, EVAL_TIMEOUT_S
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import extract_features, result_signature
from t2sql.verifier.scorer import DEFAULT_WEIGHTS, score

ROOT = Path(__file__).resolve().parents[1]
GRID = {
    "consistency": [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    "answer_in_question": [-0.5, 0.0, 0.2, 0.4, 0.8, 1.2],
    "count_shape_match": [-0.3, 0.0, 0.3, 0.6, 1.0],
    "empty": [-1.5, -1.0, -0.6, -0.4, 0.0],
    "truncated": [-1.0, -0.6, -0.3, 0.0],
    "slow": [-1.0, -0.5, -0.2, 0.0],
    "schema_link": [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
}
KEYS = list(GRID)


def selection_accuracy(items: list[list[dict]], weights: dict) -> tuple[float, int, int]:
    """(items x candidates) -> fraction where argmax-scored candidate is correct."""
    correct = 0
    has_any = 0
    for cands in items:
        if any(c["correct"] for c in cands):
            has_any += 1
        pick = max(range(len(cands)),
                   key=lambda j: score(cands[j]["feats"], weights))
        correct += int(bool(cands[pick]["correct"]))
    return correct / len(items), correct, has_any


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfile", default="reports/C_val_n100.jsonl")
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/verifier_weights.json")
    args = ap.parse_args()

    t0 = time.time()
    records = [json.loads(line) for line in (ROOT / args.cfile).open()]
    val = json.loads((ROOT / "data" / "splits" / "val.json").read_text())

    sandboxes: dict[str, Sandbox] = {}
    gold_ok: dict[tuple[str, str], tuple | None] = {}   # (db,q) -> gold rows sig exec

    items: list[list[dict]] = []
    n_bad_gold = 0
    for rec in records:
        item = val[rec["index"]]
        assert (item["db_id"], item["question"]) == (rec["db_id"], rec["question"])
        db_id = rec["db_id"]
        if db_id not in sandboxes:
            sandboxes[db_id] = Sandbox(db_file("spider", db_id))
        sb = sandboxes[db_id]

        key = (db_id, rec["question"])
        if key not in gold_ok:
            gsql = execution.prepare(item["query"])
            gres = sb.execute(gsql, timeout_s=EVAL_TIMEOUT_S, max_rows=EVAL_MAX_ROWS)
            gold_ok[key] = gres if gres.ok else None
        gres = gold_ok[key]
        if gres is None:
            n_bad_gold += 1
            continue

        cands: list[dict] = []
        results = [sb.execute(c["sql"], timeout_s=EVAL_TIMEOUT_S,
                              max_rows=EVAL_MAX_ROWS) for c in rec["candidates"]]
        sigs = [result_signature(r.columns, r.rows) if r.ok else None
                for r in results]
        counts = {s: sigs.count(s) for s in sigs if s}
        k = len(results)
        for c, r, sig in zip(rec["candidates"], results, sigs):
            feats = extract_features(
                rec["question"], c["sql"], executed=r.ok, error=r.error,
                columns=r.columns, rows=r.rows, truncated=r.truncated,
                duration_ms=r.duration_ms,
                consistency=counts.get(sig, 0) / k if r.ok else 0.0)
            if r.ok:
                psql = execution.prepare(c["sql"])
                pres = sb.execute(psql, timeout_s=EVAL_TIMEOUT_S,
                                  max_rows=EVAL_MAX_ROWS)
                correct = bool(pres.ok and execution.compare(
                    execution.prepare(item["query"]), gres.rows, psql, pres.rows))
            else:
                correct = False
            cands.append({"feats": feats, "correct": correct})
        items.append(cands)

    rng = random.Random(args.seed)
    base = {k: DEFAULT_WEIGHTS[k] for k in KEYS}
    v1_acc, v1_c, _ = selection_accuracy(items, base)
    oracle = sum(1 for c in items if any(x["correct"] for x in c)) / len(items)

    best_w, best_acc = base, v1_acc
    for _ in range(args.trials):
        w = {k: rng.choice(GRID[k]) for k in KEYS}
        w["executed"] = DEFAULT_WEIGHTS["executed"]
        acc, _, _ = selection_accuracy(items, w)
        if acc > best_acc:
            best_acc, best_w = acc, w

    payload = {
        "source": args.cfile,
        "n_items": len(items),
        "n_bad_gold_skipped": n_bad_gold,
        "trials": args.trials,
        "seed": args.seed,
        "v1_weights": base,
        "v1_selection_acc": round(v1_acc, 4),
        "tuned_weights": best_w,
        "tuned_selection_acc": round(best_acc, 4),
        "oracle_pool_acc": round(oracle, 4),
        "generated_s": round(time.time() - t0),
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("n_items", "v1_selection_acc", "tuned_selection_acc",
                       "oracle_pool_acc")}, indent=2))
    print(f"wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
