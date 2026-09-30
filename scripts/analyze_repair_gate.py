#!/usr/bin/env python3
"""Stage 4.6 analysis: which gold-free acceptance gate for repair (checklist 4.6).

The score-improvement gate of cell D is net-negative on the v2 val run
(6 replacements, -2 correct). This script replays the recorded replacements
against val gold and reports, per candidate gate rule, what D accuracy would
have been. Rules examined (all gold-free):
  - accept all (v1 score gate)
  - margin >= delta (score gain must clear delta)
  - pool-agreement veto (new result must match the pool majority signature)
  - combinations
Reads val gold (tuning side, never dev). CPU only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from t2sql.datasets import db_file
from t2sql.eval import execution
from t2sql.eval.harness import EVAL_MAX_ROWS, EVAL_TIMEOUT_S
from t2sql.sandbox.executor import Sandbox
from t2sql.verifier.features import result_signature

ROOT = Path(__file__).resolve().parents[1]
MARGINS = (0.0, 0.1, 0.25, 0.5, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfile", default="reports/C_val_n100_k5t08w.jsonl")
    ap.add_argument("--dfile", default="reports/D_val_n100_tau3.0.jsonl")
    args = ap.parse_args()

    val = json.loads((ROOT / "data" / "splits" / "val.json").read_text())
    C = {r["index"]: r for r in
         (json.loads(l) for l in (ROOT / args.cfile).open())}
    D = [json.loads(l) for l in (ROOT / args.dfile).open()]
    sand: dict[str, Sandbox] = {}

    def sandbox(db: str) -> Sandbox:
        if db not in sand:
            sand[db] = Sandbox(db_file("spider", db))
        return sand[db]

    def executes(db: str, sql: str):
        return sandbox(db).execute(sql, timeout_s=EVAL_TIMEOUT_S,
                                   max_rows=EVAL_MAX_ROWS)

    def correct(db: str, idx: int, pred: str) -> bool:
        item = val[idx]
        g = execution.prepare(item["query"])
        gr = executes(db, g)
        p = execution.prepare(pred)
        pr = executes(db, p)
        return bool(pr.ok and execution.compare(g, gr.rows, p, pr.rows))

    events = []      # per replaced event: flips, margin, veto signals
    finals: dict[int, tuple[bool, bool]] = {}   # idx -> (acc_all, ...)
    c_acc = d_acc_all = 0
    for rec in D:
        idx = rec["index"]
        db = rec["db_id"]
        before = C[idx]["prediction"]
        c_ok = correct(db, idx, before)
        c_acc += c_ok
        pool_sigs = Counter(c["result_sig"] for c in C[idx]["candidates"]
                            if c.get("result_sig"))
        majority_sig, majority_n = (pool_sigs.most_common(1)[0]
                                    if pool_sigs else (None, 0))
        cur_ok, cur_sql, cur_score = c_ok, before, C[idx]["verifier_score"]
        for rep in rec.get("repairs", []):
            if not rep.get("replaced"):
                continue
            res = executes(db, rep["sql"])
            sig = result_signature(res.columns, res.rows) if res.ok else None
            after_ok = correct(db, idx, rep["sql"])
            events.append({
                "idx": idx,
                "margin": rep["score"] - cur_score,
                "helpful": (not cur_ok) and after_ok,
                "harmful": cur_ok and (not after_ok),
                "res_ok": res.ok,
                "pool_veto": bool(sig and majority_n >= (len(C[idx]["candidates"]) + 1) // 2
                                  and sig != majority_sig),
            })
            cur_ok, cur_sql, cur_score = after_ok, rep["sql"], rep["score"]
        d_acc_all += cur_ok

    n_rep = len(events)
    print(f"C accuracy: {c_acc/len(D):.2f}  |  D (accept all): {d_acc_all/len(D):.2f}"
          f"  |  replacements: {n_rep}")
    print(f"  helpful: {sum(e['helpful'] for e in events)}  "
          f"harmful: {sum(e['harmful'] for e in events)}  "
          f"pool-veto'd: {sum(e['pool_veto'] for e in events)}")
    for e in events:
        print(f"  idx {e['idx']}: margin {e['margin']:+.2f} "
              f"helpful={e['helpful']} harmful={e['harmful']} "
              f"veto={e['pool_veto']}")

    # replay each gate rule from scratch
    print("\nrule -> D accuracy on val:")
    for margin in MARGINS:
        for use_veto in (False, True):
            correct_n = 0
            for rec in D:
                idx, db = rec["index"], rec["db_id"]
                c_ok = correct(db, idx, C[idx]["prediction"])
                pool_sigs = Counter(c["result_sig"] for c in C[idx]["candidates"]
                                    if c.get("result_sig"))
                majority_sig, majority_n = (pool_sigs.most_common(1)[0]
                                            if pool_sigs else (None, 0))
                k = len(C[idx]["candidates"])
                cur_ok, cur_score = c_ok, C[idx]["verifier_score"]
                for rep in rec.get("repairs", []):
                    if not rep.get("replaced"):
                        continue
                    gain = rep["score"] - cur_score
                    if gain < margin:
                        continue
                    res = executes(db, rep["sql"])
                    sig = result_signature(res.columns, res.rows) if res.ok else None
                    if use_veto and sig and majority_n >= (k + 1) // 2 and sig != majority_sig:
                        continue
                    cur_ok = correct(db, idx, rep["sql"])
                    cur_score = rep["score"]
                correct_n += cur_ok
            tag = f"margin>={margin:g}" + ("+pool_veto" if use_veto else "")
            print(f"  {tag:24s} {correct_n/len(D):.2f}")


if __name__ == "__main__":
    main()