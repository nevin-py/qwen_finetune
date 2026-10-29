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
from t2sql.verifier.features import (extract_features, pool_consistency,
                                      pool_signature_counts, result_signature)
from t2sql.verifier.scorer import (DEFAULT_WEIGHTS, max_reachable_score, score)

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
    ap.add_argument("--stability-seeds", type=int, default=4,
                    help="extra search seeds used to test whether the winning "
                         "weight vector reproduces (0 disables)")
    ap.add_argument("--split", default="holdout",
                    help="split the --cfile was generated over; gold is read "
                         "from here. Defaults to holdout so the verifier and "
                         "the repair threshold are tuned on unseen databases, "
                         "like the locked benchmarks")
    ap.add_argument("--flag-quantile", type=float, default=0.30,
                    help="quantile of tuning-split selection scores to emit as "
                        "flag_threshold, the repair gate on this weight scale")
    args = ap.parse_args()

    t0 = time.time()
    records = [json.loads(line) for line in (ROOT / args.cfile).open()]
    gold_split = json.loads(
        (ROOT / "data" / "splits" / f"{args.split}.json").read_text())

    sandboxes: dict[str, Sandbox] = {}
    gold_ok: dict[tuple[str, str], tuple | None] = {}   # (db,q) -> gold rows sig exec
    items: list[list[dict]] = []
    n_bad_gold = 0
    for rec in records:
        item = gold_split[rec["index"]]
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
        counts = pool_signature_counts(sigs)
        k = len(results)
        for c, r, sig in zip(rec["candidates"], results, sigs):
            feats = extract_features(
                rec["question"], c["sql"], executed=r.ok, error=r.error,
                columns=r.columns, rows=r.rows, truncated=r.truncated,
                duration_ms=r.duration_ms,
                consistency=pool_consistency(sig, counts, k))
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

    base = {k: DEFAULT_WEIGHTS[k] for k in KEYS}
    v1_acc, v1_c, _ = selection_accuracy(items, base)
    oracle = sum(1 for c in items if any(x["correct"] for x in c)) / len(items)

    def search(seed: int) -> tuple[dict, float]:
        rng = random.Random(seed)
        best_w, best_acc = base, v1_acc
        for _ in range(args.trials):
            w = {k: rng.choice(GRID[k]) for k in KEYS}
            w["executed"] = DEFAULT_WEIGHTS["executed"]
            acc, _, _ = selection_accuracy(items, w)
            if acc > best_acc:
                best_acc, best_w = acc, w
        return best_w, best_acc

    best_w, best_acc = search(args.seed)

    # Repeat the search under other seeds. A winner that does not reproduce is
    # a coin flip fitted to the tuning split, not a tuned scorer: on n=100 a
    # one-item edge produces a different argmax vector on every seed while the
    # reported accuracy stays identical. Surfacing that here stops such a
    # vector from being shipped as if it were a finding.
    alt_accs, distinct = [], set()
    for s in range(args.seed + 1, args.seed + 1 + args.stability_seeds):
        w_s, acc_s = search(s)
        alt_accs.append(round(acc_s, 4))
        distinct.add(tuple(sorted(w_s.items())))
    distinct.add(tuple(sorted(best_w.items())))
    n_items = len(items)
    # One item on the tuning split is the resolution limit of the metric.
    resolution = 1.0 / n_items if n_items else 1.0
    margin = best_acc - v1_acc
    stable = margin > resolution and len(distinct) == 1
    if margin <= resolution:
        note = (f"gain over rule weights is {margin:.4f}, at or below the "
                f"one-item resolution {resolution:.4f} of an n={n_items} "
                "split — this is noise, not tuning")
    elif len(distinct) > 1:
        note = (f"{len(distinct)} distinct weight vectors reach the same "
                f"accuracy across {args.stability_seeds + 1} seeds — the winner "
                "is arbitrary; prefer the rule weights unless a vector "
                "reproduces")
    else:
        note = "winner reproduces across all stability seeds"

    # Calibrate the repair flag threshold on this weight scale instead of
    # hardcoding one. A threshold chosen for a different weight vector can sit
    # above the reachable ceiling and flag every item.
    # Honour the preregistered rule rather than merely reporting on it: when
    # the winning vector does not reproduce across seeds, ship the rule
    # weights and calibrate the threshold on those instead. v2 shipped an
    # arbitrary vector and it lost to the plain rules on the locked set.
    shipped_w = best_w if stable else base
    shipped_acc = best_acc if stable else v1_acc
    # Threshold on the score the selected candidate would actually receive, so
    # repair is gated on the same quantity the verifier ranked with.
    scores = sorted(max(score(c["feats"], shipped_w) for c in cands)
                    for cands in items if cands)
    ceil_score = max_reachable_score(shipped_w)
    q = args.flag_quantile
    idx = min(len(scores) - 1, max(0, round(q * (len(scores) - 1))))
    flag_threshold = scores[idx] if scores else 0.0
    if flag_threshold >= ceil_score:
        flag_threshold = next((s for s in scores if s < ceil_score), ceil_score - 1e-6)

    payload = {
        "source": args.cfile,
        "n_items": len(items),
        "n_bad_gold_skipped": n_bad_gold,
        "trials": args.trials,
        "seed": args.seed,
        "v1_weights": base,
        "v1_selection_acc": round(v1_acc, 4),
        # What downstream cells actually use. Falls back to the rule weights
        # when the searched vector failed to reproduce across seeds.
        "tuned_weights": shipped_w,
        "tuned_selection_acc": round(shipped_acc, 4),
        # What the search found, kept for the record.
        "searched_weights": best_w,
        "searched_selection_acc": round(best_acc, 4),
        "shipped_rule_weights_fallback": not stable,
        "oracle_pool_acc": round(oracle, 4),
        "stability": {
            "seeds_checked": args.stability_seeds + 1,
            "alt_selection_acc": alt_accs,
            "distinct_weight_vectors": len(distinct),
            "one_item_resolution": round(resolution, 4),
            "gain_over_rules": round(margin, 4),
            "stable": stable,
            "note": note,
        },
        "max_reachable_score": round(ceil_score, 4),
        "flag_threshold": round(flag_threshold, 4),
        "flag_quantile": q,
        "generated_s": round(time.time() - t0),
    }
    out = ROOT / args.out
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in
                      ("n_items", "v1_selection_acc", "tuned_selection_acc",
                       "oracle_pool_acc", "max_reachable_score",
                       "flag_threshold")}, indent=2))
    print(f"stability: {note}")
    print(f"wrote {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
