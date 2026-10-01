#!/usr/bin/env python3
"""Phase 11: ablation matrix, failure taxonomy, CIs (design §15/§22).

CPU-only: reads prediction files + gold through the eval package. Produces
the locked results layout:

    results{suffix}/{spider,bird,sparc}/{cell}{suffix}.json  metrics + CI + taxonomy
    results{suffix}/ablations/{dataset}.json        SC-majority, oracle, paired tests
    results{suffix}/analysis_report.md              human-readable tables

``--suffix _v2`` reads ``reports/{ds}_{cell}_dev_v2.jsonl`` into ``results_v2/``
so the v1 locked layout stays untouched.

Gold never leaves the eval domain (objective §12).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from t2sql.eval.error_taxonomy import classify
from t2sql.eval.harness import evaluate
from t2sql.eval.sparc import sparc_metrics
from t2sql.eval.stats import bootstrap_ci, mcnemar_p, paired_delta_ci

from run_eval import ROOT, resolve_split

PAIRS = [("A", "B"), ("B", "C"), ("C", "D")]
MAX_EXAMPLES = 3


def preds_path(ds: str, cell: str, suffix: str = "") -> Path:
    return ROOT / "reports" / f"{ds}_{cell}_dev{suffix}.jsonl"


def load_preds(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def eval_cell(ds: str, cell: str, items, db_dir, suffix: str):
    preds = load_preds(preds_path(ds, cell, suffix))
    report = evaluate(items, preds, db_dir)
    gold_by_idx = {i: it for i, it in enumerate(items)}
    preds_by_idx = {p["index"]: p for p in preds}

    taxonomy, examples = Counter(), {}
    for r in report.scored:
        if r.correct:
            continue
        gold = gold_by_idx[r.index]
        pred_sql = preds_by_idx[r.index]["prediction"]
        cat, comps = classify(pred_sql, gold["query"], pred_error=r.pred_error)
        taxonomy[cat] += 1
        if taxonomy[cat] <= MAX_EXAMPLES:
            examples.setdefault(cat, []).append({
                "index": r.index, "pred": pred_sql[:160], "gold": gold["query"][:160],
                "differs": comps})

    acc = [float(r.correct) for r in report.scored]
    out = {
        "cell": cell, "dataset": ds, "n": report.n, "scored": len(report.scored),
        "accuracy": round(report.accuracy, 4),
        "accuracy_ci95": [round(v, 4) for v in bootstrap_ci(acc)],
        "exact_match": round(report.exact_match_rate, 4),
        "exec_valid_rate": round(1 - sum(report.error_counts.values()) / len(report.scored), 4)
        if report.scored else 0.0,
        "pred_errors": report.error_counts,
        "by_difficulty": {k: round(v, 4) for k, v in report.by_difficulty.items()},
        "failure_taxonomy": dict(taxonomy.most_common()),
        "taxonomy_examples": examples,
    }
    if ds == "sparc":
        out["conversation"] = sparc_metrics(report.results, items)
    correct = {r.index: r.correct for r in report.scored}
    return out, correct


def ablations(ds: str, cfile_items, db_dir, suffix: str = "", n_slots: int = 5) -> dict:
    """Self-consistency majority, per-slot accuracies, oracle ceiling.

    v2 pools are deduplicated, so records may hold fewer than ``n_slots``
    candidates: slots are scored on the records that have them, and the
    oracle unions every candidate that exists.
    """
    records = load_preds(ROOT / "reports" / f"{ds}_C_dev{suffix}.jsonl")

    def slot_preds(j: int) -> list[dict]:
        return [{"index": r["index"], "db_id": r["db_id"], "question": r["question"],
                 "prediction": r["candidates"][j]["sql"]}
                for r in records if j < len(r["candidates"])]

    def majority_preds() -> list[dict]:
        out = []
        for r in records:
            counts = Counter(c["result_sig"] for c in r["candidates"] if c["result_sig"])
            best_sig = counts.most_common(1)[0][0] if counts else None
            choice = next((c for c in r["candidates"] if c["result_sig"] == best_sig),
                          r["candidates"][0])
            out.append({"index": r["index"], "db_id": r["db_id"],
                        "question": r["question"], "prediction": choice["sql"]})
        return out

    slot_correct = []
    for j in range(n_slots):
        if not slot_preds(j):
            break
        rep = evaluate(cfile_items, slot_preds(j), db_dir)
        slot_correct.append({r.index: r.correct for r in rep.scored})
    union = set()
    for sc in slot_correct:
        union |= {i for i, ok in sc.items() if ok}
    rep_sc = evaluate(cfile_items, majority_preds(), db_dir)
    n = len(rep_sc.scored)
    pool_sizes = [len(r["candidates"]) for r in records]
    return {
        "n": n,
        "pool_size_mean": round(sum(pool_sizes) / len(pool_sizes), 3),
        "pool_size_min": min(pool_sizes),
        "slots_accuracy": [round(sum(sc.values()) / len(sc), 4)
                           for sc in slot_correct],
        "sc_majority_accuracy": round(sum(r.correct for r in rep_sc.scored) / n, 4),
        "oracle_pool_accuracy": round(len(union) / n, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="spider,bird,sparc")
    ap.add_argument("--cells", default="A,B,C,D")
    ap.add_argument("--suffix", default="",
                    help="prediction-file suffix, e.g. _v2 (writes results_v2/)")
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    cells = args.cells.split(",")
    out_root = ROOT / f"results{args.suffix}"
    out_root.mkdir(exist_ok=True)
    report_md = ["# Analysis report (Phase 11)\n",
                 "Generated by `scripts/analyze_results.py`. All numbers are "
                 "execution accuracy unless noted; CIs are 1000-replicate "
                 "percentile bootstrap; paired tests are exact McNemar.\n"]

    for ds in datasets:
        items, db_dir = resolve_split(ds, "dev")
        ds_dir = out_root / ds
        ds_dir.mkdir(exist_ok=True)
        correct_by_cell, table = {}, []
        for cell in cells:
            if not preds_path(ds, cell, args.suffix).exists():
                continue
            out, correct = eval_cell(ds, cell, items, db_dir, args.suffix)
            (ds_dir / f"{cell}.json").write_text(json.dumps(out, indent=2))
            correct_by_cell[cell] = correct
            table.append(f"| {cell} | {out['accuracy']:.3f} | "
                         f"[{out['accuracy_ci95'][0]:.3f}, {out['accuracy_ci95'][1]:.3f}] | "
                         f"{out['exact_match']:.3f} | {out['exec_valid_rate']:.3f} |")

        report_md += [f"\n## {ds}\n",
                      "| cell | acc | 95% CI | exact match | exec-valid |",
                      "|---|---|---|---|---|", *table]

        # paired tests
        paired = []
        for a, b in PAIRS:
            if a not in correct_by_cell or b not in correct_by_cell:
                continue
            idxs = sorted(set(correct_by_cell[a]) & set(correct_by_cell[b]))
            va = [correct_by_cell[a][i] for i in idxs]
            vb = [correct_by_cell[b][i] for i in idxs]
            delta, lo, hi = paired_delta_ci([float(x) for x in va],
                                            [float(x) for x in vb])
            p = mcnemar_p(va, vb)
            paired.append({"vs": f"{a}->{b}", "delta": round(delta, 4),
                           "ci95": [round(lo, 4), round(hi, 4)],
                           "mcnemar_p": round(p, 6),
                           "improved": sum(1 for x, y in zip(va, vb) if y and not x),
                           "regressed": sum(1 for x, y in zip(va, vb) if x and not y)})
        if paired:
            report_md += ["\n### Paired tests\n", "| pair | delta | 95% CI | p | +/− |",
                          "|---|---|---|---|---|"]
            report_md += [f"| {t['vs']} | {t['delta']:+.3f} | [{t['ci95'][0]:+.3f}, "
                          f"{t['ci95'][1]:+.3f}] | {t['mcnemar_p']:.4g} | "
                          f"{t['improved']}/{t['regressed']} |" for t in paired]

        # failure taxonomy per dataset (all cells side by side)
        cats = sorted({c for cell_dir in [ds_dir]
                       for f in cell_dir.glob("*.json")
                       for c in json.loads(f.read_text()).get("failure_taxonomy", {})})
        if cats:
            heads = [f.name[:-5] for f in sorted(ds_dir.glob("*.json"))]
            rows = [[c] + [str(json.loads((ds_dir / f"{h}.json").read_text())
                               .get("failure_taxonomy", {}).get(c, 0)) for h in heads]
                    for c in cats]
            report_md += ["\n### Failure taxonomy (incorrect items)\n",
                          "| category | " + " | ".join(heads) + " |",
                          "|---|" + "---|" * len(heads)]
            report_md += ["| " + " | ".join(r) + " |" for r in rows]

        # ablations need the verifier cell
        if (ROOT / "reports" / f"{ds}_C_dev{args.suffix}.jsonl").exists():
            abl = ablations(ds, items, db_dir, args.suffix)
            (out_root / "ablations").mkdir(exist_ok=True)
            (out_root / "ablations" / f"{ds}.json").write_text(
                json.dumps({**abl, "paired": paired}, indent=2))
            report_md += ["\n### Selection ablations\n",
                          "| strategy | accuracy |", "|---|---|",
                          f"| greedy slot 0 (B) | {abl['slots_accuracy'][0]:.3f} |",
                          f"| self-consistency majority (k~{abl['pool_size_mean']:.2f}) | "
                          f"{abl['sc_majority_accuracy']:.3f} |",
                          f"| oracle pool ceiling | {abl['oracle_pool_accuracy']:.3f} |"]

    (out_root / "analysis_report.md").write_text("\n".join(report_md) + "\n")
    print(f"wrote {out_root}/analysis_report.md and per-cell JSON")


if __name__ == "__main__":
    main()