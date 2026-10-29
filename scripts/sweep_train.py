#!/usr/bin/env python3
"""Stage 2: QLoRA sweep driver (checklist 2.1).

Cartesian product of --epochs x --lr x --seeds; each configuration runs
train_qlora.py in a subprocess (one GPU at a time) under
experiments/sweep/<tag>/. Finished runs are skipped, so the driver is
resumable. Afterwards a ranked summary is (re)written to
experiments/sweep/summary.md with val_exec_acc from val_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tag_of(ep: float, lr: float, seed: int) -> str:
    return f"ep{ep:g}_lr{lr:g}_s{seed}"


def best_acc(run_dir: Path) -> float | None:
    p = run_dir / "val_metrics.json"
    if not p.exists():
        return None
    best = json.loads(p.read_text()).get("best_val_exec_acc")
    return best if isinstance(best, (int, float)) and best >= 0 else None


def write_summary(out: Path, rows: list[tuple[str, float | None]]) -> None:
    """Rank every run directory on disk, not just this invocation's grid.

    Sweeps are run in batches (epochs x lr, then extra seeds), so a summary
    rebuilt from the current batch alone silently erased the earlier results:
    the grid rows vanished the moment a seed-only run finished.
    """
    merged: dict[str, float | None] = {r[0]: r[1] for r in rows}
    for run_dir in sorted(out.iterdir()):
        if run_dir.is_dir():
            merged.setdefault(run_dir.name, best_acc(run_dir))
    lines = ["# Sweep summary", "",
             "| run | best val_exec_acc |", "|---|---|"]
    for tag, acc in sorted(merged.items(),
                           key=lambda r: (r[1] is None, -(r[1] or 0))):
        lines.append(f"| {tag} | {acc if acc is not None else 'n/a'} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=float, nargs="+", default=[2.0])
    ap.add_argument("--lr", type=float, nargs="+", default=[2e-4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--out", default="experiments/sweep")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float | None]] = []
    for ep in args.epochs:
        for lr in args.lr:
            for seed in args.seeds:
                tag = tag_of(ep, lr, seed)
                run_dir = out / tag
                done = (run_dir / "train_config.json").exists()
                if not done:
                    print(f"=== {tag}", flush=True)
                    t0 = time.time()
                    subprocess.run(
                        [sys.executable, str(ROOT / "scripts" / "train_qlora.py"),
                         "--out", f"{args.out}/{tag}",
                         "--epochs", str(ep), "--lr", str(lr),
                         "--seed", str(seed)],
                        check=True, cwd=ROOT)
                    print(f"=== {tag} done in {(time.time()-t0)/60:.1f} min",
                          flush=True)
                else:
                    print(f"=== {tag} already finished, skipping", flush=True)
                rows.append((tag, best_acc(run_dir)))
                write_summary(out, rows)

    print("\n".join(f"{t}: {a}" for t, a in rows))


if __name__ == "__main__":
    main()
