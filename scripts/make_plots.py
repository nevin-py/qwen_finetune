#!/usr/bin/env python3
"""Publication-quality plots (design §23). Reads the results/ analysis outputs
plus the persisted training log; writes results/plots/*.png (6 figures):

  1. ablation matrix accuracy with 95% CIs (spider + bird)
  2. QLoRA training loss curve
  3. SParC per-turn accuracy (A vs B)
  4. failure-taxonomy breakdown (bird, per cell)
  5. selection ablations (greedy / self-consistency / verifier / oracle)
  6. resource usage (latency, throughput, VRAM per config)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "plots"
CELLS = ["A", "B", "C", "D"]
COLORS = {"A": "#8da0cb", "B": "#66c2a5", "C": "#fc8d62", "D": "#e78ac3"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def fig1_matrix():
    datasets = ["spider", "bird"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, ds in zip(axes, datasets):
        rows = [(c, load(ROOT / "results" / ds / f"{c}.json")) for c in CELLS]
        rows = [(c, r) for c, r in rows if r]
        xs = range(len(rows))
        acc = [r["accuracy"] for _, r in rows]
        err = [[r["accuracy"] - r["accuracy_ci95"][0] for _, r in rows],
               [r["accuracy_ci95"][1] - r["accuracy"] for _, r in rows]]
        ax.bar(xs, acc, color=[COLORS[c] for c, _ in rows], yerr=err,
               capsize=3, width=0.65)
        for x, v in zip(xs, acc):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(list(xs))
        names = {"A": "base", "B": "+QLoRA", "C": "+verifier", "D": "+repair"}
        ax.set_xticklabels([f"{c}\n{names[c]}" for c, _ in rows])
        ax.set_title(f"{ds} dev" + (" (locked test)" if ds == "bird" else ""))
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("execution accuracy")
    fig.suptitle("Experiment matrix: A → B → C → D (bars: 95% bootstrap CI)",
                 fontsize=11, y=1.02)
    fig.savefig(OUT / "fig1_ablation_matrix.png")


def fig2_training_loss():
    txt = (ROOT / "experiments" / "B" / "train_log.txt").read_text()
    rows = re.findall(r"\{'loss': '([0-9.]+)'.*?'epoch': '([0-9.]+)'", txt)
    if not rows:
        return
    loss = [float(a) for a, _ in rows]
    epoch = [float(b) for _, b in rows]
    smooth = [sum(loss[max(0, i - 14):i + 1]) / len(loss[max(0, i - 14):i + 1])
              for i in range(len(loss))]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(epoch, loss, alpha=0.25, color=COLORS["B"], label="logged loss")
    ax.plot(epoch, smooth, color="black", linewidth=1.6, label="moving avg (15)")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("QLoRA fine-tuning (2 epochs, 3,114 steps)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig2_training_loss.png")


def fig3_sparc_turns():
    a = load(ROOT / "results" / "sparc" / "A.json")
    b = load(ROOT / "results" / "sparc" / "B.json")
    if not (a and b):
        return
    keys = ["1", "2", "3", "4", ">4"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for cell, data, color in [("A", a, COLORS["A"]), ("B", b, COLORS["B"])]:
        per = data["conversation"]["per_turn_accuracy"]
        ax.plot(keys, [per.get(k) for k in keys], marker="o", color=color,
                label=f"{cell} (base/+QLoRA)")
    ax.set_xlabel("turn index"); ax.set_ylabel("query consistency (exec acc)")
    ax.set_title("SParC multi-turn: accuracy by turn (gold history)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_sparc_turns.png")


def fig4_taxonomy():
    cells = [c for c in CELLS if (ROOT / "results" / "bird" / f"{c}.json").exists()]
    if not cells:
        return
    data = {c: load(ROOT / "results" / "bird" / f"{c}.json")["failure_taxonomy"]
            for c in cells}
    cats = sorted({k for d in data.values() for k in d},
                  key=lambda k: -sum(d.get(k, 0) for d in data.values()))[:8]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(cells)
    xs = range(len(cats))
    for i, c in enumerate(cells):
        ax.bar([x + i * width for x in xs],
               [data[c].get(k, 0) for k in cats], width=width,
               color=COLORS[c], label=f"cell {c}")
    ax.set_xticks([x + width * (len(cells) - 1) / 2 for x in xs])
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("# incorrect items")
    ax.set_title("BIRD dev: failure taxonomy (objective §13)")
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig4_failure_taxonomy.png")


def fig5_selection_ablations():
    sets = [ds for ds in ["spider", "bird"]
            if (ROOT / "results" / "ablations" / f"{ds}.json").exists()]
    if not sets:
        return
    labels = ["greedy (B)", "self-cons.\nmajority", "verifier (C)", "oracle\nceiling"]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.6 * len(sets), 3.2),
                             sharey=True)
    axes = [axes] if len(sets) == 1 else list(axes)
    for ax, ds in zip(axes, sets):
        abl = load(ROOT / "results" / "ablations" / f"{ds}.json")
        c_json = load(ROOT / "results" / ds / "C.json")
        vals = [abl["slots_accuracy"][0], abl["sc_majority_accuracy"],
                c_json["accuracy"], abl["oracle_pool_accuracy"]]
        ax.bar(range(4), vals, color=[COLORS["B"], "#999999", COLORS["C"],
                                      "#444444"], width=0.65)
        for x, v in enumerate(vals):
            ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(ds); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("execution accuracy")
    fig.suptitle("Candidate-selection ablations (k=5 pool)", fontsize=11, y=1.02)
    fig.savefig(OUT / "fig5_selection_ablations.png")


def fig6_resources():
    res = load(ROOT / "results" / "resource_usage" / "benchmarks.json")
    if not res:
        return
    cfgs = res["gpu_configs"] + ([res["cpu_config"]] if "cpu_config" in res else [])
    cfgs = [c for c in cfgs if "error" not in c]
    labels = [c["config"].replace("_", "\n") for c in cfgs]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].bar(labels, [c["mean_s"] for c in cfgs], color="#8da0cb")
    axes[0].set_ylabel("mean s / query (128 tok)")
    axes[1].bar(labels, [c["output_tokens_per_s"] for c in cfgs], color="#66c2a5")
    axes[1].set_ylabel("output tokens / s")
    vram = [c.get("peak_vram_gb", 0) for c in cfgs]
    axes[2].bar(labels, vram, color="#fc8d62")
    axes[2].set_ylabel("peak VRAM (GiB)" if max(vram) else "peak VRAM (GiB)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=7)
    fig.suptitle("Resource benchmarks (design §16)", fontsize=11, y=1.02)
    fig.savefig(OUT / "fig6_resources.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_matrix(); fig2_training_loss(); fig3_sparc_turns()
    fig4_taxonomy(); fig5_selection_ablations(); fig6_resources()
    print(f"wrote plots -> {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()