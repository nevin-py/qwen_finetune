"""SParC conversation metrics (design §4).

Per the official protocol each turn is scored against gold with GOLD dialogue
history, so:
- per-turn execution accuracy == query consistency (QC) under gold history
- interaction accuracy: fraction of dialogues where EVERY turn is correct
- average progress: mean over dialogues of (correct turns / total turns)

Official turn buckets: turn 1, 2, 3, 4, ">4".
"""
from __future__ import annotations

from collections import defaultdict


def _bucket(turn: int) -> str:
    return str(turn) if turn <= 4 else ">4"


def sparc_metrics(results, items: list[dict]) -> dict:
    """results: list of ItemResult (has .index/.correct); items: gold rows
    indexed by their position in the same order as evaluate() consumed them."""
    by_index = {r.index: r for r in results}
    per_turn: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    dialogues: dict[int, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    total = correct = 0

    for item in items:
        res = by_index.get(item["index"])
        if res is None:
            continue
        b = per_turn[_bucket(item["turn"])]
        b["n"] += 1
        d = dialogues[item["dialogue"]]
        d["n"] += 1
        total += 1
        if res.correct:
            b["correct"] += 1
            d["correct"] += 1
            correct += 1

    progress = sum(d["correct"] / d["n"] for d in dialogues.values()) / len(dialogues)
    interaction_acc = sum(1 for d in dialogues.values() if d["correct"] == d["n"])
    return {
        "turns_scored": total,
        "turn_accuracy": round(correct / total, 4),
        "query_consistency": round(correct / total, 4),  # gold history (official)
        "per_turn_accuracy": {
            k: round(v["correct"] / v["n"], 4) for k, v in sorted(per_turn.items())
        },
        "interaction_accuracy": round(interaction_acc / len(dialogues), 4),
        "average_progress": round(progress, 4),
        "n_dialogues": len(dialogues),
    }