"""Execution-accuracy evaluation over gold/prediction pairs (design §4).

Gold lives here — the only module allowed to read it. Predictions must carry
the ``index`` of their question in the split file; the harness verifies the
(db_id, question) pair matches before scoring, so a shuffled or misaligned
predictions file cannot silently score the wrong gold.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..sandbox.executor import Sandbox
from . import execution

EVAL_TIMEOUT_S = 30.0   # matches official TIMEOUT=60 with headroom on this GPU box
EVAL_MAX_ROWS = 200_000


@dataclass(frozen=True)
class ItemResult:
    index: int
    correct: bool
    skipped: bool            # gold itself failed to execute -> excluded
    pred_error: str | None   # taxonomy class when prediction failed execution
    difficulty: str | None


@dataclass
class EvalReport:
    results: list[ItemResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def scored(self) -> list[ItemResult]:
        return [r for r in self.results if not r.skipped]

    @property
    def n_skipped(self) -> int:
        return self.n - len(self.scored)

    @property
    def accuracy(self) -> float:
        return sum(r.correct for r in self.scored) / len(self.scored) if self.scored else 0.0

    @property
    def error_counts(self) -> dict[str, int]:
        return dict(Counter(r.pred_error for r in self.scored if r.pred_error))

    @property
    def by_difficulty(self) -> dict[str, float]:
        buckets: dict[str, list[bool]] = {}
        for r in self.scored:
            if r.difficulty:
                buckets.setdefault(r.difficulty, []).append(r.correct)
        return {k: sum(v) / len(v) for k, v in sorted(buckets.items())}

    def to_json(self) -> dict:
        return {
            "n": self.n,
            "scored": len(self.scored),
            "skipped": self.n_skipped,
            "accuracy": round(self.accuracy, 4),
            "by_difficulty": {k: round(v, 4) for k, v in self.by_difficulty.items()},
            "pred_errors": self.error_counts,
        }


def evaluate(
    items: list[dict],
    predictions: list[dict],
    db_dir: str | Path,
    *,
    timeout_s: float = EVAL_TIMEOUT_S,
    max_rows: int = EVAL_MAX_ROWS,
) -> EvalReport:
    """Score predictions (with gold from ``items``) by execution accuracy."""
    db_dir = Path(db_dir)
    sandboxes: dict[str, Sandbox] = {}
    report = EvalReport()

    for pred in predictions:
        idx = pred["index"]
        gold = items[idx]
        if (gold["db_id"], gold["question"]) != (pred["db_id"], pred["question"]):
            raise ValueError(
                f"predictions misaligned at index {idx}: "
                f"expected {(gold['db_id'], gold['question'])!r}, "
                f"got {(pred['db_id'], pred['question'])!r}"
            )
        db_id = gold["db_id"]
        if db_id not in sandboxes:
            sandboxes[db_id] = Sandbox(
                db_dir / db_id / f"{db_id}.sqlite",
                timeout_s=timeout_s, max_rows=max_rows,
            )
        sb = sandboxes[db_id]

        # official evaluator executes postprocessed + DISTINCT-stripped SQL
        gold_sql = execution.prepare(gold["query"])
        pred_sql = execution.prepare(pred["prediction"])

        gold_res = sb.execute(gold_sql, timeout_s=timeout_s, max_rows=max_rows)
        if not gold_res.ok:  # gold broken: question not evaluable (official behavior)
            report.results.append(ItemResult(idx, correct=False, skipped=True,
                                             pred_error=None, difficulty=gold.get("difficulty")))
            continue

        pred_res = sb.execute(pred_sql, timeout_s=timeout_s, max_rows=max_rows)
        if not pred_res.ok:
            report.results.append(ItemResult(idx, correct=False, skipped=False,
                                             pred_error=pred_res.error,
                                             difficulty=gold.get("difficulty")))
            continue

        correct = execution.compare(
            gold_sql, gold_res.rows, pred_sql, pred_res.rows
        )
        report.results.append(ItemResult(idx, correct=correct, skipped=False,
                                         pred_error=None, difficulty=gold.get("difficulty")))

    for sb in sandboxes.values():
        sb.close()
    return report