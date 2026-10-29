#!/usr/bin/env python3
"""Build data/splits/train_sft.json from the training split (idempotent-ish).

Dedup (checklist 0.3) runs first: a (db_id, question) pair keeps its first
occurrence, and train items whose key also appears in val.json are dropped
(the val carve is never modified). Optional v2 prompt upgrades are applied
consistently with inference: --prompt-variants, --few-shot N (writes
data/splits/fewshot_block.txt) and --schema-samples. Defaults reproduce the
v1 build byte-for-byte.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from t2sql.training.data import build_training_examples, save_examples

ROOT = Path(__file__).resolve().parents[1]
VAL_PATH = ROOT / "data" / "splits" / "val.json"
QUARANTINE_PATH = ROOT / "data" / "splits" / "train_quarantine.json"
FEWSHOT_PATH = ROOT / "data" / "splits" / "fewshot_block.txt"
FEWSHOT_SEED = 13


def blocked_keys(path: Path) -> set[tuple[str, str]]:
    """(db_id, question) keys to drop from training: gold SQL that failed
    execution validation (checklist 0.1). Empty when the file is absent."""
    if not path.exists():
        return set()
    return {(x["db_id"], x["question"].strip().lower())
            for x in json.loads(path.read_text())}


def dedup(items: list[dict], blocked: set[tuple[str, str]],
          exclude: set[tuple[str, str]] | None = None) -> tuple[list[dict], int, int, int]:
    """First-wins dedup on (db_id, normalized question).

    Drops rows whose (db_id, question) is blocked (val overlap or gold that
    failed execution validation), and rows whose gold contradicts an earlier
    row for the same question — one conflicting pair exists in Spider train
    and a target the model cannot satisfy.

    Deliberately NOT deduplicated on the gold SQL. 78% of Spider train rows
    share their SQL with another row, almost always because genuinely
    different questions resolve to the same simple query
    (`SELECT count(*) FROM customer`). Removing those would discard real
    supervision and skew the data further toward rare queries; the training
    set is already 61% join-free against a benchmark that is 74% joins. The
    redundancy that actually corrupts measurement lives in the validation
    split, not here, and is handled by make_split.py's --holdout-dbs.

    ``exclude`` holds (db_id, question) keys used as few-shot demonstrations;
    a demonstration that is also a training target is counted twice.
    """
    exclude = exclude or set()
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    n_dup = n_blocked = n_excluded = 0
    for it in items:
        key = (it["db_id"], it["question"].strip().lower())
        if key in blocked:
            n_blocked += 1
            continue
        if key in exclude:
            n_excluded += 1
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out.append(it)
    return out, n_dup, n_blocked, n_excluded


def few_shot_examples(items: list[dict], n: int) -> list[dict]:
    """Deterministic demonstration rows for the shared few-shot block."""
    return random.Random(FEWSHOT_SEED).sample(items, min(n, len(items)))



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-variants", type=int, default=1,
                    help="instruction variants in play (1 = original text)")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="N few-shot train examples -> fewshot_block.txt")
    ap.add_argument("--question-values", action="store_true",
                    help="pick schema example values that match the question; "
                         "must match the runners")
    ap.add_argument("--schema-samples", action="store_true",
                    help="append row counts + example values to the schema")
    args = ap.parse_args()

    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    val_keys: set[tuple[str, str]] = set()
    if VAL_PATH.exists():
        val = json.loads(VAL_PATH.read_text())
        val_keys = {(x["db_id"], x["question"].strip().lower()) for x in val}
    blocked = val_keys | blocked_keys(QUARANTINE_PATH)

    # Pick the few-shot demonstrations first, then drop those rows from the
    # training targets. A demonstration that is also a target is supervised
    # once as an answer and demonstrated in every one of the ~5k prompts.
    few_shot = ""
    exclude: set[tuple[str, str]] = set()
    if args.few_shot > 0:
        chosen = few_shot_examples(items, args.few_shot)
        few_shot = "\n\n".join(f"Q: {x['question']}\nA: {x['query']}"
                               for x in chosen)
        exclude = {(x["db_id"], x["question"].strip().lower()) for x in chosen}
        FEWSHOT_PATH.write_text(few_shot)
        print(f"wrote {FEWSHOT_PATH.relative_to(ROOT)} ({args.few_shot} examples)")

    items, n_dup, n_blocked, n_excluded = dedup(items, blocked, exclude)

    examples = build_training_examples(
        items, ROOT / "data" / "spider" / "database",
        prompt_variants=args.prompt_variants, few_shot=few_shot,
        samples=args.schema_samples,
        question_values=args.question_values)
    out = ROOT / "data" / "splits" / "train_sft.json"
    save_examples(examples, out)
    n_dbs = len({x["db_id"] for x in items})
    print(f"wrote {out.relative_to(ROOT)}: {len(examples)} examples over "
          f"{n_dbs} databases "
          f"(dropped {n_dup} dup + {n_blocked} blocked + {n_excluded} "
          f"few-shot-demonstrations of {len(items) + n_dup + n_blocked + n_excluded})")


if __name__ == "__main__":
    main()
