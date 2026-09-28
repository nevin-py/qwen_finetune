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

from t2sql.prompts.schema import schema_text
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


def dedup(items: list[dict], blocked: set[tuple[str, str]]) -> tuple[list[dict], int, int]:
    """First-wins dedup on (db_id, normalized question); drops blocked keys
    (val overlap + quarantine)."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    n_dup = n_blocked = 0
    for it in items:
        key = (it["db_id"], it["question"].strip().lower())
        if key in blocked:
            n_blocked += 1
            continue
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        out.append(it)
    return out, n_dup, n_blocked


def build_fewshot_block(items: list[dict], n: int) -> str:
    rng = random.Random(FEWSHOT_SEED)
    chosen = rng.sample(items, n)
    return "\n\n".join(f"Q: {x['question']}\nA: {x['query']}" for x in chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-variants", type=int, default=1,
                    help="instruction variants in play (1 = original text)")
    ap.add_argument("--few-shot", type=int, default=0,
                    help="N few-shot train examples -> fewshot_block.txt")
    ap.add_argument("--schema-samples", action="store_true",
                    help="append row counts + example values to the schema")
    args = ap.parse_args()

    items = json.loads((ROOT / "data" / "splits" / "train.json").read_text())
    val_keys: set[tuple[str, str]] = set()
    if VAL_PATH.exists():
        val = json.loads(VAL_PATH.read_text())
        val_keys = {(x["db_id"], x["question"].strip().lower()) for x in val}
    blocked = val_keys | blocked_keys(QUARANTINE_PATH)
    items, n_dup, n_blocked = dedup(items, blocked)

    few_shot = ""
    if args.few_shot > 0:
        few_shot = build_fewshot_block(items, args.few_shot)
        FEWSHOT_PATH.write_text(few_shot)
        print(f"wrote {FEWSHOT_PATH.relative_to(ROOT)} ({args.few_shot} examples)")

    examples = build_training_examples(
        items, ROOT / "data" / "spider" / "database",
        prompt_variants=args.prompt_variants, few_shot=few_shot,
        samples=args.schema_samples)
    out = ROOT / "data" / "splits" / "train_sft.json"
    save_examples(examples, out)
    print(f"wrote {out.relative_to(ROOT)}: {len(examples)} examples "
          f"(dropped {n_dup} dup + {n_blocked} blocked of "
          f"{len(items) + n_dup + n_blocked})")


if __name__ == "__main__":
    main()
