#!/usr/bin/env python3
"""Derive SParC evaluation artifacts from the official dev split.

Writes two files with strictly separated roles (gold isolation, objective §12):

- ``data/sparc/dev_prompts.json`` — prompt contexts for generation: utterance +
  GOLD dialogue history of *previous* turns (official SParC protocol: models
  are conditioned on gold dialogue context). Contains NO current-turn gold.
- ``data/sparc/dev_gold.json`` — current-turn gold SQL for the eval harness
  only (eval domain). Generation code never reads this file.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sparc" / "dev.json"


def main() -> None:
    dialogues = json.loads(SRC.read_text())
    prompts: list[dict] = []
    gold: list[dict] = []
    idx = 0
    for d_id, dlg in enumerate(dialogues):
        db_id = dlg["database_id"]
        history: list[dict[str, str]] = []
        for t_id, turn in enumerate(dlg["interaction"], 1):
            prompts.append({
                "index": idx, "db_id": db_id,
                "question": turn["utterance"], "history": history,
            })
            gold.append({
                "index": idx, "db_id": db_id, "question": turn["utterance"],
                "query": turn["query"], "dialogue": d_id, "turn": t_id,
            })
            history = history + [{"utterance": turn["utterance"],
                                  "query": turn["query"]}]
            idx += 1

    (ROOT / "data" / "sparc" / "dev_prompts.json").write_text(
        json.dumps(prompts, indent=1))
    (ROOT / "data" / "sparc" / "dev_gold.json").write_text(
        json.dumps(gold, indent=1))
    print(f"wrote dev_prompts.json + dev_gold.json ({idx} turns, "
          f"{len(dialogues)} dialogues)")


if __name__ == "__main__":
    main()