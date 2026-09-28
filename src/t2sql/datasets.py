"""Dataset file locations for the experiment runners (Phase 10).

Runner-side paths only: files here contain dataset inputs, and split files may
carry gold SQL in ``query`` — runners never read that field (gold isolation:
``src/t2sql/eval/`` is the only module that scores against gold, objective §12).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def dataset_items(dataset: str, split: str) -> list[dict]:
    if dataset == "bird":
        return json.loads((ROOT / "data" / "bird" / "dev.json").read_text())
    path = ROOT / "data" / "splits" / f"{split}.json"
    if not path.exists():  # official dev lives outside data/splits
        path = ROOT / "data" / "spider" / f"{split}.json"
    return json.loads(path.read_text())


def db_file(dataset: str, db_id: str) -> Path:
    root = ROOT / "data" / "bird" / "databases" if dataset == "bird" \
        else ROOT / "data" / "spider" / "database"
    return root / db_id / f"{db_id}.sqlite"


def sample_indices(dataset: str, split: str, n: int, seed: int) -> list[int]:
    """Deterministic sample (or the full split when n <= 0)."""
    import random
    total = len(dataset_items(dataset, split))
    if n is None or n <= 0:
        return list(range(total))
    return random.Random(seed).sample(range(total), min(n, total))