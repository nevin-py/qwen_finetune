"""Dataset file locations for the experiment runners (Phase 10).

Runner-side paths only: files here contain dataset inputs, and split files may
carry gold SQL in ``query`` — runners never read that field (gold isolation:
``src/t2sql/eval/`` is the only module that scores against gold, objective §12).
"""
from __future__ import annotations

import json
from pathlib import Path
import random

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
 
 
def db_stratified_sample(items: list[dict], n: int, seed: int = 0) -> list[dict]:
    """``n`` items drawn round-robin across distinct databases.

    Split files are grouped by ``db_id``, so taking a prefix or a uniform
    random sample measures a handful of schemas. A probe built that way
    reports an accuracy for the model on a few memorized databases and is
    used to pick hyperparameters, checkpoints and thresholds that then have to
    work on unseen schemas. Round-robin over databases keeps every item
    independent of file order and is deterministic given ``seed``.
    """
    if n is None or n <= 0 or n >= len(items):
        return list(items)
    by_db: dict[str, list[dict]] = {}
    for it in items:
        by_db.setdefault(it["db_id"], []).append(it)
    dbs = sorted(by_db)
    # Offsetting the start keeps the choice from always favouring the
    # alphabetically first databases.
    rng = random.Random(seed)
    start = rng.randrange(len(dbs)) if dbs else 0
    order = dbs[start:] + dbs[:start]
    picked: list[dict] = []
    round_no = 0
    while len(picked) < n:
        progressed = False
        for db in order:
            bucket = by_db[db]
            if round_no < len(bucket):
                picked.append(bucket[round_no])
                progressed = True
                if len(picked) == n:
                    break
        if not progressed:
            break
        round_no += 1
    return picked