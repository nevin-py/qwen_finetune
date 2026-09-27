#!/usr/bin/env python3
"""Download and lay out Spider, SParC, and BIRD-dev. Idempotent: skips work if
the canonical layout is already present, then verifies integrity either way.

Canonical layout (see reports/design.md §3):
  data/spider/{train_spider.json, dev.json, tables.json, dev_gold.sql, database/}
  data/sparc/{train.json, dev.json, tables.json}        # uses spider/database
  data/bird/{dev.json, dev.sql, dev_tables.json, dev_tied_append.json, databases/}
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

SPIDER_DRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"   # official Spider zip (incl. DBs)
SPARC_DRIVE_ID = "1Uu7NMHTR1tdQw1t7bAuM7OPU4LElVKfg"    # official SParC zip (incl. DBs)
BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

MARKERS = (
    "spider/train_spider.json", "spider/dev.json", "spider/dev_gold.sql",
    "spider/database", "sparc/train.json", "sparc/dev.json",
    "bird/dev.json", "bird/dev.sql", "bird/databases",
)

EXPECTED_COUNTS = {  # question/interaction counts, verified against official files
    "spider/dev.json": 1034,
    "spider/train_spider.json": 7000,
    "sparc/dev.json": 422,
    "sparc/train.json": 3034,
    "bird/dev.json": 1534,
}


def _gdown(file_id: str, dest: Path) -> None:
    gdown = ROOT / ".venv/bin/gdown"
    cmd = [str(gdown), "-q", file_id, "-O", str(dest)] if gdown.exists() else \
        [sys.executable, "-m", "gdown", "-q", file_id, "-O", str(dest)]
    subprocess.run(cmd, check=True)


def _unzip(zpath: Path, dest: Path) -> None:
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(dest)


def _db_id(q: dict) -> str:
    """Spider/BIRD use `db_id`, official SParC uses `database_id`."""
    return q.get("db_id") or q["database_id"]


def layout_ok() -> bool:
    return all((DATA / m).exists() for m in MARKERS)


def verify() -> None:
    problems = []
    for rel, expected in EXPECTED_COUNTS.items():
        n = len(json.loads((DATA / rel).read_text()))
        if n != expected:
            problems.append(f"{rel}: {n} != {expected}")
    # every db_id resolves to a SQLite file in the canonical databases dir
    checks = [
        ("spider/dev.json", "spider/database"),
        ("spider/train_spider.json", "spider/database"),
        ("sparc/dev.json", "spider/database"),
        ("sparc/train.json", "spider/database"),
        ("bird/dev.json", "bird/databases"),
    ]
    for questions_rel, db_root_rel in checks:
        ids = {_db_id(q) for q in json.loads((DATA / questions_rel).read_text())}
        missing = [i for i in ids if not (DATA / db_root_rel / i / f"{i}.sqlite").exists()]
        if missing:
            problems.append(f"{questions_rel}: missing DBs {sorted(missing)[:5]}")
    if problems:
        raise SystemExit("VERIFY FAILED:\n  " + "\n  ".join(problems))
    print(f"verify OK: {len(EXPECTED_COUNTS)} files, counts and DB coverage intact")


def fetch_spider(tmp: Path) -> None:
    zpath = tmp / "spider.zip"
    _gdown(SPIDER_DRIVE_ID, zpath)
    _unzip(zpath, tmp)
    src = tmp / "spider_data"
    if not src.exists():  # zip layout guard
        src = next(p for p in tmp.iterdir() if (p / "dev.json").exists())
    (DATA / "spider").mkdir(parents=True, exist_ok=True)
    for name in ("train_spider.json", "dev.json", "tables.json", "dev_gold.sql"):
        shutil.move(str(src / name), DATA / "spider" / name)
    shutil.move(str(src / "database"), DATA / "spider" / "database")
    shutil.rmtree(src, ignore_errors=True)  # drop test split, train_others, macOS junk


def fetch_sparc(tmp: Path) -> None:
    zpath = tmp / "sparc.zip"
    _gdown(SPARC_DRIVE_ID, zpath)
    _unzip(zpath, tmp)
    src = tmp / "sparc"
    (DATA / "sparc").mkdir(parents=True, exist_ok=True)
    for name in ("train.json", "dev.json", "tables.json"):
        shutil.move(str(src / name), DATA / "sparc" / name)
    shutil.rmtree(tmp / "__MACOSX", ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)  # database/ is byte-identical to spider's


def fetch_bird(tmp: Path) -> None:
    zpath = tmp / "bird_dev.zip"
    with urllib.request.urlopen(BIRD_DEV_URL) as r, open(zpath, "wb") as f:
        shutil.copyfileobj(r, f)
    _unzip(zpath, tmp)
    inner_dir = next(p for p in tmp.iterdir() if p.is_dir() and (p / "dev.json").exists())
    (DATA / "bird").mkdir(parents=True, exist_ok=True)
    for name in ("dev.json", "dev.sql", "dev_tables.json", "dev_tied_append.json"):
        shutil.move(str(inner_dir / name), DATA / "bird" / name)
    nested = inner_dir / "dev_databases.zip"
    _unzip(nested, tmp)
    dbs = next(p for p in tmp.iterdir() if p.is_dir()
               and any(p.rglob("*.sqlite")) and p.name != "bird")
    shutil.move(str(dbs), DATA / "bird" / "databases")
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    if layout_ok():
        print("layout present — skipping download")
    else:
        with tempfile.TemporaryDirectory(dir=DATA) as t:
            tmp = Path(t)
            if not (DATA / "spider" / "dev.json").exists():
                fetch_spider(tmp)
            if not (DATA / "sparc" / "dev.json").exists():
                fetch_sparc(tmp)
            if not (DATA / "bird" / "dev.json").exists():
                fetch_bird(tmp)
    verify()


if __name__ == "__main__":
    main()