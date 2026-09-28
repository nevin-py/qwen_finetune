"""Guards: every script and module must at least compile (catches edit damage
that unit tests on src/ can't see, since scripts are CLI-only)."""

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_scripts_compile():
    targets = list((ROOT / "scripts").glob("*.py")) + \
        list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    assert targets, "nothing found"
    for path in targets:
        py_compile.compile(str(path), doraise=True)