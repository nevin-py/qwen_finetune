"""Phase 1 scaffold gate: package imports, env is py3.11, layout intact."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "t2sql"


def test_python_version_is_311():
    assert sys.version_info[:2] == (3, 11)


def test_package_layout():
    for sub in ("sandbox", "prompts", "inference", "verifier", "repair", "eval"):
        assert (SRC / sub / "__init__.py").is_file(), f"missing module: {sub}"


def test_imports():
    import t2sql  # noqa: F401

    for sub in ("sandbox", "prompts", "inference", "verifier", "repair", "eval"):
        __import__(f"t2sql.{sub}")