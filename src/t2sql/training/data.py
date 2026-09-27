"""Training-set construction: plain Python structures -> list[dict] (design §8).

No HF dataset processing APIs are used to *construct* the set (objective);
the JSON artifact is self-contained (system prompt includes the schema) so
training is reproducible from this one file.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..prompts.schema import schema_text
from ..prompts.template import build_messages


def build_training_examples(items: list[dict], db_dir: str | Path) -> list[dict]:
    """One conversational example per question; assistant turn = gold SQL."""
    db_dir = Path(db_dir)
    schemas: dict[str, str] = {}
    examples = []
    for item in items:
        db_id = item["db_id"]
        if db_id not in schemas:
            schemas[db_id] = schema_text(db_dir / db_id / f"{db_id}.sqlite")
        messages = build_messages(item["question"], schemas[db_id])
        messages.append({"role": "assistant", "content": item["query"]})
        examples.append({"messages": messages})
    return examples


def save_examples(examples: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(examples))


def load_examples(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())