"""Training-set construction: plain Python structures -> list[dict] (design §8).

No HF dataset processing APIs are used to *construct* the set (objective);
the JSON artifact is self-contained (system prompt includes the schema) so
training is reproducible from this one file.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..prompts.schema import SchemaCache
from ..prompts.template import build_messages, variant_for


def build_training_examples(items: list[dict], db_dir: str | Path,
                            *, prompt_variants: int = 1,
                            few_shot: str = "",
                            samples: bool = False,
                            question_values: bool = False) -> list[dict]:
    """One conversational example per question; assistant turn = gold SQL.

    ``prompt_variants`` / ``few_shot`` / ``samples`` / ``question_values`` must
    match the values used at inference (checklist 3.1-3.3); defaults reproduce
    v1 exactly. With ``question_values`` the schema's example values are
    selected to match the question, which is what the runners do too, so the
    training prompt and the inference prompt stay in the same format.
    """
    db_dir = Path(db_dir)
    cache = SchemaCache(samples=samples, question_values=question_values)
    examples = []
    for item in items:
        schema = cache.get(db_dir / item["db_id"] / f"{item['db_id']}.sqlite",
                           item["question"])
        messages = build_messages(
            item["question"], schema, few_shot=few_shot,
            variant=variant_for(item["question"], prompt_variants))
        messages.append({"role": "assistant", "content": item["query"]})
        examples.append({"messages": messages})
    return examples


def save_examples(examples: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(examples))


def load_examples(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())