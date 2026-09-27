"""Prompt construction for SQLite text-to-SQL (design §6).

One template: instructions + optional few-shot block live in the system
message, the user message carries the question. Few-shot examples are
content (string), not a code branch.
"""
from __future__ import annotations

INSTRUCTIONS = """\
You are a text-to-SQL assistant. Write one SQLite query that answers the
user's question given the database schema below.

Rules:
- Output exactly one SQL statement, nothing else.
- Use only the tables and columns from the schema.
- Use SQLite syntax only.
- Do not modify the database.
{examples}"""


def build_messages(
    question: str,
    schema: str,
    *,
    few_shot: str = "",
) -> list[dict[str, str]]:
    """Chat messages for text-to-SQL generation."""
    examples = f"\nExamples:\n{few_shot.strip()}\n" if few_shot.strip() else ""
    system = INSTRUCTIONS.format(examples=examples) + f"\nSchema:\n{schema}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]