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


def _system(schema: str, few_shot: str = "") -> str:
    examples = f"\nExamples:\n{few_shot.strip()}\n" if few_shot.strip() else ""
    return INSTRUCTIONS.format(examples=examples) + f"\nSchema:\n{schema}"


def build_messages(
    question: str,
    schema: str,
    *,
    few_shot: str = "",
) -> list[dict[str, str]]:
    """Chat messages for text-to-SQL generation."""
    return [
        {"role": "system", "content": _system(schema, few_shot)},
        {"role": "user", "content": question},
    ]


def build_repair_messages(
    question: str,
    schema: str,
    sql: str,
    feedback: str,
) -> list[dict[str, str]]:
    """Chat messages for one bounded repair attempt (cell D).

    Same system instructions as generation so the model stays on-distribution;
    the user turn carries the previous SQL plus verifier-derived feedback
    (execution error / observable weaknesses — never gold SQL).
    """
    user = (
        f"{question}\n\nYour previous SQL:\n{sql}\n"
        f"Problem: {feedback}\nReturn corrected SQL only."
    )
    return [
        {"role": "system", "content": _system(schema)},
        {"role": "user", "content": user},
    ]