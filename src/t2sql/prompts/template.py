"""Prompt construction for SQLite text-to-SQL (design §6).

One template: instructions + optional few-shot block live in the system
message, the user message carries the question. Few-shot examples are
content (string), not a code branch. Instruction variants (checklist 3.1)
are complete rewrites of the instruction block with identical rules;
index 0 is the original v1 text so default behavior stays byte-identical.
"""
from __future__ import annotations

import hashlib

INSTRUCTIONS = """\
You are a text-to-SQL assistant. Write one SQLite query that answers the
user's question given the database schema below.

Rules:
- Output exactly one SQL statement, nothing else.
- Use only the tables and columns from the schema.
- Use SQLite syntax only.
- Do not modify the database.
{examples}"""

INSTRUCTION_VARIANTS: tuple[str, ...] = (
    INSTRUCTIONS,
    """\
You are a database assistant. Given the SQLite schema below, write a single
query that answers the user's question.

Rules:
- Reply with exactly one SQL statement and nothing else.
- Reference only tables and columns that appear in the schema.
- Stay within SQLite syntax.
- Never write, change, or delete data.
{examples}""",
    """\
Write one SQLite SELECT statement that answers the question. The schema is
provided below.

Rules:
- Output only the single SQL statement.
- Use only schema objects (tables and columns) that exist in the given schema.
- SQLite dialect only.
- Read-only: do not alter the database.
{examples}""",
    """\
Task: translate the question into one SQLite query for the schema below.

Rules:
- Return exactly one SQL statement - no prose, no explanations.
- Only use tables and columns listed in the schema.
- SQLite syntax only.
- Read-only queries: never modify the database.
{examples}""",
)


def variant_for(key: str, n: int) -> int:
    """Deterministic instruction-variant choice for one example key.

    Same key + same n => same variant, so training and inference agree
    without either side consulting the other.
    """
    if n <= 1:
        return 0
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % n


def _system(schema: str, few_shot: str = "", variant: int = 0) -> str:
    instructions = INSTRUCTION_VARIANTS[variant % len(INSTRUCTION_VARIANTS)]
    examples = f"\nExamples:\n{few_shot.strip()}\n" if few_shot.strip() else ""
    return instructions.format(examples=examples) + f"\nSchema:\n{schema}"


def build_messages(
    question: str,
    schema: str,
    *,
    few_shot: str = "",
    variant: int = 0,
) -> list[dict[str, str]]:
    """Chat messages for text-to-SQL generation."""
    return [
        {"role": "system", "content": _system(schema, few_shot, variant)},
        {"role": "user", "content": question},
    ]


def full_question(item: dict) -> str:
    """Model-facing question text: the benchmark question plus evidence when
    the dataset provides it (BIRD). The eval-facing question stays
    ``item['question']`` — predictions record that raw form."""
    q = item["question"]
    ev = item.get("evidence")
    return f"{q}\nEvidence: {ev}" if ev else q


def build_dialogue_messages(
    question: str,
    schema: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Chat messages for a multi-turn SParC question.

    ``history`` holds GOLD (utterance, query) pairs of *previous* turns only —
    the official SParC protocol conditions on gold dialogue context; the
    current turn's gold SQL never appears here.
    """
    if history:
        lines = ["Dialogue so far:"]
        for h in history:
            lines += [f"Q: {h['utterance']}", f"SQL: {h['query']}"]
        user = "\n".join(lines) + f"\nQ: {question}"
    else:
        user = question
    return [
        {"role": "system", "content": _system(schema)},
        {"role": "user", "content": user},
    ]


def build_repair_messages(
    question: str,
    schema: str,
    sql: str,
    feedback: str,
    *,
    few_shot: str = "",
    variant: int = 0,
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
        {"role": "system", "content": _system(schema, few_shot, variant)},
        {"role": "user", "content": user},
    ]