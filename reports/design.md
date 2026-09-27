# Design Document — Text-to-SQL Fine-Tuning, Execution-Based Verification & Repair

Status: Phase 0 complete — all external sources verified live on 2026-09-27.

## 1. Research question

Does fine-tuning improve text-to-SQL correctness, and how much additional correctness is
recovered by (a) execution-based verification and (b) bounded execution-based repair?

Measured by Execution Accuracy on **Spider dev (held-out)**, confirmed once on **BIRD dev (held-out)**.

## 2. Experiment matrix

| Cell | Pipeline | Purpose |
|---|---|---|
| A | base model + prompt | zero-shot baseline |
| B | A + QLoRA adapter | fine-tuning delta |
| B2 *(optional)* | 3B base + QLoRA adapter | model-scale delta (gated by VRAM spike test) |
| C | B + verifier (filter/rank) | verification delta |
| D | C + bounded repair loop | repair delta |

All cells archived under `experiments/<cell>/` (adapter + config + seed + predictions) → every number reproducible.

## 3. Datasets (verified sources)

| Dataset | Role | Source (verified) | Size | License |
|---|---|---|---|---|
| Spider train (7,000, `train_spider.json`) | training (+ carve validation) | official: Google Drive link on yale-lily.github.io/spider (gdown, verified) | small | CC BY-SA 4.0 |
| **Spider dev (1,034)** | **final single-turn test (locked)** | same | small | CC BY-SA 4.0 |
| SParC (3,034 train / 422 dev interactions) | multi-turn **evaluation only** (Option A) | official Google Drive zip via gdown (verified) | 99 MB zip | CC BY-SA 4.0 |
| **BIRD dev (1,534 questions, 11 DBs)** | **final held-out benchmark (locked)** | direct: `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip` (verified, 346 MB) | 1.7 GB extracted | research use |
| BIRD train | **not used** (8.9 GB) | — | — | — |

Source verification notes (Phase 0/2):
- Spider zip ships `test.json` + `test_gold.sql` and `train_others.json` (1,659 non-Spider
  questions) — **all excluded**: protocol uses `train_spider.json` for training and official
  dev as the locked test. 166 DBs shipped; 160 used by train+dev.
- HF mirror `aherntech/sparc` was **rejected**: correct counts (3,034/422) but flattened
  turns with **no gold SQL**. Official zip is the only accepted SParC source.
- SParC ships a byte-identical copy of Spider's 166 databases — deleted, spider's kept.
- SParC schema: item = `{database_id, interaction: [turns], final: {query, utterance}}`;
  96% of `final` turns duplicate the last interaction query (paraphrased utterance).
- BIRD difficulty values are `simple/moderate/challenging` (not `challenge`); official
  eval script is not shipped in `dev.zip` → our harness implements the official protocol.
- Key naming: Spider/BIRD use `db_id`, SParC uses `database_id` (handled by `_db_id`).

Split protocol: official Spider **train** (7,000) → 90/10 train/validation (validation used
for all tuning, verifier thresholding, repair decisions). Official Spider **dev** and BIRD
**dev** are touched exactly once each (Phases 9/10) — never during tuning.

Google-Drive-only files (Spider, SParC): auto-download via HF mirrors where verified,
`gdown` fallback with the official Drive IDs. If a mirror's content differs from official
files (checksum/schema check in Phase 2), fall back to Drive.

## 4. Evaluation protocol

- **Primary metric: Execution Accuracy** — candidate SQL vs gold SQL executed on the same
  SQLite DB, result-set equality (order-insensitive unless ORDER BY is semantically required;
  official protocol rules applied for limit/column mismatch).
- **Spider/SParC official cross-check:** `taoyds/test-suite-sql-eval` (official metric since
  Oct 2020 = test-suite execution accuracy; original exact-set-match reported as reference).
  Our harness must match it on a sample (Phase 5 gate).
- **BIRD official metrics:** executed accuracy + valid-SQL rate, computed with the official
  evaluation code shipped in `dev.zip` when available; our harness reproduces it otherwise.
- **SParC (eval-only):** per-turn execution accuracy + query consistency/progress metrics
  from official SParC eval; conversation history is included in the prompt context.
- **Statistics:** bootstrap 95% CIs over questions; breakdown by difficulty (Spider
  easy/medium/hard/extra) and by BIRD defect/effect-category.
- **Gold-SQL isolation:** gold queries live only inside `src/eval/` + dataset files; the
  model, verifier, and repair modules never import or receive them (enforced by module
  boundaries, verified by a test).

## 5. Model selection (verified via HF API)

| Model | Params | License | Gated | Role |
|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.54B | Apache-2.0 | no | **primary training model (B, C, D)** |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54B | Apache-2.0 | no | fallback / general-domain alternative |
| `Qwen/Qwen3-1.7B` | 2.03B | Apache-2.0 | no | candidate (needs instruct variant check) |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | 3.09B | qwen-research | no | 3B spike test → optional B2; inference-only scale baseline |
| `google/gemma-3-1b-it` | 1.0B | gemma (manual gate) | **yes** | rejected (gated) |

Decision: **1.5B primary** (safe on 4 GB), **3B QLoRA spike test in Phase 5** (10 min:
seq 1024, batch 1, grad-accum 4, checkpointing). If VRAM stable < ~3.9 GB → B2 committed.

## 6. Architecture & data flow

```
datasets ──► prompts (schema serialization + template) ──► inference (base | base+adapter)
                                                              │
                     ┌────────────────────────────────────────┤
                     ▼                                        ▼
              sandbox executor ◄── candidate SQL        verifier (features → score)
                     │                                        │
                     ▼                                        ▼
              eval harness (gold SQL, locked)          repair loop (≤2 iters, judged
                                                       by verifier, budget + dedup)
```

- `src/sandbox` — read-only SQLite executor: read-only URI + file permissions +
  `authorizer` allowlist (SELECT/READ/FUNCTION only — DDL/DML/PRAGMA/ATTACH denied),
  execution timeout via progress handler, row/output caps, error taxonomy
  (syntax / no_such_table / no_such_column / ambiguous / no_such_function / forbidden /
  timeout / operational). Writes are never needed: candidates are judged by SELECT
  results only, so no temp-DB staging exists (rule 5: no unnecessary code).
- `src/prompts` — single schema serializer (tables/columns/PK-FK, optionally sample rows)
  + one template module (few-shot variant is content, not code branches).
- `src/inference` — model loading (base vs base+adapter), generation, SQL extraction.
- `src/verifier` — features only from observable signals: execution outcome, error class,
  row count, referenced DB objects vs schema, execution cost, self-consistency over k
  samples, string-match for scalar queries. Scored by rule-based scorer first; logistic
  regression only if it measurably beats it on validation (one sklearn dep).
- `src/repair` — bounded loop, ≤2 iterations, verifier is the sole judge, hard budget
  (max candidate executions per question), duplicate-prompt/loop guard.
- `src/eval` — the **only** module that reads gold SQL.

## 7. Training design (QLoRA)

4-bit NF4 base (frozen) + LoRA (r=16, α=32, target: attention + MLP projections),
batch 1, grad-accum 4–8, seq ≤1024 (spike may allow 1536), gradient checkpointing,
2–3 epochs overnight, fixed seeds, per-cell adapter checkpoints, VRAM logging per step,
OOM-safe resume.

## 8. Dependencies (minimal, pinned)

Runtime: `torch, transformers, peft, trl, bitsandbytes, accelerate, scikit-learn, pyyaml`
Dev: `pytest`  Conditional: `gdown` (only if Spider/SParC mirror check fails).
Explicitly **not** used: HF `datasets` for training-set construction (built as plain JSON),
`sqlglot` (stdlib `sqlite3` parses/executes), any web framework, any config framework.

## 9. Maintainability rules (binding for all phases)

1. No speculative code — a phase writes only what its gate requires.
2. One implementation per concern (one serializer, one template, one eval entrypoint).
3. Deps added only when a gate forces them; record reason in this document.
4. Tests guard boundaries: sandbox adversarial tests, gold-isolation test, harness-vs-official.
5. Dead code from earlier phases is deleted before Phase 12.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 4 GB OOM in training | batch 1, grad-accum, checkpointing, 1024 cap, resume; B2 gated by spike |
| Python 3.14 incompatible ML libs | isolated py3.11 `.venv` (verified working: venv+pip OK) |
| Google Drive download flakiness | HF mirror first, gdown fallback with official IDs |
| Eval drift | fixed seeds, archive per-cell predictions, cross-check vs official evaluator |
| BIRD/Spider dev leakage | one-time access protocol in Phase gates |
| Qwen 3B `qwen-research` license | research/portfolio use only — acceptable; noted in README |