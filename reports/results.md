# Results (living document)

Final held-out numbers (Spider dev, BIRD dev) are added only in Phase 10/11.
Everything below uses the **validation split** (official dev stays locked).

## Experiment A (base) vs B (QLoRA) — n=100 val, seed 0, identical samples

| Cell | Pipeline | Execution accuracy | Execution errors |
|---|---|---|---|
| A | Qwen2.5-Coder-1.5B base + prompt | **47.0%** | 27 (26 no_such_column, 1 no_such_table) |
| B | A + QLoRA adapter (2 epochs, 3114 steps) | **74.0%** | 7 (all no_such_column) |

**Paired analysis (same 100 questions):**

| | B correct | B wrong |
|---|---|---|
| **A correct** | 46 | 1 |
| **A wrong** | 28 | 25 |

- 28 improvements vs 1 regression → exact McNemar **p = 1.12e-07**
- Fine-tuning delta: **+27.0 points** (relative +57%)
- Artifacts: `experiments/B/adapter`, `experiments/B/train_config.json`,
  `reports/{A,B}_val_n100.jsonl`, `reports/eval_{A,B}_val_n100.json`

## Training details (cell B)

- QLoRA: NF4 4-bit base, LoRA r=16/α=32/dropout 0.05 on all attention + MLP projections (18.5M trainable, 1.18%)
- 6,300 examples, assistant-only loss, max_len 2048 (covers 98.8%; p95=1372, max=2352)
- 2 epochs, cosine, lr 2e-4 → 5e-5, warmup 100, paged AdamW 8-bit, seed 13
- 3114 steps × 3.5 s = **181 min**, peak VRAM **1.97 GiB**
- Loss 0.54 → 0.094, token accuracy 99.8%

## Experiment C (verifier) — same 100 val questions, k=5 candidates

| Cell | Pipeline | Execution accuracy |
|---|---|---|
| B | QLoRA, single greedy | 74.0% |
| C | B + k=5 (1 greedy + 4 sampled, T=0.8) + gold-free verifier selection | **79.0%** |

- Paired B→C: **5 improvements, 0 regressions** (74 both-right, 21 both-wrong), exact McNemar p=0.0625 (two-sided)
- Greedy candidate is **bit-identical to B** (0/100 mismatches) — determinism verified
- Self-consistency majority (k=5, offline from saved candidates): also 79.0%
- **Oracle pool ceiling: 79** — verifier selection already equals the best achievable
  over the candidate pool (whenever any candidate was correct, the verifier picked it)
- Artifacts: `reports/C_val_n100.jsonl` (all candidates + scores + result signatures),
  `reports/eval_C_val_n100.json`

## Experiment D (bounded repair) — same 100 val questions, τ=3.0, ≤2 iterations

| Cell | Pipeline | Execution accuracy |
|---|---|---|
| C | verifier selection over k=5 | 79.0% |
| D | C + verifier-gated repair (score < τ → ≤2 rounds, verifier sole judge) | **80.0%** |

- Paired C→D: **1 improvement, 0 regressions** (exact McNemar p=1.0, single discordant pair)
- 28/100 flagged by τ=3.0 (tuned on validation: catches 9/21 known-wrong C items),
  35 repair attempts, **5 replacements** — every replacement strictly score-improving,
  zero regressions by construction + empirically
- **Execution errors in final output: 0** (C had 2 no_such_column failures; both repaired)
- τ rationale: verifier scores poorly separate correct/unknown (wrong p50 = correct p50
  = 3.30 — consensus-wrong answers score high); repair is gated so blast radius is
  bounded by the strict-improvement rule
- Artifacts: `reports/D_val_n100.jsonl`, `reports/eval_D_val_n100.json`

## Full matrix (validation, n=100, seed 0)

| A (base) | B (+QLoRA) | C (+verifier) | D (+repair) |
|---|---|---|---|
| 47.0% | 74.0% | 79.0% | **80.0%** |

## Baseline context (earlier gates)

- A on n=20 val (Phase 4/5): 50.0% — official-evaluator agreement: **100%** (M2)
- 3B QLoRA (B2): **rejected** — OOM at seq 1024/768/512 (usable VRAM 3.68 GiB)
