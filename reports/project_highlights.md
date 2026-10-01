# Project Highlights — Resume Reference & Interview Prep

**Text-to-SQL Fine-Tuning, Execution-Based Verification, and Bounded Repair**
End-to-end ML research system: fine-tune a 1.5B open model, verify its SQL by
executing it, repair what fails — evaluated with publication-grade discipline
on locked benchmarks, entirely on a 4 GB laptop GPU.

**Stack:** Python 3.11 · PyTorch · HuggingFace Transformers/TRL · PEFT (QLoRA) ·
bitsandbytes (NF4) · SQLite · pytest · matplotlib · bash/git — 30 commits,
92 tests, ~4 K LOC (`src/` + `scripts/`).

---

## 1. Headline numbers (safe to put on a resume)

| Metric | Value |
|---|---|
| Fine-tuning gain (A→B, locked tests) | **+12.6 pts Spider / +9.7 pts BIRD** (v2); **+16.7 / +6.3** (v1); up to **+27 pts** on validation — all McNemar p < 5e-7 |
| Verifier gain (B→C) | **+4.6 pts BIRD, +3.0 pts Spider** (p < 5e-7 / p = 3.3e-5) |
| Repair (C→D) | +0.7 BIRD (**p = 0.0063**), +0.2 Spider (n.s.) — reported honestly |
| Final scores v2 (A→B→C→D) | Spider **54.2 / 66.8 / 69.8 / 70.0%** · BIRD **21.3 / 31.0 / 35.6 / 36.2%** |
| vs prior experiment (v1) | **BIRD +3.6 / +5.2 / +4.7 pts** from methodology upgrades alone |
| Compute budget | **RTX 3050, 4 GB VRAM** — best run 2.09 GiB peak, 591 min; QLoRA adapter = 18.5 M trainable params (1.18%) |
| Training efficiency | 6,288 examples → 4,716 steps, loss 0.53 → 0.015, best-checkpoint selected by live execution-accuracy probe |
| Experiment rigor | 2 full experiments (v1 + preregistered v2), **12/12 gates passed**, locked test sets touched once each, 3-seed variance (78/76/80, ±2.0) |

---

## 2. Resume bullets (pick 4–6, tune to the job description)

**ML-engineer flavor**

- Built an end-to-end text-to-SQL system (QLoRA fine-tuning → execution-based
  candidate verification → bounded self-repair) raising execution accuracy on
  locked BIRD dev from **21.3% base to 36.2%** (+70% relative), with every
  pipeline stage gated by paired significance tests.
- Fine-tuned Qwen2.5-Coder-1.5B with **QLoRA on a 4 GB laptop GPU** (18.5 M
  trainable params, 2.09 GiB peak VRAM, 10 h/run); built a resumable
  hyperparameter sweep (6 configs × 3 seeds) whose winner generalized at
  **78.0 ± 2.0** on validation.
- Shipped a **gold-free SQL verifier** (14 execution-derived features + rule
  scorer) that added **+3.0/+4.6 pts** on Spider/BIRD (McNemar p < 5e-7) by
  selecting the best of 5 sampled candidates — beating self-consistency
  majority where it matters and matching it where it doesn't (documented).
- Engineered a **bounded repair loop** (execution-error feedback, ≤2 rounds,
  strict acceptance gate) and discovered via replay forensics that the naive
  score gate was net-harmful (0 helpful / 2 harmful replacements) → replaced
  with margin + pool-agreement veto, eliminating regressions.

**Research-rigor flavor**

- Ran a **preregistered two-experiment study** (v1 + v2) with one-touch locked
  test sets (Spider 1,034 / BIRD 1,534), exact McNemar + paired bootstrap CIs,
  a 13-category SQL failure taxonomy, and an official-evaluator agreement gate
  (100% item-for-item match) — 12/12 experimental gates held.
- Applied data-quality engineering that out-returned modeling tricks:
  cross-split leakage audit (0 exact overlaps), execution-validation of all
  training gold (3 broken targets quarantined), duplicate removal, and a
  cluster-atomic stratified split — each with regression tests to prevent
  silent drift.
- Delivered **honest negative results**: bounded repair adds ~0 on Spider,
  learned verifier scorer unnecessary (rules hit the pool oracle) — reported
  with evidence instead of suppressed, alongside the significant gains.

**One-liner (portfolio header / LinkedIn)**

> Text-to-SQL research system on a 4 GB laptop: QLoRA fine-tuning (+9.7–12.6 pts
> locked), execution-based verifier (+3–4.6 pts, p<5e-7), bounded repair —
> preregistered, locked benchmarks, 92 tests, all gates passed.

---

## 3. Techniques & methodology (reference for interviews)

### 3.1 Data engineering (Stage 0)
- **Sources:** Spider (7,000 train → 6,300/700 carve), BIRD dev (locked, with
  `evidence` field), SParC (1,203 dialogue turns, eval-only, gold-history protocol).
- **Stratified, cluster-atomic split:** per-db largest-remainder quota; duplicate
  `(db_id, question)` clusters assigned atomically → train/val can never share a
  key; deterministic; guarded by a regression test accepting only the two known carves.
- **Gold quarantine:** executed all 6,300 training targets in the sandbox →
  3 unrunnable gold SQL excluded (missing table; `ORDER BY` before `INTERSECT`).
- **Dedup:** 10 duplicate questions removed → final 6,288 training examples.
- **Leakage audit:** exact (db, question) overlap train/val/dev = **0**;
  5-word-shingle near-dup scan; same-db SQL-template sharing documented.
- **Gold isolation:** only `src/t2sql/eval/` may read gold; verifier, repair, and
  runners are structurally tested to never import it; prediction files carry no gold.

### 3.2 Training (QLoRA)
- Base: Qwen2.5-Coder-1.5B-Instruct; **NF4 4-bit**, LoRA **r=16, α=32, dropout .05**
  on all 7 projection modules (q/k/v/o/gate/up/down); assistant-only loss
  (schema/instruction tokens unmasked for loss); paged AdamW 8-bit, cosine schedule.
- **Frozen recipe from a real sweep:** epochs {1,2,3} × lr {1e-4, 2e-4} →
  **ep3/lr1e-4** (78%) beat ep2 (76%) and ep1 (68–70%); ep3/lr2e-4 overfit (74%,
  lower loss but worse exec — memorization signature).
- **3 seeds:** 78 / 76 / 80 → mean **78.0 ± 2.0** (seed 13 pre-registered as final).
- **Trainer hardening built in:** per-eval val loss + **execution-accuracy probe**
  (50 val items every 500 steps) → **best-checkpoint selection** (final step had
  regressed 80→78; probe caught it); **overflow fail-fast** (no silent right-truncation
  of gold SQL — longest example 3,021 ≤ 3,072); CSV learning curves; resumable runs.
- 3B variant spike-tested and **rejected for OOM** at all context lengths (documented).

### 3.3 Prompt engineering (train = inference, byte-consistent)
- **4 deterministic instruction variants** (sha1(question) % n — both training and
  inference derive the identical variant with no shared state).
- **2-shot few-shot block** built from train only (seed 13), threaded through
  generation, repair, and training.
- **Schema enrichment:** verbatim DDL + row counts + ≤3 non-null sample values
  per column, **gold-free**, capped at 30 lines/schema (prompt-size budget guard —
  caught an 8,040-token blow-up before it cost a training run).

### 3.4 Verification (cell C)
- Generate **k=5 candidates**: 1 greedy (bit-identical to cell B) + sampled at
  T=0.8 until 5 *unique* SQLs or 2× budget (dedup — pools averaged ~3–4 unique,
  saving GPU); `max_new_tokens=512`.
- Execute every candidate in the sandbox; extract **14 gold-free features**:
  executed, result-signature consistency (self-consistency share),
  answer-in-question, count-shape match, empty, truncated, slow, table count,
  question↔schema table-link overlap, execution-error taxonomy one-hots…
- **Rule scorer** (hand weights + tuned variant): score ranks candidates; failed
  execution hard-zero. Tuned by seeded random search (5,000 trials) offline on
  recorded features → 79→80 selection accuracy; **pool oracle 81** — rules nearly
  saturate the pool, so no learned scorer was justified (design §6 condition met).
- Ablations vs **self-consistency majority**: Spider 0.700 (SC) ≈ 0.698 (verifier);
  BIRD 0.343 (SC) < 0.356 (verifier) — verifier's edge appears exactly where
  candidate diversity is high.

### 3.5 Bounded repair (cell D)
- Gate: verifier score < τ=3.0 (val-tuned) → at most **2 repair rounds**;
  prompt = same system prompt + previous SQL + feature-derived feedback
  (execution error verbatim, empty/count-shape/consistency critiques,
  pool-disagreement signal "3 of 5 attempts produced a different result").
- **Acceptance gate (v2):** score gain ≥ **0.1** AND result must not contradict
  the candidate-pool majority — derived from replay forensics showing the bare
  score gate was net-harmful (0 helpful / 2 harmful replacements on val).
- Blast radius bounded by construction: unflagged items keep C's prediction;
  the rule scorer is the sole judge (never gold).

### 3.6 Evaluation & statistics
- **Official-fidelity gates:** ported Spider test-suite-sql-eval (agreement test
  = 100% item-for-item on real predictions) and SParC's official evaluator;
  BIRD execution accuracy; exact-match reported as secondary only.
- **Locked protocol:** dev sets touched **once** per experiment; v2 preregistered
  in `reports/v2_protocol.md` before the run; predictions carry index/db/question,
  harness verifies pairing before scoring.
- **Stats:** paired bootstrap CIs, exact McNemar on discordant pairs, per-cell
  execution-valid rate, difficulty breakdown, oracle-pool ceilings, SC ablations.
- **13-category failure taxonomy** (no_such_column, wrong table, join missing,
  aggregate misuse, syntax, literal errors…) via null-whitened clause spans —
  drove the diagnosis: BIRD `no_such_column` collapses 643 → 193 A→D.

### 3.7 Security engineering
- Read-only SQLite sandbox: `mode=ro` URI, authorizer denying file writes /
  ATTACH / dangerous functions, execution timeouts, row caps, schema-cache
  per DB; **adversarial test suite** (SQL injection, path traversal, zip-slip
  on download, forbidden functions) — a project where the untrusted input is
  *model-generated SQL* by design.

### 3.8 Testing & process
- **92 tests:** sandbox adversarial, official-evaluator agreement, structural gold
  isolation, split-regression guards, prompt byte-identity, repair loop
  (budget/guards/gating), stats correctness, `py_compile` guard over every script.
- Process discipline: one commit per stage, incident log (a `--help` footgun
  silently rewrote the split → restored, argparse added, regression test written),
  transformer-5.x API breakages found by crash-not-silent-failure.

---

## 4. Research findings — best interview talking points

1. **Specialization beats prompting:** 1.5B + 18.5 M LoRA params beat its own
   base by +16 pts locked / +27 pts validation — the highest-leverage move on
   consumer hardware.
2. **Verification's value is diversity-dependent:** execution-based selection
   adds +3–4.6 pts significant; it matches SC on Spider but wins on BIRD where
   candidates disagree most.
3. **Repair is weak by design — bound it, don't hide it:** forensics (0 helpful /
   2 harmful) led to a dual acceptance gate; final result: neutral on Spider,
   **significant +0.7 (p=0.0063) on BIRD**.
4. **Process > tricks:** quarantine + dedup + stratified split + train/infer
   prompt consistency produced the largest single-version jump (BIRD +4.7 pts
   end-to-end) — none of it was a modeling idea.
5. **My own gate was nearly contaminated:** the "reference" adapter had seen 91%
   of the new val set during training — caught before trusting the comparison;
   re-baselined cleanly (A→B delta reproduced: +26 vs +27 in the clean era).
6. **OOD is the frontier:** BIRD needs evidence/schema supervision; prompts alone
   bought +4.7; declared next levers: evidence augmentation, wider sample budgets,
   4.9-pt unclaimed pool headroom (oracle 0.405 vs 0.356).

---

## 5. Artifacts map (where to point reviewers)

| What | Where |
|---|---|
| TL;DR results table | `README.md` |
| Preregistered protocol (written before the locked run) | `reports/v2_protocol.md` |
| Full methodology plan w/ checked-off stages | `reports/v2_checklist.md` |
| v1 vs v2 results narrative | `reports/results.md` |
| Locked predictions (one-touch) | `reports/{spider,bird}_{A..D}_dev_v2.jsonl` |
| Statistics, CIs, taxonomy, ablations | `results_v2/analysis_report.md` + per-cell JSON |
| Publication figures (6) | `results_v2/plots/fig1..fig6.png` |
| Training evidence (curve, config, best step) | `experiments/B_v2/{val_metrics.json,train_config.json,train_log.csv}` |
| Sweep grid + seeds | `experiments/sweep/summary.md` |
| Leakage audit / data profile | `reports/leakage_audit.md`, `reports/data_profile.md` |
| Design doc (§ references used throughout) | `reports/design.md` |
| Tests | `tests/` (92) |

## 6. Reproduce in 6 commands

```bash
.venv/bin/python scripts/download_data.py && .venv/bin/python scripts/make_split.py --stratify
.venv/bin/python scripts/build_training_set.py --prompt-variants 4 --few-shot 2 --schema-samples
.venv/bin/python scripts/train_qlora.py --out experiments/B_v2 --epochs 3 --lr 0.0001 --seed 13 \
    --prompt-variants 4 --few-shot-file data/splits/fewshot_block.txt --schema-samples
bash scripts/run_final_benchmarks_v2.sh          # locked, one touch
.venv/bin/python scripts/analyze_results.py --suffix _v2 && .venv/bin/python scripts/make_plots.py --suffix _v2
.venv/bin/python -m pytest -q                    # 92 passed
```

## 7. Limits to state proactively (they will ask)

- One model family/size (1.5B; 3B OOM'd on 4 GB — spike-tested, documented).
- BIRD absolute ceiling still low (36.2%); `evidence` never used in training
  (scoped out, declared in protocol) — the obvious next lever.
- Repair gain small (n.s. on Spider); verifier/repair thresholds tuned on
  n=100 validation only; seed variance measured (±2.0) but not on the locked set.
- SParC eval-only (no multi-turn training — locked "Option A").