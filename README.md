# Text-to-SQL Fine-Tuning, Execution-Based Verification, and Bounded Repair

Can a small open-source LLM be fine-tuned for text-to-SQL, and does
execution-based verification + bounded repair improve end-to-end correctness?
This repo answers both empirically with an experiment matrix evaluated on
Spider and BIRD under the benchmarks' own **execution-accuracy** methodology
(our harness agrees item-for-item with the official
`taoyds/test-suite-sql-eval` evaluator — see `tests/test_official_agreement.py`).

## Status

**v2 is the current result.** An audit found that three of v2's reported
results were produced by code that did not do what the protocol said, and that
every v2 tuning decision was made on a measurement unable to detect the
problem. Those defects are fixed (see *Audit fixes* below); the fixes change
mechanism, not the numbers in the table.

> **Negative result — do not repeat this.** A follow-up experiment retuned the
> adapter to the QLoRA paper's reference settings (rank 64, lr 2e-4) on a
> 20-database holdout. Spider improved by ~0.4 pts, but **BIRD lost ~8 pts**
> and the fine-tuning gain over the base model fell from +9.7 to +4.0. The
> evidence points to the more aggressive adapter fitting Spider's schemas
> harder and transferring them worse: `no_such_column` errors on BIRD stopped
> falling after fine-tuning, and the candidate pool ceiling dropped. It was
> invisible on the holdout, which was disjoint in schema but still Spider in
> domain. **Treat rank/lr as BIRD-sensitive, and do not judge OOD transfer from
> a Spider-only holdout.**

All artifacts of that experiment were discarded. This note is the only record
of it.

## TL;DR results (locked test sets, full splits) — v2

| Cell | Pipeline | Spider dev (1,034) | BIRD dev (1,534) |
|---|---|---|---|
| A | Qwen2.5-Coder-1.5B base + prompt | **54.2%** | **21.3%** |
| B | A + QLoRA fine-tuning | **66.8%** | **31.0%** |
| C | B + execution-based verifier (k=5) | **69.8%** | **35.6%** |
| D | C + "bounded repair" (≤2 rounds) | **69.8%** | **36.2%** |

Cell D re-scored with the corrected harness reads 69.83% / 36.20% (it was
reported as 70.0% / 36.2% while two Spider and one BIRD question were being
skipped as broken gold). The corrected harness scores three more items, all of
which the model gets wrong, so the true figure is marginally lower.

All three gates (B≥A, C≥B, D≥C) hold on both locked tests. Fine-tuning
dominates (McNemar p < 5e-7 on both); the verifier adds +3.0 on Spider
(**p = 3.3e-5**, not p < 5e-7) and +4.6 on BIRD (p < 5e-7); repair adds +0.2
(n.s.) on Spider and +0.7 (p = 0.0063) on BIRD. Multi-turn SParC (eval-only):
57.4% → 60.2% turn accuracy.

Three caveats attached to these numbers, all fixed in the code since (see
*Audit fixes* below):

1. **Cell D was not bounded.** `--tau 3.0` was compared against a weight vector
   whose maximum reachable score is 2.0, so every item was flagged
   (1034/1034 and 1534/1534). D is "repair everything, accept only strict
   improvements", not "repair the untrusted tail".
2. **The verifier could rank a crashing query above a working one** under those
   same weights; 37 Spider / 323 BIRD selections took a failed candidate.
3. **The verifier weights are a coin flip.** Twelve search seeds give twelve
   different vectors at the same 0.80 on n=100 — one item of difference — and
   the un-tuned rule weights score *higher* on locked Spider.

Full statistics, CIs, failure taxonomy and ablations: `results_v2/analysis_report.md`,
`reports/results.md`.


## Hardware & software requirements

- **GPU**: any CUDA GPU with ≥ 4 GB VRAM for training (RTX 3050 Laptop 4 GB
  used). Evaluation/analysis/plots are CPU-only.
- **VRAM headroom is real but capped by gradient checkpointing.** Training peaks
  at ~2.1-2.3 GiB of the 3.68 GiB usable on a 4 GB card. Disabling gradient
  checkpointing — the obvious way to use the spare memory — OOMs at every batch
  size and rank tested, so it stays on.
- **RAM**: ≥ 8 GB (15 GB available here). **Disk**: ~10 GB
  (model cache 5.8 GB, datasets ~2.3 GB, outputs < 1 GB).
- **Python 3.11** (pinned), then:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # torch, transformers, trl, peft, bnb, ...
```

Key versions used: torch 2.14, transformers 5.17, trl 1.14, peft 0.21,
bitsandbytes 0.50.2, datasets 5.0.1, sqlparse 0.6, matplotlib 3.11.
(`pyproject.toml` documents which phase forced each dependency.)

## Repository layout

```
src/t2sql/
  prompts/        schema text + prompt templates (incl. dialogue/repair prompts)
  inference/      single LLM wrapper (4-bit NF4, adapter-aware) + SQL extraction
  sandbox/        read-only SQLite executor, timeouts, error taxonomy
  verifier/       gold-free features + rule scorer (cells C/D)
  repair/         verifier feedback builder (cell D)
  training/       SFT example builder
  eval/           GOLD-LIVE: official execution-accuracy port, harness, stats,
                  SParC metrics, failure taxonomy, dataset resolution
scripts/          CLI runners (download, split, train, run_*, eval, analyze, plots)
                  + sweep_train (summary accumulates across invocations),
                  tune_verifier (seed-stability + calibrated threshold),
                  validate_train_gold, audit_leakage, analyze_repair_gate
tests/            107 tests (adversarial sandbox, official-eval agreement, gold
                  isolation, taxonomy, stats, split guards, compile guard,
                  and tests/test_audit_fixes.py — one test per audit defect)
reports/          design.md, results.md, v2_protocol.md, v2_checklist.md,
                  raw eval JSONs, prediction files
results/          v1 analysis outputs; results_v2/ = same layout for v2
experiments/B/    v1 adapter; experiments/B_v2/ = v2 adapter + train_config,
                  val_metrics.json, train_log.csv; experiments/sweep/ = Stage-2 grid
tools/            cloned official evaluators (gitignored)
```

## Reproduce (in order)

```bash
# 1. data (downloads + verifies; spider/sparc via Google Drive, bird via OSS)
.venv/bin/python scripts/download_data.py
# build the SParC prompt/gold split: run_sparc.py and run_eval.py both read
# these, and nothing else creates them
.venv/bin/python scripts/build_sparc_eval.py
# stratify the val carve AND hold out whole databases for the probe. The
# holdout is required: val.json shares all 140 databases with training, so it
# cannot measure generalization to unseen schemas.
.venv/bin/python scripts/make_split.py --stratify --holdout-dbs 20
.venv/bin/python scripts/audit_leakage.py           # -> reports/leakage_audit.md
.venv/bin/python scripts/validate_train_gold.py     # -> data/splits/train_quarantine.json
.venv/bin/python scripts/profile_data.py            # -> reports/data_profile.md

# 2. training data (prompt flags must match the runners exactly)
.venv/bin/python scripts/build_training_set.py \
    --prompt-variants 4 --few-shot 2 --schema-samples --question-values
#    -> data/splits/train_sft.json

# 3. train — v2 recipe (r=16, lr 1e-4, 3 epochs). Do NOT raise the rank or the
#    learning rate casually: see the negative-result note above.
#    (smoke path: --max-steps 2 --limit 64)
.venv/bin/python scripts/train_qlora.py --out experiments/B_v2 \
    --rank 16 --epochs 3 --lr 0.0001 --seed 13 --max-len 3072 \
    --prompt-variants 4 --schema-samples \
    --few-shot-file data/splits/fewshot_block.txt

# 4. verifier weights + the repair threshold, calibrated together on the
#    database-disjoint holdout. The tuner ships the rule weights unless the
#    searched vector reproduces across seeds, and emits a feasible
#    flag_threshold; run_repair reads it and rejects any tau at or above the
#    reachable ceiling (which is what made v2's tau=3.0 flag every item).
.venv/bin/python scripts/run_verify.py --split holdout --n 0 --k 5 \
    --adapter experiments/B_v2/adapter --prompt-variants 4 --schema-samples \
    --few-shot-file data/splits/fewshot_block.txt \
    --out reports/C_holdout.jsonl
.venv/bin/python scripts/tune_verifier.py --cfile reports/C_holdout.jsonl \
    --split holdout --out experiments/verifier_weights.json
# 5. locked benchmarks — ONE run each (prereg: reports/v2_protocol.md)
bash scripts/run_final_benchmarks_v2.sh

# 6. analysis, figures
.venv/bin/python scripts/analyze_results.py --suffix _v2   # -> results_v2/
.venv/bin/python scripts/benchmark_resources.py           # fig6 input
.venv/bin/python scripts/make_plots.py --suffix _v2       # -> results_v2/plots/

# 7. tests
.venv/bin/python -m pytest -q                    # 107 passed
```

Hyperparameter sweep (optional, resumable; the summary accumulates across
invocations):

```bash
.venv/bin/python scripts/sweep_train.py --epochs 1 2 3 --lr 0.0001 0.0002 --seeds 13
```

Single evaluation of any predictions file (never needs a GPU):

```bash
.venv/bin/python scripts/run_eval.py --dataset bird --split dev --preds reports/bird_D_dev_v2.jsonl
```


## Datasets

| Dataset | Split used | Size | Source |
|---|---|---|---|
| Spider train | train (5,427) / val (700, carved) / **holdout (873, 20 whole databases)** | 7,000 | official zip (gdown) |
| Spider dev | **locked test**, touched once | 1,034 | official zip |
| BIRD dev | **held-out benchmark**, touched once | 1,534 | official OSS release |
| SParC dev | eval-only, multi-turn | 422 dialogues / 1,203 turns | official zip (gdown) |

`scripts/download_data.py` is idempotent and verifies sizes; test splits and
`train_others` are excluded by design. All datasets are CC BY-SA 4.0.

**The holdout exists because `val.json` cannot measure generalization**: all
140 of its databases appear in training and 532/700 of its questions have an
exact `(database, gold SQL)` match there, while both locked benchmarks consist
entirely of unseen databases (0/20 Spider, 1/11 BIRD).

## Training configuration

QLoRA throughout: NF4 4-bit base (double quantization), LoRA on all attention
and MLP projections, assistant-only loss, paged AdamW 8-bit. Training aborts
rather than truncating an example whose gold SQL would be clipped.

**v2 (current):** r=16/α=32/dropout 0.05, 3 epochs, lr 1e-4, max_len 3072,
seed 13, 4,716 steps, 591 min, peak 2.09 GiB, loss 0.53 → 0.0155.
Full config: `experiments/B_v2/train_config.json`, curve: `val_metrics.json`
+ `train_log.csv`.

**Measured recipe facts** (RTX 3050 Laptop, 3.68 GiB usable): rank 64 costs
+2.8% per step versus rank 16; batch size 1 is the fastest per sample (1.71 vs
2.29 s); sequence packing is 3.2× *slower* per step and buys nothing at batch
size 1, so it is off; gradient checkpointing cannot be disabled without OOM.

**v1 (provenance):** 6,300 examples, 2 epochs (3,114 steps), lr 2e-4,
max_len 2048, seed 13, 181 min, 1.97 GiB, loss 0.55 → 0.018
(`experiments/B/`).

## Audit fixes

An audit of the pipeline found defects that changed what cells C and D
measured. All are fixed, with one regression test each in
`tests/test_audit_fixes.py`. None of them changes a number in the results
table above; they change whether the code did what the protocol said.

| Defect | Fix |
|---|---|
| A failed execution floored the verifier score at `0.0`, so under penalty weights argmax could rank **crashing SQL above working SQL** (37 Spider / 323 BIRD picks) | failures return a sentinel below any reachable executed score |
| `tau=3.0` was compared against a weight vector whose reachable maximum is 2.0, so **every item was flagged** (1034/1034, 1534/1534) — "bounded repair" repaired everything | the tuner emits a `flag_threshold` calibrated on its own weight scale; `run_repair` refuses a `tau` at or above the ceiling, before loading the GPU |
| Cell C scored pool self-consistency as `count/k` and cell D as `(count+1)/(k+1)`, so the repair margin compared **two different scales** (drift 0.267 vs a 0.1 margin) | one shared `pool_consistency()`; `repair_item` recomputes the incumbent instead of trusting the stored C score |
| Tuning used `val[:50]` — **10 of 140 databases** — and 532/700 of `val.json` has an exact `(db, gold SQL)` match in training, so tuning measured recall | `make_split.py --holdout-dbs` carves whole databases out, with assertions; the probe draws round-robin across them |
| Verifier weights were tuned on n=100 where 12 seeds give 12 different vectors at the same accuracy | the tuner reports seed stability and **ships the rule weights unless the searched vector reproduces** |
| Schema "value hints" used `ORDER BY 1 LIMIT 3`, recovering **8.2%** of BIRD gold literals | question-conditioned value retrieval recovers **42.1%** (`--question-values`, opt-in) |
| The harness dropped 3 questions the official evaluator scores (non-UTF-8 text; a 30 s timeout vs the official 60 s) | `text_factory` mirrors the official connection; timeout matches `TIMEOUT = 60` |
| Stages were skipped on `[ -f output ]`, so an **interrupted run's partial file looked finished** and B/D were built from a truncated pool | stages check record count; generators take `--resume` |


## Evaluation protocol

- **Execution accuracy** against gold SQL with the official postprocessing
  (DISTINCT stripping, column-permutation search, bag equality, ORDER BY
  row-order rule) — ported from `taoyds/test-suite-sql-eval` and verified
  identical item-for-item; secondary metric: normalized exact match.
- **Gold isolation**: only `src/t2sql/eval/` reads gold. Prediction files
  carry no gold; generation/verifier/repair scripts never see the current
  turn's SQL (structural tests enforce this). SParC uses the official
  gold-history protocol (previous turns only).
- **Statistics**: 1,000-replicate paired bootstrap CIs and exact McNemar on
  the same question indices for every cell comparison.
- **Determinism**: greedy decoding for A/B (bit-identical, verified), fixed
  seeds for sampling (k=5, T=0.8) and bootstraps.

## Limitations

- One model family/size (1.5B); a 3B variant was rejected for OOM on this
  GPU at every context length (documented in `reports/design.md` §10).
- **All v2 thresholds were tuned on a measurement that could not detect the
  problem**: `val.json` shares all 140 databases with training, 532/700 of its
  questions have an exact `(db, gold SQL)` match there, and the probe sampled
  the first 50 items — 10 databases. k/T, the verifier weights, τ and the
  best checkpoint all inherit that. The holdout fixes the measurement going
  forward; it does not retroactively repair the v2 tuning history.
- **The v2 verifier weights are not reproducible**: 12 search seeds give 12
  different vectors at the same n=100 accuracy (one item), and the plain rule
  weights beat them on locked Spider. `tune_verifier.py` now reports this and
  `tune_verifier.py` now reports this automatically.
- BIRD still shows ~4.9 pts of unclaimed pool headroom (oracle 0.405 vs C
  0.356); the pool, not the sampler, is the binding constraint.
- Repair gains are small: n.s. on Spider (2↑/0↓), significant on BIRD v2
  (11↑/1↓, p=0.0063). Repair feedback uses a documented component-level
  approximation of SQL structure (keyword-anchored spans, not a full parse),
  so the taxonomy cannot isolate every objective-§13 category.
- **Evidence is a train/inference format mismatch**: BIRD prompts carry
  `evidence` on 1,386/1,534 items and 0/6,288 training examples do. 54.2% of
  BIRD gold literals appear only in that field, and 45.8% of v2's BIRD errors
  have one. Deliberately unimplemented — synthesizing hints from gold SQL is
  gold-derived supervision and would break the gold-isolation property this
  repo is built on.
- Training-seed variance is measured on validation only (3 seeds, ±2.0 pts),
  and under the broken probe; bootstrap CIs cover question sampling, not seed
  variance.
- Trainer computes in bf16 while inference loads fp16. Unchanged: it alters
  inference numerics and belongs to a declared run, not a patch.

## Security notes

SQL runs in-process but strictly read-only: `file:...?mode=ro` URIs,
`PRAGMA query_only`, an allowlist authorizer (SELECT-only, no PRAGMA /
ATTACH / writes / dynamic loading), execution timeouts, and row caps —
see `tests/test_sandbox.py` for the adversarial suite.
