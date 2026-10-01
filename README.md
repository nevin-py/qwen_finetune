# Text-to-SQL Fine-Tuning, Execution-Based Verification, and Bounded Repair

Can a small open-source LLM be fine-tuned for text-to-SQL, and does
execution-based verification + bounded repair improve end-to-end correctness?
This repo answers both empirically with an experiment matrix evaluated on
Spider and BIRD under the benchmarks' own **execution-accuracy** methodology
(our harness agrees item-for-item with the official
`taoyds/test-suite-sql-eval` evaluator — see `tests/test_official_agreement.py`).

## TL;DR results (locked test sets, full splits)

Final experiment **v2** (prereg: `reports/v2_protocol.md`), v1 in brackets:

| Cell | Pipeline | Spider dev (1,034) | BIRD dev (1,534) |
|---|---|---|---|
| A | Qwen2.5-Coder-1.5B base + prompt | **54.2%** [50.8] | **21.3%** [21.1] |
| B | A + QLoRA fine-tuning | **66.8%** [67.5] | **31.0%** [27.4] |
| C | B + execution-based verifier (k=5) | **69.8%** [70.4] | **35.6%** [30.4] |
| D | C + bounded repair (≤2 rounds) | **70.0%** [70.8] | **36.2%** [31.5] |

All three gates (B≥A, C≥B, D≥C) hold on **both** locked tests in both
experiments. v2's prompt/schema upgrades trade a flat Spider (−0.7 pts,
within noise) for **+3.6/+5.2/+4.7 pts on BIRD** — the OOD benchmark.
Fine-tuning dominates (McNemar p < 5e-7; stored rounded to 6 dp so JSON
reads 0.0); verifier +3.0/+4.6 (p < 5e-7); repair +0.2 (n.s.) on Spider,
+0.7 (p = 0.0063) on BIRD. Multi-turn SParC (eval-only): v2 57.4% → 60.2%
turn accuracy. Full statistics, CIs, failure taxonomy, and ablations:
`results_v2/analysis_report.md`, `reports/results.md`.

## Hardware & software requirements

- **GPU**: any CUDA GPU with ≥ 4 GB VRAM for training (RTX 3050 Laptop 4 GB
  used; QLoRA peaks at **1.97 GiB**). Evaluation/analysis/plots are CPU-only.
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
                  + v2 tooling: sweep_train, tune_verifier, validate_train_gold,
                  audit_leakage, analyze_repair_gate, run_final_benchmarks_v2.sh
tests/            92 tests (adversarial sandbox, official-eval agreement, gold
                  isolation, taxonomy, stats, split guards, compile guard)
reports/          design.md, results.md, v2_protocol.md, v2_checklist.md, raw
                  eval JSONs, prediction files
results/          v1 analysis outputs; results_v2/ = same layout for v2
experiments/B/    v1 adapter; experiments/B_v2/ = v2 adapter + train_config,
                  val_metrics.json, train_log.csv; experiments/sweep/ = Stage-2 grid
tools/            cloned official evaluators (gitignored)
```

## Reproduce (in order)

```bash
# 1. data (downloads + verifies; spider/sparc via Google Drive, bird via OSS)
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/make_split.py --stratify   # 6300 train / 700 val (v2, cluster-atomic)
.venv/bin/python scripts/audit_leakage.py           # -> reports/leakage_audit.md
.venv/bin/python scripts/validate_train_gold.py     # -> data/splits/train_quarantine.json
.venv/bin/python scripts/profile_data.py            # -> reports/data_profile.md

# 2. train (overnight on 4 GB; smoke path: --max-steps 2)
.venv/bin/python scripts/build_training_set.py \
    --prompt-variants 4 --few-shot 2 --schema-samples   # -> train_sft.json (6288)
#   stage-2 sweep (epochs x lr x seeds, resumable) then the frozen recipe:
.venv/bin/python scripts/sweep_train.py --epochs 1 2 3 --lr 0.0001 0.0002
.venv/bin/python scripts/train_qlora.py --out experiments/B_v2 \
    --epochs 3 --lr 0.0001 --seed 13 --prompt-variants 4 \
    --few-shot-file data/splits/fewshot_block.txt --schema-samples
#   verifier weights (offline, from a val C run):
.venv/bin/python scripts/tune_verifier.py --cfile reports/C_val_n100_v2up.jsonl

# 3. locked benchmarks — ONE run each (v2 prereg: reports/v2_protocol.md)
bash scripts/run_final_benchmarks_v2.sh          # spider + bird A->D + sparc A/B

# 4. analysis, figures
.venv/bin/python scripts/analyze_results.py --suffix _v2   # -> results_v2/
.venv/bin/python scripts/make_plots.py --suffix _v2        # -> results_v2/plots/

# 5. tests
.venv/bin/python -m pytest -q                    # 92 passed
```

Single evaluation of any predictions file (never needs a GPU):

```bash
.venv/bin/python scripts/run_eval.py --dataset bird --split dev --preds reports/bird_D_dev.jsonl
```

## Datasets

| Dataset | Split used | Size | Source |
|---|---|---|---|
| Spider train | train (6,300) / val (700, carved, seed 13) | 7,000 | official zip (gdown) |
| Spider dev | **locked test**, touched once | 1,034 | official zip |
| BIRD dev | **held-out benchmark**, touched once | 1,534 | official OSS release |
| SParC dev | eval-only, multi-turn | 422 dialogues / 1,203 turns | official zip (gdown) |

`scripts/download_data.py` is idempotent and verifies sizes; test splits and
`train_others` are excluded by design. All datasets are CC BY-SA 4.0
(see dataset providers' pages for terms).

## Training configuration (cell B)

QLoRA: NF4 4-bit base + LoRA r=16/α=32/dropout 0.05 on all attention and MLP
projections (18.5M trainable, 1.18%); assistant-only loss, cosine lr, warmup
100, paged AdamW 8-bit.

**v2 (final):** 6,288 examples (stratified split, dedup + gold-quarantine),
prompts = 4 instruction variants + 2-shot block + schema sample values
(30-line budget), **3 epochs, lr 1e-4, max_len 3072, seed 13** (sweep of
6 configs × 3 seeds: 78/76/80, mean 78.0 ± 2.0 on val), per-500-step
execution-accuracy probe with best-checkpoint selection (step 2500,
probe 80%). 4,716 steps, wall time 591 min, peak VRAM 2.09 GiB,
loss 0.53 → 0.015. Full config: `experiments/B_v2/train_config.json`,
curve: `val_metrics.json` + `train_log.csv`.

**v1 (provenance):** 6,300 examples, 2 epochs (3,114 steps), lr 2e-4,
max_len 2048, seed 13, 181 min, 1.97 GiB, loss 0.55 → 0.018
(`experiments/B/`).

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
- All thresholds are **validation-tuned only**: k=5/T=0.8, verifier
  weights (grid, 5,000 trials), τ=3.0, repair margin 0.1 + pool veto
  (replay showed the bare score gate was net-negative). BIRD still shows
  ~4.9 pts of unclaimed pool headroom (oracle 0.405 vs C 0.356).
- Repair gains remain small: n.s. on Spider (2↑/0↓), significant on BIRD
  v2 (11↑/1↓, p=0.0063). Repair feedback uses a documented component-level
  approximation of SQL structure (keyword-anchored spans, not a full parse),
  so the taxonomy cannot isolate every objective-§13 category.
- BIRD prompts include the benchmark's `evidence` field but training never
  saw evidence-annotated data (augmentation was scoped out) — `evidence`
  use remains the obvious next lever; schema sample context is capped at
  30 lines per schema (prompt-size guard), which under-serves BIRD's wide
  schemas. SParC is evaluation-only (no multi-turn training data).
- Training-seed variance is measured on validation only (3 seeds,
  ±2.0 pts); bootstrap CIs cover question sampling, not seed variance.
  v2's Spider deltas vs v1 (−0.6…−0.8) are within that noise band.

## Security notes

SQL runs in-process but strictly read-only: `file:...?mode=ro` URIs,
`PRAGMA query_only`, an allowlist authorizer (SELECT-only, no PRAGMA /
ATTACH / writes / dynamic loading), execution timeouts, and row caps —
see `tests/test_sandbox.py` for the adversarial suite.
