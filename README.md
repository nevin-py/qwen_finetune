# Text-to-SQL Fine-Tuning, Execution-Based Verification, and Bounded Repair

Can a small open-source LLM be fine-tuned for text-to-SQL, and does
execution-based verification + bounded repair improve end-to-end correctness?
This repo answers both empirically with an experiment matrix evaluated on
Spider and BIRD under the benchmarks' own **execution-accuracy** methodology
(our harness agrees item-for-item with the official
`taoyds/test-suite-sql-eval` evaluator — see `tests/test_official_agreement.py`).

## TL;DR results (locked test sets, full splits)

| Cell | Pipeline | Spider dev (1,034) | BIRD dev (1,534) |
|---|---|---|---|
| A | Qwen2.5-Coder-1.5B base + prompt | 50.8% | 21.1% |
| B | A + QLoRA fine-tuning | 67.5% | 27.4% |
| C | B + execution-based verifier (k=5) | 70.4% | 30.4% |
| D | C + bounded repair (≤2 rounds) | **70.8%** | **31.5%** |

All three gates (B≥A, C≥B, D≥C) hold on **both** locked tests. Fine-tuning
contributes the largest gain (McNemar p < 1e-6 on Spider, p = 4e-8 on BIRD);
the verifier adds 2.8/3.0 points with significant paired improvement; repair
adds 0.5 (n.s.) on Spider and 1.1 (p = 7.6e-5) on BIRD. Multi-turn SParC
(eval-only): 56.1% → 61.3% turn accuracy A→B. Full statistics, CIs, failure
taxonomy, and ablations: `results/analysis_report.md`, `reports/results.md`.

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
tests/            78 tests (adversarial sandbox, official-eval agreement, gold
                  isolation, taxonomy, stats, compile guard)
reports/          design.md, results.md, raw eval JSONs, prediction files
results/          analysis outputs: per-cell metrics, ablations, resource
                  usage, plots, analysis_report.md
experiments/B/    QLoRA adapter + train_config.json + train_log.txt
tools/            cloned official evaluators (gitignored)
```

## Reproduce (in order)

```bash
# 1. data (downloads + verifies; spider/sparc via Google Drive, bird via OSS)
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/make_split.py           # 6300 train / 700 val, seed 13
.venv/bin/python scripts/profile_data.py         # -> reports/data_profile.md

# 2. train (overnight on 4 GB; smoke path: --max-steps 2)
.venv/bin/python scripts/build_training_set.py   # -> data/splits/train_sft.json
.venv/bin/python scripts/train_qlora.py --out experiments/B

# 3. locked benchmarks — ONE run each, full splits, A→B→C→D
bash scripts/run_final_benchmarks.sh             # spider dev + bird dev
.venv/bin/python scripts/run_sparc.py --adapter experiments/B/adapter \
    --out reports/sparc_B_dev.jsonl              # multi-turn (gold history)

# 4. analysis, benchmarks, figures
.venv/bin/python scripts/analyze_results.py      # -> results/
.venv/bin/python scripts/benchmark_resources.py  # -> results/resource_usage/
.venv/bin/python scripts/make_plots.py           # -> results/plots/

# 5. tests
.venv/bin/python -m pytest -q                    # 78 passed
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
projections (18.5M trainable, 1.18%); 6,300 examples, assistant-only loss,
max_len 2048, 2 epochs (3,114 steps), cosine lr 2e-4→5e-5, warmup 100,
paged AdamW 8-bit, seed 13. Wall time **181 min**, peak VRAM **1.97 GiB**,
loss 0.55 → 0.018. Full config: `experiments/B/train_config.json`.

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
- Verifier threshold τ=3.0 and k=5 were tuned on **validation only**; the
  rule-based verifier reached the pool oracle on validation, so no learned
  (logistic) scorer was needed — but BIRD shows ~4 pts of unclaimed pool
  headroom, so a learned scorer may help elsewhere.
- Repair gains are small and not significant on Spider (p=0.125); BIRD-only
  significance. Repair feedback uses a documented component-level
  approximation of SQL structure (keyword-anchored spans, not a full parse),
  so the taxonomy cannot isolate every objective-§13 category (e.g. pure
  date/NULL errors rarely appear as single-component diffs).
- BIRD prompts include the benchmark's `evidence` field; Spider prompts do
  not have one. SParC is evaluation-only (no multi-turn training data).
- Bootstrap CIs cover question sampling, not training-seed variance (single
  training seed by design).

## Security notes

SQL runs in-process but strictly read-only: `file:...?mode=ro` URIs,
`PRAGMA query_only`, an allowlist authorizer (SELECT-only, no PRAGMA /
ATTACH / writes / dynamic loading), execution timeouts, and row caps —
see `tests/test_sandbox.py` for the adversarial suite.
