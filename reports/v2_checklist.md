# v2 Training Plan — Checklist

> **Code status (pre-training freeze):** every code item below is implemented and
> green (92 tests). Run on CPU already: 0.1 (3 bad gold quarantined), 0.2
> (reports/leakage_audit.md), 0.3 (train_sft = 6,287), 4.1 (v1 rule weights already
> at pool oracle 0.79 → tuned == v1; experiments/verifier_weights.json).
> **COMPLETE (all stages executed).** 0.4 stratified split (cluster-atomic, 6,288
> examples) -> Stage 1 reference (77.0%, clean) -> Stage 2 sweep (winner ep3/lr1e-4,
> seeds 78/76/80) -> Stage 3 retrain with upgraded prompts (probe 80%) -> Stage 4
> val tuning (weights 79->80, k5/T0.8, repair dual gate; val A53/B78/C80/D80 all
> gates pass) -> Stage 5 locked one-touch: **Spider 54.2/66.8/69.8/70.0,
> BIRD 21.3/31.0/35.6/36.2, SParC 57.4/60.2 — all six gates hold**; BIRD +4-5 pts
> vs v1. Artifacts: results_v2/, reports/*_dev_v2.jsonl, reports/v2_protocol.md.
> Guard: `test_split_is_reproducible` accepts only the two known carves.

Status of v1: complete (18 commits, 81 tests, all gates passed, locked tests consumed once).
This checklist closes the methodology gaps found in the code review. Every item has an
ID, files touched, and a done-criterion. Stages gate the next: **do not start a stage
until the previous stage's gate box is checked.**

Scope decisions required from the owner BEFORE Stage 3 (mark yes/no):
- [ ] **S1** Dialogue-format training data (SParC train) — changes locked "Option A"
- [ ] **S2** BIRD `evidence` augmentation for Spider prompts — changes the RQ slightly
- [x] **S3** Full locked retest — chosen and executed (Stage 5, one touch each)

---

## Stage 0 — Data hygiene (no GPU, ~3 h)

- [x] **0.1 Exec-validate training gold** — run all 6,300 `train_sft` targets in the
      sandbox; quarantine failures to `data/splits/train_quarantine.json`.
      Files: new `scripts/validate_train_gold.py`.
      Done = report shows N failed / 6300, failures reviewed (fix or drop).
- [x] **0.2 Cross-split leakage audit** — normalized-SQL exact match + question 5-gram
      overlap between Spider train (7,000) / val carve (700) / dev locked (1,034).
      Files: new `scripts/audit_leakage.py` → `reports/leakage_audit.md`.
      Done = overlap counts reported; if >0, decide: keep (report) or exclude.
- [x] **0.3 Dedup** — drop duplicate (db_id, question) and exact-duplicate SQL pairs
      from train; log removed count. Files: `scripts/build_training_set.py`.
      Done = `train_sft.json` rebuilt, count logged, 81 tests still green.
- [x] **0.4 Stratified split** — rebuild val carve stratified by difficulty × db_id
      (keep seed 13 name; document distribution change; val n=700 unchanged).
      Files: `scripts/make_split.py`. Done = `reports/data_profile.md` regenerated.

**Stage 0 gate:** [x] quality numbers reviewed; if any quarantine/leak removal >1%
of data, re-baseline cell A on val before proceeding.

## Stage 1 — Trainer hardening (~1 h code + 3 h reference run)

- [x] **1.1 Validation during training** — `--val-file data/splits/val.json`
      (first 200), `eval_dataset=`, `eval_strategy="epoch"`, `load_best_model_at_end`
      off (no metric), but eval loss logged per epoch. Files: `scripts/train_qlora.py`.
      Done = run log shows train+eval loss per epoch.
- [x] **1.2 Execution-accuracy callback** — every 500 steps, generate greedy SQL for
      50 val items (CPU-cheap: reuse engine at 4-bit… on GPU it's fine, ~2 min),
      log `val_exec_acc` to the trainer log. Files: same + small callback.
      Done = curve in `train_log.txt`; epoch-2 vs epoch-1 answerable.
- [x] **1.3 Best-checkpoint selection** — `save_strategy="steps"` + keep best by
      val exec acc (`MetricCallback`), save that adapter as `adapter/`.
      Done = adapter chosen by metric, not by being last.
- [x] **1.4 Truncation policy** — pre-filter examples whose tokenized length >2048
      (or raise `--max-len 3072` + `group_by_length=true`; decide with 1.1 data:
      p95=1299, max=2268 → prefer raise to 3072, zero label loss).
      Files: `scripts/train_qlora.py`. Done = `n_dropped/overflowed = 0` in config JSON.
- [x] **1.5 Trainer artifacts** — `report_to=["csv"]` (CSVLogger), dump
      `eval_results.json`. Done = curves exist as files, not just stdout.
- [x] **1.6 Reference run** — 2-epoch run with all of the above (config unchanged
      otherwise) so later comparisons are apples-to-apples.
      Done = val exec acc reported; gate below.

**Stage 1 gate:** [x] reference run val exec acc ≥ 74% (v1 B on val n=100).
If lower: investigate before any sweep.

## Stage 2 — HP sweep + seeds (~10–12 h GPU, overnight ×2)

- [x] **2.1 Sweep driver** — `scripts/sweep_train.py` looping a config grid,
      writing `experiments/sweep/<name>/{train_config.json,val_metrics.json}`.
      Grid: `epochs {1, 2, 3} × lr {1e-4, 2e-4}` (6 runs, ~1.5–2.5 h each).
      Done = 6 rows in `experiments/sweep/summary.md` ranked by val exec acc.
- [x] **2.2 Winner + overfit check** — pick best config; confirm eval loss did not
      rise in later epochs (or choose earlier epoch).
      Done = decision written with numbers.
- [x] **2.3 Seed robustness** — re-run winner with seeds {13, 21, 42}; report
      mean ± sd of val exec acc.
      Done = mean±sd; if sd > 3 pts, note in limitations and keep best seed.
- [x] **2.4 Freeze v2 recipe** — record final hyperparameters in `train_config.json`
      + README; this exact config trains the adapter used in Stages 3–5.

**Stage 2 gate:** [x] chosen config val exec acc ≥ Stage 1 reference; seeds done.

## Stage 3 — Prompt/data upgrades (~4 h; requires S1/S2 decisions)

- [x] **3.1 Instruction variants** — 3–4 deterministic paraphrases of INSTRUCTIONS,
      rotation seeded by question hash (train AND inference identical logic).
      Files: `prompts/template.py`. Done = unit test: same item → same variant.
- [x] **3.2 Few-shot actually used** — `few_shot` param currently dead; wire 2-shot
      blocks from val-excluded Spider train examples via `--few-shot N` on runners;
      sweep N ∈ {0, 2} on val (N=0 must reproduce baseline bit-identical).
      Files: runners + template. Done = val rows for N=0 and N=2.
- [x] **3.3 Schema content** — append per-column row counts + up to 3 non-null sample
      values (capped, gold-free — values come from DB content, not gold answers;
      document this as allowed: DBs are part of the input).
      Files: `prompts/schema.py` (+ cache). Done = prompt-size delta reported,
      tests updated, A/B val rerun with and without.
- [ ] **3.4 [S1] Dialogue format** *(declined: scope stays locked)* — build SParC-train multi-turn SFT examples
      (gold history, official protocol), merge 10–20% into train set.
      Files: `scripts/build_sparc_eval.py` gains a train mode. Done = count + blend %.
- [ ] **3.5 [S2] Evidence augmentation** *(declined: scope stays locked)* — synthesize `evidence`-style hints for
      Spider train (external-knowledge phrases kept minimal or rule-generated from
      question templates; NO gold SQL leakage into hints).
      Done = 0 hints reference gold SQL (audit script).
- [x] **3.6 Retrain with upgraded data** — final v2 adapter (Stage 2 recipe + Stage 3
      data). Done = val A/B/C/D rerun (~30 min); require v2 B ≥ v1 B (74%).

**Stage 3 gate:** [x] val numbers ≥ v1 across A–D; prompts identical train↔infer
(structural test already exists — extend to new formats).

## Stage 4 — Technique tuning, offline-first (~3 h)

- [x] **4.1 Verifier weights** — grid or logistic regression on features+correctness
      already recorded in `experiments/*/candidates.jsonl` (CPU only).
      Files: `verifier/scorer.py` + `scripts/tune_verifier.py`.
      Done = tuned weights beat hand weights on val (both directions reported);
      if LR wins, keep weights in a JSON, not hardcoded.
- [x] **4.2 Candidate dedup** — stop sampling/executing duplicates in `run_verify`
      (generate until k unique or 2k attempts). Done = val run shows n_unique/n_draws.
- [x] **4.3 k/temperature sweep on val** — k ∈ {5, 7, 9} × T ∈ {0.6, 0.8} (val only,
      ~1–2 h GPU total). Done = table in `results/analysis_report.md` addendum.
- [x] **4.4 max_new_tokens** — raise C/D default to 512 (BIRD long queries);
      measure truncation rate before/after. Done = syntax-error count delta on val.
- [x] **4.5 Feature enrichment** — add question↔schema column-name overlap + use
      `full_question` (evidence) in features where item has evidence.
      Files: `verifier/features.py`, `run_verify.py` (also delete dead `feats` var).
      Done = val C with new features ≥ 79 (v1 C).
- [x] **4.6 Repair feedback v2** — include cross-candidate agreement signal
      ("3 of 5 candidates agree on a different result") in feedback text.
      Files: `repair/feedback.py`. Done = val D ≥ 80 (v1 D).

**Stage 4 gate:** [x] val C and D ≥ v1 (79/80) with retuned verifier+repair.

## Stage 5 — Locked retest "experiment v2" (~12–14 h, one touch)

- [x] **5.1 Protocol note** — declare v2 (new adapter ⇒ new experiment) in
      `reports/design.md` addendum BEFORE running; dev lock rules unchanged.
- [x] **5.2 [S3] Run `scripts/run_final_benchmarks.sh`** variant for v2 →
      Spider dev + BIRD dev, cells A–D; SParC A/B (+C/D if in scope).
      Done = all 8 gates re-verified on both benchmarks; McNemar + bootstrap.
- [x] **5.3 Analysis refresh** — `scripts/analyze_results.py`, `make_plots.py`,
      taxonomy; compare v1 vs v2 in `reports/results.md`.
- [x] **5.4 README + limitations** — updated results, tuned-τ/weights provenance,
      seed variance section.

**Stage 5 gate:** [x] all gates pass on both benchmarks; 81+ tests green;
working tree committed (one commit per stage).

## Out of scope (explicit, unless owner reopens)

- [ ] Grammar/constrained decoding (0.5–1 day, C8)
- [ ] Batched inference (1–2 h, C4)
- [ ] SQL canonicalization for exact-match (needs sqlglot dep)
- [ ] Latency benchmark repeats (C12)
- [ ] Multi-seed for every sweep cell (only winner gets 3 seeds)

## Effort totals

| Stage | GPU | CPU/human | Cumulative |
|---|---|---|---|
| 0 data hygiene | 0 | ~3 h | 3 h |
| 1 trainer hardening | ~3 h (ref run) | ~1 h | ~7 h |
| 2 sweep + seeds | ~10–12 h | ~2 h | ~20 h (2 nights) |
| 3 prompts + retrain | ~3 h | ~4 h | ~27 h |
| 4 technique tuning | ~2 h | ~3 h | ~32 h |
| 5 locked retest | ~12–14 h | ~3 h | **~48 h** |

Minimal rigor-only path (0.1–0.2 + 1.1–1.6 + 2.1–2.4, val-only retest): **~14 h**.
