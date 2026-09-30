# Experiment v2 — pre-registration (written BEFORE the locked run)

Status: val tuning complete (Stages 0–4). This file freezes every choice that
touches the locked benchmarks. The official Spider dev (1,034) and BIRD dev
(1,534) sets are each evaluated ONCE for v2, exactly as in v1 (design §4).
Re-running with any change below = a new declared experiment, not a v2 rerun.

## Training

- Data: `data/splits/train_sft.json` — stratified carve (`make_split --stratify`,
  cluster-atomic), dedup (9) + gold-quarantine (3), 6,288 examples.
- Recipe: frozen from sweep — **epochs=3, lr=1e-4, r=16 (α=32, dropout .05),
  accum=4, max_len=3072, seed=13**, QLoRA NF4, all 7 proj modules.
- Selection: best execution-accuracy probe on val (n=50 greedy, every 500
  steps) → adapter at **step 2500** (probe 80%); val loss flat (no overfit).
- Artifact: `experiments/B_v2/adapter` (config: `train_config.json`, curve:
  `val_metrics.json`, `train_log.csv`).
- Seeds (winner config): 78/76/80 → mean 78.0 ± 2.0; seed 13 pre-registered.

## Prompt configuration (identical for train, val, A–D, both benchmarks)

- `--prompt-variants 4` (deterministic `variant_for` on the question)
- `--few-shot-file data/splits/fewshot_block.txt` (2 Spider-train examples, seed 13)
- `--schema-samples` (row counts + ≤3 sample values/column, 30-line budget, gold-free)
- SParC: `--schema-samples` only (dialogue system prompt, no few-shot/variants)

## Verifier / repair (all val-tuned, declared)

- Weights: `experiments/verifier_weights_v2.json` (seeded grid, 5,000 trials,
  val selection acc 79→80; pool oracle 81).
- Candidates: **k=5** (1 greedy + sampled until 5 unique or 8 draws),
  **T=0.8**, `max_new_tokens=512`.
- Repair: **τ=3.0**, max 2 iterations, acceptance = score gain ≥ **0.1** AND
  no pool-majority contradiction (replay on val: bare score gate was −2;
  dual gate restores D=C).

## Val gates (all passed on new val, n=100)

A 53.0 / B 78.0 / C 80.0 / D 80.0 — B ≥ v1-recipe ref (77), C ≥ B, D ≥ C.

## Locked protocol (Stage 5)

- Files (new names, v1 artifacts untouched): `reports/{spider,bird}_{A,B,C,D}_dev_v2.jsonl`,
  `reports/sparc_{A,B}_dev_v2.jsonl`.
- Matrix: A = base+greedy; B = greedy slot of C; C = verifier over k=5;
  D = gated repair of C. SParC = A/B only (eval-only, v1 parity).
- Metrics/gates: execution accuracy + paired stats (McNemar, bootstrap),
  same harness as v1 (official-agreement gated). Success = every v1 gate
  re-holds on both benchmarks (B>A, C≥B, D≥C; C→D may be n.s.).
- Known a-priori weakness: BIRD still has zero evidence-supervised training
  data (S2 declined) — absolute BIRD numbers expected far below Spider.
