#!/usr/bin/env bash
# Stage 5 (v2): final locked benchmarks — official Spider dev (1034), BIRD dev
# (1534) and SParC dev (eval-only), each touched exactly ONCE for experiment v2
# (reports/v2_protocol.md). Full matrix A/B/C/D; B's predictions come free from
# the verifier run (candidate 0 = greedy). Fails fast (set -e); resumable
# (existing v2 outputs are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."

# Completeness guard: a bare `-f` test treats an interrupted run as finished,
# since the generators flush every record. Compare against the split size.
complete() {  # $1 = path, $2 = expected records
  [ -f "$1" ] && [ "$(wc -l < "$1")" -eq "$2" ]
}
N_SPIDER=1034
N_BIRD=1534

PY=.venv/bin/python
ADAPTER=experiments/B_v2/adapter
WEIGHTS=experiments/verifier_weights_v2.json
K=5
TAU=3.0
PROMPTS="--prompt-variants 4 --few-shot-file data/splits/fewshot_block.txt --schema-samples"

for ds in spider bird; do
  if [ "$ds" = spider ]; then N=$N_SPIDER; else N=$N_BIRD; fi

  echo "=== [$ds] A: base + greedy (v2 prompts) ==="
  complete reports/${ds}_A_dev_v2.jsonl $N || \
    $PY scripts/run_baseline.py --dataset $ds --split dev --n 0 $PROMPTS \
        --out reports/${ds}_A_dev_v2.jsonl

  echo "=== [$ds] C: verifier over k=$K candidates (tuned weights) ==="
  complete reports/${ds}_C_dev_v2.jsonl $N || \
    $PY scripts/run_verify.py --dataset $ds --split dev --n 0 --k $K \
        --adapter $ADAPTER --weights $WEIGHTS $PROMPTS --resume \
        --out reports/${ds}_C_dev_v2.jsonl

  echo "=== [$ds] B: greedy slot extracted from C candidates ==="
  complete reports/${ds}_B_dev_v2.jsonl $N || \
    $PY -c "
import json, pathlib
src = pathlib.Path('reports/${ds}_C_dev_v2.jsonl')
dst = pathlib.Path('reports/${ds}_B_dev_v2.jsonl')
with dst.open('w') as out:
    for line in src.open():
        r = json.loads(line)
        out.write(json.dumps({
            'index': r['index'], 'db_id': r['db_id'],
            'question': r['question'],
            'prediction': r['candidates'][0]['sql']}) + '\n')
print('wrote', dst)
"

  echo "=== [$ds] D: bounded repair (tau=$TAU, dual gate) ==="
  complete reports/${ds}_D_dev_v2.jsonl $N || \
    $PY scripts/run_repair.py --dataset $ds --cfile reports/${ds}_C_dev_v2.jsonl \
        --tau $TAU --adapter $ADAPTER --weights $WEIGHTS $PROMPTS \
        --out reports/${ds}_D_dev_v2.jsonl

  echo "=== [$ds] evaluate A/B/C/D ==="
  for cell in A B C D; do
    $PY scripts/run_eval.py --dataset $ds --split dev \
        --preds reports/${ds}_${cell}_dev_v2.jsonl
  done
done

echo "=== [sparc] A/B (eval-only dialogue, gold history) ==="
N_SPARC=1203
complete reports/sparc_A_dev_v2.jsonl $N_SPARC || \
  $PY scripts/run_sparc.py --n 0 --schema-samples --out reports/sparc_A_dev_v2.jsonl
complete reports/sparc_B_dev_v2.jsonl $N_SPARC || \
  $PY scripts/run_sparc.py --n 0 --adapter $ADAPTER --schema-samples \
      --out reports/sparc_B_dev_v2.jsonl
$PY scripts/run_eval.py --dataset sparc --preds reports/sparc_A_dev_v2.jsonl
$PY scripts/run_eval.py --dataset sparc --preds reports/sparc_B_dev_v2.jsonl

echo "=== Stage 5 (v2) complete ==="
