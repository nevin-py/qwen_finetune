#!/usr/bin/env bash
# Phase 10: final locked benchmarks — official Spider dev (1034) and BIRD dev
# (1534), each touched exactly ONCE, full matrix A/B/C/D (design §4, objective
# §2/§14). Cell B's predictions come free from the verifier run (candidate 0 =
# greedy = B, verified bit-identical on validation). Fails fast (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."

# Completeness guard. A bare `-f` test treats an interrupted run as finished:
# the generators flush every record, so a partial file exists as soon as the
# first question completes, and B/D would then be built from a truncated pool.
complete() {  # $1 = path, $2 = expected records
  [ -f "$1" ] && [ "$(wc -l < "$1")" -eq "$2" ]
}
N_SPIDER=1034
N_BIRD=1534

PY=.venv/bin/python
ADAPTER=experiments/B/adapter
K=5
TAU=3.0

for ds in spider bird; do
  if [ "$ds" = spider ]; then N=$N_SPIDER; else N=$N_BIRD; fi

  echo "=== [$ds] A: base + greedy ==="
  complete reports/${ds}_A_dev.jsonl $N || \
    $PY scripts/run_baseline.py --dataset $ds --split dev --n 0 \
        --out reports/${ds}_A_dev.jsonl

  echo "=== [$ds] C: verifier over k=$K candidates ==="
  complete reports/${ds}_C_dev.jsonl $N || \
    $PY scripts/run_verify.py --dataset $ds --split dev --n 0 --k $K \
        --adapter $ADAPTER --resume --out reports/${ds}_C_dev.jsonl

  echo "=== [$ds] B: greedy slot extracted from C candidates ==="
  complete reports/${ds}_B_dev.jsonl $N || \
  $PY -c "
import json, pathlib
src = pathlib.Path('reports/${ds}_C_dev.jsonl')
dst = pathlib.Path('reports/${ds}_B_dev.jsonl')
with dst.open('w') as out:
    for line in src.open():
        r = json.loads(line)
        out.write(json.dumps({
            'index': r['index'], 'db_id': r['db_id'],
            'question': r['question'],
            'prediction': r['candidates'][0]['sql']}) + '\n')
print('wrote', dst)
"

  echo "=== [$ds] D: bounded repair (tau=$TAU) ==="
  complete reports/${ds}_D_dev.jsonl $N || \
    $PY scripts/run_repair.py --dataset $ds --cfile reports/${ds}_C_dev.jsonl \
        --tau $TAU --adapter $ADAPTER --out reports/${ds}_D_dev.jsonl

  echo "=== [$ds] evaluate A/B/C/D ==="
  for cell in A B C D; do
    $PY scripts/run_eval.py --dataset $ds --split dev \
        --preds reports/${ds}_${cell}_dev.jsonl
  done
done
echo "=== Phase 10 complete ==="