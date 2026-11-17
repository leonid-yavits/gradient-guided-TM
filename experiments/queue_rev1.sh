#!/bin/bash
# Runs of the first revision, in order (experiments/PROTOCOL_rev1.md). Every step skips finished runs.
set -e
cd "$(dirname "$0")/.."
export GGTH_DATA=/home/claude/data/treetm/TreeTM_datasets GGTH_BIN=/home/claude/data/treetm/bin
export TMU_PYTHON=/home/claude/venv_tmu/bin/python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[ "${SKIP_LINEAR:-0}" = 1 ] || python3 experiments/linear_baselines.py
for p in dump abl_tune abl_final batch ext_tune ext_final seeds tm_ext_tune tm_ext_final tmucotm_tune tmucotm_final; do
  echo "== $p"
  python3 experiments/rev1.py $p --workers ${WORKERS:-2}
done
echo QUEUE_DONE
