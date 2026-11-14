#!/bin/bash
# Remaining runs of the paper revision, in order. Every step skips finished runs, so the script can be restarted.
#   1. TMU check: CoTM implementation check and multiclass TM tuning   (experiments/PROTOCOL_tmu.md)
#   2. GG-linear with post-hoc rounding: tuning                         (experiments/PROTOCOL_gglin.md)
#   3. TMU multiclass TM: final runs
#   4. GG-linear with post-hoc rounding: final runs and 120-epoch reruns
set -e
cd "$(dirname "$0")/.."
export GGTH_DATA=/home/claude/data/treetm/TreeTM_datasets GGTH_BIN=/home/claude/data/treetm/bin
export FPTM_SRC=/home/claude/boobsd/fuzzypatterntm/src/FuzzyPatternTM.jl JULIA=/home/claude/mamba/jl/bin/julia
export TMU_PYTHON=/home/claude/venv_tmu/bin/python
python3 experiments/tmu_check.py tune --workers 2
python3 experiments/es.py tune --workers 2 --models gg_linr
python3 experiments/tmu_check.py final --workers 2
python3 experiments/es.py final --workers 2 --models gg_linr
python3 experiments/es.py ext --workers 2 --models gg_linr
echo QUEUE_DONE
