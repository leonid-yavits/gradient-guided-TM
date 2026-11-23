#!/bin/bash
# Local launcher for the early-stopping study (paths of this machine).
export GGTH_DATA=/home/claude/data/treetm/TreeTM_datasets
export GGTH_BIN=/home/claude/data/treetm/bin
export FPTM_SRC=/home/claude/boobsd/fuzzypatterntm/src/FuzzyPatternTM.jl
export JULIA=/home/claude/mamba/jl/bin/julia
cd /home/claude/gg-th && exec python3 experiments/es.py "$@"
