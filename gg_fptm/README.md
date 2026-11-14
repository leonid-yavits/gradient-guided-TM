# Sub-project: gradient-guided feedback with Fuzzy-Pattern clauses (GG + FPTM)

Started 14 Nov 2026 from branch `es-study` (commit noted in the git history). Separate from the GG paper revision.

## Question

Can GG feedback improve a TM whose clauses match partially, as in the Fuzzy-Pattern TM (FPTM)? In the early-stopping
study (branch `es-study`, `results_es/summary.md`), FPTM and a clause-free ternary network (TNN) are more accurate than
GG with hard-AND clauses on the sparse text datasets. GG improves on the CoTM at the same clause count. FPTM changes
the clause evaluation (graded match, length limit L, output cap L_F), and GG changes the feedback. Whether the two
gains add up is open. It is not known whether a combination would beat FPTM or TNN.

## Plan (to be fixed in PROTOCOL.md before any run)

1. **Reproduce the FPTM clause in this code base.** Read the FPTM reference implementation (Julia, BooBSD/FuzzyPatternTM,
   commit e25a51e, path in `experiments/launch_es.sh`). Implement its clause evaluation, length limit and feedback in
   NumPy, next to `ggth/tm.py`.
2. **Verify the reproduction.** With FPTM's own feedback, per-class organization and the selected FPTM settings of the
   early-stopping study, the NumPy version must match the Julia FPTM within the seed spread on Sector and 20NG at two
   sizes. If it does not, stop and report. No GG result is meaningful without this step.
3. **GG feedback on graded clauses.** Shared clause pool with graded clause outputs, ternary linear readout as in GG-TL,
   GG feedback from the readout gradient. Define how Type I and Type II feedback act on a graded clause and record that
   definition in PROTOCOL.md before the runs.
4. **Comparison.** Sector and 20NG at 200 and 1000 clauses, tuning plus 3 seeds and early stopping as in `es.py`.
   Compare against FPTM, GG-TL and TNN from `results_es`.

## Status

Not started. No code, no runs.
