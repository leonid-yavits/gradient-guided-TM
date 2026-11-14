# Protocol: baseline check with the official TMU library

Written and committed on 14 Nov 2026, before any accuracy result of these runs was seen (only the time per
epoch of single smoke-test epochs was measured). Code: `experiments/tmu_check.py`.

## Purpose

The early-stopping study compares GG against CoTM implemented in this repository (`ggth/tm.py`). This check answers
two questions a reader will ask:

1. Does our CoTM reproduce the reference implementation at the same settings?
2. How does GG compare with the standard multiclass Tsetlin Machine, as implemented in the reference library?

## Library

TMU, https://github.com/cair/tmu, installed from GitHub at commit 17f17231b944d1aa7ae1a36541ba0bc9f0e52bc4
(the PyPI release 0.7.1 fails under NumPy 2). CPU clause banks. Default options apart from the clause count, T, s
and the seed.

## Data, sampling and stopping

Identical to `experiments/es.py`: the six datasets of `C.DS_ES`, the same splits, class-balanced sampling with P
samples per class per epoch (`C.PCLS`), test metrics read at the best validation macro F1 epoch.

## Runs

* **CoTM implementation check (`tmu_cotm`, TMCoalescedClassifier).** n clauses shared by all classes,
  T = Tm n. The setting (s, Tm) is the one selected for our CoTM by `es.py` at the same budget, not retuned.
  Budget 200 on all six datasets, and additionally 1000 on R52 and Sector. Seeds 1 to 3, up to 60 epochs,
  patience 10. A full grid is not run: one epoch on WOS at n = 1000 takes about 430 s on this machine.
* **Multiclass Tsetlin Machine (`tmu_tm`, TMClassifier).** cpc(ds, n) clauses per class (half of each polarity),
  about n clauses in total; at small n with many classes the total exceeds n (as for FPTM) and is reported.
  T = Tm cpc. Grid: s in {1.5, 2, 5, 10, 20} x Tm in {0.1, 0.3, 1, 2.5, 5, 10} (30 settings). Tuning at the anchors
  200 and 1000 (budgets up to 500 use the 200 setting), seed 1, up to 20 epochs, patience 5. Final: all six
  budgets, seeds 1 to 3, up to 60 epochs, patience 10.

## Frozen choices and reporting

* The grid is not extended. Optima on its boundary are reported.
* All results are reported: test macro F1, balanced accuracy and accuracy (mean and sample standard deviation over
  seeds), literals per clause, and best epochs. Welch t-tests: TMU CoTM against our CoTM at the same setting;
  the multiclass TM against GG with a ternary linear readout, GG-TH and our CoTM.
* If our CoTM is significantly below TMU CoTM at a checked budget, this is stated in the paper, and the CoTM
  comparison is qualified accordingly.

## Amendment, 14 Nov 2026 (before any accuracy result was read)

A WOS check run takes about 2 hours, the time limit of one background job here. Runs are therefore checkpointed after
every epoch (pickled TMU model, sampler state and log) and resumed after an interruption. A resumed run is not
bit-identical to an uninterrupted one, because the TMU C code keeps its own random state. Resumed runs list the
epochs after which they resumed in the field `resumed_after` of their log.
