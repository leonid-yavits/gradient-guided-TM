# Protocol: GG-linear with post-hoc ternary rounding, full study

Written and committed on 14 Nov 2026, before any run of this study. Code: `experiments/es.py` (model `gg_linr`) and
`train.py --select f1_round`.

## Motivation

In the early-stopping study, GG-linear (GG feedback with a real-valued linear readout trained by Adam) was run only at
n = 1000, with the setting selected for the ternary linear readout trained with the STE (gg_lintern). After rounding
its readout weights to {-1, 0, +1} (ternary weight network rule with one scale, `GGReadout.scores_rounded`), it
scored higher than gg_lintern on R52, Sector, RCV1 and 20NG, and lower on WOS and ALeRCE. The rounded model has the
same inference structure as gg_lintern: ternary clause votes per class, one scale and a bias per class. This study
gives it its own tuning and the full budget range, so the two ways of training the ternary linear readout can be
compared on equal terms.

## Model and selection

* Model `gg_linr`: clause bank with GG feedback, real-valued linear readout trained by Adam (`--readout gg_lin`).
  The deployed model is the rounded readout. All reported test metrics are those of the rounded readout.
* Selection: early stopping and the best epoch use the validation macro F1 of the rounded readout
  (`--select f1_round`). Tuning selects on the same quantity.
* Grid: the gg_lintern grid, s in {1.5, 2, 5, 10, 20} x learning rate in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2}.

## Protocol

Identical to the early-stopping study (`experiments/es.py`, `PROTOCOL_text.md`): the six datasets, budgets 50 to 2000,
tuning at the anchors 200 and 1000 with seed 1 (up to 20 epochs, patience 5), final runs with seeds 1 to 3 (up to 60
epochs, patience 10), and 120-epoch reruns of settings whose best epoch falls within the last 5 epochs of the cap.

## Frozen choices and reporting

* The grid is not extended. Optima on its boundary are reported.
* All results are reported. Welch t-tests of gg_linr against gg_lintern (same inference structure), CoTM, FPTM, TNN
  and GG-TH at every budget.
* The earlier n = 1000 runs with the gg_lintern setting (`results_es/gg_lin`) stay in the record and are not
  replaced.
