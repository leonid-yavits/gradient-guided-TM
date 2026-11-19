# Protocol: first revision (reviewer round 1)

Written and committed on 17 Nov 2026, before any run of this revision. Code: `experiments/rev1.py`,
`experiments/linear_baselines.py`, `train.py` (flags `--fb`, `--no_pfac`, `--no_mag`, `--no_bias`, `--dump`, `--batch`),
queue `experiments/queue_rev1.sh`. Results: `results_rev1/`. The data, sampling, early stopping and metrics are those
of `experiments/es.py` (`PROTOCOL_text.md`) unless stated otherwise. TUNE: seed 1, up to 20 epochs, patience 5.
FINAL: up to 60 epochs, patience 10, test metrics at the best validation macro F1 epoch.

The main tables of the paper keep the pre-registered protocols. The runs below are additional analyses.

## A. Ablations of GG-TL (reviewer M1)

At the anchors 200 and 1000, on all six datasets, seeds 1 to 3, FINAL. Compared with the GG-TL FINAL runs (60-epoch
cap) of the same cells by Welch's t-test.

* `fb_cotm`: the GG-TL readout (ternary, Adam, STE, offsets) with the clause feedback of the CoTM rule applied to the
  ternary votes (`GGReadout.cotm_feedback`): one random negative class per sample, selection probabilities
  (T - v_y)/2T and (T + v_q)/2T, Type I or II by the sign of the ternary weight (zero counts as positive, as in our
  CoTM), Type II suppressed where Type I is selected. This isolates the clause-feedback rule from the readout learner.
  Learning rate of the selected GG-TL setting, T = Tm n with Tm of the selected CoTM setting, s tuned over
  {1.5, 2, 5, 10, 20} (TUNE, seed 1, selection on validation macro F1).
* `no_pfac`: GG-TL without the factor (1 - p_y). `no_mag`: GG-TL with min(1, |g|/mean|g|) replaced by 1.
  `no_bias`: GG-TL with the class offsets fixed at 0. All at the selected GG-TL setting, not retuned.

The opposite cell of the 2 x 2 design (GG clause feedback with the integer CoTM readout) is not run: GG feedback
needs a differentiable readout, and the CoTM readout has no probabilistic output without an additional, arbitrary
logit scale.

## Addendum A2 (17 Nov 2026, written after the part-A results and before the runs below)

In part A, `fb_cotm` used T = Tm n with the Tm selected for the CoTM. The CoTM selects Tm for integer weights whose
magnitude grows during training, whereas the ternary votes of the GG-TL readout satisfy |v| <= n. With Tm >= 1, the
selection probabilities (T - v_y)/2T and (T + v_q)/2T then stay close to 1/2, so the part-A margin is not calibrated
for the ternary readout. Tm of `fb_cotm` is therefore retuned over the CoTM grid {0.03, 0.1, 0.3, 1, 2.5, 5, 10, 20}
at the s and learning rate selected in part A (TUNE, seed 1). The setting with the best validation macro F1 over the
part-A and part-A2 settings is run with seeds 1 to 3 (FINAL). The paper reports this setting and states the part-A
result. Code: `rev1.py` phases `abl2_tune` and `abl2_final`.

## B. Clause dumps (reviewer M6)

GG-TL, GG-TH and CoTM at the anchors 200 and 1000, six datasets, seed 1, FINAL, with the include masks and readout
weights saved at the best validation epoch. These runs repeat the seed-1 FINAL runs of `results_es` (deterministic
seeds); their logs are checked against the originals. Reported: literals per clause split into plain and negated
literals, and the number of nonzero ternary votes per clause (GG-TL).

## C. CoTM mini-batch size (reviewer M2)

Our CoTM computes the votes once per mini-batch of 64 samples. CoTM with mini-batch size 1 (per-sample updates) at
n = 200 on ALeRCE, R52 and WOS, seeds 1 to 3, FINAL, at the selected CoTM setting. Compared with the B = 64 runs by
Welch's t-test.

## D. One grid step beyond boundary optima (reviewer M2, M4)

For GG-TL, CoTM and the TMU multi-class TM, at every anchor where the selected setting lies on the lower or upper edge
of a grid axis with at least three values, one setting one step beyond the edge is tuned (TUNE, seed 1), the other axes
held at the optimum. Steps: s 1.5 -> 1.0, s upper -> 2x; learning rate /3 or x3; Tm /3 or x2. If the extension changes
the selected setting, FINAL runs (seeds 1 to 3) are made at the budgets mapped to that anchor and reported as a
sensitivity analysis next to the pre-registered results.

## E. Additional seeds (reviewer M3)

GG-TL and CoTM, all 36 cells, seeds 4 to 8 at the selected settings. Settings that were rerun with 120 epochs in the
early-stopping study (all three seeds in `ext_*` files) use the 120-epoch protocol for the new seeds too, the others
FINAL. Analysis: seeds 2 to 8 (seven seeds, disjoint from the tuning seed 1). Per cell: mean difference, 95%
confidence interval (Welch), Welch's p, Holm correction over the 36 cells. Across cells: Wilcoxon signed-rank test
on the 36 cell means and on the six dataset means. Macro F1, balanced accuracy and accuracy are all reported.

## F. Linear reference baselines (reviewer M7)

Multinomial logistic regression and a linear SVM on the same binarized features (`experiments/linear_baselines.py`):
C in {0.01, 0.1, 1, 10, 100} x class_weight in {None, balanced}, selected on validation macro F1, refit with seeds 1 to 3.

## G. TMU coalesced classifier, tuned (reviewer M2)

The TMU coalesced classifier tuned at n = 200 (TUNE, seed 1) over the CoTM grid of `es.py` (s in {1.5, 2, 5, 10, 20} x
Tm in {0.03, 0.1, 0.3, 1, 2.5, 5, 10, 20}) on ALeRCE, R52, Sector, RCV1 and 20NG, and over s in {1.5, 2, 5} x
Tm in {2.5, 5, 10, 20} on WOS (one epoch on WOS takes about 110 s at n = 200). FINAL runs at the selected setting,
seeds 1 to 3, compared with our CoTM at n = 200.

## Reporting

All results are reported, including those that weaken the paper's claims.
