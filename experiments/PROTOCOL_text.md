# Protocol: four additional sparse text datasets

Written and committed on 13 Nov 2026, before any run on these datasets. It extends the early-stopping study of
`experiments/es.py` (WOS and ALeRCE) without changing its models, grids or selection rules.

## Motivation

In the early-stopping study, GG-TH and GG with a ternary linear readout show their clearest gains on WOS (133 classes,
2,000 sparse binary features). ALeRCE (15 classes, dense thermometer-coded features) shows smaller or no gains. The study
below tests whether the WOS results hold on other text datasets with many classes and sparse binary inputs. All results
are reported, whatever their direction.

## Datasets

Fixed before any run. Downloaded files and checksums: `data_prep/SHA256SUMS_text`. Encoders: `data_prep/prep_text.py`, which follows the QMTM study encoders with V = 2,000 features
(the WOS input size). Chi-square selection on the reduced training part only.

| Name | Source | Classes | Train | Validation | Test | Active features per sample (train) |
|---|---|---|---|---|---|---|
| r52 | Reuters-21578 ApteMod, NLTK copy, single-topic, R52 definition | 52 | 5,576 | 984 | 2,569 | 24.1 |
| sector | LIBSVM sector.scale | 105 | 5,770 | 642 | 3,207 | 11.1 |
| rcv1 | LIBSVM rcv1.multiclass, test subsampled to 50,000 (seed 0) | 51 | 14,007 | 1,557 | 50,000 | 29.8 |
| news20 | LIBSVM news20 | 20 | 14,341 | 1,594 | 3,993 | 22.0 |

For reference, WOS has 55.1 active features per sample. rcv1: the training file holds 51 of the 53 classes, and 86 test
documents of the two absent classes are dropped (as in QMTM).

Samples per class per epoch (class-balanced sampling, as for WOS and ALeRCE): P = training size / classes, rounded to
the nearest 50. r52: 100, sector: 50, rcv1: 300, news20: 700.

## Models, budgets, tuning and final runs

Identical to `experiments/es.py` at commit c83f1b0 (the WOS and ALeRCE study):

* Models: GG-TH (gg_hidtern), GG with a ternary linear readout (gg_lintern), clause-free ternary network (tnn),
  CoTM, FPTM, and GG-linear with post-hoc ternary rounding (gg_lin, n = 1000 only, with the gg_lintern setting).
* Clause budgets: 50, 100, 200, 500, 1000, 2000. FPTM: clauses per class = max(2, 2 round(n / 2m)), each distinct
  size run once.
* Grids: the final `GRIDS` of es.py (GG-TH 24, gg_lintern 30, tnn 12, CoTM 40 settings), the FPTM base grid of
  config.py (24 per size) and the FPTM one-step extension beyond boundary optima (`FPTM_EXT`).
* Tuning: seed 1, up to 20 epochs, patience 5, selection on the best validation macro F1. Clause and unit models at the
  anchors 200 and 1000 (budgets up to 500 use the 200 setting). FPTM at every size.
* Final: seeds 1 to 3, up to 60 epochs, patience 10. Test metrics at the best validation epoch.
* Extension: every final setting with a seed whose best epoch falls within the last 5 epochs of the 60-epoch cap is
  rerun from scratch with up to 120 epochs (es.py `ext`). The rerun replaces the 60-epoch result in the tables.

## Frozen choices

* The grids are not extended for these datasets. Optima on a grid boundary are reported, not acted on.
* No dataset is added or removed after the first run. A dataset that cannot be run to completion (for example, out of
  memory) is reported as such.

## Reporting

* Every dataset, budget and model: test macro F1 (primary), balanced accuracy and accuracy, mean and sample standard
  deviation over seeds 1 to 3.
* Welch t-tests against GG-TH at each budget. In addition, GG with a ternary linear readout against CoTM and against
  FPTM, which bears on the choice of the main model of the paper. That choice is made by the authors after all results
  are in. No decision rule is fixed here.
* Literals per clause (nonzero first-layer weights per unit for the ternary network), stored bits, and best epochs.
