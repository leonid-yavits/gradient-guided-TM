# Early-stopping study, results summary

Phase: final. Test metrics in %, read at the best validation epoch; mean ± sample std over seeds (1, 2, 3).
p: Welch t-test against GG-TH at the same budget. FPTM rows are listed at their nearest clause budget (clauses per class matched to the total budget).

## Search grids (seed 1, validation macro F1)

| Model | Grid size | Axes |
|---|---|---|
| GG-TH | 24 | s ∈ [1.5, 2, 5, 10, 20, 40]; lr ∈ [0.001, 0.003]; H ∈ [128, 256] |
| GG, ternary linear readout | 30 | s ∈ [1.5, 2, 5, 10, 20]; lr ∈ [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03] |
| Ternary net, no clauses | 12 | lr ∈ [3e-05, 0.0001, 0.0003, 0.001, 0.003, 0.01]; H ∈ [128, 256] |
| CoTM, tuned | 40 | s ∈ [1.5, 2, 5, 10, 20]; Tm ∈ [0.03, 0.1, 0.3, 1.0, 2.5, 5.0, 10.0, 20.0] |
| FPTM | 24 per size | (L, LF), S, T multiplier |

| Model | Dataset | Anchor | Best setting | Validation F1 |
|---|---|---|---|---|
| GG-TH | wos | 1000 | s20_lr0.003_H128 | 60.62 |
| GG-TH | wos | 200 | s1.5_lr0.001_H256 | 55.40 |
| GG-TH | alerce | 1000 | s5_lr0.003_H256 | 66.14 |
| GG-TH | alerce | 200 | s1.5_lr0.001_H256 | 60.54 |
| GG, ternary linear readout | wos | 1000 | s1.5_lr0.01_H0 | 57.86 |
| GG, ternary linear readout | wos | 200 | s1.5_lr0.0003_H0 | 54.97 |
| GG, ternary linear readout | alerce | 1000 | s5_lr0.003_H0 | 63.82 |
| GG, ternary linear readout | alerce | 200 | s2_lr0.0003_H0 | 58.63 |
| Ternary net, no clauses | wos | 1000 | s0_lr0.0003_H256 | 56.81 |
| Ternary net, no clauses | wos | 200 | s0_lr0.003_H256 | 53.15 |
| Ternary net, no clauses | alerce | 1000 | s0_lr0.0001_H128 | 68.33 |
| Ternary net, no clauses | alerce | 200 | s0_lr0.0001_H256 | 63.35 |
| CoTM, tuned | wos | 1000 | s2_Tm5.0_H0 | 52.27 |
| CoTM, tuned | wos | 200 | s2_Tm10.0_H0 | 44.55 |
| CoTM, tuned | alerce | 1000 | s1.5_Tm0.1_H0 | 62.08 |
| CoTM, tuned | alerce | 200 | s5_Tm0.3_H0 | 52.28 |

## Macro F1, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 50.01 ± 0.22 | 43.91 ± 0.29 (p=0.000) | 46.00 ± 0.22 (p=0.000) | 30.77 ± 0.68 (p=0.000) | 40.49 ± 1.11 (p=0.003) [266 cl.] |
| 100 | 53.44 ± 0.49 | 51.91 ± 0.47 (p=0.018) | 49.90 ± 0.43 (p=0.001) | 36.09 ± 0.67 (p=0.000) | n/a |
| 200 | 54.09 ± 0.36 | 54.02 ± 0.52 (p=0.860) | 51.39 ± 0.46 (p=0.002) | 43.50 ± 0.96 (p=0.001) | n/a |
| 500 | 55.31 ± 0.41 | 55.69 ± 0.33 (p=0.277) | 53.38 ± 0.24 (p=0.005) | 50.12 ± 0.09 (p=0.001) | 54.58 ± 0.39 (p=0.090) [532 cl.] |
| 1000 | 60.14 ± 0.68 | 56.19 ± 1.54 (p=0.031) | 56.04 ± 0.78 (p=0.002) | 51.79 ± 0.66 (p=0.000) | 60.68 ± 0.75 (p=0.410) [1064 cl.] |
| 2000 | 60.84 ± 0.62 | 57.87 ± 0.95 (p=0.015) | 56.23 ± 0.57 (p=0.001) | 54.71 ± 0.46 (p=0.000) | 63.30 ± 0.40 (p=0.007) [2128 cl.] |

## Macro F1, ALeRCE, 15 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 54.12 ± 0.44 | 49.52 ± 1.31 (p=0.018) | 59.07 ± 0.70 (p=0.001) | 44.35 ± 1.97 (p=0.010) | 55.93 ± 0.80 (p=0.040) [60 cl.] |
| 100 | 57.70 ± 3.38 | 54.72 ± 1.33 (p=0.263) | 60.79 ± 1.26 (p=0.249) | 51.58 ± 1.13 (p=0.076) | 59.57 ± 1.92 (p=0.462) [90 cl.] |
| 200 | 59.45 ± 0.96 | 56.63 ± 0.86 (p=0.020) | 64.48 ± 1.44 (p=0.010) | 51.68 ± 2.37 (p=0.018) | 60.71 ± 1.27 (p=0.247) [210 cl.] |
| 500 | 64.37 ± 0.78 | 59.10 ± 0.50 (p=0.001) | 64.82 ± 0.94 (p=0.560) | 55.77 ± 0.62 (p=0.000) | 62.86 ± 3.42 (p=0.527) [510 cl.] |
| 1000 | 64.74 ± 0.70 | 59.66 ± 1.87 (p=0.030) | 66.05 ± 2.47 (p=0.461) | 58.30 ± 1.56 (p=0.009) | 66.06 ± 1.02 (p=0.147) [990 cl.] |
| 2000 | 67.33 ± 1.81 | 65.09 ± 2.31 (p=0.260) | 64.87 ± 2.18 (p=0.210) | 64.54 ± 1.90 (p=0.140) | 66.85 ± 1.16 (p=0.720) [2010 cl.] |

## Balanced accuracy, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 51.89 ± 0.28 | 46.46 ± 0.30 (p=0.000) | 46.62 ± 0.51 (p=0.000) | 31.95 ± 0.55 (p=0.000) | 41.48 ± 0.89 (p=0.001) [266 cl.] |
| 100 | 54.36 ± 0.31 | 53.33 ± 0.65 (p=0.096) | 49.96 ± 0.58 (p=0.001) | 36.23 ± 0.71 (p=0.000) | n/a |
| 200 | 54.96 ± 0.50 | 54.77 ± 0.42 (p=0.631) | 51.43 ± 0.31 (p=0.001) | 42.95 ± 1.21 (p=0.001) | n/a |
| 500 | 55.64 ± 0.41 | 56.28 ± 0.11 (p=0.105) | 53.13 ± 0.11 (p=0.006) | 49.73 ± 0.09 (p=0.001) | 56.61 ± 0.46 (p=0.054) [532 cl.] |
| 1000 | 61.06 ± 0.67 | 56.40 ± 1.28 (p=0.011) | 55.72 ± 0.81 (p=0.001) | 51.46 ± 0.88 (p=0.000) | 62.32 ± 0.70 (p=0.087) [1064 cl.] |
| 2000 | 61.55 ± 0.45 | 58.11 ± 1.35 (p=0.036) | 55.77 ± 0.48 (p=0.000) | 54.47 ± 0.51 (p=0.000) | 64.54 ± 0.39 (p=0.001) [2128 cl.] |

## Balanced accuracy, ALeRCE, 15 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 67.24 ± 1.12 | 65.18 ± 3.99 (p=0.469) | 67.12 ± 0.92 (p=0.894) | 59.74 ± 2.99 (p=0.036) | 69.96 ± 0.94 (p=0.033) [60 cl.] |
| 100 | 69.82 ± 1.66 | 68.39 ± 2.44 (p=0.456) | 66.86 ± 0.82 (p=0.072) | 68.04 ± 1.63 (p=0.256) | 72.64 ± 1.75 (p=0.114) [90 cl.] |
| 200 | 70.05 ± 0.93 | 70.76 ± 1.48 (p=0.529) | 67.07 ± 1.12 (p=0.025) | 67.60 ± 2.37 (p=0.209) | 70.61 ± 0.18 (p=0.409) [210 cl.] |
| 500 | 70.55 ± 1.09 | 69.90 ± 1.11 (p=0.507) | 67.57 ± 0.68 (p=0.022) | 68.99 ± 1.47 (p=0.218) | 70.89 ± 0.65 (p=0.678) [510 cl.] |
| 1000 | 69.68 ± 1.98 | 69.97 ± 2.54 (p=0.882) | 66.88 ± 2.67 (p=0.224) | 67.33 ± 0.42 (p=0.171) | 70.19 ± 0.23 (p=0.698) [990 cl.] |
| 2000 | 69.72 ± 1.11 | 71.08 ± 0.81 (p=0.168) | 67.60 ± 1.53 (p=0.130) | 65.83 ± 3.55 (p=0.191) | 70.95 ± 0.32 (p=0.189) [2010 cl.] |

## Accuracy, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 52.99 ± 0.33 | 47.71 ± 0.41 (p=0.000) | 48.39 ± 0.23 (p=0.000) | 32.70 ± 0.67 (p=0.000) | 43.36 ± 1.34 (p=0.004) [266 cl.] |
| 100 | 56.06 ± 0.48 | 54.54 ± 0.58 (p=0.026) | 52.07 ± 0.48 (p=0.001) | 37.90 ± 0.41 (p=0.000) | n/a |
| 200 | 56.81 ± 0.14 | 56.43 ± 0.25 (p=0.102) | 54.09 ± 0.38 (p=0.003) | 44.88 ± 1.19 (p=0.003) | n/a |
| 500 | 57.73 ± 0.31 | 58.02 ± 0.25 (p=0.278) | 55.89 ± 0.02 (p=0.009) | 52.20 ± 0.33 (p=0.000) | 58.01 ± 0.36 (p=0.372) [532 cl.] |
| 1000 | 62.20 ± 0.80 | 58.50 ± 1.19 (p=0.015) | 58.44 ± 1.18 (p=0.014) | 54.18 ± 0.64 (p=0.000) | 64.15 ± 0.59 (p=0.031) [1064 cl.] |
| 2000 | 63.06 ± 0.44 | 60.35 ± 0.68 (p=0.007) | 58.40 ± 0.37 (p=0.000) | 57.25 ± 0.37 (p=0.000) | 66.23 ± 0.43 (p=0.001) [2128 cl.] |

## Accuracy, ALeRCE, 15 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 81.47 ± 0.98 | 76.24 ± 0.71 (p=0.002) | 85.53 ± 1.96 (p=0.050) | 72.66 ± 0.48 (p=0.001) | 83.24 ± 0.64 (p=0.068) [60 cl.] |
| 100 | 83.48 ± 3.37 | 82.00 ± 1.49 (p=0.543) | 87.08 ± 1.03 (p=0.198) | 79.73 ± 3.14 (p=0.232) | 85.62 ± 0.36 (p=0.384) [90 cl.] |
| 200 | 85.68 ± 0.26 | 81.95 ± 1.99 (p=0.081) | 90.20 ± 0.36 (p=0.000) | 79.24 ± 2.63 (p=0.050) | 87.04 ± 0.66 (p=0.056) [210 cl.] |
| 500 | 88.60 ± 0.58 | 84.48 ± 1.31 (p=0.019) | 90.23 ± 0.57 (p=0.026) | 82.27 ± 0.35 (p=0.000) | 88.71 ± 2.14 (p=0.938) [510 cl.] |
| 1000 | 88.82 ± 1.00 | 86.10 ± 0.60 (p=0.023) | 90.01 ± 0.78 (p=0.182) | 84.61 ± 0.26 (p=0.014) | 90.92 ± 0.37 (p=0.054) [990 cl.] |
| 2000 | 90.61 ± 1.11 | 88.67 ± 1.40 (p=0.138) | 90.33 ± 0.91 (p=0.757) | 88.45 ± 1.30 (p=0.096) | 91.23 ± 0.66 (p=0.465) [2010 cl.] |

## GG-linear trained in floating point, evaluated with weights rounded to {-1, 0, +1} (n = 1000)

| Dataset | GG-linear float F1 | GG-linear rounded F1 | GG, ternary readout (STE) F1 | GG-TH F1 |
|---|---|---|---|---|
| wos | 54.90 ± 0.82 | 45.93 ± 1.66 | 56.19 ± 1.54 | 60.14 ± 0.68 |
| alerce | 61.73 ± 1.61 | 58.96 ± 1.04 | 59.66 ± 1.87 | 64.74 ± 0.70 |

## Sparsity and memory

Literals: mean included literals per clause (clause models) or mean nonzero first-layer weights per unit (ternary net), at the best epoch. Bits: stored bits of the clause or first layer plus ternary readout layers.

| Dataset | Budget | Model | Literals per clause or unit | Bits | Best epoch (mean) |
|---|---|---|---|---|---|
| wos | 50 | GG-TH | 58.4 | 293,696 | 18.3 |
| wos | 50 | GG, ternary linear readout | 58.9 | 213,300 | 23.0 |
| wos | 50 | Ternary net, no clauses | 832.5 | 293,696 | 23.0 |
| wos | 50 | CoTM, tuned | 111.7 | 200,000 | 54.3 |
| wos | 50 | FPTM | 30.4 | 1,064,000 | 38.3 |
| wos | 100 | GG-TH | 70.6 | 519,296 | 18.7 |
| wos | 100 | GG, ternary linear readout | 67.7 | 426,600 | 20.3 |
| wos | 100 | Ternary net, no clauses | 929.3 | 519,296 | 15.0 |
| wos | 100 | CoTM, tuned | 117.4 | 400,000 | 35.3 |
| wos | 200 | GG-TH | 75.6 | 970,496 | 9.0 |
| wos | 200 | GG, ternary linear readout | 71.3 | 853,200 | 13.3 |
| wos | 200 | Ternary net, no clauses | 1000.6 | 970,496 | 12.7 |
| wos | 200 | CoTM, tuned | 116.8 | 800,000 | 31.7 |
| wos | 500 | GG-TH | 76.8 | 2,324,096 | 6.3 |
| wos | 500 | GG, ternary linear readout | 70.7 | 2,133,000 | 2.7 |
| wos | 500 | Ternary net, no clauses | 1065.6 | 2,324,096 | 11.3 |
| wos | 500 | CoTM, tuned | 117.2 | 2,000,000 | 34.7 |
| wos | 500 | FPTM | 15.5 | 2,128,000 | 53.0 |
| wos | 1000 | GG-TH | 4.1 | 4,290,048 | 12.3 |
| wos | 1000 | GG, ternary linear readout | 69.9 | 4,266,000 | 3.0 |
| wos | 1000 | Ternary net, no clauses | 1140.7 | 4,580,096 | 2.0 |
| wos | 1000 | CoTM, tuned | 115.4 | 4,000,000 | 21.7 |
| wos | 1000 | FPTM | 21.7 | 4,256,000 | 36.7 |
| wos | 2000 | GG-TH | 5.7 | 8,546,048 | 16.7 |
| wos | 2000 | GG, ternary linear readout | 73.2 | 8,532,000 | 4.0 |
| wos | 2000 | Ternary net, no clauses | 1146.3 | 9,092,096 | 2.3 |
| wos | 2000 | CoTM, tuned | 115.2 | 8,000,000 | 32.0 |
| wos | 2000 | FPTM | 24.3 | 8,512,000 | 26.0 |
| alerce | 50 | GG-TH | 18.0 | 150,480 | 14.0 |
| alerce | 50 | GG, ternary linear readout | 16.2 | 118,700 | 9.7 |
| alerce | 50 | Ternary net, no clauses | 612.1 | 150,480 | 32.3 |
| alerce | 50 | CoTM, tuned | 2.9 | 117,200 | 13.0 |
| alerce | 50 | FPTM | 84.5 | 140,640 | 14.0 |
| alerce | 100 | GG-TH | 32.6 | 293,280 | 20.7 |
| alerce | 100 | GG, ternary linear readout | 23.1 | 237,400 | 8.7 |
| alerce | 100 | Ternary net, no clauses | 649.8 | 293,280 | 18.0 |
| alerce | 100 | CoTM, tuned | 6.9 | 234,400 | 18.0 |
| alerce | 100 | FPTM | 89.6 | 210,960 | 27.3 |
| alerce | 200 | GG-TH | 40.4 | 578,880 | 7.0 |
| alerce | 200 | GG, ternary linear readout | 28.8 | 474,800 | 9.7 |
| alerce | 200 | Ternary net, no clauses | 653.4 | 578,880 | 35.0 |
| alerce | 200 | CoTM, tuned | 16.3 | 468,800 | 11.7 |
| alerce | 200 | FPTM | 80.2 | 492,240 | 14.0 |
| alerce | 500 | GG-TH | 49.7 | 1,435,680 | 25.3 |
| alerce | 500 | GG, ternary linear readout | 38.8 | 1,187,000 | 11.3 |
| alerce | 500 | Ternary net, no clauses | 667.9 | 1,435,680 | 21.0 |
| alerce | 500 | CoTM, tuned | 33.0 | 1,172,000 | 17.7 |
| alerce | 500 | FPTM | 101.2 | 1,195,440 | 32.3 |
| alerce | 1000 | GG-TH | 16.8 | 2,863,680 | 20.7 |
| alerce | 1000 | GG, ternary linear readout | 14.1 | 2,374,000 | 16.0 |
| alerce | 1000 | Ternary net, no clauses | 673.1 | 2,603,840 | 13.7 |
| alerce | 1000 | CoTM, tuned | 48.8 | 2,344,000 | 17.0 |
| alerce | 1000 | FPTM | 103.2 | 2,320,560 | 35.0 |
| alerce | 2000 | GG-TH | 20.1 | 5,719,680 | 22.3 |
| alerce | 2000 | GG, ternary linear readout | 17.7 | 4,748,000 | 17.7 |
| alerce | 2000 | Ternary net, no clauses | 673.6 | 5,203,840 | 22.0 |
| alerce | 2000 | CoTM, tuned | 49.5 | 4,688,000 | 13.0 |
| alerce | 2000 | FPTM | 109.9 | 4,711,440 | 47.3 |

Runs whose best epoch lies within the last 5 epochs of the cap (60): 6.
- alerce FPTM n=2000
- wos CoTM, tuned n=50
- wos FPTM n=500
- wos FPTM n=1000
