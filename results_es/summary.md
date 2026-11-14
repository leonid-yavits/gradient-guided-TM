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
| GG-TH | r52 | 1000 | s1.5_lr0.001_H256 | 69.54 |
| GG-TH | r52 | 200 | s10_lr0.003_H256 | 63.87 |
| GG-TH | sector | 1000 | s1.5_lr0.001_H256 | 52.99 |
| GG-TH | sector | 200 | s1.5_lr0.001_H256 | 49.14 |
| GG-TH | rcv1 | 1000 | s1.5_lr0.001_H128 | 64.34 |
| GG-TH | rcv1 | 200 | s1.5_lr0.003_H256 | 59.03 |
| GG-TH | news20 | 1000 | s1.5_lr0.003_H256 | 73.98 |
| GG-TH | news20 | 200 | s1.5_lr0.001_H128 | 72.02 |
| GG, ternary linear readout | wos | 1000 | s1.5_lr0.01_H0 | 57.86 |
| GG, ternary linear readout | wos | 200 | s1.5_lr0.0003_H0 | 54.97 |
| GG, ternary linear readout | alerce | 1000 | s5_lr0.003_H0 | 63.82 |
| GG, ternary linear readout | alerce | 200 | s2_lr0.0003_H0 | 58.63 |
| GG, ternary linear readout | r52 | 1000 | s1.5_lr0.001_H0 | 74.46 |
| GG, ternary linear readout | r52 | 200 | s1.5_lr0.001_H0 | 68.17 |
| GG, ternary linear readout | sector | 1000 | s1.5_lr0.001_H0 | 60.58 |
| GG, ternary linear readout | sector | 200 | s1.5_lr0.01_H0 | 54.81 |
| GG, ternary linear readout | rcv1 | 1000 | s2_lr0.001_H0 | 65.86 |
| GG, ternary linear readout | rcv1 | 200 | s1.5_lr0.0001_H0 | 59.86 |
| GG, ternary linear readout | news20 | 1000 | s1.5_lr0.0001_H0 | 75.03 |
| GG, ternary linear readout | news20 | 200 | s1.5_lr0.0001_H0 | 74.02 |
| Ternary net, no clauses | wos | 1000 | s0_lr0.0003_H256 | 56.81 |
| Ternary net, no clauses | wos | 200 | s0_lr0.003_H256 | 53.15 |
| Ternary net, no clauses | alerce | 1000 | s0_lr0.0001_H128 | 68.33 |
| Ternary net, no clauses | alerce | 200 | s0_lr0.0001_H256 | 63.35 |
| Ternary net, no clauses | r52 | 1000 | s0_lr0.001_H128 | 74.68 |
| Ternary net, no clauses | r52 | 200 | s0_lr0.0003_H256 | 73.78 |
| Ternary net, no clauses | sector | 1000 | s0_lr0.001_H256 | 69.53 |
| Ternary net, no clauses | sector | 200 | s0_lr0.01_H256 | 67.31 |
| Ternary net, no clauses | rcv1 | 1000 | s0_lr0.0003_H256 | 63.21 |
| Ternary net, no clauses | rcv1 | 200 | s0_lr0.01_H128 | 62.99 |
| Ternary net, no clauses | news20 | 1000 | s0_lr0.003_H128 | 76.88 |
| Ternary net, no clauses | news20 | 200 | s0_lr0.01_H128 | 76.57 |
| CoTM, tuned | wos | 1000 | s2_Tm5.0_H0 | 52.27 |
| CoTM, tuned | wos | 200 | s2_Tm10.0_H0 | 44.55 |
| CoTM, tuned | alerce | 1000 | s1.5_Tm0.1_H0 | 62.08 |
| CoTM, tuned | alerce | 200 | s5_Tm0.3_H0 | 52.28 |
| CoTM, tuned | r52 | 1000 | s1.5_Tm5.0_H0 | 75.43 |
| CoTM, tuned | r52 | 200 | s1.5_Tm5.0_H0 | 66.79 |
| CoTM, tuned | sector | 1000 | s2_Tm2.5_H0 | 54.31 |
| CoTM, tuned | sector | 200 | s1.5_Tm5.0_H0 | 46.21 |
| CoTM, tuned | rcv1 | 1000 | s5_Tm1.0_H0 | 60.01 |
| CoTM, tuned | rcv1 | 200 | s2_Tm1.0_H0 | 53.06 |
| CoTM, tuned | news20 | 1000 | s2_Tm10.0_H0 | 73.07 |
| CoTM, tuned | news20 | 200 | s1.5_Tm20.0_H0 | 70.53 |

Selected settings on a grid boundary, axes with more than two values (grids not extended for these, see experiments/PROTOCOL_text.md):

- GG-TH, wos, anchor 200: s=1.5 (lower edge)
- GG-TH, alerce, anchor 200: s=1.5 (lower edge)
- GG-TH, r52, anchor 1000: s=1.5 (lower edge)
- GG-TH, sector, anchor 1000: s=1.5 (lower edge)
- GG-TH, sector, anchor 200: s=1.5 (lower edge)
- GG-TH, rcv1, anchor 1000: s=1.5 (lower edge)
- GG-TH, rcv1, anchor 200: s=1.5 (lower edge)
- GG-TH, news20, anchor 1000: s=1.5 (lower edge)
- GG-TH, news20, anchor 200: s=1.5 (lower edge)
- GG, ternary linear readout, wos, anchor 1000: s=1.5 (lower edge)
- GG, ternary linear readout, wos, anchor 200: s=1.5 (lower edge)
- GG, ternary linear readout, r52, anchor 1000: s=1.5 (lower edge)
- GG, ternary linear readout, r52, anchor 200: s=1.5 (lower edge)
- GG, ternary linear readout, sector, anchor 1000: s=1.5 (lower edge)
- GG, ternary linear readout, sector, anchor 200: s=1.5 (lower edge)
- GG, ternary linear readout, rcv1, anchor 200: s=1.5 (lower edge), lr=0.0001 (lower edge)
- GG, ternary linear readout, news20, anchor 1000: s=1.5 (lower edge), lr=0.0001 (lower edge)
- GG, ternary linear readout, news20, anchor 200: s=1.5 (lower edge), lr=0.0001 (lower edge)
- Ternary net, no clauses, sector, anchor 200: lr=0.01 (upper edge)
- Ternary net, no clauses, rcv1, anchor 200: lr=0.01 (upper edge)
- Ternary net, no clauses, news20, anchor 200: lr=0.01 (upper edge)
- CoTM, tuned, alerce, anchor 1000: s=1.5 (lower edge)
- CoTM, tuned, r52, anchor 1000: s=1.5 (lower edge)
- CoTM, tuned, r52, anchor 200: s=1.5 (lower edge)
- CoTM, tuned, sector, anchor 200: s=1.5 (lower edge)
- CoTM, tuned, news20, anchor 200: s=1.5 (lower edge), Tm=20.0 (upper edge)

## Macro F1, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 50.01 ± 0.22 | 43.91 ± 0.29 (p=0.000) | 46.00 ± 0.22 (p=0.000) | 31.66 ± 2.02 (p=0.004) | 40.49 ± 1.11 (p=0.003) [266 cl.] |
| 100 | 53.44 ± 0.49 | 51.91 ± 0.47 (p=0.018) | 49.90 ± 0.43 (p=0.001) | 36.09 ± 0.67 (p=0.000) | n/a |
| 200 | 54.09 ± 0.36 | 54.02 ± 0.52 (p=0.860) | 51.39 ± 0.46 (p=0.002) | 43.50 ± 0.96 (p=0.001) | n/a |
| 500 | 55.31 ± 0.41 | 55.69 ± 0.33 (p=0.277) | 53.38 ± 0.24 (p=0.005) | 50.12 ± 0.09 (p=0.001) | 55.19 ± 0.85 (p=0.843) [532 cl.] |
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
| 2000 | 67.33 ± 1.81 | 65.09 ± 2.31 (p=0.260) | 64.87 ± 2.18 (p=0.210) | 64.54 ± 1.90 (p=0.140) | 66.77 ± 1.11 (p=0.674) [2010 cl.] |

## Macro F1, R52, 52 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 28.95 ± 1.79 | 37.30 ± 1.26 (p=0.004) | 49.13 ± 0.55 (p=0.001) | 37.01 ± 1.41 (p=0.004) | 44.91 ± 2.14 (p=0.001) [104 cl.] |
| 100 | 41.67 ± 0.95 | 51.86 ± 1.64 (p=0.002) | 51.76 ± 2.72 (p=0.015) | 43.08 ± 1.36 (p=0.224) | n/a |
| 200 | 50.25 ± 1.80 | 55.02 ± 1.86 (p=0.033) | 54.09 ± 4.53 (p=0.278) | 52.96 ± 1.17 (p=0.106) | 54.80 ± 2.40 (p=0.063) [208 cl.] |
| 500 | 51.96 ± 1.09 | 61.26 ± 2.75 (p=0.017) | 55.91 ± 1.78 (p=0.041) | 57.65 ± 0.26 (p=0.009) | 60.35 ± 2.44 (p=0.015) [520 cl.] |
| 1000 | 59.13 ± 1.10 | 64.62 ± 1.16 (p=0.004) | 57.20 ± 0.70 (p=0.073) | 60.50 ± 2.35 (p=0.429) | 57.87 ± 0.73 (p=0.184) [1040 cl.] |
| 2000 | 60.59 ± 1.92 | 63.51 ± 1.14 (p=0.101) | 57.89 ± 3.45 (p=0.318) | 61.25 ± 0.87 (p=0.627) | 58.80 ± 0.70 (p=0.243) [1976 cl.] |

## Macro F1, Sector, 105 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 38.50 ± 2.62 | 36.20 ± 0.53 (p=0.266) | 59.40 ± 0.75 (p=0.003) | 28.47 ± 0.86 (p=0.015) | 42.73 ± 6.79 (p=0.399) [210 cl.] |
| 100 | 45.23 ± 0.40 | 46.78 ± 0.61 (p=0.027) | 61.07 ± 0.82 (p=0.000) | 38.45 ± 1.35 (p=0.009) | n/a |
| 200 | 47.70 ± 0.73 | 52.82 ± 0.81 (p=0.001) | 62.80 ± 0.02 (p=0.001) | 44.42 ± 1.57 (p=0.050) | n/a |
| 500 | 49.86 ± 1.14 | 56.61 ± 0.45 (p=0.004) | 64.68 ± 0.17 (p=0.002) | 50.28 ± 0.75 (p=0.625) | 54.22 ± 1.03 (p=0.008) [420 cl.] |
| 1000 | 52.24 ± 0.67 | 57.51 ± 0.66 (p=0.001) | 64.70 ± 0.19 (p=0.000) | 53.67 ± 1.25 (p=0.178) | 59.88 ± 1.56 (p=0.006) [1050 cl.] |
| 2000 | 53.20 ± 0.36 | 59.94 ± 0.48 (p=0.000) | 64.70 ± 0.29 (p=0.000) | 56.34 ± 0.55 (p=0.002) | 64.29 ± 0.74 (p=0.000) [2100 cl.] |

## Macro F1, RCV1, 51 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 33.43 ± 1.87 | 29.05 ± 0.28 (p=0.053) | 44.27 ± 0.29 (p=0.009) | 25.66 ± 1.16 (p=0.006) | 41.94 ± 0.51 (p=0.011) [102 cl.] |
| 100 | 39.68 ± 0.52 | 37.69 ± 1.17 (p=0.081) | 45.35 ± 0.57 (p=0.000) | 31.49 ± 1.18 (p=0.002) | n/a |
| 200 | 42.50 ± 0.79 | 42.27 ± 0.12 (p=0.673) | 46.18 ± 0.13 (p=0.013) | 35.11 ± 1.11 (p=0.001) | 46.23 ± 3.16 (p=0.171) [204 cl.] |
| 500 | 45.19 ± 0.10 | 46.43 ± 0.38 (p=0.023) | 46.88 ± 0.49 (p=0.023) | 40.84 ± 1.45 (p=0.035) | 49.04 ± 0.72 (p=0.010) [510 cl.] |
| 1000 | 44.55 ± 0.28 | 48.21 ± 0.53 (p=0.002) | 48.40 ± 0.13 (p=0.000) | 43.52 ± 0.98 (p=0.206) | 48.84 ± 0.91 (p=0.010) [1020 cl.] |
| 2000 | 46.15 ± 1.75 | 49.77 ± 0.86 (p=0.051) | 48.49 ± 0.81 (p=0.132) | 46.61 ± 0.24 (p=0.693) | 50.15 ± 1.06 (p=0.037) [2040 cl.] |

## Macro F1, 20 Newsgroups, 20 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 66.89 ± 1.45 | 69.08 ± 1.03 (p=0.107) | 75.86 ± 0.20 (p=0.008) | 62.12 ± 0.55 (p=0.019) | 70.24 ± 1.40 (p=0.045) [40 cl.] |
| 100 | 70.04 ± 1.12 | 72.34 ± 0.78 (p=0.050) | 76.53 ± 0.48 (p=0.004) | 67.01 ± 0.31 (p=0.035) | 73.84 ± 0.38 (p=0.019) [80 cl.] |
| 200 | 71.96 ± 0.52 | 74.77 ± 0.17 (p=0.007) | 76.63 ± 0.61 (p=0.001) | 70.63 ± 0.67 (p=0.057) | 77.24 ± 0.32 (p=0.000) [200 cl.] |
| 500 | 73.97 ± 0.49 | 75.15 ± 0.18 (p=0.041) | 77.46 ± 0.59 (p=0.002) | 72.05 ± 0.84 (p=0.038) | 78.50 ± 0.13 (p=0.002) [480 cl.] |
| 1000 | 74.68 ± 0.98 | 75.39 ± 0.61 (p=0.357) | 77.14 ± 0.47 (p=0.031) | 73.63 ± 0.44 (p=0.196) | 79.26 ± 0.52 (p=0.005) [1000 cl.] |
| 2000 | 74.88 ± 0.48 | 76.07 ± 0.46 (p=0.036) | 77.04 ± 0.29 (p=0.005) | 74.47 ± 0.22 (p=0.281) | 79.30 ± 0.18 (p=0.002) [2000 cl.] |

## Balanced accuracy, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 51.89 ± 0.28 | 46.46 ± 0.30 (p=0.000) | 46.62 ± 0.51 (p=0.000) | 33.08 ± 2.08 (p=0.004) | 41.48 ± 0.89 (p=0.001) [266 cl.] |
| 100 | 54.36 ± 0.31 | 53.33 ± 0.65 (p=0.096) | 49.96 ± 0.58 (p=0.001) | 36.23 ± 0.71 (p=0.000) | n/a |
| 200 | 54.96 ± 0.50 | 54.77 ± 0.42 (p=0.631) | 51.43 ± 0.31 (p=0.001) | 42.95 ± 1.21 (p=0.001) | n/a |
| 500 | 55.64 ± 0.41 | 56.28 ± 0.11 (p=0.105) | 53.13 ± 0.11 (p=0.006) | 49.73 ± 0.09 (p=0.001) | 57.12 ± 0.91 (p=0.091) [532 cl.] |
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
| 2000 | 69.72 ± 1.11 | 71.08 ± 0.81 (p=0.168) | 67.60 ± 1.53 (p=0.130) | 65.83 ± 3.55 (p=0.191) | 70.84 ± 0.48 (p=0.216) [2010 cl.] |

## Balanced accuracy, R52, 52 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 34.77 ± 1.13 | 45.53 ± 1.03 (p=0.000) | 50.54 ± 1.70 (p=0.000) | 42.92 ± 2.22 (p=0.011) | 50.63 ± 2.50 (p=0.003) [104 cl.] |
| 100 | 46.87 ± 1.44 | 54.95 ± 2.55 (p=0.016) | 52.14 ± 2.23 (p=0.034) | 47.92 ± 1.02 (p=0.369) | n/a |
| 200 | 54.03 ± 2.25 | 55.76 ± 2.73 (p=0.446) | 53.20 ± 3.22 (p=0.737) | 55.61 ± 2.25 (p=0.437) | 60.20 ± 0.81 (p=0.030) [208 cl.] |
| 500 | 55.08 ± 1.03 | 59.91 ± 1.74 (p=0.022) | 54.18 ± 2.05 (p=0.546) | 60.54 ± 2.31 (p=0.038) | 63.51 ± 1.91 (p=0.006) [520 cl.] |
| 1000 | 58.43 ± 0.38 | 62.87 ± 0.88 (p=0.006) | 55.21 ± 0.69 (p=0.005) | 63.31 ± 3.95 (p=0.164) | 59.25 ± 0.61 (p=0.134) [1040 cl.] |
| 2000 | 59.67 ± 1.88 | 61.37 ± 0.79 (p=0.254) | 56.23 ± 2.13 (p=0.105) | 62.87 ± 1.12 (p=0.078) | 60.74 ± 0.34 (p=0.428) [1976 cl.] |

## Balanced accuracy, Sector, 105 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 38.83 ± 2.56 | 36.75 ± 1.06 (p=0.295) | 58.27 ± 0.75 (p=0.003) | 26.48 ± 0.76 (p=0.009) | 37.69 ± 7.14 (p=0.815) [210 cl.] |
| 100 | 45.00 ± 0.84 | 46.30 ± 0.63 (p=0.105) | 59.90 ± 0.75 (p=0.000) | 36.52 ± 1.11 (p=0.001) | n/a |
| 200 | 47.09 ± 0.89 | 52.27 ± 0.56 (p=0.002) | 61.77 ± 0.14 (p=0.001) | 41.43 ± 1.14 (p=0.003) | n/a |
| 500 | 49.28 ± 0.82 | 55.80 ± 0.37 (p=0.002) | 63.92 ± 0.20 (p=0.001) | 46.36 ± 1.57 (p=0.064) | 53.23 ± 1.07 (p=0.008) [420 cl.] |
| 1000 | 51.35 ± 0.58 | 57.95 ± 0.30 (p=0.000) | 63.95 ± 0.39 (p=0.000) | 51.68 ± 1.60 (p=0.760) | 57.23 ± 1.61 (p=0.015) [1050 cl.] |
| 2000 | 51.79 ± 0.59 | 60.18 ± 0.24 (p=0.000) | 63.78 ± 0.48 (p=0.000) | 54.09 ± 0.44 (p=0.007) | 63.31 ± 0.71 (p=0.000) [2100 cl.] |

## Balanced accuracy, RCV1, 51 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 37.75 ± 1.28 | 34.28 ± 1.11 (p=0.025) | 45.17 ± 0.32 (p=0.007) | 29.59 ± 0.66 (p=0.002) | 45.60 ± 0.58 (p=0.003) [102 cl.] |
| 100 | 42.10 ± 0.37 | 41.42 ± 0.68 (p=0.223) | 45.52 ± 0.55 (p=0.002) | 34.59 ± 1.11 (p=0.004) | n/a |
| 200 | 44.15 ± 0.36 | 45.60 ± 0.26 (p=0.006) | 46.23 ± 0.49 (p=0.005) | 37.72 ± 1.05 (p=0.005) | 51.08 ± 1.84 (p=0.020) [204 cl.] |
| 500 | 45.96 ± 0.27 | 49.75 ± 1.10 (p=0.022) | 46.90 ± 1.00 (p=0.239) | 41.96 ± 1.26 (p=0.027) | 52.05 ± 0.21 (p=0.000) [510 cl.] |
| 1000 | 46.78 ± 0.90 | 48.89 ± 2.06 (p=0.210) | 47.38 ± 0.23 (p=0.368) | 44.56 ± 0.19 (p=0.046) | 53.51 ± 0.71 (p=0.001) [1020 cl.] |
| 2000 | 47.14 ± 2.35 | 50.10 ± 0.57 (p=0.154) | 47.85 ± 0.61 (p=0.658) | 46.84 ± 0.19 (p=0.847) | 51.95 ± 0.18 (p=0.070) [2040 cl.] |

## Balanced accuracy, 20 Newsgroups, 20 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 66.86 ± 1.28 | 68.97 ± 0.83 (p=0.085) | 75.75 ± 0.05 (p=0.007) | 61.88 ± 0.73 (p=0.008) | 70.50 ± 1.30 (p=0.026) [40 cl.] |
| 100 | 69.89 ± 1.33 | 72.35 ± 0.76 (p=0.065) | 76.38 ± 0.52 (p=0.007) | 66.95 ± 0.32 (p=0.055) | 74.15 ± 0.61 (p=0.018) [80 cl.] |
| 200 | 71.88 ± 0.44 | 74.81 ± 0.18 (p=0.003) | 76.71 ± 0.59 (p=0.001) | 70.59 ± 0.79 (p=0.086) | 77.66 ± 0.11 (p=0.001) [200 cl.] |
| 500 | 73.63 ± 0.50 | 75.14 ± 0.13 (p=0.028) | 77.39 ± 0.62 (p=0.001) | 71.80 ± 1.29 (p=0.119) | 78.69 ± 0.27 (p=0.000) [480 cl.] |
| 1000 | 74.62 ± 0.76 | 75.49 ± 0.55 (p=0.189) | 77.01 ± 0.50 (p=0.014) | 73.51 ± 0.89 (p=0.177) | 79.52 ± 0.43 (p=0.002) [1000 cl.] |
| 2000 | 74.83 ± 0.42 | 76.32 ± 0.29 (p=0.010) | 77.03 ± 0.23 (p=0.004) | 74.71 ± 0.33 (p=0.729) | 79.52 ± 0.23 (p=0.000) [2000 cl.] |

## Accuracy, WOS, 133 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 52.99 ± 0.33 | 47.71 ± 0.41 (p=0.000) | 48.39 ± 0.23 (p=0.000) | 33.93 ± 2.24 (p=0.004) | 43.36 ± 1.34 (p=0.004) [266 cl.] |
| 100 | 56.06 ± 0.48 | 54.54 ± 0.58 (p=0.026) | 52.07 ± 0.48 (p=0.001) | 37.90 ± 0.41 (p=0.000) | n/a |
| 200 | 56.81 ± 0.14 | 56.43 ± 0.25 (p=0.102) | 54.09 ± 0.38 (p=0.003) | 44.88 ± 1.19 (p=0.003) | n/a |
| 500 | 57.73 ± 0.31 | 58.02 ± 0.25 (p=0.278) | 55.89 ± 0.02 (p=0.009) | 52.20 ± 0.33 (p=0.000) | 58.72 ± 0.97 (p=0.212) [532 cl.] |
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
| 2000 | 90.61 ± 1.11 | 88.67 ± 1.40 (p=0.138) | 90.33 ± 0.91 (p=0.757) | 88.45 ± 1.30 (p=0.096) | 91.20 ± 0.64 (p=0.483) [2010 cl.] |

## Accuracy, R52, 52 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 44.58 ± 14.23 | 70.91 ± 3.88 (p=0.076) | 86.52 ± 0.98 (p=0.036) | 75.33 ± 1.81 (p=0.062) | 81.20 ± 3.12 (p=0.041) [104 cl.] |
| 100 | 68.65 ± 4.99 | 84.12 ± 0.28 (p=0.033) | 87.57 ± 1.05 (p=0.019) | 79.77 ± 0.99 (p=0.056) | n/a |
| 200 | 80.41 ± 1.27 | 86.71 ± 0.40 (p=0.008) | 88.57 ± 0.90 (p=0.001) | 84.03 ± 0.69 (p=0.021) | 85.69 ± 1.73 (p=0.016) [208 cl.] |
| 500 | 82.57 ± 0.21 | 87.78 ± 1.46 (p=0.023) | 88.39 ± 0.96 (p=0.007) | 86.03 ± 0.68 (p=0.008) | 88.61 ± 0.32 (p=0.000) [520 cl.] |
| 1000 | 87.31 ± 0.60 | 89.50 ± 0.97 (p=0.039) | 88.85 ± 0.42 (p=0.026) | 86.79 ± 0.33 (p=0.277) | 90.04 ± 0.27 (p=0.007) [1040 cl.] |
| 2000 | 85.87 ± 2.90 | 85.93 ± 7.54 (p=0.990) | 88.58 ± 0.50 (p=0.245) | 88.08 ± 0.21 (p=0.318) | 89.71 ± 0.66 (p=0.143) [1976 cl.] |

## Accuracy, Sector, 105 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 37.52 ± 2.44 | 35.39 ± 1.06 (p=0.268) | 57.13 ± 1.05 (p=0.002) | 25.70 ± 0.77 (p=0.009) | 35.99 ± 6.82 (p=0.744) [210 cl.] |
| 100 | 44.27 ± 0.75 | 44.96 ± 0.70 (p=0.306) | 59.03 ± 0.88 (p=0.000) | 35.63 ± 1.09 (p=0.001) | n/a |
| 200 | 46.27 ± 0.95 | 50.68 ± 0.60 (p=0.004) | 60.94 ± 0.18 (p=0.001) | 40.83 ± 1.16 (p=0.004) | n/a |
| 500 | 48.16 ± 1.29 | 54.68 ± 0.42 (p=0.008) | 63.01 ± 0.29 (p=0.002) | 45.37 ± 1.17 (p=0.050) | 52.06 ± 1.06 (p=0.017) [420 cl.] |
| 1000 | 50.27 ± 0.70 | 56.58 ± 0.53 (p=0.000) | 62.83 ± 0.41 (p=0.000) | 50.56 ± 1.78 (p=0.811) | 56.20 ± 1.45 (p=0.009) [1050 cl.] |
| 2000 | 50.68 ± 0.64 | 59.00 ± 0.19 (p=0.001) | 62.94 ± 0.48 (p=0.000) | 52.69 ± 0.54 (p=0.015) | 62.26 ± 0.85 (p=0.000) [2100 cl.] |

## Accuracy, RCV1, 51 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 64.27 ± 1.51 | 56.88 ± 1.18 (p=0.003) | 77.82 ± 0.64 (p=0.001) | 50.02 ± 6.84 (p=0.063) | 71.72 ± 0.50 (p=0.008) [102 cl.] |
| 100 | 70.87 ± 2.26 | 68.16 ± 0.99 (p=0.163) | 79.58 ± 0.41 (p=0.019) | 62.09 ± 3.05 (p=0.019) | n/a |
| 200 | 74.39 ± 0.67 | 71.98 ± 0.64 (p=0.011) | 80.20 ± 0.78 (p=0.001) | 61.98 ± 2.53 (p=0.010) | 75.69 ± 4.20 (p=0.645) [204 cl.] |
| 500 | 77.24 ± 0.38 | 75.30 ± 1.45 (p=0.138) | 80.89 ± 0.97 (p=0.013) | 68.13 ± 1.77 (p=0.010) | 78.57 ± 0.65 (p=0.050) [510 cl.] |
| 1000 | 74.88 ± 2.26 | 78.25 ± 1.19 (p=0.106) | 81.62 ± 0.26 (p=0.034) | 75.84 ± 1.12 (p=0.558) | 77.72 ± 1.04 (p=0.149) [1020 cl.] |
| 2000 | 77.16 ± 0.21 | 77.67 ± 0.80 (p=0.384) | 80.96 ± 0.36 (p=0.000) | 79.21 ± 0.57 (p=0.015) | 80.65 ± 1.29 (p=0.039) [2040 cl.] |

## Accuracy, 20 Newsgroups, 20 classes

| Budget | GG-TH | GG, ternary linear readout | Ternary net, no clauses | CoTM, tuned | FPTM |
|---|---|---|---|---|---|
| 50 | 66.88 ± 1.27 | 68.98 ± 0.82 (p=0.084) | 75.76 ± 0.04 (p=0.007) | 61.90 ± 0.72 (p=0.008) | 70.51 ± 1.30 (p=0.026) [40 cl.] |
| 100 | 69.90 ± 1.33 | 72.36 ± 0.76 (p=0.065) | 76.39 ± 0.53 (p=0.007) | 66.96 ± 0.31 (p=0.056) | 74.16 ± 0.61 (p=0.018) [80 cl.] |
| 200 | 71.89 ± 0.43 | 74.82 ± 0.18 (p=0.003) | 76.72 ± 0.59 (p=0.001) | 70.60 ± 0.79 (p=0.086) | 77.67 ± 0.12 (p=0.001) [200 cl.] |
| 500 | 73.64 ± 0.50 | 75.16 ± 0.13 (p=0.027) | 77.39 ± 0.61 (p=0.001) | 71.82 ± 1.29 (p=0.121) | 78.70 ± 0.27 (p=0.000) [480 cl.] |
| 1000 | 74.62 ± 0.75 | 75.51 ± 0.55 (p=0.182) | 77.02 ± 0.50 (p=0.014) | 73.53 ± 0.89 (p=0.180) | 79.53 ± 0.43 (p=0.002) [1000 cl.] |
| 2000 | 74.83 ± 0.42 | 76.33 ± 0.29 (p=0.009) | 77.03 ± 0.23 (p=0.004) | 74.72 ± 0.33 (p=0.742) | 79.52 ± 0.23 (p=0.000) [2000 cl.] |

## GG-linear trained in floating point, evaluated with weights rounded to {-1, 0, +1} (n = 1000)

| Dataset | GG-linear float F1 | GG-linear rounded F1 | GG, ternary readout (STE) F1 | GG-TH F1 |
|---|---|---|---|---|
| wos | 54.90 ± 0.82 | 45.93 ± 1.66 | 56.19 ± 1.54 | 60.14 ± 0.68 |
| alerce | 61.73 ± 1.61 | 58.96 ± 1.04 | 59.66 ± 1.87 | 64.74 ± 0.70 |
| r52 | 63.31 ± 1.81 | 65.76 ± 2.64 | 64.62 ± 1.16 | 59.13 ± 1.10 |
| sector | 60.43 ± 0.23 | 61.18 ± 0.73 | 57.51 ± 0.66 | 52.24 ± 0.67 |
| rcv1 | 50.90 ± 0.40 | 50.79 ± 1.90 | 48.21 ± 0.53 | 44.55 ± 0.28 |
| news20 | 78.07 ± 0.13 | 78.22 ± 0.52 | 75.39 ± 0.61 | 74.68 ± 0.98 |

## Sparsity and memory

Literals: mean included literals per clause (clause models) or mean nonzero first-layer weights per unit (ternary net), at the best epoch. Bits: stored bits of the clause or first layer plus ternary readout layers.

| Dataset | Budget | Model | Literals per clause or unit | Bits | Best epoch (mean) |
|---|---|---|---|---|---|
| wos | 50 | GG-TH | 58.4 | 293,696 | 18.3 |
| wos | 50 | GG, ternary linear readout | 58.9 | 213,300 | 23.0 |
| wos | 50 | Ternary net, no clauses | 832.5 | 293,696 | 23.0 |
| wos | 50 | CoTM, tuned | 112.1 | 200,000 | 72.0 |
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
| wos | 500 | FPTM | 16.3 | 2,128,000 | 70.7 |
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
| alerce | 2000 | FPTM | 110.2 | 4,711,440 | 54.0 |
| r52 | 50 | GG-TH | 10.3 | 252,224 | 21.7 |
| r52 | 50 | GG, ternary linear readout | 72.4 | 205,200 | 20.7 |
| r52 | 50 | Ternary net, no clauses | 1144.7 | 252,224 | 13.7 |
| r52 | 50 | CoTM, tuned | 287.7 | 200,000 | 19.3 |
| r52 | 50 | FPTM | 113.2 | 416,000 | 18.3 |
| r52 | 100 | GG-TH | 31.5 | 477,824 | 32.7 |
| r52 | 100 | GG, ternary linear readout | 104.4 | 410,400 | 19.7 |
| r52 | 100 | Ternary net, no clauses | 1146.7 | 477,824 | 17.7 |
| r52 | 100 | CoTM, tuned | 313.5 | 400,000 | 20.7 |
| r52 | 200 | GG-TH | 72.4 | 929,024 | 20.3 |
| r52 | 200 | GG, ternary linear readout | 142.4 | 820,800 | 11.0 |
| r52 | 200 | Ternary net, no clauses | 1149.4 | 929,024 | 14.7 |
| r52 | 200 | CoTM, tuned | 309.4 | 800,000 | 25.0 |
| r52 | 200 | FPTM | 40.6 | 832,000 | 22.7 |
| r52 | 500 | GG-TH | 130.9 | 2,282,624 | 12.3 |
| r52 | 500 | GG, ternary linear readout | 142.4 | 2,052,000 | 11.0 |
| r52 | 500 | Ternary net, no clauses | 1151.4 | 2,282,624 | 10.0 |
| r52 | 500 | CoTM, tuned | 322.9 | 2,000,000 | 13.3 |
| r52 | 500 | FPTM | 21.2 | 2,080,000 | 18.0 |
| r52 | 1000 | GG-TH | 218.9 | 4,538,624 | 12.3 |
| r52 | 1000 | GG, ternary linear readout | 162.2 | 4,104,000 | 11.3 |
| r52 | 1000 | Ternary net, no clauses | 1143.5 | 4,269,312 | 8.7 |
| r52 | 1000 | CoTM, tuned | 357.2 | 4,000,000 | 13.0 |
| r52 | 1000 | FPTM | 167.2 | 4,160,000 | 43.0 |
| r52 | 2000 | GG-TH | 170.3 | 9,050,624 | 6.0 |
| r52 | 2000 | GG, ternary linear readout | 116.3 | 8,208,000 | 6.0 |
| r52 | 2000 | Ternary net, no clauses | 1149.0 | 8,525,312 | 5.0 |
| r52 | 2000 | CoTM, tuned | 327.6 | 8,000,000 | 21.3 |
| r52 | 2000 | FPTM | 403.5 | 7,904,000 | 18.7 |
| sector | 50 | GG-TH | 428.8 | 279,360 | 26.0 |
| sector | 50 | GG, ternary linear readout | 468.1 | 210,500 | 33.0 |
| sector | 50 | Ternary net, no clauses | 1088.6 | 279,360 | 8.0 |
| sector | 50 | CoTM, tuned | 1281.4 | 200,000 | 28.7 |
| sector | 50 | FPTM | 113.2 | 840,000 | 37.0 |
| sector | 100 | GG-TH | 773.7 | 504,960 | 24.0 |
| sector | 100 | GG, ternary linear readout | 651.7 | 421,000 | 19.3 |
| sector | 100 | Ternary net, no clauses | 1054.6 | 504,960 | 8.3 |
| sector | 100 | CoTM, tuned | 1348.3 | 400,000 | 40.0 |
| sector | 200 | GG-TH | 1002.4 | 956,160 | 15.7 |
| sector | 200 | GG, ternary linear readout | 858.2 | 842,000 | 14.3 |
| sector | 200 | Ternary net, no clauses | 1005.3 | 956,160 | 15.3 |
| sector | 200 | CoTM, tuned | 1258.3 | 800,000 | 20.7 |
| sector | 500 | GG-TH | 1002.1 | 2,309,760 | 5.3 |
| sector | 500 | GG, ternary linear readout | 943.4 | 2,105,000 | 8.0 |
| sector | 500 | Ternary net, no clauses | 977.3 | 2,309,760 | 25.7 |
| sector | 500 | CoTM, tuned | 1134.9 | 2,000,000 | 13.3 |
| sector | 500 | FPTM | 106.9 | 1,680,000 | 37.3 |
| sector | 1000 | GG-TH | 979.1 | 4,565,760 | 5.3 |
| sector | 1000 | GG, ternary linear readout | 1081.7 | 4,210,000 | 13.3 |
| sector | 1000 | Ternary net, no clauses | 1130.9 | 4,565,760 | 12.0 |
| sector | 1000 | CoTM, tuned | 1151.7 | 4,000,000 | 25.7 |
| sector | 1000 | FPTM | 40.1 | 4,200,000 | 29.7 |
| sector | 2000 | GG-TH | 1061.8 | 9,077,760 | 4.7 |
| sector | 2000 | GG, ternary linear readout | 1084.9 | 8,420,000 | 11.0 |
| sector | 2000 | Ternary net, no clauses | 1122.0 | 9,077,760 | 16.7 |
| sector | 2000 | CoTM, tuned | 1217.9 | 8,000,000 | 24.7 |
| sector | 2000 | FPTM | 18.0 | 8,400,000 | 42.0 |
| rcv1 | 50 | GG-TH | 35.0 | 251,712 | 19.7 |
| rcv1 | 50 | GG, ternary linear readout | 110.4 | 205,100 | 32.0 |
| rcv1 | 50 | Ternary net, no clauses | 1061.2 | 225,856 | 18.7 |
| rcv1 | 50 | CoTM, tuned | 380.2 | 200,000 | 18.3 |
| rcv1 | 50 | FPTM | 100.3 | 408,000 | 27.7 |
| rcv1 | 100 | GG-TH | 154.7 | 477,312 | 13.7 |
| rcv1 | 100 | GG, ternary linear readout | 242.8 | 410,200 | 25.0 |
| rcv1 | 100 | Ternary net, no clauses | 1061.9 | 438,656 | 11.7 |
| rcv1 | 100 | CoTM, tuned | 455.3 | 400,000 | 22.3 |
| rcv1 | 200 | GG-TH | 262.8 | 928,512 | 14.3 |
| rcv1 | 200 | GG, ternary linear readout | 292.3 | 820,400 | 9.7 |
| rcv1 | 200 | Ternary net, no clauses | 1044.1 | 864,256 | 9.7 |
| rcv1 | 200 | CoTM, tuned | 452.5 | 800,000 | 9.0 |
| rcv1 | 200 | FPTM | 134.5 | 816,000 | 25.7 |
| rcv1 | 500 | GG-TH | 319.9 | 2,282,112 | 10.7 |
| rcv1 | 500 | GG, ternary linear readout | 255.3 | 2,051,000 | 4.0 |
| rcv1 | 500 | Ternary net, no clauses | 1035.6 | 2,141,056 | 9.3 |
| rcv1 | 500 | CoTM, tuned | 455.0 | 2,000,000 | 6.7 |
| rcv1 | 500 | FPTM | 362.8 | 2,040,000 | 14.7 |
| rcv1 | 1000 | GG-TH | 305.2 | 4,269,056 | 9.3 |
| rcv1 | 1000 | GG, ternary linear readout | 254.5 | 4,102,000 | 9.7 |
| rcv1 | 1000 | Ternary net, no clauses | 1143.9 | 4,538,112 | 13.0 |
| rcv1 | 1000 | CoTM, tuned | 376.9 | 4,000,000 | 11.0 |
| rcv1 | 1000 | FPTM | 391.6 | 4,080,000 | 13.7 |
| rcv1 | 2000 | GG-TH | 351.3 | 8,525,056 | 15.0 |
| rcv1 | 2000 | GG, ternary linear readout | 289.8 | 8,204,000 | 7.3 |
| rcv1 | 2000 | Ternary net, no clauses | 1148.1 | 9,050,112 | 9.3 |
| rcv1 | 2000 | CoTM, tuned | 369.8 | 8,000,000 | 10.0 |
| rcv1 | 2000 | FPTM | 132.6 | 8,160,000 | 10.3 |
| news20 | 50 | GG-TH | 321.3 | 217,920 | 12.7 |
| news20 | 50 | GG, ternary linear readout | 393.9 | 202,000 | 36.0 |
| news20 | 50 | Ternary net, no clauses | 1132.0 | 217,920 | 7.3 |
| news20 | 50 | CoTM, tuned | 605.0 | 200,000 | 23.7 |
| news20 | 50 | FPTM | 45.7 | 160,000 | 28.0 |
| news20 | 100 | GG-TH | 410.5 | 430,720 | 26.7 |
| news20 | 100 | GG, ternary linear readout | 445.3 | 404,000 | 27.0 |
| news20 | 100 | Ternary net, no clauses | 1136.9 | 430,720 | 7.7 |
| news20 | 100 | CoTM, tuned | 607.3 | 400,000 | 17.7 |
| news20 | 100 | FPTM | 52.4 | 320,000 | 36.3 |
| news20 | 200 | GG-TH | 363.8 | 856,320 | 5.7 |
| news20 | 200 | GG, ternary linear readout | 368.6 | 808,000 | 8.0 |
| news20 | 200 | Ternary net, no clauses | 1141.1 | 856,320 | 5.0 |
| news20 | 200 | CoTM, tuned | 609.0 | 800,000 | 23.3 |
| news20 | 200 | FPTM | 290.6 | 800,000 | 34.3 |
| news20 | 500 | GG-TH | 395.8 | 2,133,120 | 6.3 |
| news20 | 500 | GG, ternary linear readout | 404.4 | 2,020,000 | 12.0 |
| news20 | 500 | Ternary net, no clauses | 1136.6 | 2,133,120 | 5.7 |
| news20 | 500 | CoTM, tuned | 531.2 | 2,000,000 | 9.0 |
| news20 | 500 | FPTM | 154.9 | 1,920,000 | 24.3 |
| news20 | 1000 | GG-TH | 438.1 | 4,522,240 | 5.7 |
| news20 | 1000 | GG, ternary linear readout | 374.4 | 4,040,000 | 7.7 |
| news20 | 1000 | Ternary net, no clauses | 1137.3 | 4,261,120 | 6.0 |
| news20 | 1000 | CoTM, tuned | 554.6 | 4,000,000 | 6.7 |
| news20 | 1000 | FPTM | 96.6 | 4,000,000 | 31.7 |
| news20 | 2000 | GG-TH | 477.0 | 9,034,240 | 7.0 |
| news20 | 2000 | GG, ternary linear readout | 385.2 | 8,080,000 | 11.3 |
| news20 | 2000 | Ternary net, no clauses | 1120.6 | 8,517,120 | 11.0 |
| news20 | 2000 | CoTM, tuned | 591.4 | 8,000,000 | 9.0 |
| news20 | 2000 | FPTM | 217.6 | 8,000,000 | 32.3 |

Runs whose best epoch lies within the last 5 epochs of the cap (60 or 120): 0.

## Settings rerun with up to 120 epochs

All seeds of every final setting with at least one seed whose best epoch fell within the last 5 epochs of the 60-epoch cap were rerun from scratch with up to 120 epochs and the same patience. The tables above use the rerun.

| Dataset | Budget | Model | Macro F1, 60-epoch cap | Macro F1, 120-epoch cap | Best epochs (120-epoch cap) |
|---|---|---|---|---|---|
| wos | 50 | CoTM, tuned | 30.77 ± 0.68 | 31.66 ± 2.02 | 105, 50, 61 |
| wos | 500 | FPTM | 54.58 ± 0.39 | 55.19 ± 0.85 | 82, 48, 82 |
| wos | 1000 | FPTM | 60.68 ± 0.75 | 60.68 ± 0.75 | 58, 18, 34 |
| alerce | 2000 | FPTM | 66.85 ± 1.16 | 66.77 ± 1.11 | 23, 69, 70 |
| r52 | 100 | GG-TH | 42.69 ± 2.70 | 41.67 ± 0.95 | 64, 18, 16 |
| r52 | 1000 | FPTM | 58.51 ± 1.82 | 57.87 ± 0.73 | 15, 67, 47 |
| sector | 2000 | FPTM | 64.29 ± 0.74 | 64.29 ± 0.74 | 60, 32, 34 |

## Pairwise comparisons per dataset (test macro F1)

Mean difference A − B over the budgets where both have results (FPTM: budgets with an FPTM size). Wins and losses: budgets with p < 0.05 (Welch) in favour of A or of B.

| A | B | Dataset | Budgets | Mean A − B | Range of A − B | A better (p<0.05) | B better (p<0.05) |
|---|---|---|---|---|---|---|---|
| GG-TH | GG, ternary linear readout | wos | 6 | +2.37 | -0.38 to +6.10 | 4 | 0 |
| GG-TH | GG, ternary linear readout | alerce | 6 | +3.83 | +2.24 to +5.27 | 4 | 0 |
| GG-TH | GG, ternary linear readout | r52 | 6 | -6.84 | -10.19 to -2.92 | 0 | 5 |
| GG-TH | GG, ternary linear readout | sector | 6 | -3.86 | -6.76 to +2.30 | 0 | 5 |
| GG-TH | GG, ternary linear readout | rcv1 | 6 | -0.32 | -3.66 to +4.38 | 0 | 2 |
| GG-TH | GG, ternary linear readout | news20 | 6 | -1.73 | -2.81 to -0.71 | 0 | 4 |
| GG-TH | Ternary net, no clauses | wos | 6 | +3.48 | +1.93 to +4.61 | 6 | 0 |
| GG-TH | Ternary net, no clauses | alerce | 6 | -2.06 | -5.03 to +2.46 | 0 | 2 |
| GG-TH | Ternary net, no clauses | r52 | 6 | -5.57 | -20.18 to +2.70 | 0 | 3 |
| GG-TH | Ternary net, no clauses | sector | 6 | -15.10 | -20.90 to -11.50 | 0 | 6 |
| GG-TH | Ternary net, no clauses | rcv1 | 6 | -4.68 | -10.85 to -1.68 | 0 | 5 |
| GG-TH | Ternary net, no clauses | news20 | 6 | -4.71 | -8.96 to -2.16 | 0 | 6 |
| GG-TH | CoTM, tuned | wos | 6 | +10.99 | +5.19 to +18.35 | 6 | 0 |
| GG-TH | CoTM, tuned | alerce | 6 | +6.91 | +2.79 to +9.76 | 4 | 0 |
| GG-TH | CoTM, tuned | r52 | 6 | -3.32 | -8.06 to -0.66 | 0 | 2 |
| GG-TH | CoTM, tuned | sector | 6 | +2.52 | -3.14 to +10.03 | 3 | 1 |
| GG-TH | CoTM, tuned | rcv1 | 6 | +4.71 | -0.46 to +8.19 | 4 | 0 |
| GG-TH | CoTM, tuned | news20 | 6 | +2.08 | +0.41 to +4.77 | 3 | 0 |
| GG-TH | FPTM | wos | 4 | +1.66 | -2.47 to +9.52 | 1 | 1 |
| GG-TH | FPTM | alerce | 6 | -0.70 | -1.87 to +1.51 | 0 | 1 |
| GG-TH | FPTM | r52 | 5 | -5.17 | -15.95 to +1.78 | 0 | 2 |
| GG-TH | FPTM | sector | 4 | -6.83 | -11.09 to -4.23 | 0 | 3 |
| GG-TH | FPTM | rcv1 | 5 | -4.88 | -8.52 to -3.73 | 0 | 4 |
| GG-TH | FPTM | news20 | 6 | -4.33 | -5.28 to -3.34 | 0 | 6 |
| GG, ternary linear readout | CoTM, tuned | wos | 6 | +8.62 | +3.16 to +15.82 | 6 | 0 |
| GG, ternary linear readout | CoTM, tuned | alerce | 6 | +3.08 | +0.55 to +5.16 | 3 | 0 |
| GG, ternary linear readout | CoTM, tuned | r52 | 6 | +3.52 | +0.28 to +8.78 | 1 | 0 |
| GG, ternary linear readout | CoTM, tuned | sector | 6 | +6.37 | +3.60 to +8.41 | 6 | 0 |
| GG, ternary linear readout | CoTM, tuned | rcv1 | 6 | +5.03 | +3.16 to +7.17 | 6 | 0 |
| GG, ternary linear readout | CoTM, tuned | news20 | 6 | +3.81 | +1.60 to +6.96 | 6 | 0 |
| GG, ternary linear readout | FPTM | wos | 4 | -1.50 | -5.43 to +3.42 | 1 | 2 |
| GG, ternary linear readout | FPTM | alerce | 6 | -4.53 | -6.41 to -1.68 | 0 | 4 |
| GG, ternary linear readout | FPTM | r52 | 5 | +1.00 | -7.61 to +6.76 | 2 | 1 |
| GG, ternary linear readout | FPTM | sector | 4 | -2.71 | -6.53 to +2.40 | 1 | 1 |
| GG, ternary linear readout | FPTM | rcv1 | 5 | -4.10 | -12.89 to -0.38 | 0 | 2 |
| GG, ternary linear readout | FPTM | news20 | 6 | -2.60 | -3.87 to -1.16 | 0 | 4 |
| GG, ternary linear readout | Ternary net, no clauses | wos | 6 | +1.11 | -2.09 to +2.63 | 3 | 1 |
| GG, ternary linear readout | Ternary net, no clauses | alerce | 6 | -5.90 | -9.56 to +0.22 | 0 | 5 |
| GG, ternary linear readout | Ternary net, no clauses | r52 | 6 | +1.27 | -11.84 to +7.42 | 1 | 1 |
| GG, ternary linear readout | Ternary net, no clauses | sector | 6 | -11.25 | -23.20 to -4.76 | 0 | 6 |
| GG, ternary linear readout | Ternary net, no clauses | rcv1 | 6 | -4.36 | -15.22 to +1.28 | 0 | 3 |
| GG, ternary linear readout | Ternary net, no clauses | news20 | 6 | -2.98 | -6.77 to -0.97 | 0 | 6 |
