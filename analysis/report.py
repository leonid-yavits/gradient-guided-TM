"""Tables and figure of the paper from the stored run logs.

  python analysis/report.py      # writes results/summary.md and figures/fig_budget.{pdf,png}

All numbers are test metrics after epoch 20, mean and sample standard deviation over seeds 1 to 3.
Differences are tested with Welch's t-test. Hyperparameters are re-derived from the tune logs
(validation macro F1), exactly as experiments/run.py selects them.
"""
import json
import os
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments'))
import config as C

NAME = {'wos': 'WOS, 133 classes', 'alerce': 'ALeRCE, 15 classes'}
FIG = os.path.join(C.ROOT, 'figures')


def last(path, ep=C.FINAL_EPOCHS):
    e = json.load(open(path))['epochs']
    return e[ep - 1] if len(e) >= ep else None


def ms(runs, key):
    v = np.array([r[key] for r in runs]) * 100
    return v.mean(), v.std(ddof=1), v


def welch(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


# ------------------------------------------------------------------ collect
rows = {}
for ds in C.DS:
    m, sizes = C.M[ds], C.fptm_sizes(ds)
    for n in C.BUDGETS:
        c = C.ours_best(ds, n)
        r = [last(C.ours_file('final', ds, n, c, s)) for s in C.SEEDS]
        mem = 2 * C.O[ds] * n + 2 * c['H'] * n + 2 * m * c['H']          # bits, Eq. (8) of the paper
        rows.setdefault((ds, 'ours'), []).append(dict(n=n, clauses=n, mem=mem, cfg=c, r=r))
        if n in sizes:
            cf, cp = C.fptm_best(ds, n), C.cpc(ds, n)
            r = [last(C.fptm_file('final', ds, n, cf, s)) for s in C.SEEDS]
            rows.setdefault((ds, 'fptm'), []).append(dict(n=n, clauses=cp * m, mem=cp * m * 2 * C.O[ds], cfg=cf, r=r))
for v in rows.values():
    for x in v:
        x['lit'] = np.mean([q['mean_literals'] for q in x['r']])

# ------------------------------------------------------------------ tables
L = ['# Results summary', '', 'Test metrics in %, epoch 20, mean ± std over 3 seeds. Welch t-test p values.', '']
L += ['## Ablation, n = 1000 clauses, fixed settings (s = 10, lr = 2e-3, H = 256)', '',
      '| Dataset | Model | Macro F1 | Bal. acc. | Acc. | ΔF1 vs previous (p) | Literals/clause |', '|---|---|---|---|---|---|---|']
for ds in C.DS:
    prev = None
    for r, lab in (('cotm', 'CoTM'), ('gg_lin', 'GG-linear'), ('gg_hidtern', 'GG-TH')):
        runs = [last(f'{C.R_ABL}/{ds}_{r}_1000_s{s}.json') for s in C.SEEDS]
        f, b, a = ms(runs, 'test_f1'), ms(runs, 'test_bacc'), ms(runs, 'test_acc')
        d = f'{f[0] - prev[0]:+.2f} ({welch(f[2], prev[2]):.3f})' if prev else ''
        L.append(f'| {ds} | {lab} | {f[0]:.2f} ± {f[1]:.2f} | {b[0]:.2f} | {a[0]:.2f} | {d} | '
                 f'{np.mean([q["mean_literals"] for q in runs]):.1f} |')
        prev = f
for ds in C.DS:
    L += ['', f'## Clause-budget sweep, {NAME[ds]}', '',
          '| GG-TH clauses | Macro F1 | Bal. acc. | Mem. (kbit) | Total literals | FPTM clauses | Macro F1 | Bal. acc. | Mem. (kbit) | Total literals | ΔF1 (p) |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    F = rows[(ds, 'fptm')]
    for xo in rows[(ds, 'ours')]:
        xf = min(F, key=lambda v: abs(np.log(v['clauses'] / xo['clauses'])))
        fo, ff = ms(xo['r'], 'test_f1'), ms(xf['r'], 'test_f1')
        bo, bf = ms(xo['r'], 'test_bacc'), ms(xf['r'], 'test_bacc')
        L.append(f'| {xo["clauses"]} | {fo[0]:.2f} ± {fo[1]:.2f} | {bo[0]:.2f} | {xo["mem"] / 1e3:.0f} | {xo["clauses"] * xo["lit"]:.0f} | '
                 f'{xf["clauses"]} | {ff[0]:.2f} ± {ff[1]:.2f} | {bf[0]:.2f} | {xf["mem"] / 1e3:.0f} | {xf["clauses"] * xf["lit"]:.0f} | '
                 f'{fo[0] - ff[0]:+.2f} ({welch(fo[2], ff[2]):.3f}) |')
    L += ['', 'Selected settings: ' + '; '.join(f'GG-TH n={x["n"]}: {x["cfg"]}' for x in rows[(ds, 'ours')]) + '.',
          'FPTM: ' + '; '.join(f'{x["clauses"]} clauses: {x["cfg"]}' for x in F) + '.']
os.makedirs(C.RES, exist_ok=True)
open(os.path.join(C.RES, 'summary.md'), 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))

# ------------------------------------------------------------------ figure
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42})
STY = {'fptm': dict(color='#2e7d32', marker='s', label='FPTM, tuned per budget'),
       'ours': dict(color='#1f4e9c', marker='o', label='GG-TH (this work)')}
fig = plt.figure(figsize=(6.6, 6.3))
gs = GridSpec(3, 2, height_ratios=[0.12, 1, 1], hspace=0.8, wspace=0.28, figure=fig)
lax = fig.add_subplot(gs[0, :])
lax.axis('off')
lax.legend(handles=[Line2D([], [], lw=1.8, ms=6, **STY[k]) for k in ('ours', 'fptm')], loc='center', ncol=2, frameon=False)
tags = iter('abcd')
for i, ds in enumerate(C.DS):
    for j, key in enumerate(('clauses', 'lits')):
        ax = fig.add_subplot(gs[i + 1, j])
        for meth in ('fptm', 'ours'):
            xs = rows[(ds, meth)]
            x = np.array([v['clauses'] * (v['lit'] if key == 'lits' else 1) for v in xs], float)
            y = np.array([ms(v['r'], 'test_f1')[0] for v in xs])
            e = np.array([ms(v['r'], 'test_f1')[1] for v in xs])
            o = np.argsort(x)
            ax.errorbar(x[o], y[o], yerr=e[o], color=STY[meth]['color'], marker=STY[meth]['marker'], ms=5,
                        lw=1.6 if key == 'clauses' else 0, elinewidth=1.2, capsize=2.5,
                        ls='-' if key == 'clauses' else 'none')
        ax.set_xscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.xaxis.set_major_locator(tk.LogLocator(base=10, subs=(1, 2, 5) if key == 'clauses' else (1, 3)))
        ax.xaxis.set_major_formatter(tk.FuncFormatter(lambda v, _: f'{v / 1e3:g}k' if v >= 1e3 else f'{v:g}'))
        ax.xaxis.set_minor_formatter(tk.NullFormatter())
        ax.set_xlabel('Total clauses' if key == 'clauses' else 'Total included literals')
        ax.set_ylabel('Test macro F1 (%)')
        ax.set_title(f'({next(tags)}) {NAME[ds]}')
        if ds == 'wos' and key == 'clauses':
            minc = 2 * C.M['wos']
            ax.axvline(minc, color='#2e7d32', ls=':', lw=1)
            ax.text(minc * 1.05, 0.05, 'FPTM minimum,\n2 clauses/class', transform=ax.get_xaxis_transform(),
                    fontsize=8, color='#2e7d32', va='bottom')
os.makedirs(FIG, exist_ok=True)
fig.savefig(os.path.join(FIG, 'fig_budget.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(FIG, 'fig_budget.png'), dpi=150, bbox_inches='tight')
print('saved figures/fig_budget.pdf')
