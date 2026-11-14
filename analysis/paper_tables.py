"""LaTeX tables, number macros and the budget figure of the paper, generated from the run logs.

  python analysis/paper_tables.py <out_dir>

Reads the early-stopping study (results_es, through analysis/report_es.py) and the TMU check (results_tmu).
Writes <out_dir>/tab_data.tex, tab_f1.tex, tab_pairs.tex, tab_round.tex, tab_tmu.tex, tab_sparsity.tex,
results_macros.tex and fig_budget.{pdf,png}. Every number in these files is computed here; none is typed by hand.
"""
import io
import contextlib
import json
import os
import sys

import numpy as np

OUT = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]                                  # report_es parses its own arguments
with contextlib.redirect_stdout(io.StringIO()):
    sys.path.insert(0, HERE)
    import report_es as R                                 # noqa: E402  (also refreshes summary.md)
import config as C                                        # noqa: E402
import es                                                 # noqa: E402
sys.path.insert(0, os.path.join(HERE, '..', 'experiments'))
import tmu_check as TC                                    # noqa: E402

import matplotlib                                         # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                           # noqa: E402
import matplotlib.ticker as tk                            # noqa: E402
from matplotlib.gridspec import GridSpec                  # noqa: E402
from matplotlib.lines import Line2D                       # noqa: E402

DSN = {'wos': 'WOS', 'alerce': 'ALeRCE', 'r52': 'R52', 'sector': 'Sector', 'rcv1': 'RCV1', 'news20': '20NG'}
NAME = {'gg_lintern': 'GG-TL', 'gg_hidtern': 'GG-TH', 'cotm': 'CoTM', 'tmu_tm': 'TM', 'fptm': 'FPTM', 'tnn': 'TNN'}
COLS = ('gg_lintern', 'gg_hidtern', 'cotm', 'tmu_tm', 'fptm', 'tnn')
MAC = {}
ALIAS = {'wos': 'wos', 'alerce': 'alerce', 'r52': 'reuters', 'sector': 'sector', 'rcv1': 'rcv', 'news20': 'news'}


def mac(name, val):
    """LaTeX macro \\r<name>. Command names must consist of letters only."""
    for k, v in ALIAS.items():
        name = name.replace(k, v) if k != v else name
    assert name.isalpha(), name
    MAC[name] = val.replace('-', '$-$') if val.startswith('-') else val


# ------------------------------------------------------------------ TMU runs
def tmu_runs(phase, model, ds, n, cfg):
    r = []
    for s in C.SEEDS:
        x = R.at_best(TC.ofile(phase, model, ds, n, cfg, s))
        if x:
            r.append(x)
    return r if len(r) == len(C.SEEDS) else []


for ds in C.DS_ES:
    for n in C.BUDGETS:
        try:
            cfg = TC.best_cfg('tmu_tm', ds, n)
        except Exception:
            continue
        r = tmu_runs('final', 'tmu_tm', ds, n, cfg)
        if r:
            cl = C.cpc(ds, n) * C.M[ds]
            R.rows.setdefault((ds, 'tmu_tm'), []).append(dict(n=n, clauses=cl, cfg=cfg, r=r, bits=2 * C.O[ds] * cl,
                                                             lit=np.mean([q['mean_literals'] for q in r]),
                                                             ep=np.mean([q['best_epoch'] for q in r]),
                                                             f1=R.ms(r, 'test_f1')))


def f1(ds, m, n):
    x = R.get(ds, m, n)
    return None if x is None else R.ms(x['r'], 'test_f1')


def pval(ds, a, b, n, key='test_f1'):
    xa, xb = R.get(ds, a, n), R.get(ds, b, n)
    if xa is None or xb is None:
        return float('nan')
    return R.welch(R.ms(xa['r'], key)[2], R.ms(xb['r'], key)[2])


def w(name, text):
    open(os.path.join(OUT, name), 'w').write(text)


# ------------------------------------------------------------------ Table: datasets
L = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
     r'Dataset & Bits $o$ & Classes $m$ & Train & Val. & Test & Smallest / largest class & Active bits \\', r'\midrule']
for ds in C.DS_ES:
    from ggth.data import load_prepared
    D = load_prepared(ds)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = D['train'], D['val'], D['test']
    cnt = np.bincount(ytr, minlength=D['m'])
    L.append(f'{DSN[ds]} & {Xtr.shape[1]:,} & {D["m"]} & {len(ytr):,} & {len(yva):,} & {len(yte):,} & '
             f'{cnt.min():,} / {cnt.max():,} & {Xtr.sum(1).mean():.1f} \\\\'.replace(',', '{,}'))
L += [r'\bottomrule', r'\end{tabular}']
w('tab_data.tex', '\n'.join(L) + '\n')


# ------------------------------------------------------------------ Table: macro F1 for all models
def fmt_cell(ds, m, n, ref='gg_lintern'):
    v = f1(ds, m, n)
    if v is None:
        return '--'
    s = f'{v[0]:.1f}'
    if m != ref:
        p = pval(ds, m, ref, n)
        if p < 0.05:
            s += r'$^{+}$' if v[0] > f1(ds, ref, n)[0] else r'$^{-}$'
    return s


L = [r'\begin{tabular}{llrrrrrr}', r'\toprule',
     r'Dataset & $n$ & ' + ' & '.join(NAME[m] for m in COLS) + r' \\', r'\midrule']
for i, ds in enumerate(C.DS_ES):
    for j, n in enumerate(C.BUDGETS):
        vals = {m: f1(ds, m, n) for m in COLS}
        best = max(v[0] for v in vals.values() if v is not None)
        cells = []
        for m in COLS:
            c = fmt_cell(ds, m, n)
            if vals[m] is not None and abs(vals[m][0] - best) < 1e-9:
                c = r'\textbf{' + c.split('$')[0] + '}' + c[len(c.split('$')[0]):]
            cells.append(c)
        L.append((DSN[ds] if j == 0 else '') + f' & {n} & ' + ' & '.join(cells) + r' \\')
    if i < len(C.DS_ES) - 1:
        L.append(r'\midrule')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_f1.tex', '\n'.join(L) + '\n')


# ------------------------------------------------------------------ Table: pairwise summary per dataset
def pair(a, b, ds):
    d, win, loss = [], 0, 0
    for n in C.BUDGETS:
        va, vb = f1(ds, a, n), f1(ds, b, n)
        if va is None or vb is None:
            continue
        d.append(va[0] - vb[0])
        p = R.welch(va[2], vb[2])
        win += int(p < 0.05 and va[0] > vb[0])
        loss += int(p < 0.05 and va[0] < vb[0])
    return d, win, loss


PAIRS = [('gg_lintern', 'cotm'), ('gg_hidtern', 'cotm'), ('gg_lintern', 'tmu_tm'), ('gg_hidtern', 'gg_lintern'),
         ('gg_lintern', 'fptm'), ('gg_lintern', 'tnn')]
L = [r'\begin{tabular}{l' + 'r' * len(C.DS_ES) + 'r}', r'\toprule',
     r'Comparison & ' + ' & '.join(DSN[d] for d in C.DS_ES) + r' & Cells \\', r'\midrule']
for a, b in PAIRS:
    cells, W, Lo, N = [], 0, 0, 0
    for ds in C.DS_ES:
        d, win, loss = pair(a, b, ds)
        if not d:
            cells.append('--')
            continue
        cells.append(f'{np.mean(d):+.1f} ({win}/{loss})'.replace('-', '$-$'))
        W, Lo, N = W + win, Lo + loss, N + len(d)
        mac(f'pair{a}{b}{ds}'.replace('_', ''), f'{np.mean(d):+.1f}'.replace('-', '$-$'))
    L.append(f'{NAME[a]} $-$ {NAME[b]} & ' + ' & '.join(cells) + f' & {W}/{Lo} of {N}' + r' \\')
    mac(f'pair{a}{b}win'.replace('_', ''), str(W))
    mac(f'pair{a}{b}loss'.replace('_', ''), str(Lo))
    mac(f'pair{a}{b}cells'.replace('_', ''), str(N))
L += [r'\bottomrule', r'\end{tabular}']
w('tab_pairs.tex', '\n'.join(L) + '\n')

# Range of the GG-TL gain over CoTM, over all cells
g = [f1(ds, 'gg_lintern', n)[0] - f1(ds, 'cotm', n)[0] for ds in C.DS_ES for n in C.BUDGETS
     if f1(ds, 'gg_lintern', n) and f1(ds, 'cotm', n)]
mac('tlcotmmin', f'{min(g):.1f}')
mac('tlcotmmax', f'{max(g):.1f}')
mac('tlcotmmean', f'{np.mean(g):.1f}')
mac('tlcotmpos', str(sum(x > 0 for x in g)))

# ------------------------------------------------------------------ Table: post-hoc rounding at n = 1000
L = [r'\begin{tabular}{lrrrr}', r'\toprule',
     r'Dataset & GG-linear, real & GG-linear, rounded & GG-TL & GG-TH \\', r'\midrule']
for ds in C.DS_ES:
    x = R.get(ds, 'gg_lin', 1000)
    if x is None:
        continue
    rk = next(k for k in x['r'][0] if k.startswith('test_f1') and 'round' in k)
    a, b = R.ms(x['r'], 'test_f1'), R.ms(x['r'], rk)
    t, h = f1(ds, 'gg_lintern', 1000), f1(ds, 'gg_hidtern', 1000)
    pm = lambda v: f'{v[0]:.1f} $\\pm$ {v[1]:.1f}'
    L.append(f'{DSN[ds]} & {pm(a)} & {pm(b)} & {pm(t)} & {pm(h)} \\\\')
    mac(f'round{ds}', f'{a[0] - b[0]:.1f}')
    mac(f'roundtl{ds}', f'{t[0] - b[0]:+.1f}'.replace('-', '$-$'))
L += [r'\bottomrule', r'\end{tabular}']
w('tab_round.tex', '\n'.join(L) + '\n')

# ------------------------------------------------------------------ Table: CoTM implementation check
L = [r'\begin{tabular}{lrlrrr}', r'\toprule',
     r'Dataset & $n$ & Setting ($s$, $T/n$) & CoTM, ours & CoTM, TMU & $p$ \\', r'\midrule']
nchk, nsig = 0, 0
for ds in C.DS_ES:
    for n in TC.CHECK[ds]:
        cfg = TC.ours_cotm_cfg(ds, n)
        r = tmu_runs('check', 'tmu_cotm', ds, n, cfg)
        o = f1(ds, 'cotm', n)
        if not r or o is None:
            L.append(f'{DSN[ds]} & {n} & {cfg["s"]}, {cfg["Tm"]} & {o[0]:.1f} $\\pm$ {o[1]:.1f} & running & \\\\')
            continue
        t = R.ms(r, 'test_f1')
        p = R.welch(o[2], t[2])
        nchk += 1
        nsig += int(p < 0.05 and o[0] < t[0])
        L.append(f'{DSN[ds]} & {n} & {cfg["s"]}, {cfg["Tm"]} & {o[0]:.1f} $\\pm$ {o[1]:.1f} & '
                 f'{t[0]:.1f} $\\pm$ {t[1]:.1f} & {p:.3f} \\\\')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_tmu.tex', '\n'.join(L) + '\n')
mac('tmuchecked', str(nchk))
mac('tmuoursworse', str(nsig))

# ------------------------------------------------------------------ Table: clause length and memory
SP = ('gg_lintern', 'gg_hidtern', 'cotm', 'tmu_tm', 'fptm')
L = [r'\begin{tabular}{lr' + 'r' * len(SP) + r'r}', r'\toprule',
     r'Dataset & $n$ & ' + ' & '.join(NAME[m] for m in SP) + r' & Literals $2o$ \\', r'\midrule']
for ds in C.DS_ES:
    for n in (200, 1000):
        cells = []
        for m in SP:
            x = R.get(ds, m, n)
            cells.append('--' if x is None else f'{x["lit"]:.0f}')
        L.append(f'{DSN[ds]} & {n} & ' + ' & '.join(cells) + f' & {2 * C.O[ds]:,}'.replace(',', '{,}') + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_sparsity.tex', '\n'.join(L) + '\n')

# Clause length: GG-TL against CoTM at the same budget
fewer = [(R.get(ds, 'gg_lintern', n)['lit'], R.get(ds, 'cotm', n)['lit']) for ds in C.DS_ES for n in C.BUDGETS
         if R.get(ds, 'gg_lintern', n) and R.get(ds, 'cotm', n)]
mac('litfewer', str(sum(a < b for a, b in fewer)))
mac('litcells', str(len(fewer)))

# ------------------------------------------------------------------ macros
w('results_macros.tex', '%% Generated by gg-th/analysis/paper_tables.py. Do not edit by hand.\n' +
  '\n'.join(f'\\newcommand{{\\r{k}}}{{{v}}}' for k, v in sorted(MAC.items())) + '\n')

# ------------------------------------------------------------------ figure
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42})
STY = {'gg_lintern': ('#1b7837', 'o', '-'), 'gg_hidtern': ('#7fbf7b', 's', '--'), 'cotm': ('#b2182b', 'v', ':'),
       'tmu_tm': ('#e08214', 'P', ':'), 'fptm': ('#2166ac', 'D', '-.'), 'tnn': ('#762a83', '^', '-')}
fig = plt.figure(figsize=(7.2, 8.6))
gs = GridSpec(4, 2, height_ratios=[0.12, 1, 1, 1], hspace=0.55, wspace=0.28, figure=fig)
lax = fig.add_subplot(gs[0, :])
lax.axis('off')
present = [m for m in COLS if any(k[1] == m for k in R.rows)]
lax.legend([Line2D([], [], color=STY[m][0], marker=STY[m][1], ls=STY[m][2], lw=1.5, ms=5) for m in present],
           [NAME[m] for m in present], loc='center', ncol=len(present), frameon=False, fontsize=9.5)
for i, ds in enumerate(C.DS_ES):
    ax = fig.add_subplot(gs[1 + i // 2, i % 2])
    for m in present:
        v = R.rows.get((ds, m), [])
        if not v:
            continue
        col, mk, ls = STY[m]
        ax.errorbar([q['clauses'] for q in v], [q['f1'][0] for q in v],
                    yerr=[0 if np.isnan(q['f1'][1]) else q['f1'][1] for q in v],
                    color=col, marker=mk, ls=ls, ms=4.5, lw=1.4, capsize=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(tk.ScalarFormatter())
    ax.set_xticks([50, 100, 200, 500, 1000, 2000])
    ax.xaxis.set_minor_formatter(tk.NullFormatter())
    ax.set_title(f'({chr(97 + i)}) {R.NAME[ds]}')
    if i >= 4:
        ax.set_xlabel('Total clauses (first-layer units for TNN)')
    if i % 2 == 0:
        ax.set_ylabel('Test macro F1 (%)')
    ax.grid(alpha=0.3)
fig.savefig(os.path.join(OUT, 'fig_budget.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_budget.png'), dpi=200, bbox_inches='tight')
print('wrote tables, macros and figure to', OUT)
