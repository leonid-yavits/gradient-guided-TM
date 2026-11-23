"""LaTeX tables, number macros and the budget figure of the paper, generated from the run logs.

  python analysis/paper_tables.py <out_dir>

Reads the early-stopping study (results_es, through analysis/report_es.py) and the TMU check (results_tmu).
Writes <out_dir>/tab_data.tex, tab_f1.tex, tab_pairs.tex, tab_round.tex, tab_tmu.tex, tab_sparsity.tex,
results_macros.tex and fig_budget.{pdf,png}. Every number in these files is computed here; none is typed by hand.
"""
import io
import contextlib
import glob
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
NAME = {'gg_lintern': 'GG-TL', 'gg_linr': 'GG-TL$_\\mathrm{R}$', 'gg_hidtern': 'GG-TH', 'cotm': 'CoTM', 'tmu_tm': 'TM', 'fptm': 'FPTM', 'tnn': 'TNN'}
COLS = ('gg_lintern', 'gg_linr', 'gg_hidtern', 'cotm', 'tmu_tm', 'fptm', 'tnn')
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


TM_DUP = []                                               # (ds, n) cells that repeat an earlier TM experiment
for ds in C.DS_ES:
    seen = set()
    for n in C.BUDGETS:
        try:
            cfg = TC.best_cfg('tmu_tm', ds, n)
        except Exception:
            continue
        key = (C.cpc(ds, n), tuple(sorted(cfg.items())))
        if key in seen:                                   # same clauses per class and setting as a smaller budget:
            TM_DUP.append((ds, n))                        # the same experiment, listed once (as for FPTM)
            continue
        seen.add(key)
        r = tmu_runs('final', 'tmu_tm', ds, n, cfg)
        if r:
            cl = C.cpc(ds, n) * C.M[ds]
            R.rows.setdefault((ds, 'tmu_tm'), []).append(dict(n=n, clauses=cl, cfg=cfg, r=r, bits=2 * C.O[ds] * cl,
                                                             lit=np.mean([q['mean_literals'] for q in r]),
                                                             ep=np.mean([q['best_epoch'] for q in r]),
                                                             f1=R.ms(r, 'test_f1')))


def f1(ds, m, n, key='test_f1'):
    x = R.get(ds, m, n)
    return None if x is None else R.ms(x['r'], key)


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


L = [r'\begin{tabular}{ll' + 'r' * len(COLS) + '}', r'\toprule',
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


PAIRS = [('gg_lintern', 'cotm'), ('gg_hidtern', 'cotm'), ('gg_lintern', 'tmu_tm'), ('cotm', 'tmu_tm'), ('gg_hidtern', 'gg_lintern'),
         ('gg_lintern', 'fptm'), ('gg_lintern', 'tnn'), ('gg_hidtern', 'tnn'),
         ('gg_linr', 'gg_lintern'), ('gg_linr', 'cotm'), ('gg_linr', 'tmu_tm'), ('gg_linr', 'fptm'), ('gg_linr', 'tnn')]
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


# GG-TL_R against GG-TL, split by budget
for lo, hi, tag in ((0, 200, 'small'), (500, 10 ** 9, 'large')):
    W = Lo = N = 0
    for ds in C.DS_ES:
        for n in C.BUDGETS:
            if not lo <= n <= hi:
                continue
            va, vb = f1(ds, 'gg_linr', n), f1(ds, 'gg_lintern', n)
            if va is None or vb is None:
                continue
            p = R.welch(va[2], vb[2])
            W, Lo, N = W + int(p < 0.05 and va[0] > vb[0]), Lo + int(p < 0.05 and va[0] < vb[0]), N + 1
    mac(f'rnd{tag}win', str(W))
    mac(f'rnd{tag}loss', str(Lo))
    mac(f'rnd{tag}cells', str(N))

# Multi-class TM: settings selected on the lower or upper edge of the grid
sa = sorted({c['s'] for c in TC.GRID})
ta = sorted({c['Tm'] for c in TC.GRID})
bs = bt = na = 0
for ds in C.DS_ES:
    for a in (200, 1000):
        c = TC.best_cfg('tmu_tm', ds, a)
        na += 1
        bs += int(c['s'] in (sa[0], sa[-1]))
        bt += int(c['Tm'] in (ta[0], ta[-1]))
mac('tmedges', str(bs))
mac('tmedget', str(bt))
mac('tmanchors', str(na))

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
nchk, nsig, nbet = 0, 0, 0
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
        nbet += int(p < 0.05 and o[0] > t[0])
        L.append(f'{DSN[ds]} & {n} & {cfg["s"]}, {cfg["Tm"]} & {o[0]:.1f} $\\pm$ {o[1]:.1f} & '
                 f'{t[0]:.1f} $\\pm$ {t[1]:.1f} & ' + ('$<$0.001' if p < 0.001 else f'{p:.3f}') + ' \\\\')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_tmu.tex', '\n'.join(L) + '\n')
mac('tmuchecked', str(nchk))
mac('tmuoursworse', str(nsig))
mac('tmuoursbetter', str(nbet))
# TMU runs resumed from a checkpoint after an interruption
_tj = [os.path.join(dp, f) for dp, _, fs in os.walk(TC.RES) for f in fs if f.endswith('.json')]
_tj += [f for f in glob.glob(os.path.join(C.ROOT, 'results_rev1', '**', '*.json'), recursive=True)
        if str(json.load(open(f)).get('model', '')).startswith('tmu_') and json.load(open(f)).get('done')]
mac('tmuruns', str(len(_tj)))
mac('tmuresumed', str(sum(bool(json.load(open(f)).get('resumed_after')) for f in _tj)))


# ------------------------------------------------------------------ Table: clause length and memory
SP = ('gg_lintern', 'gg_linr', 'gg_hidtern', 'cotm', 'tmu_tm', 'fptm')
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


# Cells where GG-TL clauses are not shorter than CoTM clauses
exc = [(ds, n) for ds in C.DS_ES for n in C.BUDGETS if R.get(ds, 'gg_lintern', n) and R.get(ds, 'cotm', n)
       and R.get(ds, 'gg_lintern', n)['lit'] >= R.get(ds, 'cotm', n)['lit']]
mac('litexc', '; '.join(f'{DSN[d]}, $n={n}$' for d, n in exc) if exc else 'none')
mac('tmdup', str(len(TM_DUP)))


# ------------------------------------------------------------------ Tables: other metrics (appendix)
def metric_table(key, fname):
    L = [r'\begin{tabular}{ll' + 'r' * len(COLS) + '}', r'\toprule',
         r'Dataset & $n$ & ' + ' & '.join(NAME[m] for m in COLS) + r' \\', r'\midrule']
    for i, ds in enumerate(C.DS_ES):
        for j, n in enumerate(C.BUDGETS):
            vals = {m: f1(ds, m, n, key) for m in COLS}
            best = max(v[0] for v in vals.values() if v is not None)
            cells = []
            for m in COLS:
                v = vals[m]
                if v is None:
                    cells.append('--')
                    continue
                c = f'{v[0]:.1f}'
                if m != 'gg_lintern':
                    ref = vals['gg_lintern']
                    if R.welch(v[2], ref[2]) < 0.05:
                        c += r'$^{+}$' if v[0] > ref[0] else r'$^{-}$'
                if abs(v[0] - best) < 1e-9:
                    c = r'\textbf{' + c.split('$')[0] + '}' + c[len(c.split('$')[0]):]
                cells.append(c)
            L.append((DSN[ds] if j == 0 else '') + f' & {n} & ' + ' & '.join(cells) + r' \\')
        if i < len(C.DS_ES) - 1:
            L.append(r'\midrule')
    L += [r'\bottomrule', r'\end{tabular}']
    w(fname, '\n'.join(L) + '\n')


metric_table('test_bacc', 'tab_bacc.tex')
metric_table('test_acc', 'tab_acc.tex')

# GG-TL against CoTM and TM under the three metrics
MET = (('test_f1', 'Macro F1'), ('test_bacc', 'Balanced accuracy'), ('test_acc', 'Accuracy'))
L = [r'\begin{tabular}{ll' + 'r' * len(C.DS_ES) + 'r}', r'\toprule',
     r'Comparison & Metric & ' + ' & '.join(DSN[d] for d in C.DS_ES) + r' & Cells \\', r'\midrule']
for a, b in (('gg_lintern', 'cotm'), ('gg_lintern', 'tmu_tm')):
    for k, (key, lab) in enumerate(MET):
        cells, W, Lo, N, pos = [], 0, 0, 0, 0
        for ds in C.DS_ES:
            d, win, loss = [], 0, 0
            for n in C.BUDGETS:
                va, vb = f1(ds, a, n, key), f1(ds, b, n, key)
                if va is None or vb is None:
                    continue
                d.append(va[0] - vb[0])
                p = R.welch(va[2], vb[2])
                win += int(p < 0.05 and va[0] > vb[0])
                loss += int(p < 0.05 and va[0] < vb[0])
            cells.append(f'{np.mean(d):+.1f} ({win}/{loss})'.replace('-', '$-$'))
            W, Lo, N, pos = W + win, Lo + loss, N + len(d), pos + sum(x > 0 for x in d)
        mk = {'test_f1': 'mf', 'test_bacc': 'bacc', 'test_acc': 'acc'}[key]
        mac(f'met{a}{b}{mk}'.replace('_', '') + 'pos', str(pos))
        mac(f'met{a}{b}{mk}'.replace('_', '') + 'win', str(W))
        mac(f'met{a}{b}{mk}'.replace('_', '') + 'loss', str(Lo))
        L.append((f'{NAME[a]} $-$ {NAME[b]}' if k == 0 else '') + f' & {lab} & ' + ' & '.join(cells) +
                 f' & {W}/{Lo} of {N}' + r' \\')
    L.append(r'\midrule')
L = L[:-1] + [r'\bottomrule', r'\end{tabular}']
w('tab_metrics.tex', '\n'.join(L) + '\n')


# ------------------------------------------------------------------ Table: selected settings on a grid edge
def edges(grid, cfg):
    out = []
    for ax in ('s', 'lr', 'Tm', 'H'):
        vals = sorted({g[ax] for g in grid if ax in g and g[ax]})
        if len(vals) < 3 or not cfg.get(ax):
            continue
        nm = {'s': '$s', 'lr': '$\\eta', 'Tm': '$T/n', 'H': '$H'}[ax]
        if cfg[ax] == vals[0]:
            out.append(nm + '{\\downarrow}$')
        elif cfg[ax] == vals[-1]:
            out.append(nm + '{\\uparrow}$')
    return out


AXN = {'s': '$s$', 'lr': 'lr', 'Tm': '$T/n$'}
L = [r'\begin{tabular}{l' + 'l' * len(C.DS_ES) + '}', r'\toprule',
     r'Model & ' + ' & '.join(DSN[d] for d in C.DS_ES) + r' \\', r'\midrule']
nedge = {}
for m in ('gg_lintern', 'gg_linr', 'gg_hidtern', 'cotm', 'tnn', 'tmu_tm'):
    cells = []
    for ds in C.DS_ES:
        parts = []
        for a in (200, 1000):
            if m == 'tmu_tm':
                e = edges(TC.GRID, TC.best_cfg('tmu_tm', ds, a))
            else:
                e = edges(es.GRIDS[m], es.best_cfg(m, ds, a))
            nedge[m] = nedge.get(m, 0) + int(bool(e))
            if e:
                parts.append(f'{a}: ' + ', '.join(e))
        cells.append('; '.join(parts) if parts else '--')
    L.append(NAME[m] + ' & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_edges.tex', '\n'.join(L).replace('lr$', 'lr$') + '\n')
for m, k in nedge.items():
    mac(f'edge{m}'.replace('_', ''), str(k))


# ------------------------------------------------------------------ Table: training time per epoch at the anchors
def sec_per_epoch(paths, key):
    v = []
    for p in paths:
        try:
            v += [e[key] for e in json.load(open(p))['epochs']]
        except Exception:
            return None
    return np.mean(v) if v else None


def time_paths(m, ds, n):
    if m in ('gg_lintern', 'cotm', 'gg_hidtern', 'tnn'):
        c = es.best_cfg(m, ds, n)
        return [es.ofile('final', m, ds, n, c, s) for s in C.SEEDS], 'sec'
    if m == 'fptm':
        nf = next(b for b in C.fptm_sizes(ds) if C.cpc(ds, b) == C.cpc(ds, n))   # budget at which this size was run
        c = es.fptm_best(ds, nf)
        return [es.ffile('final', ds, nf, c, s) for s in C.SEEDS], 'sec'
    c = TC.best_cfg('tmu_tm', ds, n)
    return [TC.ofile('final', 'tmu_tm', ds, n, c, s) for s in C.SEEDS], 'train_s'


TM_ = ('gg_lintern', 'cotm', 'gg_hidtern', 'tnn', 'tmu_tm', 'fptm')
L = [r'\begin{tabular}{lr' + 'r' * len(TM_) + '}', r'\toprule',
     r'Dataset & $n$ & ' + ' & '.join(NAME[m] for m in TM_) + r' \\', r'\midrule']
for ds in C.DS_ES:
    for n in (200, 1000):
        cells = []
        for m in TM_:
            ps, key = time_paths(m, ds, n)
            v = sec_per_epoch(ps, key) if ps else None
            cells.append('--' if v is None else (f'{v:.1f}' if v < 100 else f'{v:.0f}'))
        L.append(f'{DSN[ds]} & {n} & ' + ' & '.join(cells) + r' \\')
L += [r'\bottomrule', r'\end{tabular}']
w('tab_time.tex', '\n'.join(L) + '\n')

# ================================================================== first revision (experiments/PROTOCOL_rev1.md)
# Each analysis is written only when all its runs are finished; otherwise it is skipped with a note.
sys.path.insert(0, os.path.join(HERE, '..', 'experiments'))
import rev1 as RV                                         # noqa: E402
from scipy import stats as ST                             # noqa: E402

REV = os.path.join(C.ROOT, 'results_rev1')
ANCH = (200, 1000)


def runs(paths):
    r = [R.at_best(p) for p in paths]
    return r if r and all(r) else None


def mv(r, key='test_f1'):
    return np.array([q[key] for q in r], dtype=float) * 100


def sig(a, b):
    """'+' or '-' if a differs from b at p < 0.05 (Welch), else ''."""
    p = R.welch(a, b)
    return '' if not p < 0.05 else ('+' if a.mean() > b.mean() else '-')


def fmt(v, mark=''):
    s = f'{v.mean():.1f}'
    return s + (f'$^{{{mark}}}$' if mark else '')


def ours60(model, ds, n, seeds=C.SEEDS):
    c = es.best_cfg(model, ds, n)
    return runs([es.ofile('final', model, ds, n, c, s) for s in seeds])


def note(x):
    print('rev1:', x)


# ---- F. linear reference models
LIN = {}
for ds in C.DS_ES:
    for mdl in ('logreg', 'linsvm'):
        try:
            d = json.load(open(os.path.join(REV, 'linear', f'{mdl}_{ds}.json')))
            LIN[(ds, mdl)] = (np.array([r['test']['f1'] for r in d['final']]) * 100,
                              np.array([r['test']['bacc'] for r in d['final']]) * 100, d['best'])
        except Exception:
            pass
if len(LIN) == 2 * len(C.DS_ES):
    L = [r'\begin{tabular}{lrrrrl}', r'\toprule',
         r'Dataset & LogReg & Lin.\ SVM & GG-TL & CoTM & Best clause model \\', r'\midrule']
    above, gap = 0, []
    for ds in C.DS_ES:
        best = max(((m, x) for m in ('gg_lintern', 'gg_linr', 'gg_hidtern', 'cotm', 'tmu_tm', 'fptm')
                    for x in R.rows.get((ds, m), [])), key=lambda t: t[1]['f1'][0])
        lin = max(LIN[(ds, 'logreg')][0].mean(), LIN[(ds, 'linsvm')][0].mean())
        above += int(lin > best[1]['f1'][0])
        gap.append(lin - f1(ds, 'gg_lintern', 2000)[0])
        L.append(f'{DSN[ds]} & {LIN[(ds, "logreg")][0].mean():.1f} & {LIN[(ds, "linsvm")][0].mean():.1f} & '
                 f'{f1(ds, "gg_lintern", 2000)[0]:.1f} & {f1(ds, "cotm", 2000)[0]:.1f} & '
                 f'{best[1]["f1"][0]:.1f} ({NAME[best[0]]}, $n={best[1]["n"]}$)' + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_linear.tex', '\n'.join(L) + '\n')
    mac('linabove', str(above))
    mac('linggmin', f'{min(gap):.1f}')
    mac('linggmax', f'{max(gap):.1f}')
else:
    note(f'linear baselines incomplete ({len(LIN)} of {2 * len(C.DS_ES)})')

# ---- A. ablations of GG-TL at the anchors (60-epoch FINAL runs of GG-TL as reference)
ABLN = (('fb_cotm', 'CoTM clause feedback'), ('fb_cotm_s', r'CoTM clause feedback, $s\ge 1$'),
        ('fb_cotm_z', 'CoTM clause feedback, zero weights skipped'), ('no_pfac', r'No factor $1-p_y$'),
        ('no_mag', r'No magnitude $|g|$'), ('no_bias', 'No class offsets'))


def abl_runs(name, ds, a):
    if name in ('fb_cotm_s', 'fb_cotm_z'):   # addendum A3: s = 1 added (fb_cotm), zero weights skipped (fb_cotm0)
        try:
            c = RV.fbcotm3_best(ds, a) if name == 'fb_cotm_s' else RV.fbcotm0_best(ds, a)
        except Exception:
            return None
        return runs([RV.rfile('abl', 'fb_cotm' if name == 'fb_cotm_s' else 'fb_cotm0', 'final', ds, a, c, s)
                     for s in C.SEEDS])
    if name in ('fb_cotm', 'fb_cotm_a'):     # fb_cotm: selected over parts A and A2; fb_cotm_a: part A only
        try:
            c = RV.fbcotm2_best(ds, a) if name == 'fb_cotm' else RV.fbcotm_best(ds, a)
        except Exception:
            return None
        return runs([RV.rfile('abl', 'fb_cotm', 'final', ds, a, c, s) for s in C.SEEDS])
    else:
        c = es.best_cfg('gg_lintern', ds, a)
    return runs([RV.rfile('abl', name, 'final', ds, a, c, s) for s in C.SEEDS])


ABL_OK = all(abl_runs(nm, ds, a) for nm, _ in ABLN for ds in C.DS_ES for a in ANCH)
if ABL_OK:
    L = [r'\begin{tabular}{l' + 'r' * len(C.DS_ES) + 'r}', r'\toprule',
         r'Variant & ' + ' & '.join(DSN[d] for d in C.DS_ES) + r' & Cells \\', r'\midrule']
    rows_ = [(nm, lab, lambda ds, a, nm=nm: abl_runs(nm, ds, a)) for nm, lab in ABLN]
    rows_.append(('cotm', 'CoTM (reference)', lambda ds, a: ours60('cotm', ds, a)))
    for nm, lab, get in rows_:
        cells, W, Lo, N, D = [], 0, 0, 0, []
        for ds in C.DS_ES:
            d, wi, lo = [], 0, 0
            for a in ANCH:
                x, g = mv(get(ds, a)), mv(ours60('gg_lintern', ds, a))
                d.append(x.mean() - g.mean())
                s = sig(x, g)
                wi, lo = wi + (s == '+'), lo + (s == '-')
            cells.append(f'{np.mean(d):+.1f} ({wi}/{lo})'.replace('-', '$-$'))
            W, Lo, N, D = W + wi, Lo + lo, N + len(d), D + d
        mac(f'abl{nm}'.replace('_', '') + 'win', str(W))
        mac(f'abl{nm}'.replace('_', '') + 'loss', str(Lo))
        mac(f'abl{nm}'.replace('_', '') + 'mean', f'{np.mean(D):+.1f}'.replace('-', '$-$'))  # signed, variant - GG-TL
        mac(f'abl{nm}'.replace('_', '') + 'abs', f'{abs(np.mean(D)):.1f}')
        L.append(f'{lab} & ' + ' & '.join(cells) + f' & {W}/{Lo} of {N}' + r' \\')
        if nm == 'no_bias':
            L.append(r'\midrule')
    # part A only (Tm taken from the CoTM, before addendum A2)
    D = [mv(abl_runs('fb_cotm_a', ds, a)).mean() - mv(ours60('gg_lintern', ds, a)).mean() for ds in C.DS_ES
         for a in ANCH]
    mac('ablfbcotmAmean', f'{np.mean(D):+.1f}'.replace('-', '$-$'))
    # Tm of the selected fb_cotm setting relative to the Tm of the CoTM
    tms = [(RV.fbcotm2_best(ds, a)['Tm'], es.best_cfg('cotm', ds, a)['Tm']) for ds in C.DS_ES for a in ANCH]
    mac('ablfbcotmTmsmaller', str(sum(x < y for x, y in tms)))
    # CoTM clause feedback with the GG readout against our CoTM, and the share of the GG-TL - CoTM difference removed
    DG = np.mean([mv(ours60('gg_lintern', ds, a)).mean() - mv(ours60('cotm', ds, a)).mean()
                  for ds in C.DS_ES for a in ANCH])
    for nm in ('fb_cotm', 'fb_cotm_s', 'fb_cotm_z'):
        k = nm.replace('_', '')
        W = Lo = 0
        D, DV = [], []
        for ds in C.DS_ES:
            for a in ANCH:
                x, c = mv(abl_runs(nm, ds, a)), mv(ours60('cotm', ds, a))
                D.append(x.mean() - c.mean())
                DV.append(x.mean() - mv(ours60('gg_lintern', ds, a)).mean())
                s = sig(x, c)
                W, Lo = W + (s == '+'), Lo + (s == '-')
        mac(f'{k}vscotmwin', str(W))
        mac(f'{k}vscotmloss', str(Lo))
        mac(f'{k}vscotmmean', f'{np.mean(D):+.1f}'.replace('-', '$-$'))  # signed, variant - CoTM
        mac(f'{k}vscotmpos', str(sum(d > 0 for d in D)))
        mac(f'{k}vscotmabs', f'{abs(np.mean(D)):.1f}')
        mac(f'{k}share', f'{100 * -np.mean(DV) / DG:.0f}')                # % of the GG-TL - CoTM difference removed
    mac('ablfbcotmzsone', str(sum(RV.fbcotm0_best(ds, a)['s'] == 1 for ds in C.DS_ES for a in ANCH)))
    mac('ablfbcotmzsedge', str(sum(RV.fbcotm0_best(ds, a)['s'] in (1, 20) for ds in C.DS_ES for a in ANCH)))
    mac('ablfbcotmschg', str(sum(RV.fbcotm3_best(ds, a) != RV.fbcotm2_best(ds, a) for ds in C.DS_ES for a in ANCH)))
    mac('ablfbcotmsone', str(sum(RV.fbcotm2_best(ds, a)['s'] == 1.5 for ds in C.DS_ES for a in ANCH)))
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_abl.tex', '\n'.join(L) + '\n')
else:
    note('ablations incomplete')

# ---- C and G. CoTM checks at n = 200
B1 = {ds: runs([RV.rfile('batch1', 'cotm', 'final', ds, 200, es.best_cfg('cotm', ds, 200), s) for s in C.SEEDS])
      for ds in RV.BATCH_DS}
TCO = {}
for ds in C.DS_ES:
    try:
        TCO[ds] = runs([RV.tfile('tmucotm', 'tmu_cotm', 'final', ds, 200, RV.tmucotm_best(ds), s) for s in C.SEEDS])
    except Exception:
        TCO[ds] = None
if all(B1.values()):
    for ds in RV.BATCH_DS:
        x, c = mv(B1[ds]), mv(ours60('cotm', ds, 200))
        mac(f'bone{ds}', f'{x.mean():.1f}')
        mac(f'bsixtyfour{ds}', f'{c.mean():.1f}')
        pv = R.welch(x, c); mac(f'bonep{ds}', '$<$0.01' if pv < 0.01 else f'{pv:.2f}')
        mac(f'boneepoch{ds}', f'{np.mean([q["sec"] for q in B1[ds]]):.0f}' if 'sec' in B1[ds][0] else '--')
    mac('bonesig', str(sum(sig(mv(B1[ds]), mv(ours60('cotm', ds, 200))) != '' for ds in RV.BATCH_DS)))
else:
    note('batch-1 runs incomplete')
if all(TCO.values()):
    L = [r'\begin{tabular}{lrrrrl}', r'\toprule',
         r'Dataset & GG-TL & CoTM (ours) & TMU CoTM, tuned & TMU CoTM, our setting & Tuned setting \\',
         r'\midrule']
    W = Lo = 0
    for ds in C.DS_ES:
        g, c, t = mv(ours60('gg_lintern', ds, 200)), mv(ours60('cotm', ds, 200)), mv(TCO[ds])
        st = sig(t, c)
        W, Lo = W + (st == '+'), Lo + (st == '-')
        bc = RV.tmucotm_best(ds)
        _o = tmu_runs('check', 'tmu_cotm', ds, 200, TC.ours_cotm_cfg(ds, 200))
        old = R.ms(_o, 'test_f1')[0] if _o else float('nan')
        L.append(f'{DSN[ds]} & {g.mean():.1f} & {c.mean():.1f} & {fmt(t, st)} & {old:.1f} & '
                 f'$s={bc["s"]:g}$, $T/n={bc["Tm"]:g}$' + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_tmucotm.tex', '\n'.join(L) + '\n')
    mac('tcowin', str(W))
    mac('tcoloss', str(Lo))
    mac('tcoggwin', str(sum(sig(mv(ours60('gg_lintern', ds, 200)), mv(TCO[ds])) == '+' for ds in C.DS_ES)))
    mac('tcoggloss', str(sum(sig(mv(ours60('gg_lintern', ds, 200)), mv(TCO[ds])) == '-' for ds in C.DS_ES)))
else:
    note('tuned TMU CoTM incomplete')

# ---- D. one step beyond boundary optima
EXT_OK = True
chg, ext_rows = {}, []
for m in RV.EXT_MODELS + ('tmu_tm',):
    chg[m] = 0
    for ds in C.DS_ES:
        for a in ANCH:
            b = RV.tm_ext_best(ds, a) if m == 'tmu_tm' else RV.ext_best(m, ds, a)
            if b is None:
                EXT_OK = False
                continue
            base = TC.best_cfg('tmu_tm', ds, a) if m == 'tmu_tm' else es.best_cfg(m, ds, a)
            chg[m] += int(b != base)
if EXT_OK:
    for m, k in chg.items():
        mac(f'extchg{m}'.replace('_', ''), str(k))
    # final runs of changed settings: difference to the pre-registered setting, per budget
    for m in RV.EXT_MODELS + ('tmu_tm',):
        for ds in C.DS_ES:
            for n in C.BUDGETS:
                if m == 'tmu_tm':
                    b, base = RV.tm_ext_best(ds, C.anchor(n)), TC.best_cfg('tmu_tm', ds, n)
                    if b == base or (ds, n) in TM_DUP:
                        continue
                    x = runs([RV.tfile('ext', 'tmu_tm', 'final', ds, n, b, s) for s in C.SEEDS])
                    y = tmu_runs('final', 'tmu_tm', ds, n, base)
                else:
                    b, base = RV.ext_best(m, ds, C.anchor(n)), es.best_cfg(m, ds, n)
                    if b == base:
                        continue
                    x = runs([RV.rfile('ext', m, 'final', ds, n, b, s) for s in C.SEEDS])
                    y = ours60(m, ds, n)
                if x is None or not y:
                    EXT_OK = False
                    continue
                ext_rows.append((m, ds, n, mv(x), mv(y)))
if EXT_OK:
    for m in RV.EXT_MODELS + ('tmu_tm',):
        e = [r for r in ext_rows if r[0] == m]
        mac(f'extcells{m}'.replace('_', ''), str(len(e)))
        mac(f'extmean{m}'.replace('_', ''), f'{np.mean([r[3].mean() - r[4].mean() for r in e]):+.1f}' if e else '0')
        mac(f'extup{m}'.replace('_', ''), str(sum(sig(r[3], r[4]) == '+' for r in e)))
        mac(f'extdown{m}'.replace('_', ''), str(sum(sig(r[3], r[4]) == '-' for r in e)))
        mac(f'extmax{m}'.replace('_', ''), f'{max([r[3].mean() - r[4].mean() for r in e], default=0):.1f}')

    # GG-TL against CoTM and TM with the extended settings wherever they changed
    def ext_or_base(m, ds, n):
        r = next((q for q in ext_rows if q[:3] == (m, ds, n)), None)
        if r is not None:
            return r[3]
        if m == 'tmu_tm':
            x = R.get(ds, 'tmu_tm', n)
            return None if x is None else R.ms(x['r'], 'test_f1')[2]
        return mv(ours60(m, ds, n))
    for b in ('cotm', 'tmu_tm'):
        W = Lo = N = 0
        for ds in C.DS_ES:
            for n in C.BUDGETS:
                if b == 'tmu_tm' and R.get(ds, 'tmu_tm', n) is None:
                    continue
                x, y = ext_or_base('gg_lintern', ds, n), ext_or_base(b, ds, n)
                s = sig(x, y)
                W, Lo, N = W + (s == '+'), Lo + (s == '-'), N + 1
        mac(f'extgg{b}win'.replace('_', ''), str(W))
        mac(f'extgg{b}loss'.replace('_', ''), str(Lo))
        mac(f'extgg{b}n'.replace('_', ''), str(N))
else:
    note('grid extension incomplete')

# ---- E. seeds 2 to 8
SEED_EVAL = (2, 3) + RV.NEW_SEEDS


def seed_runs(model, ds, n):
    c = es.best_cfg(model, ds, n)
    ext = RV.has_ext(model, ds, n, c)
    ph = 'ext' if ext else 'final'
    old = [es.ofile(ph, model, ds, n, c, s) for s in (2, 3)]
    new = [RV.rfile('seeds', model, ph, ds, n, c, s) for s in RV.NEW_SEEDS]
    return runs(old + new)


SR = {(m, ds, n): seed_runs(m, ds, n) for m in ('gg_lintern', 'cotm') for ds in C.DS_ES for n in C.BUDGETS}
if all(SR.values()):
    L = [r'\begin{tabular}{l' + 'r' * len(C.DS_ES) + 'rrr}', r'\toprule',
         r'Metric & ' + ' & '.join(DSN[d] for d in C.DS_ES) +
         r' & Holm & $p_\mathrm{W}$, cells & $p_\mathrm{W}$, datasets \\', r'\midrule']
    for key, lab in MET:
        mk = {'test_f1': 'mf', 'test_bacc': 'bacc', 'test_acc': 'acc'}[key]
        P, D, cellsd = [], [], {}
        for ds in C.DS_ES:
            for n in C.BUDGETS:
                a, b = mv(SR[('gg_lintern', ds, n)], key), mv(SR[('cotm', ds, n)], key)
                P.append(R.welch(a, b))
                D.append(a.mean() - b.mean())
                cellsd.setdefault(ds, []).append(len(P) - 1)
        P, D = np.array(P), np.array(D)
        order = np.argsort(P)                             # Holm step-down over the 36 cells
        rej = np.zeros(len(P), bool)
        for i, j in enumerate(order):
            if P[j] <= 0.05 / (len(P) - i):
                rej[j] = True
            else:
                break
        W, Lo = int(np.sum(rej & (D > 0))), int(np.sum(rej & (D < 0)))
        dsm = [D[cellsd[ds]].mean() for ds in C.DS_ES]
        pw_c = ST.wilcoxon(D).pvalue
        pw_d = ST.wilcoxon(dsm).pvalue
        cells = [f'{D[cellsd[ds]].mean():+.1f} ({int(np.sum(rej[cellsd[ds]] & (D[cellsd[ds]] > 0)))}/'
                 f'{int(np.sum(rej[cellsd[ds]] & (D[cellsd[ds]] < 0)))})'.replace('-', '$-$') for ds in C.DS_ES]
        L.append(f'{lab} & ' + ' & '.join(cells) + f' & {W}/{Lo} & {pw_c:.1g} & {pw_d:.2f}'.replace('e-0', 'e-')
                 + r' \\')
        mac(f'seed{mk}holmwin', str(W))
        mac(f'seed{mk}holmloss', str(Lo))
        mac(f'seed{mk}pos', str(int(np.sum(D > 0))))
        mac(f'seed{mk}rawwin', str(int(np.sum((P < 0.05) & (D > 0)))))
        mac(f'seed{mk}rawloss', str(int(np.sum((P < 0.05) & (D < 0)))))
        e = f'{pw_c:.1e}'.split('e')
        mac(f'seed{mk}pwcells', f'{float(e[0]):.1f}\\times 10^{{{int(e[1])}}}')
        mac(f'seed{mk}pwds', f'{pw_d:.3f}')
        mac(f'seed{mk}meandiff', f'{D.mean():.1f}')
        if key == 'test_f1':
            # 95% confidence interval of the mean difference per cell (Welch); smallest lower bound reported
            lo = []
            for ds in C.DS_ES:
                for n in C.BUDGETS:
                    a, b = mv(SR[('gg_lintern', ds, n)]), mv(SR[('cotm', ds, n)])
                    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                    lo.append(a.mean() - b.mean() - ST.t.ppf(0.975, df) * np.sqrt(va + vb))
            mac('seedcipos', str(int(np.sum(np.array(lo) > 0))))
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_seeds.tex', '\n'.join(L) + '\n')
else:
    note(f'seeds incomplete ({sum(bool(v) for v in SR.values())} of {len(SR)} cells)')

# ---- B. clause dumps: plain and negated literals, nonzero votes, CoTM weight range
DUMP = {}
for m in ('gg_lintern', 'cotm', 'gg_hidtern'):
    for ds in C.DS_ES:
        for a in ANCH:
            p = RV.rfile('dump', m, 'final', ds, a, es.best_cfg(m, ds, a), 1).replace('.json', '.npz')
            if os.path.exists(p) and R.at_best(p.replace('.npz', '.json')):
                DUMP[(m, ds, a)] = np.load(p)
if len(DUMP) == 3 * len(C.DS_ES) * len(ANCH):
    L = [r'\begin{tabular}{lr' + 'r' * len(C.DS_ES) + '}', r'\toprule',
         r'Quantity & $n$ & ' + ' & '.join(DSN[d] for d in C.DS_ES) + r' \\', r'\midrule']
    negfrac = {}
    for m in ('gg_lintern', 'cotm'):
        for a in ANCH:
            cells = []
            for ds in C.DS_ES:
                z = DUMP[(m, ds, a)]
                inc = np.unpackbits(z['include'], axis=1)[:, :int(z['n_lit'])]
                o = inc.shape[1] // 2
                pl, ng = inc[:, :o].sum(1), inc[:, o:].sum(1)
                negfrac[(m, ds, a)] = ng.sum() / max(1, (pl + ng).sum())
                cells.append(f'{pl.mean():.1f} / {ng.mean():.1f}')
            L.append((f'{NAME[m]}: plain / negated literals' if a == ANCH[0] else '') + f' & {a} & ' +
                     ' & '.join(cells) + r' \\')
    votefrac = []
    for a in ANCH:
        cells = []
        for ds in C.DS_ES:
            Wt = DUMP[('gg_lintern', ds, a)]['W']
            votefrac.append((Wt != 0).sum(0).mean() / Wt.shape[0])
            cells.append(f'{(Wt != 0).sum(0).mean():.1f} / {Wt.shape[0]}')
        L.append(('GG-TL: nonzero votes / $m$' if a == ANCH[0] else '') + f' & {a} & ' + ' & '.join(cells) +
                 r' \\')
    for a in ANCH:
        cells = []
        for ds in C.DS_ES:
            Wc = DUMP[('cotm', ds, a)]['W']
            r_ = int(np.abs(Wc).max())
            cells.append(f'{r_} ({int(np.ceil(np.log2(2 * r_ + 1)))})')
        L.append(('CoTM: largest $|w|$ (bits)' if a == ANCH[0] else '') + f' & {a} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    w('tab_dump.tex', '\n'.join(L) + '\n')
    gneg = [negfrac[('gg_lintern', ds, a)] for ds in C.DS_ES for a in ANCH]
    cneg = [negfrac[('cotm', ds, a)] for ds in C.DS_ES for a in ANCH]
    mac('dumpggnegmin', f'{100 * min(gneg):.0f}')
    mac('dumpggnegmax', f'{100 * max(gneg):.0f}')
    mac('dumpcotmnegmin', f'{100 * min(cneg):.0f}')
    mac('dumpcotmnegmax', f'{100 * max(cneg):.0f}')
    for _m, _k in (('gg_lintern', 'gg'), ('cotm', 'cotm')):
        _v = [negfrac[(_m, 'alerce', a)] for a in ANCH]
        mac(f'dump{_k}negalcmin', f'{100 * min(_v):.0f}')
        mac(f'dump{_k}negalcmax', f'{100 * max(_v):.0f}')
    mac('dumpggzeromin', f'{100 * (1 - max(votefrac)):.0f}')
    mac('dumpggzeromax', f'{100 * (1 - min(votefrac)):.0f}')
    bits = [int(np.ceil(np.log2(2 * int(np.abs(DUMP[('cotm', ds, a)]['W']).max()) + 1))) for ds in C.DS_ES
            for a in ANCH]
    mac('dumpggvotemin', f'{100 * min(votefrac):.0f}')
    mac('dumpggvotemax', f'{100 * max(votefrac):.0f}')
    mac('dumpcotmbitsmin', str(min(bits)))
    mac('dumpcotmbitsmax', str(max(bits)))
    # the dump runs repeat the seed-1 FINAL runs: largest deviation of the best validation F1 from the originals
    dev = []
    for (m, ds, a) in DUMP:
        x = R.at_best(RV.rfile('dump', m, 'final', ds, a, es.best_cfg(m, ds, a), 1))
        y = R.at_best(es.ofile('final', m, ds, a, es.best_cfg(m, ds, a), 1))
        if x and y:
            dev.append(abs(x['val_f1'] - y['val_f1']) * 100)
    mac('dumpdev', f'{max(dev):.2f}' if dev else '--')
else:
    note(f'dumps incomplete ({len(DUMP)} of {3 * len(C.DS_ES) * len(ANCH)})')

# ------------------------------------------------------------------ macros
w('results_macros.tex', '%% Generated by gg-th/analysis/paper_tables.py. Do not edit by hand.\n' +
  '\n'.join(f'\\newcommand{{\\r{k}}}{{{v}}}' for k, v in sorted(MAC.items())) + '\n')

# ------------------------------------------------------------------ figure
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'pdf.fonttype': 42})
STY = {'gg_lintern': ('#1b7837', 'o', '-'), 'gg_linr': ('#35978f', 'X', '-'), 'gg_hidtern': ('#7fbf7b', 's', '--'), 'cotm': ('#b2182b', 'v', ':'),
       'tmu_tm': ('#e08214', 'P', ':'), 'fptm': ('#2166ac', 'D', '-.'), 'tnn': ('#762a83', '^', '-')}
fig = plt.figure(figsize=(7.2, 8.6))
gs = GridSpec(4, 2, height_ratios=[0.12, 1, 1, 1], hspace=0.55, wspace=0.28, figure=fig)
lax = fig.add_subplot(gs[0, :])
lax.axis('off')
present = [m for m in COLS if m != 'gg_linr' and any(k[1] == m for k in R.rows)]
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
