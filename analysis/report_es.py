"""Tables and figure of the early-stopping study (experiments/es.py) from the stored run logs.

  python analysis/report_es.py [--phase final|tune]
      writes results_es/summary.md and figures/fig_es_budget.{pdf,png}

Test metrics are read at the best validation epoch of each run (early stopping on validation macro F1).
Values are mean and sample standard deviation over the available seeds (1 to 3 in the final phase).
Differences against GG-TH are tested with Welch's t-test. Hyperparameters are re-derived from the
tune logs exactly as experiments/es.py selects them. '--phase tune' is a smoke test on the seed-1 tune
runs at the anchor budgets.
"""
import argparse
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
import es

NAME = {'wos': 'WOS, 133 classes', 'alerce': 'ALeRCE, 15 classes', 'r52': 'R52, 52 classes',
        'sector': 'Sector, 105 classes', 'rcv1': 'RCV1, 51 classes', 'news20': '20 Newsgroups, 20 classes'}
PAIRS = (('gg_hidtern', 'gg_lintern'), ('gg_hidtern', 'tnn'), ('gg_hidtern', 'cotm'), ('gg_hidtern', 'fptm'),
         ('gg_lintern', 'cotm'), ('gg_lintern', 'fptm'), ('gg_lintern', 'tnn'))
LABEL = {'gg_hidtern': 'GG-TH', 'gg_lintern': 'GG, ternary linear readout', 'tnn': 'Ternary net, no clauses',
         'cotm': 'CoTM, tuned', 'fptm': 'FPTM', 'gg_lin': 'GG-linear, rounded to ternary'}
STYLE = {'gg_hidtern': ('#1b7837', 'o', '-'), 'gg_lintern': ('#5aae61', 's', '--'), 'tnn': ('#762a83', '^', '-'),
         'cotm': ('#b2182b', 'v', ':'), 'fptm': ('#2166ac', 'D', '-.')}
FIG = os.path.join(C.ROOT, 'figures')


def at_best(path):
    """Epoch record at the best validation epoch, or None if the run is missing or unfinished."""
    try:
        d = json.load(open(path))
    except Exception:
        return None
    if not d.get('done'):
        return None
    e = {r['epoch']: r for r in d['epochs']}[d['best_epoch']]
    cap = es.EXT['epochs'] if '/ext_' in path else es.FINAL['epochs'] if '/final_' in path else es.TUNE['epochs']
    return dict(e, best_epoch=d['best_epoch'], n_epochs=len(d['epochs']), cap=cap)


def ms(runs, key):
    v = np.array([r[key] for r in runs], dtype=float) * 100
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan')), v


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def dense_bits(model, ds, n, cfg):
    """Stored ternary/Boolean bits of the clause (or first) layer plus the ternary readout layers.
    CoTM integer weights and FPTM are reported by clause memory only (2o bits per clause)."""
    o, m = C.O[ds], C.M[ds]
    if model in ('gg_hidtern', 'tnn'):
        return 2 * o * n + 2 * cfg['H'] * n + 2 * m * cfg['H']
    if model in ('gg_lintern', 'gg_lin'):
        return 2 * o * n + 2 * m * n
    return 2 * o * n


ap = argparse.ArgumentParser()
ap.add_argument('--phase', choices=['final', 'tune'], default='final')
A = ap.parse_args()
PH = A.phase
SEEDS = C.SEEDS if PH == 'final' else (1,)
BUD = C.BUDGETS if PH == 'final' else (200, 1000)

# ------------------------------------------------------------------ collect
rows = {}                                          # (ds, model) -> list of dicts per budget
EXTENDED = []                                      # (ds, model, n, 60-epoch runs, 120-epoch runs)


def load(path):
    """Runs of one setting over the seeds. path(phase, seed) gives the log file. In the final phase a setting
    rerun with up to 120 epochs (es.py ext) replaces its 60-epoch runs once all its seeds are done."""
    r = [x for x in (at_best(path(PH, s)) for s in SEEDS) if x]
    if PH == 'final':
        e = [at_best(path('ext', s)) for s in SEEDS]
        if all(e):
            return e, r
    return r, None


def add(ds, model, n, clauses, c, path, bits):
    r, r60 = load(path)
    if r:
        rows.setdefault((ds, model), []).append(dict(n=n, clauses=clauses, cfg=c, r=r, bits=bits))
        if r60:
            EXTENDED.append((ds, model, n, r60, r))


for ds in C.DS_ES:
    for n in BUD:
        nn = n if PH == 'final' else C.anchor(n)
        for model in es.MODELS:
            c = es.best_cfg(model, ds, n)
            add(ds, model, n, n, c, lambda ph, s: es.ofile(ph, model, ds, nn, c, s), dense_bits(model, ds, n, c))
        if n == 1000:
            c = dict(es.best_cfg('gg_lintern', ds, n))
            add(ds, 'gg_lin', n, n, c, lambda ph, s: es.ofile(ph, 'gg_lin', ds, n, c, s), dense_bits('gg_lin', ds, n, c))
        if n in C.fptm_sizes(ds):
            c, cp = es.fptm_best(ds, n), C.cpc(ds, n)
            add(ds, 'fptm', n, cp * C.M[ds], c, lambda ph, s: es.ffile(ph, ds, n, c, s), 2 * C.O[ds] * cp * C.M[ds])
for v in rows.values():
    for x in v:
        x['lit'] = np.mean([q['mean_literals'] for q in x['r']])
        x['ep'] = np.mean([q['best_epoch'] for q in x['r']])
        x['f1'] = ms(x['r'], 'test_f1')


def get(ds, model, n):
    return next((x for x in rows.get((ds, model), []) if x['n'] == n), None)


def cell(x, key='test_f1', ref=None):
    if x is None:
        return 'n/a'
    mu, sd, v = ms(x['r'], key)
    s = f'{mu:.2f} ± {sd:.2f}' if not np.isnan(sd) else f'{mu:.2f}'
    if ref is not None:
        p = welch(v, ms(ref['r'], key)[2])
        if not np.isnan(p):
            s += f' (p={p:.3f})'
    return s


# ------------------------------------------------------------------ tables
L = ['# Early-stopping study, results summary', '',
     f'Phase: {PH}. Test metrics in %, read at the best validation epoch; mean ± sample std over seeds {SEEDS}.',
     'p: Welch t-test against GG-TH at the same budget. FPTM rows are listed at their nearest clause budget '
     '(clauses per class matched to the total budget).', '']

L += ['## Search grids (seed 1, validation macro F1)', '', '| Model | Grid size | Axes |', '|---|---|---|']
for m in es.MODELS:
    ax = {k: sorted({g[k] for g in es.GRIDS[m]}) for k in es.GRIDS[m][0] if len({g[k] for g in es.GRIDS[m]}) > 1}
    L.append(f'| {LABEL[m]} | {len(es.GRIDS[m])} | ' + '; '.join(f'{k} ∈ {v}' for k, v in ax.items()) + ' |')
L.append(f'| FPTM | {len(C.fptm_grid(10))} per size | (L, LF), S, T multiplier |')
L += ['', '| Model | Dataset | Anchor | Best setting | Validation F1 |', '|---|---|---|---|---|']
for m in es.MODELS:
    for ds in C.DS_ES:
        for a in (1000, 200):
            c = es.best_cfg(m, ds, a)
            L.append(f'| {LABEL[m]} | {ds} | {a} | {es.tag(c)} | {es.best_val(es.ofile("tune", m, ds, a, c, 1)) * 100:.2f} |')

L += ['', 'Selected settings on a grid boundary, axes with more than two values (grids not extended for these, see '
      'experiments/PROTOCOL_text.md):', '']
for m in es.MODELS:
    ax = {k: sorted({g[k] for g in es.GRIDS[m]}) for k in es.GRIDS[m][0] if len({g[k] for g in es.GRIDS[m]}) > 1}
    for ds in C.DS_ES:
        for a in (1000, 200):
            c = es.best_cfg(m, ds, a)
            e = [f'{k}={c[k]} ({"lower" if c[k] == v[0] else "upper"} edge)' for k, v in ax.items()
                 if len(v) > 2 and c[k] in (v[0], v[-1])]
            if e:
                L.append(f'- {LABEL[m]}, {ds}, anchor {a}: ' + ', '.join(e))

for key, title in (('test_f1', 'Macro F1'), ('test_bacc', 'Balanced accuracy'), ('test_acc', 'Accuracy')):
    for ds in C.DS_ES:
        L += ['', f'## {title}, {NAME[ds]}', '',
              '| Budget | ' + ' | '.join(LABEL[m] for m in es.MODELS) + ' | FPTM |',
              '|---' * (len(es.MODELS) + 2) + '|']
        for n in BUD:
            ref = get(ds, 'gg_hidtern', n)
            cells = [cell(get(ds, m, n), key, None if m == 'gg_hidtern' else ref) for m in es.MODELS]
            f = get(ds, 'fptm', n)
            fc = cell(f, key, ref) + (f' [{f["clauses"]} cl.]' if f else '')
            L.append(f'| {n} | ' + ' | '.join(cells) + f' | {fc} |')

# Point 3: post-hoc rounding of GG-linear at n = 1000
L += ['', '## GG-linear trained in floating point, evaluated with weights rounded to {-1, 0, +1} (n = 1000)', '',
      '| Dataset | GG-linear float F1 | GG-linear rounded F1 | GG, ternary readout (STE) F1 | GG-TH F1 |',
      '|---|---|---|---|---|']
for ds in C.DS_ES:
    x = get(ds, 'gg_lin', 1000)
    if x is None:
        continue
    rk = next((k for k in x['r'][0] if k.startswith('test_f1') and 'round' in k), None)
    L.append(f'| {ds} | {cell(x)} | {cell(x, rk) if rk else "n/a"} | {cell(get(ds, "gg_lintern", 1000))} | '
             f'{cell(get(ds, "gg_hidtern", 1000))} |')

# Point 1: what the clauses add, sparsity and memory
L += ['', '## Sparsity and memory', '',
      'Literals: mean included literals per clause (clause models) or mean nonzero first-layer weights per unit '
      '(ternary net), at the best epoch. Bits: stored bits of the clause or first layer plus ternary readout layers.', '',
      '| Dataset | Budget | Model | Literals per clause or unit | Bits | Best epoch (mean) |', '|---|---|---|---|---|---|']
for ds in C.DS_ES:
    for n in BUD:
        for m in es.MODELS + ('fptm',):
            x = get(ds, m, n)
            if x:
                L.append(f'| {ds} | {n} | {LABEL[m]} | {x["lit"]:.1f} | {x["bits"]:,} | {x["ep"]:.1f} |')

# Convergence: runs that stopped at the epoch cap
hit = [(ds, m, x['n']) for (ds, m), v in rows.items() for x in v for q in x['r']
       if q['n_epochs'] >= q['cap'] and q['best_epoch'] > q['cap'] - es.FINAL['patience'] // 2]
cap = '60 or 120' if EXTENDED else (es.FINAL['epochs'] if PH == 'final' else es.TUNE['epochs'])
L += ['', f'Runs whose best epoch lies within the last {es.FINAL["patience"] // 2} epochs of the cap ({cap}): {len(hit)}.']
L += [f'- {ds} {LABEL[m]} n={n}' for ds, m, n in sorted(set(hit))]
if EXTENDED:
    L += ['', f'## Settings rerun with up to {es.EXT["epochs"]} epochs', '',
          'All seeds of every final setting with at least one seed whose best epoch fell within the last '
          f'{es.FINAL["patience"] // 2} epochs of the {es.FINAL["epochs"]}-epoch cap were rerun from scratch with up to '
          f'{es.EXT["epochs"]} epochs and the same patience. The tables above use the rerun.', '',
          '| Dataset | Budget | Model | Macro F1, 60-epoch cap | Macro F1, 120-epoch cap | Best epochs (120-epoch cap) |',
          '|---|---|---|---|---|---|']
    for ds, m, n, r60, r in EXTENDED:
        f = lambda rr: f'{ms(rr, "test_f1")[0]:.2f} ± {ms(rr, "test_f1")[1]:.2f}'
        L.append(f'| {ds} | {n} | {LABEL[m]} | {f(r60)} | {f(r)} | {", ".join(str(q["best_epoch"]) for q in r)} |')

# Pairwise comparisons (PROTOCOL_text.md: GG-TH against all models; GG with a ternary linear readout against CoTM,
# FPTM and the ternary net). Per dataset over the budgets where both models have results.
L += ['', '## Pairwise comparisons per dataset (test macro F1)', '',
      'Mean difference A − B over the budgets where both have results (FPTM: budgets with an FPTM size). '
      'Wins and losses: budgets with p < 0.05 (Welch) in favour of A or of B.', '',
      '| A | B | Dataset | Budgets | Mean A − B | Range of A − B | A better (p<0.05) | B better (p<0.05) |',
      '|---|---|---|---|---|---|---|---|']
for a, b in PAIRS:
    for ds in C.DS_ES:
        d, w, l = [], 0, 0
        for n in BUD:
            xa, xb = get(ds, a, n), get(ds, b, n)
            if xa is None or xb is None:
                continue
            va, vb = ms(xa['r'], 'test_f1')[2], ms(xb['r'], 'test_f1')[2]
            d.append(va.mean() - vb.mean())
            p = welch(va, vb)
            w += int(p < 0.05 and va.mean() > vb.mean())
            l += int(p < 0.05 and va.mean() < vb.mean())
        if d:
            L.append(f'| {LABEL[a]} | {LABEL[b]} | {ds} | {len(d)} | {np.mean(d):+.2f} | '
                     f'{min(d):+.2f} to {max(d):+.2f} | {w} | {l} |')

out_md = os.path.join(es.RES, 'summary.md' if PH == 'final' else 'summary_tune.md')
open(out_md, 'w').write('\n'.join(L) + '\n')
print('wrote', out_md)

# ------------------------------------------------------------------ figure: macro F1 against clause budget
DSF = [ds for ds in C.DS_ES if any(k[0] == ds for k in rows)]
NC = 3 if len(DSF) > 2 else len(DSF)
NR = -(-len(DSF) // NC)
fig = plt.figure(figsize=(4.4 * NC, 3.7 * NR + 0.6))
gs = GridSpec(NR + 1, NC, height_ratios=[1] * NR + [0.12], hspace=0.45, wspace=0.25)
for i, ds in enumerate(DSF):
    ax = fig.add_subplot(gs[i // NC, i % NC])
    for m in es.MODELS + ('fptm',):
        v = rows.get((ds, m), [])
        if not v:
            continue
        col, mk, ls = STYLE[m]
        x = [q['clauses'] for q in v]
        y = [q['f1'][0] for q in v]
        e = [0 if np.isnan(q['f1'][1]) else q['f1'][1] for q in v]
        ax.errorbar(x, y, yerr=e, color=col, marker=mk, ls=ls, ms=5, lw=1.5, capsize=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(tk.ScalarFormatter())
    ax.set_xticks([50, 100, 200, 500, 1000, 2000])
    if i // NC == NR - 1 or i + NC >= len(DSF):
        ax.set_xlabel('Total clauses (units for the ternary net)')
    if i % NC == 0:
        ax.set_ylabel('Test macro F1 (%)')
    ax.set_title(NAME[ds])
    ax.grid(alpha=0.3)
lax = fig.add_subplot(gs[NR, :])
lax.axis('off')
h = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls=STYLE[m][2], lw=1.5) for m in es.MODELS + ('fptm',)]
lax.legend(h, [LABEL[m] for m in es.MODELS + ('fptm',)], loc='center', ncol=5, frameon=False, fontsize=9)
os.makedirs(FIG, exist_ok=True)
stem = os.path.join(FIG, 'fig_es_budget' if PH == 'final' else 'fig_es_budget_tune')
fig.savefig(stem + '.pdf', bbox_inches='tight')
fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
print('wrote', stem + '.{pdf,png}')
