"""Shared experiment configuration: datasets, budgets, search grids and result paths."""
import itertools
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RES = os.path.join(ROOT, 'results')
R_ABL = os.path.join(RES, 'ablation')
R_OURS = os.path.join(RES, 'budget', 'ours')
R_FPTM = os.path.join(RES, 'budget', 'fptm')

DS = ('wos', 'alerce')                                         # original study (experiments/run.py)
DS_ES = DS + ('r52', 'sector', 'rcv1', 'news20')               # early-stopping study; the last four: experiments/PROTOCOL_text.md
M = {'wos': 133, 'alerce': 15, 'r52': 52, 'sector': 105, 'rcv1': 51, 'news20': 20}            # classes
O = {'wos': 2000, 'alerce': 1172, 'r52': 2000, 'sector': 2000, 'rcv1': 2000, 'news20': 2000}  # Boolean features
PCLS = {'wos': 300, 'alerce': 4000, 'r52': 100, 'sector': 50, 'rcv1': 300, 'news20': 700}     # samples per class per epoch
BUDGETS = (50, 100, 200, 500, 1000, 2000)  # total clauses
SEEDS = (1, 2, 3)
TUNE_EPOCHS, FINAL_EPOCHS = 5, 20

# GG-TH search grid, tuned at the anchor budgets 200 and 1000
OURS_GRID = [dict(s=s, lr=lr, H=H) for s, lr, H in itertools.product((5, 10, 20), (1e-3, 3e-3), (128, 256))]
ANCHORS = (200, 1000)


def anchor(n):
    return 200 if n <= 500 else 1000


def ours_file(tag, ds, n, cfg, seed):
    return f'{R_OURS}/{tag}_{ds}_{n}_s{cfg["s"]}_lr{cfg["lr"]}_H{cfg["H"]}_seed{seed}.json'


def ours_best(ds, n):
    a = anchor(n)
    return max(OURS_GRID, key=lambda c: json.load(open(ours_file('tune', ds, a, c, 1)))['epochs'][-1]['val_f1'])


# FPTM: clauses per class matched to the total budget, at least 2 (one positive, one negative)
def cpc(ds, n):
    return max(2, 2 * round(n / M[ds] / 2))


def fptm_grid(c):
    out = []
    for (L, LF), S, tm in itertools.product(((8, 4), (16, 8), (64, 32), (200, 200)), (100, 200, 1000), (1, 2)):
        out.append(dict(T=max(2, round(tm * math.sqrt(c / 2 * LF))), S=S, L=L, LF=LF))
    return out


def fptm_file(tag, ds, n, cfg, seed):
    return f'{R_FPTM}/{tag}_{ds}_{n}_T{cfg["T"]}_S{cfg["S"]}_L{cfg["L"]}_LF{cfg["LF"]}_s{seed}.json'


def fptm_best(ds, n):
    return max(fptm_grid(cpc(ds, n)), key=lambda c: json.load(open(fptm_file('tune', ds, n, c, 1)))['epochs'][-1]['val_f1'])


def fptm_sizes(ds):
    """Budgets that map to distinct FPTM sizes (the first budget of each size is the one that is run)."""
    seen, out = set(), []
    for n in BUDGETS:
        if cpc(ds, n) not in seen:
            seen.add(cpc(ds, n))
            out.append(n)
    return out


def done(path, epochs):
    try:
        return len(json.load(open(path))['epochs']) >= epochs
    except Exception:
        return False
