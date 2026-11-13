"""Revised comparison with early stopping for every model.

Models: GG-TH (gg_hidtern), GG with a ternary linear readout (gg_lintern), clause-free ternary network of the
same shape and memory (tnn), tuned CoTM (cotm), FPTM, and GG-linear with post-hoc ternary rounding (gg_lin, n=1000).

Protocol
  tune : seed 1, up to 20 epochs, patience 5, selection on the best validation macro F1 over epochs.
         Clause/unit models are tuned at the anchors 200 and 1000 (budgets <= 500 use the 200 setting);
         FPTM is tuned at every distinct size.
  final: seeds 1 to 3, up to 60 epochs, patience 10; test metrics are read at the best validation epoch.

  ext  : rerun, with up to 120 epochs, every final setting with a seed whose best epoch fell near the cap.

  python experiments/es.py tune|final|ext|all [--workers 2] [--dry]
Runs are checkpointed every epoch and resumed after an interruption; completed runs are skipped.
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import config as C

RES = os.path.join(C.ROOT, 'results_es')
TUNE = dict(epochs=20, patience=5)
FINAL = dict(epochs=60, patience=10)
EXT = dict(epochs=120, patience=10)            # rerun of final settings whose best epoch fell near the 60-epoch cap
ORDER = (1000, 200, 500, 2000, 100, 50)      # final budgets, most informative first

GRIDS = {
    # Grids extended where tuning optima fell on the boundary. All clause models share the s range 1.5 to 20 (40).
    'gg_hidtern': [dict(s=s, lr=lr, H=H) for s, lr, H in itertools.product((1.5, 2, 5, 10, 20, 40), (1e-3, 3e-3), (128, 256))],
    'gg_lintern': [dict(s=s, lr=lr, H=0) for s, lr in itertools.product((1.5, 2, 5, 10, 20), (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2))],
    'tnn':        [dict(s=0, lr=lr, H=H) for lr, H in itertools.product((3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2), (128, 256))],
    'cotm':       [dict(s=s, Tm=t, H=0) for s, t in itertools.product((1.5, 2, 5, 10, 20), (0.03, 0.1, 0.3, 1.0, 2.5, 5.0, 10.0, 20.0))],
}
MODELS = tuple(GRIDS)


def tag(cfg):
    return '_'.join(f'{k}{v}' for k, v in cfg.items())


def ofile(phase, model, ds, n, cfg, seed):
    return f'{RES}/{model}/{phase}_{ds}_{n}_{tag(cfg)}_seed{seed}.json'


def ffile(phase, ds, n, cfg, seed):
    return f'{RES}/fptm/{phase}_{ds}_{n}_T{cfg["T"]}_S{cfg["S"]}_L{cfg["L"]}_LF{cfg["LF"]}_s{seed}.json'


def finished(path):
    try:
        return bool(json.load(open(path)).get('done'))
    except Exception:
        return False


def best_val(path):
    return max(e['val_f1'] for e in json.load(open(path))['epochs'])


def best_cfg(model, ds, n):
    a = C.anchor(n)
    return max(GRIDS[model], key=lambda c: best_val(ofile('tune', model, ds, a, c, 1)))


# FPTM: base grid of config.py, then one step beyond the base grid along every axis on which the base optimum
# lies at the grid boundary (other axes held at the base optimum).
FPTM_EXT = dict(tm={1: 0.5, 2: 3}, S={100: 50, 1000: 2000}, L={(8, 4): (4, 2)})


def fptm_T(c, LF, tm):
    return max(2, round(tm * math.sqrt(c / 2 * LF)))


def fptm_cands(ds, n):
    c = C.cpc(ds, n)
    base = C.fptm_grid(c)
    if not all(finished(ffile('tune', ds, n, g, 1)) for g in base):
        return base
    b = max(base, key=lambda g: best_val(ffile('tune', ds, n, g, 1)))
    tm = next(t for t in (1, 2) if fptm_T(c, b['LF'], t) == b['T'])
    ext = []
    if tm in FPTM_EXT['tm']:
        ext.append(dict(b, T=fptm_T(c, b['LF'], FPTM_EXT['tm'][tm])))
    if b['S'] in FPTM_EXT['S']:
        ext.append(dict(b, S=FPTM_EXT['S'][b['S']]))
    if (b['L'], b['LF']) in FPTM_EXT['L']:
        L, LF = FPTM_EXT['L'][(b['L'], b['LF'])]
        ext.append(dict(b, L=L, LF=LF, T=fptm_T(c, LF, tm)))
    return base + [e for e in ext if e not in base]


def fptm_best(ds, n):
    return max(fptm_cands(ds, n), key=lambda c: best_val(ffile('tune', ds, n, c, 1)))


def ours_job(phase, model, ds, n, cfg, seed, readout=None):
    out = ofile(phase, model, ds, n, cfg, seed)
    if finished(out):
        return None
    P = {'tune': TUNE, 'final': FINAL, 'ext': EXT}[phase]
    cmd = [sys.executable, os.path.join(C.ROOT, 'train.py'), '--dataset', ds, '--readout', readout or model,
           '--clauses', str(n), '--lr', str(cfg.get('lr', 2e-3)), '--H', str(cfg['H'] or 256),
           '--P', str(C.PCLS[ds]), '--epochs', str(P['epochs']), '--patience', str(P['patience']),
           '--seed', str(seed), '--out', out, '--resume']
    if cfg.get('s'):
        cmd += ['--s', str(cfg['s'])]
    if 'Tm' in cfg:
        cmd += ['--T', str(cfg['Tm'] * n)]
    return cmd, out


def fptm_job(phase, ds, n, cfg, seed):
    out = ffile(phase, ds, n, cfg, seed)
    if finished(out):
        return None
    P = {'tune': TUNE, 'final': FINAL, 'ext': EXT}[phase]
    jl = [os.environ.get('JULIA', 'julia'), '-t', '1', os.path.join(C.ROOT, 'fptm', 'fptm_run.jl')]
    return jl + [ds, str(C.cpc(ds, n)), str(cfg['T']), str(cfg['S']), str(cfg['L']), str(cfg['LF']),
                 str(P['epochs']), str(C.PCLS[ds]), str(seed), out, str(P['patience'])], out


def tune_jobs():
    jobs = []
    for a in (1000, 200):
        for ds in C.DS:
            for model in MODELS:
                jobs += [ours_job('tune', model, ds, a, c, 1) for c in GRIDS[model]]
    for ds in C.DS:
        for n in C.fptm_sizes(ds):
            jobs += [fptm_job('tune', ds, n, c, 1) for c in fptm_cands(ds, n)]
    return [j for j in jobs if j]


def final_jobs(phase='final'):
    """phase 'final': every selected setting. phase 'ext': only settings with at least one final seed whose best
    epoch fell near the 60-epoch cap; all seeds of such a setting are rerun from scratch with up to 120 epochs."""
    def keep(paths):
        return phase == 'final' or any(near_cap(p) for p in paths)

    jobs = []
    for n in ORDER:
        for ds in C.DS:
            for model in MODELS:
                c = best_cfg(model, ds, n)
                if keep([ofile('final', model, ds, n, c, s) for s in C.SEEDS]):
                    jobs += [ours_job(phase, model, ds, n, c, s) for s in C.SEEDS]
            if n == 1000:                         # GG-linear with post-hoc rounding, setting of gg_lintern
                c = dict(best_cfg('gg_lintern', ds, n))
                if keep([ofile('final', 'gg_lin', ds, n, c, s) for s in C.SEEDS]):
                    jobs += [ours_job(phase, 'gg_lin', ds, n, c, s, readout='gg_lin') for s in C.SEEDS]
            if n in C.fptm_sizes(ds):
                c = fptm_best(ds, n)
                if keep([ffile('final', ds, n, c, s) for s in C.SEEDS]):
                    jobs += [fptm_job(phase, ds, n, c, s) for s in C.SEEDS]
    return [j for j in jobs if j]


def near_cap(path):
    """Final run whose best epoch lies within the last patience/2 epochs of the 60-epoch cap."""
    d = json.load(open(path))
    return len(d['epochs']) >= FINAL['epochs'] and d['best_epoch'] > FINAL['epochs'] - FINAL['patience'] // 2


def execute(jobs, workers, dry):
    print(len(jobs), 'runs', flush=True)
    if dry:
        for cmd, _ in jobs:
            print(' '.join(cmd))
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def one(job):
        cmd, out = job
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out.replace('.json', '.log'), 'a') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        return out, rc

    with ThreadPoolExecutor(workers) as ex:
        for out, rc in ex.map(one, jobs):
            print('done' if rc == 0 else f'FAILED ({rc})', os.path.relpath(out, C.ROOT), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['tune', 'final', 'ext', 'all'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for ph in (('tune', 'final') if a.phase == 'all' else (a.phase,)):
        execute(tune_jobs() if ph == 'tune' else final_jobs(ph), a.workers, a.dry)
