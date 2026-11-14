"""Baseline check with the official TMU library (https://github.com/cair/tmu, commit 17f17231, installed from
GitHub; the PyPI release 0.7.1 fails under NumPy 2).

Models
  tmu_cotm : TMCoalescedClassifier, n clauses shared by all classes (the reference implementation of CoTM).
  tmu_tm   : TMClassifier, the multiclass Tsetlin Machine with cpc(ds, n) clauses per class (half positive,
             half negative polarity), i.e. about n clauses in total.

Protocol: experiments/PROTOCOL_tmu.md. Same splits, class-balanced sampling (P per class per epoch), tuning,
final and early-stopping rules as experiments/es.py. Logs have the format of train.py, so analysis/report_es.py
helpers apply.

  python experiments/tmu_check.py tune|final [--workers 2] [--dry]
  python experiments/tmu_check.py one <model> <ds> <n> <s> <Tm> <seed> <epochs> <patience> <out>
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config as C

RES = os.path.join(C.ROOT, 'results_tmu')
TMU_VERSION = 'tmu, github.com/cair/tmu commit 17f17231b944d1aa7ae1a36541ba0bc9f0e52bc4'
PY = os.environ.get('TMU_PYTHON', sys.executable)         # environment with TMU installed
TUNE = dict(epochs=20, patience=5)
FINAL = dict(epochs=60, patience=10)
ANCHORS = (200, 1000)
# T = Tm * (clauses voting for one class): n for tmu_cotm, cpc for tmu_tm.
GRID = [dict(s=s, Tm=t) for s, t in itertools.product((1.5, 2, 5, 10, 20), (0.1, 0.3, 1.0, 2.5, 5.0, 10.0))]
MODELS = ('tmu_cotm', 'tmu_tm')
# tmu_cotm is too slow on this machine for a full grid (WOS, n = 1000: about 430 s per epoch). It is run as an
# implementation check: the setting selected for our CoTM (es.py), 3 seeds, final protocol, at these budgets.
CHECK = {ds: (200,) for ds in C.DS_ES}
CHECK.update(r52=(200, 1000), sector=(200, 1000))


def tag(c):
    return f's{c["s"]}_Tm{c["Tm"]}'


def ofile(phase, model, ds, n, c, seed):
    return f'{RES}/{model}/{phase}_{ds}_{n}_{tag(c)}_seed{seed}.json'


def finished(p):
    try:
        return bool(json.load(open(p)).get('done'))
    except Exception:
        return False


def best_val(p):
    return max(e['val_f1'] for e in json.load(open(p))['epochs'])


def best_cfg(model, ds, n):
    a = C.anchor(n)
    return max(GRID, key=lambda c: best_val(ofile('tune', model, ds, a, c, 1)))


# ------------------------------------------------------------------ one run
def one(model, ds, n, s, Tm, seed, epochs, patience, out):
    from tmu.models.classification.vanilla_classifier import TMClassifier
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    from ggth.data import load_prepared, balanced_perm, metrics
    np.random.seed(seed)                                   # TMU draws from the global NumPy generator
    rng = np.random.default_rng(seed)
    D = load_prepared(ds)
    (Xtr, ytr), (Xva, yva), (Xte, yte), m = D['train'], D['val'], D['test'], D['m']
    Xtr, Xva, Xte = (np.ascontiguousarray(x, dtype=np.uint32) for x in (Xtr, Xva, Xte))
    if model == 'tmu_cotm':
        T = max(1, int(round(Tm * n)))
        tm = TMCoalescedClassifier(n, T, s, platform='CPU', seed=seed)
        total = n
    else:
        c = C.cpc(ds, n)
        T = max(1, int(round(Tm * c)))
        tm = TMClassifier(c, T, s, platform='CPU', seed=seed)
        total = c * m
    log = dict(model=model, dataset=ds, clauses=n, total_clauses=total, s=s, Tm=Tm, T=T, seed=seed,
               epochs=[], done=False, library=TMU_VERSION)
    best, bad = -1.0, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = balanced_perm(ytr, C.PCLS[ds], m, rng)
        tm.fit(Xtr[idx], ytr[idx].astype(np.uint32), shuffle=False)
        tt = time.time() - t0
        a_va, b_va, f_va = metrics(yva, tm.predict(Xva))
        a_te, b_te, f_te = metrics(yte, tm.predict(Xte))
        lit = literals(tm, model, m)
        log['epochs'].append(dict(epoch=ep, val_acc=a_va, val_bacc=b_va, val_f1=f_va, test_acc=a_te,
                                  test_bacc=b_te, test_f1=f_te, mean_literals=lit, train_s=tt))
        if f_va > best:
            best, bad, log['best_epoch'] = f_va, 0, ep
        else:
            bad += 1
        json.dump(log, open(out, 'w'))
        if bad >= patience:
            break
    log['done'] = True
    json.dump(log, open(out, 'w'))


def literals(tm, model, m):
    """Mean included literals per clause."""
    try:
        if model == 'tmu_cotm':
            return float(np.mean([tm.number_of_include_actions(j) for j in range(tm.number_of_clauses)]))
        return float(np.mean([tm.number_of_include_actions(k, j) for k in range(m) for j in range(tm.number_of_clauses)]))
    except Exception:
        return float('nan')


# ------------------------------------------------------------------ job lists
def job(phase, model, ds, n, c, seed):
    out = ofile(phase, model, ds, n, c, seed)
    if finished(out):
        return None
    P = TUNE if phase == 'tune' else FINAL                 # 'check' uses the final protocol
    return [PY, os.path.abspath(__file__), 'one', model, ds, str(n), str(c['s']), str(c['Tm']),
            str(seed), str(P['epochs']), str(P['patience']), out], out


def ours_cotm_cfg(ds, n):
    import es
    c = es.best_cfg('cotm', ds, n)
    return dict(s=c['s'], Tm=c['Tm'])


def tune_jobs():
    """Vanilla TM tuning, interleaved with the CoTM implementation check (long runs first)."""
    chk = [j for ds in C.DS_ES for n in CHECK[ds] for s in C.SEEDS
           if (j := job('check', 'tmu_cotm', ds, n, ours_cotm_cfg(ds, n), s))]
    tun = [j for a in (1000, 200) for ds in C.DS_ES for c in GRID if (j := job('tune', 'tmu_tm', ds, a, c, 1))]
    return chk + tun


def final_jobs():
    return [j for n in C.BUDGETS for ds in C.DS_ES for s in C.SEEDS
            if (j := job('final', 'tmu_tm', ds, n, best_cfg('tmu_tm', ds, n), s))]


def execute(jobs, workers, dry):
    from concurrent.futures import ThreadPoolExecutor
    print(len(jobs), 'runs', flush=True)
    if dry:
        return
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(j):
        cmd, out = j
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out):
            os.remove(out)                                 # no checkpoints in TMU: restart unfinished runs
        with open(out.replace('.json', '.log'), 'a') as f:
            return out, subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode

    with ThreadPoolExecutor(workers) as ex:
        for out, rc in ex.map(run, jobs):
            print('done' if rc == 0 else f'FAILED ({rc})', os.path.relpath(out, C.ROOT), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'one':
        a = sys.argv[2:]
        one(a[0], a[1], int(a[2]), float(a[3]), float(a[4]), int(a[5]), int(a[6]), int(a[7]), a[8])
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['tune', 'final'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dry', action='store_true')
    A = ap.parse_args()
    execute(tune_jobs() if A.phase == 'tune' else final_jobs(), A.workers, A.dry)
