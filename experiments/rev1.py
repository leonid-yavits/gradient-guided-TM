"""Runs of the first revision (experiments/PROTOCOL_rev1.md). Every phase skips finished runs and can be restarted.

  python experiments/rev1.py PHASE [--workers 2] [--dry]

Phases, in the order of experiments/queue_rev1.sh:
  dump        GG-TL, GG-TH and CoTM, anchors 200 and 1000, seed 1, final protocol, clause masks saved (part B)
  abl_tune    CoTM clause feedback with the GG-TL readout: tuning of s (part A)
  abl_final   ablations of GG-TL at the anchors, seeds 1 to 3 (part A)
  batch       CoTM with mini-batch size 1 at n = 200 (part C)
  ext_tune    one grid step beyond every boundary optimum of GG-TL and CoTM (part D)
  ext_final   final runs where the extended grid changes the selected setting (part D)
  seeds       GG-TL and CoTM, seeds 4 to 8, all budgets (part E)
  tm_ext_tune, tm_ext_final   part D for the multi-class TM of TMU
  tmucotm_tune, tmucotm_final TMU coalesced classifier tuned at n = 200 (part G)
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config as C
import es
import tmu_check as TC

RES = os.path.join(C.ROOT, 'results_rev1')
NEW_SEEDS = (4, 5, 6, 7, 8)
ANCHORS = (1000, 200)


def rfile(part, model, phase, ds, n, cfg, seed):
    return f'{RES}/{part}/{model}/{phase}_{ds}_{n}_{es.tag(cfg)}_seed{seed}.json'


def our_cmd(model, ds, n, cfg, seed, out, prot, extra=()):
    """Command of experiments/es.py (ours_job) with an output path in results_rev1 and extra flags."""
    if es.finished(out):
        return None
    readout = 'gg_lin' if model == 'gg_linr' else model
    cmd = [sys.executable, os.path.join(C.ROOT, 'train.py'), '--dataset', ds, '--readout', readout,
           '--clauses', str(n), '--lr', str(cfg.get('lr', 2e-3)), '--H', str(cfg.get('H') or 256),
           '--P', str(C.PCLS[ds]), '--epochs', str(prot['epochs']), '--patience', str(prot['patience']),
           '--seed', str(seed), '--out', out, '--resume']
    if cfg.get('s'):
        cmd += ['--s', str(cfg['s'])]
    if 'Tm' in cfg:
        cmd += ['--T', str(cfg['Tm'] * n)]
    return cmd + list(extra), out


# ------------------------------------------------------------------ part B: clause dumps
def dump_jobs():
    jobs = []
    for a in ANCHORS:
        for ds in C.DS_ES:
            for model in ('gg_lintern', 'cotm', 'gg_hidtern'):
                c = es.best_cfg(model, ds, a)
                out = rfile('dump', model, 'final', ds, a, c, 1)
                jobs.append(our_cmd(model, ds, a, c, 1, out, es.FINAL, ['--dump', out.replace('.json', '.npz')]))
    return [j for j in jobs if j]


# ------------------------------------------------------------------ part A: ablations of GG-TL
ABL = {'no_pfac': ['--no_pfac'], 'no_mag': ['--no_mag'], 'no_bias': ['--no_bias']}
S_GRID = (1.5, 2, 5, 10, 20)


def fbcotm_cfgs(ds, a):
    g, t = es.best_cfg('gg_lintern', ds, a), es.best_cfg('cotm', ds, a)
    return [dict(s=s, lr=g['lr'], H=0, Tm=t['Tm']) for s in S_GRID]


def fbcotm_best(ds, a):
    return max(fbcotm_cfgs(ds, a), key=lambda c: es.best_val(rfile('abl', 'fb_cotm', 'tune', ds, a, c, 1)))


def abl_tune_jobs():
    jobs = []
    for a in ANCHORS:
        for ds in C.DS_ES:
            for c in fbcotm_cfgs(ds, a):
                jobs.append(our_cmd('gg_lintern', ds, a, c, 1, rfile('abl', 'fb_cotm', 'tune', ds, a, c, 1), es.TUNE,
                                    ['--fb', 'cotm']))
    return [j for j in jobs if j]


def abl_final_jobs():
    jobs = []
    for a in ANCHORS:
        for ds in C.DS_ES:
            c = fbcotm_best(ds, a)
            for s in C.SEEDS:
                jobs.append(our_cmd('gg_lintern', ds, a, c, s, rfile('abl', 'fb_cotm', 'final', ds, a, c, s), es.FINAL,
                                    ['--fb', 'cotm']))
            g = es.best_cfg('gg_lintern', ds, a)
            for name, flags in ABL.items():
                for s in C.SEEDS:
                    jobs.append(our_cmd('gg_lintern', ds, a, g, s, rfile('abl', name, 'final', ds, a, g, s), es.FINAL,
                                        flags))
    return [j for j in jobs if j]


# ------------------------------------------------------------------ part C: CoTM mini-batch size 1
BATCH_DS = ('alerce', 'r52', 'wos')


def batch_jobs():
    jobs = []
    for ds in BATCH_DS:
        c = es.best_cfg('cotm', ds, 200)
        for s in C.SEEDS:
            jobs.append(our_cmd('cotm', ds, 200, c, s, rfile('batch1', 'cotm', 'final', ds, 200, c, s), es.FINAL,
                                ['--batch', '1']))
    return [j for j in jobs if j]


# ------------------------------------------------------------------ part D: one step beyond boundary optima
STEP = {'s': {'lo': lambda v: 1.0 if v == 1.5 else v / 2, 'hi': lambda v: 2 * v},
        'lr': {'lo': lambda v: round(v / 3, 10), 'hi': lambda v: round(3 * v, 10)},
        'Tm': {'lo': lambda v: round(v / 3, 10), 'hi': lambda v: 2 * v}}


def ext_cands(grid, best):
    """Settings one step beyond the grid along every axis on which `best` lies at the boundary, the other axes
    held at the optimum."""
    out = []
    for ax in ('s', 'lr', 'Tm'):
        vals = sorted({g[ax] for g in grid if ax in g})
        if len(vals) < 3 or ax not in best:
            continue
        if best[ax] == vals[0]:
            out.append(dict(best, **{ax: STEP[ax]['lo'](best[ax])}))
        if best[ax] == vals[-1]:
            out.append(dict(best, **{ax: STEP[ax]['hi'](best[ax])}))
    return out


EXT_MODELS = ('gg_lintern', 'cotm')


def ext_best(model, ds, a):
    """Best setting over the base grid and its extension (None while extension runs are missing)."""
    base = es.best_cfg(model, ds, a)
    c = [(es.best_val(es.ofile('tune', model, ds, a, base, 1)), base)]
    for e in ext_cands(es.GRIDS[model], base):
        p = rfile('ext', model, 'tune', ds, a, e, 1)
        if not es.finished(p):
            return None
        c.append((es.best_val(p), e))
    return max(c, key=lambda x: x[0])[1]


def ext_tune_jobs():
    jobs = []
    for a in ANCHORS:
        for ds in C.DS_ES:
            for model in EXT_MODELS:
                for e in ext_cands(es.GRIDS[model], es.best_cfg(model, ds, a)):
                    jobs.append(our_cmd(model, ds, a, e, 1, rfile('ext', model, 'tune', ds, a, e, 1), es.TUNE))
    return [j for j in jobs if j]


def ext_final_jobs():
    jobs = []
    for n in C.BUDGETS:
        for ds in C.DS_ES:
            for model in EXT_MODELS:
                b = ext_best(model, ds, C.anchor(n))
                if b is None or b == es.best_cfg(model, ds, n):
                    continue
                for s in C.SEEDS:
                    jobs.append(our_cmd(model, ds, n, b, s, rfile('ext', model, 'final', ds, n, b, s), es.FINAL))
    return [j for j in jobs if j]


# ------------------------------------------------------------------ part E: seeds 4 to 8
def has_ext(model, ds, n, c):
    return all(os.path.exists(es.ofile('ext', model, ds, n, c, s)) for s in C.SEEDS)


def seed_jobs():
    jobs = []
    for n in es.ORDER:
        for ds in C.DS_ES:
            for model in ('gg_lintern', 'cotm'):
                c = es.best_cfg(model, ds, n)
                ph, prot = ('ext', es.EXT) if has_ext(model, ds, n, c) else ('final', es.FINAL)
                for s in NEW_SEEDS:
                    jobs.append(our_cmd(model, ds, n, c, s, rfile('seeds', model, ph, ds, n, c, s), prot))
    return [j for j in jobs if j]


# ------------------------------------------------------------------ TMU: part D for the TM, part G
def tmu_cmd(model, ds, n, c, seed, out, prot):
    if TC.finished(out):
        return None
    return [TC.PY, os.path.abspath(TC.__file__), 'one', model, ds, str(n), str(c['s']), str(c['Tm']),
            str(seed), str(prot['epochs']), str(prot['patience']), out], out


def tfile(part, model, phase, ds, n, c, seed):
    return f'{RES}/{part}/{model}/{phase}_{ds}_{n}_{TC.tag(c)}_seed{seed}.json'


def tm_ext_best(ds, a):
    base = TC.best_cfg('tmu_tm', ds, a)
    c = [(TC.best_val(TC.ofile('tune', 'tmu_tm', ds, a, base, 1)), base)]
    for e in ext_cands(TC.GRID, base):
        p = tfile('ext', 'tmu_tm', 'tune', ds, a, e, 1)
        if not TC.finished(p):
            return None
        c.append((TC.best_val(p), e))
    return max(c, key=lambda x: x[0])[1]


def tm_ext_tune_jobs():
    return [j for a in ANCHORS for ds in C.DS_ES for e in ext_cands(TC.GRID, TC.best_cfg('tmu_tm', ds, a))
            if (j := tmu_cmd('tmu_tm', ds, a, e, 1, tfile('ext', 'tmu_tm', 'tune', ds, a, e, 1), TC.TUNE))]


def tm_ext_final_jobs():
    jobs = []
    for n in C.BUDGETS:
        for ds in C.DS_ES:
            b = tm_ext_best(ds, C.anchor(n))
            if b is None or b == TC.best_cfg('tmu_tm', ds, n):
                continue
            jobs += [tmu_cmd('tmu_tm', ds, n, b, s, tfile('ext', 'tmu_tm', 'final', ds, n, b, s), TC.FINAL)
                     for s in C.SEEDS]
    return [j for j in jobs if j]


COTM_GRID = [dict(s=g['s'], Tm=g['Tm']) for g in es.GRIDS['cotm']]
WOS_SUB = [dict(s=s, Tm=t) for s, t in itertools.product((1.5, 2, 5), (2.5, 5.0, 10.0, 20.0))]


def tmucotm_grid(ds):
    return WOS_SUB if ds == 'wos' else COTM_GRID


def tmucotm_best(ds):
    return max(tmucotm_grid(ds), key=lambda c: TC.best_val(tfile('tmucotm', 'tmu_cotm', 'tune', ds, 200, c, 1)))


def tmucotm_tune_jobs():
    order = ('r52', 'alerce', 'rcv1', 'sector', 'news20', 'wos')
    return [j for ds in order for c in tmucotm_grid(ds)
            if (j := tmu_cmd('tmu_cotm', ds, 200, c, 1, tfile('tmucotm', 'tmu_cotm', 'tune', ds, 200, c, 1), TC.TUNE))]


def tmucotm_final_jobs():
    return [j for ds in C.DS_ES for s in C.SEEDS
            if (j := tmu_cmd('tmu_cotm', ds, 200, tmucotm_best(ds), s,
                             tfile('tmucotm', 'tmu_cotm', 'final', ds, 200, tmucotm_best(ds), s), TC.FINAL))]


PHASES = dict(dump=dump_jobs, abl_tune=abl_tune_jobs, abl_final=abl_final_jobs, batch=batch_jobs,
              ext_tune=ext_tune_jobs, ext_final=ext_final_jobs, seeds=seed_jobs, tm_ext_tune=tm_ext_tune_jobs,
              tm_ext_final=tm_ext_final_jobs, tmucotm_tune=tmucotm_tune_jobs, tmucotm_final=tmucotm_final_jobs)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=list(PHASES))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dry', action='store_true')
    A = ap.parse_args()
    J = PHASES[A.phase]()
    if A.dry:
        print(len(J), 'runs')
        for cmd, _ in J[:4]:
            print(' '.join(cmd[2:]))
    else:
        es.execute(J, A.workers, False)
