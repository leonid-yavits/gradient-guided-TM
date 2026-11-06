"""Run the experiments of the paper. Completed runs are skipped, so an interrupted phase can be restarted.

  python experiments/run.py ablation        # Table 2: CoTM, GG-linear, GG-TH at n = 1000, fixed settings
  python experiments/run.py tune            # validation search: GG-TH at the anchors, FPTM at every budget
  python experiments/run.py final           # selected settings, 20 epochs, seeds 1 to 3, both models
  python experiments/run.py all             # the three phases in order

Options: --workers K (parallel runs, default 2), --dry (print the commands only).
FPTM runs need Julia, the FPTM sources (FPTM_SRC) and the exported binaries (fptm/export_bin.py).
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import config as C

PY = [sys.executable, os.path.join(C.ROOT, 'train.py')]
JL = [os.environ.get('JULIA', 'julia'), '-t', '1', os.path.join(C.ROOT, 'fptm', 'fptm_run.jl')]


def ours_cmd(ds, n, readout, s, lr, H, epochs, seed, out):
    return PY + ['--dataset', ds, '--readout', readout, '--clauses', str(n), '--T', str(2.5 * n),
                 '--s', str(s), '--lr', str(lr), '--H', str(H), '--P', str(C.PCLS[ds]),
                 '--epochs', str(epochs), '--seed', str(seed), '--out', out]


def ours_job(tag, ds, n, cfg, epochs, seed):
    out = C.ours_file(tag, ds, n, cfg, seed)
    if C.done(out, epochs):
        return None
    return ours_cmd(ds, n, 'gg_hidtern', cfg['s'], cfg['lr'], cfg['H'], epochs, seed, out), out


def fptm_job(tag, ds, n, cfg, epochs, seed):
    out = C.fptm_file(tag, ds, n, cfg, seed)
    if C.done(out, epochs):
        return None
    cmd = JL + [ds, str(C.cpc(ds, n)), str(cfg['T']), str(cfg['S']), str(cfg['L']), str(cfg['LF']),
                str(epochs), str(C.PCLS[ds]), str(seed), out]
    return cmd, out


def ablation_jobs():
    jobs = []
    for ds in C.DS:
        for r in ('cotm', 'gg_lin', 'gg_hidtern'):
            for seed in C.SEEDS:
                out = f'{C.R_ABL}/{ds}_{r}_1000_s{seed}.json'
                if not C.done(out, C.FINAL_EPOCHS):
                    jobs.append((ours_cmd(ds, 1000, r, 10, 2e-3, 256, C.FINAL_EPOCHS, seed, out), out))
    return jobs


def tune_jobs():
    jobs = []
    for ds in C.DS:
        for n in C.ANCHORS:
            jobs += [ours_job('tune', ds, n, cfg, C.TUNE_EPOCHS, 1) for cfg in C.OURS_GRID]
        for n in C.fptm_sizes(ds):
            jobs += [fptm_job('tune', ds, n, cfg, C.TUNE_EPOCHS, 1) for cfg in C.fptm_grid(C.cpc(ds, n))]
    return [j for j in jobs if j]


def final_jobs():
    jobs = []
    for ds in C.DS:
        sizes = C.fptm_sizes(ds)
        for n in C.BUDGETS:
            co = C.ours_best(ds, n)
            print(ds, n, 'GG-TH', co, flush=True)
            jobs += [ours_job('final', ds, n, co, C.FINAL_EPOCHS, s) for s in C.SEEDS]
            if n in sizes:
                cf = C.fptm_best(ds, n)
                print(ds, n, 'FPTM clauses/class', C.cpc(ds, n), cf, flush=True)
                jobs += [fptm_job('final', ds, n, cf, C.FINAL_EPOCHS, s) for s in C.SEEDS]
    return [j for j in jobs if j]


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
        with open(out.replace('.json', '.log'), 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        return out, rc

    with ThreadPoolExecutor(workers) as ex:
        for out, rc in ex.map(one, jobs):
            print('done' if rc == 0 else f'FAILED ({rc})', os.path.relpath(out, C.ROOT), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('phase', choices=['ablation', 'tune', 'final', 'all'])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for ph in (('ablation', 'tune', 'final') if a.phase == 'all' else (a.phase,)):
        execute({'ablation': ablation_jobs, 'tune': tune_jobs, 'final': final_jobs}[ph](), a.workers, a.dry)
