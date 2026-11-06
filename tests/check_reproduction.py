"""Check that the code reproduces the stored logs bit for bit (first epoch of three stored runs).

  GGTH_DATA=/path/to/TreeTM_datasets python tests/check_reproduction.py

Takes about one minute on one CPU core. Exit code 0 if every metric matches exactly.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CASES = [  # train.py arguments, stored log
    (['--dataset', 'wos', '--readout', 'gg_hidtern', '--clauses', '200', '--s', '10', '--lr', '0.003', '--H', '256', '--P', '300', '--seed', '1'],
     'results/budget/ours/final_wos_200_s10_lr0.003_H256_seed1.json'),
    (['--dataset', 'wos', '--readout', 'cotm', '--clauses', '1000', '--P', '300', '--seed', '1'],
     'results/ablation/wos_cotm_1000_s1.json'),
    (['--dataset', 'wos', '--readout', 'gg_lin', '--clauses', '1000', '--P', '300', '--seed', '2'],
     'results/ablation/wos_gg_lin_1000_s2.json'),
]

ok = True
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
with tempfile.TemporaryDirectory() as tmp:
    for i, (args, ref) in enumerate(CASES):
        out = os.path.join(tmp, f'{i}.json')
        subprocess.run([sys.executable, os.path.join(ROOT, 'train.py'), *args, '--epochs', '1', '--out', out],
                       check=True, stdout=subprocess.DEVNULL, env=env)
        a = json.load(open(out))['epochs'][0]
        b = json.load(open(os.path.join(ROOT, ref)))['epochs'][0]
        same = all(a[k] == b[k] for k in b if k != 'sec')
        ok &= same
        print('MATCH   ' if same else 'MISMATCH', ref)
sys.exit(0 if ok else 1)
