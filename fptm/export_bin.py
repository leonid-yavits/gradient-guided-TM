"""Export the prepared splits to raw binaries read by fptm_run.jl.

Writes <bin>/<ds>_B{tr,va,te}.u8 (uint8, row-major n x o), <bin>/<ds>_y{tr,va,te}.i64 (int64)
and <bin>/<ds>.json with the class count m and the split shapes.

  python fptm/export_bin.py wos alerce
"""
import json
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ggth.data import load_prepared

BIN = os.environ.get('GGTH_BIN', os.path.join(os.path.dirname(__file__), '..', 'data', 'bin'))


def export(ds):
    os.makedirs(BIN, exist_ok=True)
    d = load_prepared(ds)
    info = {'m': d['m']}
    for tag, key in (('tr', 'train'), ('va', 'val'), ('te', 'test')):
        X, y = d[key]
        np.ascontiguousarray(X, dtype=np.uint8).tofile(f'{BIN}/{ds}_B{tag}.u8')
        np.ascontiguousarray(y, dtype=np.int64).tofile(f'{BIN}/{ds}_y{tag}.i64')
        info[tag] = list(X.shape)
    json.dump(info, open(f'{BIN}/{ds}.json', 'w'))
    print(ds, info)


if __name__ == '__main__':
    for ds in sys.argv[1:] or ('wos', 'alerce'):
        export(ds)
