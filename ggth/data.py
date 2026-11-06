"""Prepared TreeTM splits, class-balanced sampling and metrics."""
import json
import os
import numpy as np

DATA_ROOT = os.environ.get('GGTH_DATA', os.path.join(os.path.dirname(__file__), '..', 'data', 'TreeTM_datasets'))


def load_prepared(name, root=None):
    """Booleanized inputs and leaf labels: train, validation and test splits, plus the class count."""
    root = root or DATA_ROOT
    P = np.load(os.path.join(root, name, 'prepared.npz'))
    meta = json.load(open(os.path.join(root, name, 'meta.json')))
    leaves = meta['leaves']
    if isinstance(leaves, str):
        import ast
        leaves = ast.literal_eval(leaves)
    return dict(train=(P['Btr'], P['ytr']), val=(P['Bva'], P['yva']), test=(P['Bte'], P['yte']), m=len(leaves))


def balanced_perm(y, P, m, rng):
    """P samples per class per epoch, drawn with replacement only when the class is smaller than P."""
    idx = []
    for k in range(m):
        ik = np.flatnonzero(y == k)
        if len(ik) == 0:
            continue
        idx.append(rng.choice(ik, P, replace=len(ik) < P))
    idx = np.concatenate(idx)
    return idx[rng.permutation(len(idx))]


def metrics(y, yp):
    """Accuracy, balanced accuracy (macro recall) and macro F1 over the classes present in y."""
    rec, f1 = [], []
    for k in np.unique(y):
        tp = np.sum((yp == k) & (y == k))
        fn = np.sum((yp != k) & (y == k))
        fp = np.sum((yp == k) & (y != k))
        r = tp / (tp + fn)
        p = tp / (tp + fp) if tp + fp else 0.0
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(yp == y)), float(np.mean(rec)), float(np.mean(f1))
