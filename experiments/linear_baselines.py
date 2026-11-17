"""Linear reference baselines on the same binarized features (experiments/PROTOCOL_rev1.md, part F).

Multinomial logistic regression and a linear SVM (one-vs-rest, squared hinge), scikit-learn. Grid: C in
{0.01, 0.1, 1, 10, 100} x class_weight in {None, 'balanced'}, selected on the validation macro F1. The selected
setting is refit with seeds 1 to 3 (the solvers are deterministic up to the seed of their internal shuffling).

  python experiments/linear_baselines.py
Writes results_rev1/linear/<model>_<ds>.json.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config as C
from ggth.data import load_prepared, metrics

OUT = os.path.join(C.ROOT, 'results_rev1', 'linear')
GRID = list(itertools.product((0.01, 0.1, 1.0, 10.0, 100.0), (None, 'balanced')))


def make(model, Cv, cw, seed):
    if model == 'logreg':
        return LogisticRegression(C=Cv, class_weight=cw, max_iter=2000, random_state=seed)
    return LinearSVC(C=Cv, class_weight=cw, max_iter=20000, random_state=seed)


def evaluate(clf, X, y):
    a, ba, f = metrics(y, clf.predict(X))
    return dict(acc=a, bacc=ba, f1=f)


def main():
    os.makedirs(OUT, exist_ok=True)
    for ds in C.DS_ES:
        d = load_prepared(ds)
        (Xtr, ytr), (Xva, yva), (Xte, yte) = d['train'], d['val'], d['test']
        Xtr, Xva, Xte = (X.astype(np.float32) for X in (Xtr, Xva, Xte))
        for model in ('logreg', 'linsvm'):
            path = os.path.join(OUT, f'{model}_{ds}.json')
            if os.path.exists(path):
                continue
            tune = []
            for Cv, cw in GRID:
                t0 = time.time()
                clf = make(model, Cv, cw, 1).fit(Xtr, ytr)
                tune.append(dict(C=Cv, class_weight=cw, val=evaluate(clf, Xva, yva), sec=time.time() - t0))
                print(ds, model, Cv, cw, round(tune[-1]['val']['f1'], 4), flush=True)
            b = max(tune, key=lambda r: r['val']['f1'])
            final = []
            for seed in C.SEEDS:
                t0 = time.time()
                clf = make(model, b['C'], b['class_weight'], seed).fit(Xtr, ytr)
                final.append(dict(seed=seed, val=evaluate(clf, Xva, yva), test=evaluate(clf, Xte, yte),
                                  sec=time.time() - t0))
            json.dump(dict(model=model, ds=ds, tune=tune, best=dict(C=b['C'], class_weight=b['class_weight']),
                           final=final, done=True), open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
