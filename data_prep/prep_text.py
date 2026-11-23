"""Binary bag-of-words encodings of R52, Sector, RCV1 and 20 Newsgroups for the GG-TH experiments.

The encoders follow the QMTM study (https://github.com/leonid-yavits/QMTM, study1/prep_reuters.py and
study1/prep_more.py). The only change is the vocabulary size V = 2000, which matches WOS.

  r52     Reuters-21578 ApteMod (NLTK copy), single-topic documents; R52 = topics with at least one such training and
          one such test document (Cardoso-Cachopo's definition). Lowercase alphabetic tokens of length >= 2, NLTK English
          stopwords removed, document frequency >= 2 in the reduced training split. Validation: stratified 15 % of the
          training split (seed 0).
  sector, news20, rcv1
          LIBSVM multiclass files (sector.scale, news20, rcv1_train/test.multiclass). Binary presence of each feature.
          Validation: stratified 10 % of the training file (seed 0). rcv1: the official 518,571-document test file is
          subsampled to 50,000 documents (seed 0). Test documents of classes absent from the training file are dropped.
  All:    classes with a single training document stay in the training part. Chi-square selection of V features on the
          reduced training part only. Labels are renumbered 0..m-1 in sorted order of the training labels.

Writes <GGTH_DATA>/<name>/prepared.npz (Btr, ytr, Bva, yva, Bte, yte) and meta.json, the format of ggth.data.

  python data_prep/prep_text.py <dl_dir> <raw_dir> r52 sector rcv1 news20
dl_dir holds the downloaded LIBSVM .bz2 files; raw_dir holds the extracted NLTK corpora (reuters/, stopwords/).
"""
import bz2
import collections
import json
import os
import re
import sys
import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

V = 2000
OUT = os.environ.get('GGTH_DATA', os.path.join(os.path.dirname(__file__), '..', 'data', 'TreeTM_datasets'))


def split_val(y, frac, seed=0):
    cnt = collections.Counter(y)
    idx = np.arange(len(y))
    rare = np.array([cnt[v] < 2 for v in y])
    a, b = train_test_split(idx[~rare], test_size=frac, random_state=seed, stratify=y[~rare])
    return np.concatenate([a, idx[rare]]), b


def save(name, Xa, ya, Xb, yb, Xt, yt, classes, source):
    sc, _ = chi2(Xa, ya)
    sel = np.sort(np.argsort(-np.nan_to_num(sc), kind='stable')[:V])
    f = lambda X: np.asarray(X[:, sel].todense(), dtype=np.uint8)
    o = dict(Btr=f(Xa), ytr=ya.astype(np.int64), Bva=f(Xb), yva=yb.astype(np.int64), Bte=f(Xt), yte=yt.astype(np.int64))
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    np.savez(os.path.join(OUT, name, 'prepared.npz'), **o)
    meta = dict(name=name, leaves=[str(c) for c in classes], features=V, source=source,
                sizes={k: int(o['y' + k].shape[0]) for k in ('tr', 'va', 'te')},
                density=float(o['Btr'].mean()), active_per_sample=float(o['Btr'].sum(1).mean()),
                empty_test_docs=int((o['Bte'].sum(1) == 0).sum()),
                train_class_min=int(np.bincount(o['ytr'], minlength=len(classes)).min()),
                train_class_max=int(np.bincount(o['ytr'], minlength=len(classes)).max()))
    json.dump(meta, open(os.path.join(OUT, name, 'meta.json'), 'w'), indent=1)
    print(name, {k: v for k, v in meta.items() if k != 'leaves'}, 'classes', len(classes), flush=True)


def libsvm(name, dl):
    fn = {'news20': ('news20.bz2', 'news20.t.bz2'), 'sector': ('sector.scale.bz2', 'sector.t.scale.bz2'),
          'rcv1': ('rcv1_train.multiclass.bz2', 'rcv1_test.multiclass.bz2')}[name]
    Xtr, ytr = load_svmlight_file(bz2.open(os.path.join(dl, fn[0])))
    Xte, yte = load_svmlight_file(bz2.open(os.path.join(dl, fn[1])), n_features=Xtr.shape[1])
    if Xte.shape[1] > Xtr.shape[1]:
        Xte = Xte[:, :Xtr.shape[1]]
    classes = np.unique(ytr)
    lab = {v: i for i, v in enumerate(classes)}
    ytr = np.array([lab[v] for v in ytr])
    keep = np.array([v in lab for v in yte])
    print(name, 'test documents of classes absent from training:', int((~keep).sum()), flush=True)
    Xte, yte = Xte[keep], np.array([lab[v] for v in yte[keep]])
    if name == 'rcv1':
        s = np.sort(np.random.default_rng(0).choice(len(yte), 50000, replace=False))
        Xte, yte = Xte[s], yte[s]
    Xtr.data[:] = 1
    Xte.data[:] = 1
    Xtr, Xte = Xtr.tocsr().astype(np.uint8), Xte.tocsr().astype(np.uint8)
    a, b = split_val(ytr, 0.1)
    save(name, Xtr[a], ytr[a], Xtr[b], ytr[b], Xte, yte, [c if c != int(c) else int(c) for c in classes],
         f'LIBSVM {fn[0]}, {fn[1]}')


def r52(raw):
    R = os.path.join(raw, 'reuters')
    stop = set(open(os.path.join(raw, 'stopwords', 'english')).read().split())
    cats = {}
    for line in open(os.path.join(R, 'cats.txt')):
        f = line.split()
        cats[f[0]] = f[1:]
    single = {d: c[0] for d, c in cats.items() if len(c) == 1}
    tr = collections.Counter(c for d, c in single.items() if d.startswith('training/'))
    te = collections.Counter(c for d, c in single.items() if d.startswith('test/'))
    keep = sorted(c for c in tr if te[c] > 0)
    D = sorted(d for d, c in single.items() if c in keep)
    lab = {c: i for i, c in enumerate(keep)}
    trd = [d for d in D if d.startswith('training/')]
    ted = [d for d in D if d.startswith('test/')]
    ytr_all = np.array([lab[single[d]] for d in trd])
    yte = np.array([lab[single[d]] for d in ted])
    text = lambda d: open(os.path.join(R, d), encoding='latin-1').read()
    tok = lambda s: [w for w in re.findall(r'[a-z]{2,}', s.lower()) if w not in stop]
    a, b = split_val(ytr_all, 0.15)
    cv = CountVectorizer(analyzer=tok, binary=True, min_df=2)
    Xa = cv.fit_transform([text(trd[i]) for i in a])
    Xb = cv.transform([text(trd[i]) for i in b])
    Xt = cv.transform([text(d) for d in ted])
    save('r52', Xa, ytr_all[a], Xb, ytr_all[b], Xt, yte, keep, 'Reuters-21578 ApteMod, NLTK copy')


if __name__ == '__main__':
    dl, raw = sys.argv[1], sys.argv[2]
    for nm in sys.argv[3:]:
        r52(raw) if nm == 'r52' else libsvm(nm, dl)
