"""Train a Tsetlin Machine with a CoTM, GG-linear or GG-TH readout on a prepared TreeTM dataset.

Example (GG-TH, WOS, 200 clauses):
  python train.py --dataset wos --readout gg_hidtern --clauses 200 --s 10 --lr 3e-3 --H 256 \
                  --P 300 --epochs 20 --seed 1 --out results/run.json

Every epoch logs validation and test accuracy, balanced accuracy and macro F1, the mean number of
included literals per clause and the epoch time. No model selection is done on the test split.
"""
import argparse
import json
import os
import pickle
import time
import numpy as np
from ggth.tm import ClauseBank, make_readout
from ggth.data import load_prepared, balanced_perm, metrics
from ggth.tnn import TernaryNet


def predict(bank, ro, X, chunk=2000, rounded=False):
    out = []
    for j in range(0, len(X), chunk):
        if bank is None:                                   # clause-free ternary network
            z = ro.scores(X[j:j + chunk])
        else:
            c = bank.evaluate(ClauseBank.literals(X[j:j + chunk]), train=False)
            z = ro.scores_rounded(c) if rounded else ro.scores(c)
        out.append(np.argmax(z, 1))
    return np.concatenate(out)


def _save(path, state):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def run(args):
    ckpt = args.out + '.ckpt' if args.out and args.resume else None
    d = load_prepared(args.dataset, args.data)
    (Xtr, ytr), (Xva, yva), (Xte, yte), m = d['train'], d['val'], d['test'], d['m']
    if ckpt and os.path.exists(ckpt):                      # resume an interrupted run bit-exactly
        with open(ckpt, 'rb') as f:
            st = pickle.load(f)
        bank, ro, rng, log = st['bank'], st['ro'], st['rng'], st['log']
        print(f'resumed after epoch {len(log["epochs"])}', flush=True)
    else:
        rng = np.random.default_rng(args.seed)
        if args.readout == 'tnn':
            bank, ro = None, TernaryNet(Xtr.shape[1], args.clauses, m, rng, H=args.H, lr=args.lr)
        else:
            bank = ClauseBank(args.clauses, Xtr.shape[1], N=args.N, s=args.s, rng=rng)
            ro = make_readout(args.readout, args.clauses, m, rng, T=args.T, H=args.H, lr=args.lr)
        log = {'args': vars(args), 'epochs': []}
    best = max([(e['val_f1'], e['epoch']) for e in log['epochs']], default=(-1.0, 0))
    for ep in range(len(log['epochs']), args.epochs):
        if args.patience and ep - best[1] >= args.patience:
            break
        t0 = time.time()
        perm = balanced_perm(ytr, args.P, m, rng) if args.P else rng.permutation(len(ytr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            if bank is None:
                ro.train_step(Xtr[idx], ytr[idx])
                continue
            Lit = ClauseBank.literals(Xtr[idx])
            c = bank.evaluate(Lit, train=True)
            SI, SII = ro.train_step(c, ytr[idx])
            bank.feedback(Lit, c, SI, SII)
        rec = {'epoch': ep + 1, 'sec': time.time() - t0,
               'mean_literals': ro.mean_nonzero() if bank is None else bank.mean_literals()}
        for split, X, y in (('val', Xva, yva), ('test', Xte, yte)):
            a, ba, f = metrics(y, predict(bank, ro, X))
            rec.update({f'{split}_acc': a, f'{split}_bacc': ba, f'{split}_f1': f})
            if args.readout == 'gg_lin':                   # post-hoc ternary rounding of the readout
                a, ba, f = metrics(y, predict(bank, ro, X, rounded=True))
                rec.update({f'{split}_acc_round': a, f'{split}_bacc_round': ba, f'{split}_f1_round': f})
        log['epochs'].append(rec)
        best = max(best, (rec['val_f1'], rec['epoch']))
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(log, f)
        if ckpt:
            _save(ckpt, {'bank': bank, 'ro': ro, 'rng': rng, 'log': log})
    log['best_epoch'] = best[1]
    log['done'] = True
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(log, f)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    return log


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='wos | alerce (folder name under the data root)')
    ap.add_argument('--data', default=None, help='data root holding <dataset>/prepared.npz (default: $GGTH_DATA or data/TreeTM_datasets)')
    ap.add_argument('--readout', default='gg_hidtern', choices=['cotm', 'gg_lin', 'gg_lintern', 'gg_hidtern', 'tnn'])
    ap.add_argument('--clauses', type=int, default=200, help='size n of the shared clause pool')
    ap.add_argument('--T', type=float, default=None, help='CoTM vote margin (default 2.5 * clauses); unused by GG readouts')
    ap.add_argument('--s', type=float, default=10.0, help='specificity')
    ap.add_argument('--N', type=int, default=128, help='TA states per action (2N states in total)')
    ap.add_argument('--H', type=int, default=256, help='hidden units of the ternary layer (gg_hidtern)')
    ap.add_argument('--lr', type=float, default=2e-3, help='Adam learning rate (GG readouts)')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--P', type=int, default=0, help='samples per class per epoch (0: one pass in random order)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='', help='JSON log, rewritten after every epoch')
    ap.add_argument('--patience', type=int, default=0, help='stop after this many epochs without a validation macro F1 gain (0: off)')
    ap.add_argument('--resume', action='store_true', help='checkpoint every epoch to <out>.ckpt and resume from it')
    return ap


if __name__ == '__main__':
    run(parser().parse_args())
