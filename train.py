"""Train a Tsetlin Machine with a CoTM, GG-linear or GG-TH readout on a prepared TreeTM dataset.

Example (GG-TH, WOS, 200 clauses):
  python train.py --dataset wos --readout gg_hidtern --clauses 200 --s 10 --lr 3e-3 --H 256 \
                  --P 300 --epochs 20 --seed 1 --out results/run.json

Every epoch logs validation and test accuracy, balanced accuracy and macro F1, the mean number of
included literals per clause and the epoch time. No model selection is done on the test split.
"""
import argparse
import json
import time
import numpy as np
from ggth.tm import ClauseBank, make_readout
from ggth.data import load_prepared, balanced_perm, metrics


def predict(bank, ro, X, chunk=2000):
    out = []
    for j in range(0, len(X), chunk):
        c = bank.evaluate(ClauseBank.literals(X[j:j + chunk]), train=False)
        out.append(np.argmax(ro.scores(c), 1))
    return np.concatenate(out)


def run(args):
    rng = np.random.default_rng(args.seed)
    d = load_prepared(args.dataset, args.data)
    (Xtr, ytr), (Xva, yva), (Xte, yte), m = d['train'], d['val'], d['test'], d['m']
    bank = ClauseBank(args.clauses, Xtr.shape[1], N=args.N, s=args.s, rng=rng)
    ro = make_readout(args.readout, args.clauses, m, rng, T=args.T, H=args.H, lr=args.lr)
    log = {'args': vars(args), 'epochs': []}
    for ep in range(args.epochs):
        t0 = time.time()
        perm = balanced_perm(ytr, args.P, m, rng) if args.P else rng.permutation(len(ytr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            Lit = ClauseBank.literals(Xtr[idx])
            c = bank.evaluate(Lit, train=True)
            SI, SII = ro.train_step(c, ytr[idx])
            bank.feedback(Lit, c, SI, SII)
        rec = {'epoch': ep + 1, 'sec': time.time() - t0, 'mean_literals': bank.mean_literals()}
        for split, X, y in (('val', Xva, yva), ('test', Xte, yte)):
            a, ba, f = metrics(y, predict(bank, ro, X))
            rec.update({f'{split}_acc': a, f'{split}_bacc': ba, f'{split}_f1': f})
        log['epochs'].append(rec)
        print(json.dumps(rec), flush=True)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(log, f)
    return log


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='wos | alerce (folder name under the data root)')
    ap.add_argument('--data', default=None, help='data root holding <dataset>/prepared.npz (default: $GGTH_DATA or data/TreeTM_datasets)')
    ap.add_argument('--readout', default='gg_hidtern', choices=['cotm', 'gg_lin', 'gg_hidtern'])
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
    return ap


if __name__ == '__main__':
    run(parser().parse_args())
