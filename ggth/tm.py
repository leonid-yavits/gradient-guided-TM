"""Tsetlin Machine clause bank with pluggable readouts.

Readouts
  cotm       : Coalesced TM update (Glimsdal and Granmo, arXiv:2108.07594), integer weights.
  gg_lin     : gradient-guided (GG) TA feedback with a real-valued linear readout trained by Adam.
  gg_hidtern : GG feedback with a ternary hidden layer (ternary weights and activations)
               and a ternary output layer. This is the GG-TH model of the paper.

GG clause feedback, per sample and clause:
  g_j  = d log p_y / d c_j through the readout (straight-through estimator for quantizers)
  pi_j = (1 - p_y) * min(1, |g_j| / mean_i |g_i|)
  clause j is selected with probability pi_j; Type I if g_j > 0, Type II if g_j < 0.
"""
import numpy as np


# ------------------------------------------------------------------ clause layer
class ClauseBank:
    """n clauses over 2*o literals. One TA per (clause, literal), states 1..2N, include if state > N."""

    def __init__(self, n_clauses, n_feat, N=128, s=10.0, rng=None):
        self.n, self.o, self.N, self.s = n_clauses, n_feat, N, s
        self.rng = rng
        self.C = np.full((n_clauses, 2 * n_feat), N, dtype=np.int16)  # all literals excluded at the boundary

    def include(self):
        return self.C > self.N

    @staticmethod
    def literals(Xb):
        X = Xb.astype(np.float32)
        return np.concatenate([X, 1.0 - X], axis=1)

    def evaluate(self, Lit, train=True):
        inc = self.include().astype(np.float32)
        viol = (1.0 - Lit) @ inc.T                   # number of included literals that are false
        c = (viol < 0.5).astype(np.float32)
        if not train:                                # empty clauses output 0 at inference
            c[:, inc.sum(1) == 0] = 0.0
        return c

    def feedback(self, Lit, c, SI, SII):
        """Batched TA update. SI, SII: (B, n) {0,1} selections for Type I and Type II feedback."""
        A = SI * c                                   # firing clauses selected for Type I
        dIa = A.T @ Lit                              # Type Ia: +1 for true literals (boost true positive)
        M_ib = SI.sum(0)[:, None] - dIa              # Type Ib opportunities
        dec = np.floor(M_ib / self.s + self.rng.random(M_ib.shape, dtype=np.float32))
        dII = (SII * c).T @ (1.0 - Lit)              # Type II: +1 for false literals ...
        dII *= (self.C <= self.N)                    # ... that are currently excluded
        newC = self.C.astype(np.float32) + dIa - dec + dII
        np.clip(newC, 1, 2 * self.N, out=newC)
        self.C = newC.astype(np.int16)

    def mean_literals(self):
        return float(self.include().sum(1).mean())


# ------------------------------------------------------------------ helpers
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ternarize(Wr, frac=0.7):
    """Ternary weight network rule: sign(W) * 1[|W| > 0.7 mean|W|], per tensor."""
    delta = frac * np.mean(np.abs(Wr))
    return np.sign(Wr) * (np.abs(Wr) > delta)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ------------------------------------------------------------------ readouts
class CoTMReadout:
    """Coalesced TM update (Glimsdal and Granmo 2021, arXiv:2108.07594, App. A.2)."""

    def __init__(self, n, m, T, rng):
        self.W = rng.choice([-1, 1], size=(m, n)).astype(np.float32)
        self.T, self.m, self.rng = T, m, rng

    def scores(self, c):
        return c @ self.W.T

    def train_step(self, c, y):
        B, n = c.shape
        v = np.clip(self.scores(c), -self.T, self.T)
        q = (y + self.rng.integers(1, self.m, size=B)) % self.m    # one random negative class
        ar = np.arange(B)
        d_y = (self.T - v[ar, y]) / (2 * self.T)
        d_q = (self.T + v[ar, q]) / (2 * self.T)
        Wy, Wq = self.W[y], self.W[q]
        sel_y = (self.rng.random((B, n)) <= d_y[:, None]).astype(np.float32)
        sel_q = (self.rng.random((B, n)) <= d_q[:, None]).astype(np.float32)
        pos_y, pos_q = (Wy >= 0), (Wq >= 0)
        SI = np.clip(sel_y * pos_y + sel_q * (~pos_q), 0, 1)
        SII = np.clip(sel_y * (~pos_y) + sel_q * pos_q, 0, 1)
        SII = SII * (1 - SI)                                        # avoid double feedback
        np.add.at(self.W, y, sel_y * c)                             # target class: +c_j
        np.add.at(self.W, q, -sel_q * c)                            # negative class: -c_j
        return SI, SII


class GGReadout:
    """Differentiable readout trained by Adam. Its clause gradients drive GG feedback."""

    def __init__(self, kind, n, m, rng, H=256, lr=2e-3):
        assert kind in ('gg_lin', 'gg_hidtern'), kind
        self.kind, self.n, self.m, self.H, self.rng = kind, n, m, H, rng
        sc = 1.0 / np.sqrt(n)
        if kind == 'gg_lin':
            self.W = (rng.standard_normal((m, n)) * sc).astype(np.float32)
            self.b = np.zeros(m, np.float32)
            params = [self.W, self.b]
        else:
            self.W1 = (rng.standard_normal((H, n)) * sc).astype(np.float32)
            self.b1 = np.zeros(H, np.float32)
            self.W2 = (rng.standard_normal((m, H)) * (1 / np.sqrt(H))).astype(np.float32)
            self.b2 = np.zeros(m, np.float32)
            self.a1 = np.array([1.0 / np.sqrt(n)], np.float32)      # pre-activation scale alpha_1
            self.a2 = np.array([4.0 / np.sqrt(H)], np.float32)      # logit scale alpha_2
            params = [self.W1, self.b1, self.W2, self.b2, self.a1, self.a2]
        self.opt = Adam(params, lr=lr)

    def forward(self, c):
        if self.kind == 'gg_lin':
            return c @ self.W.T + self.b, None
        W1t, W2t = ternarize(self.W1), ternarize(self.W2)
        pre = c @ W1t.T                                             # integer signed popcount
        a = self.a1[0] * pre + self.b1
        h = np.where(a > 0.5, 1.0, np.where(a < -0.5, -1.0, 0.0)).astype(np.float32)
        z = self.a2[0] * (h @ W2t.T) + self.b2
        return z, (W1t, W2t, pre, a, h)

    def scores(self, c):
        return self.forward(c)[0]

    def train_step(self, c, y):
        B = c.shape[0]
        z, cache = self.forward(c)
        p = softmax(z)
        Y = np.zeros_like(p)
        Y[np.arange(B), y] = 1
        dz = (p - Y) / B                                            # d(mean CE)/dz
        if self.kind == 'gg_lin':
            gW = dz.T @ c
            gb = dz.sum(0)
            dc = dz @ self.W
            self.opt.step([gW, gb])
        else:
            W1t, W2t, pre, a, h = cache
            hw = h @ W2t.T
            gW2 = self.a2[0] * (dz.T @ h)
            gb2 = dz.sum(0)
            ga2 = np.array([np.sum(dz * hw)], np.float32)
            dh = self.a2[0] * (dz @ W2t)
            da = dh * (np.abs(a) < 1.5)                             # STE window
            gW1 = self.a1[0] * (da.T @ c)
            gb1 = da.sum(0)
            ga1 = np.array([np.sum(da * pre)], np.float32)
            dc = self.a1[0] * (da @ W1t)
            self.opt.step([gW1, gb1, gW2, gb2, ga1, ga2])
            self.a1[0] = max(self.a1[0], 1e-4)
            self.a2[0] = max(self.a2[0], 1e-4)
        # GG feedback: g = d log p_y / dc = -B * dc
        g = -dc * B
        d = 1.0 - p[np.arange(B), y]
        mag = np.abs(g)
        norm = mag.mean(1, keepdims=True) + 1e-12
        prob = d[:, None] * np.minimum(1.0, mag / norm)
        sel = (self.rng.random(prob.shape, dtype=np.float32) <= prob).astype(np.float32)
        return sel * (g > 0), sel * (g < 0)


def make_readout(kind, n, m, rng, T=None, H=256, lr=2e-3):
    if kind == 'cotm':
        return CoTMReadout(n, m, T if T is not None else 2.5 * n, rng)
    return GGReadout(kind, n, m, rng, H=H, lr=lr)
