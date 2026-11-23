"""Clause-free ternary network with the same shape and memory as GG-TH.

The clause layer of GG-TH is replaced by a dense ternary layer of n units over the o input bits:
  x in {0,1}^o (the Boolean input; inputs are sparse, so a +-1 encoding would add a large constant to every unit)
  h0 = tern(alpha0 * (W0t x) + b0)        W0t in {-1,0,+1}^{n x o}   (2 bits per weight = 2o bits per unit,
                                                                       the size of one clause include mask)
  h1 = tern(alpha1 * (W1t h0) + b1)        W1t in {-1,0,+1}^{H x n}
  z  = alpha2 * (W2t h1) + b2              W2t in {-1,0,+1}^{m x H}
tern(a) = +1 if a > 1/2, -1 if a < -1/2, else 0. Weights are ternarized with the TWN rule.
All parameters are trained end to end with Adam and straight-through estimators (window |a| < 3/2).
alpha0 and b0 are initialized from the first training batch so that the first-layer pre-activations
have zero mean per unit and unit variance (data-dependent initialization).
Dense memory equals GG-TH: 2*o*n + 2*H*n + 2*m*H bits (plus thresholds and biases).
"""
import numpy as np
from .tm import ternarize, softmax, Adam


def tern(a):
    return np.where(a > 0.5, 1.0, np.where(a < -0.5, -1.0, 0.0)).astype(np.float32)


class TernaryNet:
    def __init__(self, o, n, m, rng, H=256, lr=2e-3):
        self.o, self.n, self.m, self.H, self.rng = o, n, m, H, rng
        self.W0 = (rng.standard_normal((n, o)) / np.sqrt(o)).astype(np.float32)
        self.b0 = np.zeros(n, np.float32)
        self.W1 = (rng.standard_normal((H, n)) / np.sqrt(n)).astype(np.float32)
        self.b1 = np.zeros(H, np.float32)
        self.W2 = (rng.standard_normal((m, H)) / np.sqrt(H)).astype(np.float32)
        self.b2 = np.zeros(m, np.float32)
        self.a0 = np.array([1.0 / np.sqrt(o)], np.float32)
        self.a1 = np.array([1.0 / np.sqrt(n)], np.float32)
        self.a2 = np.array([4.0 / np.sqrt(H)], np.float32)
        self.initialized = False
        self.opt = Adam([self.W0, self.b0, self.W1, self.b1, self.W2, self.b2, self.a0, self.a1, self.a2], lr=lr)

    def forward(self, Xb):
        x = Xb.astype(np.float32)
        W0t, W1t, W2t = ternarize(self.W0), ternarize(self.W1), ternarize(self.W2)
        u0 = x @ W0t.T
        p0 = self.a0[0] * u0 + self.b0
        h0 = tern(p0)
        u1 = h0 @ W1t.T
        p1 = self.a1[0] * u1 + self.b1
        h1 = tern(p1)
        v = h1 @ W2t.T
        z = self.a2[0] * v + self.b2
        return z, (x, W0t, W1t, W2t, u0, p0, h0, u1, p1, h1, v)

    def scores(self, Xb):
        return self.forward(Xb)[0]

    def train_step(self, Xb, y):
        B = Xb.shape[0]
        if not self.initialized:                       # data-dependent init of the first layer
            u0 = Xb.astype(np.float32) @ ternarize(self.W0).T
            self.a0[0] = 1.0 / (u0.std() + 1e-6)
            self.b0[:] = -self.a0[0] * u0.mean(0)
            self.initialized = True
        z, (x, W0t, W1t, W2t, u0, p0, h0, u1, p1, h1, v) = self.forward(Xb)
        p = softmax(z)
        Y = np.zeros_like(p)
        Y[np.arange(B), y] = 1
        dz = (p - Y) / B
        gW2 = self.a2[0] * (dz.T @ h1)
        gb2 = dz.sum(0)
        ga2 = np.array([np.sum(dz * v)], np.float32)
        dp1 = self.a2[0] * (dz @ W2t) * (np.abs(p1) < 1.5)
        gW1 = self.a1[0] * (dp1.T @ h0)
        gb1 = dp1.sum(0)
        ga1 = np.array([np.sum(dp1 * u1)], np.float32)
        dp0 = self.a1[0] * (dp1 @ W1t) * (np.abs(p0) < 1.5)
        gW0 = self.a0[0] * (dp0.T @ x)
        gb0 = dp0.sum(0)
        ga0 = np.array([np.sum(dp0 * u0)], np.float32)
        self.opt.step([gW0, gb0, gW1, gb1, gW2, gb2, ga0, ga1, ga2])
        for a in (self.a0, self.a1, self.a2):
            a[0] = max(a[0], 1e-4)

    def mean_nonzero(self):
        """Mean number of nonzero first-layer weights per unit (the analogue of included literals per clause)."""
        return float((ternarize(self.W0) != 0).sum(1).mean())
