"""Exp 22 pre-registration amendment: what do Gumbel (tau=0.5) and Dirichlet (gamma=1) actually feed at forks?

Mirrors neuralese_r1.concept_token: Gumbel = softmax((log w + g) / tau); Dirichlet = Dir(gamma * w).
Two-candidate Gumbel has a closed form. With d = ln(w1/w2) and L = g1 - g2 ~ Logistic(0, 1):
    fed top >= 0.9  <=>  |d + L| >= tau * ln 9  (= ln 3 at tau = 0.5)
    P = 1 - sigmoid(tau*ln9 - d) + sigmoid(-tau*ln9 - d)
so P is ~0.50 at a 50/50 fork and ~0.60 at a 75/25 fork, below the 70% threshold in P1. Run: python exp22_closed_form.py
"""
import math

import numpy as np

TAU, GAMMA, N = 0.5, 1.0, 200_000
rng = np.random.default_rng(0)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def gumbel_closed_form(w1, w2):
    d, t = math.log(w1 / w2), TAU * math.log(9)
    return 1 - sigmoid(t - d) + sigmoid(-t - d)


def gumbel(w):
    z = (np.log(w) + rng.gumbel(size=(N, len(w)))) / TAU
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def dirichlet(w):
    return rng.dirichlet(np.maximum(np.asarray(w) * GAMMA, 1e-6), size=N)


cases = {"0.50/0.50": [.5, .5], "0.60/0.40": [.6, .4], "0.70/0.30": [.7, .3], "0.75/0.25": [.75, .25],
         "0.55/0.30/0.15": [.55, .3, .15], "0.45/0.30/0.25": [.45, .3, .25]}
print(f"{'fork (pre-noise)':16s} | Gumbel P(top>=.9): sim  closed | median fed top | Dirichlet P(top>=.9)")
for name, w in cases.items():
    w = np.array(w)
    gt, dt = gumbel(w).max(1), dirichlet(w).max(1)
    cf = gumbel_closed_form(w[0], w[1]) if len(w) == 2 else float("nan")
    if len(w) == 2:  # check: simulation agrees with the closed form
        assert abs((gt >= .9).mean() - cf) < 0.01, name
    print(f"{name:16s} | {(gt >= .9).mean():22.2f} {cf:6.2f} | {np.median(gt):14.2f} | {(dt >= .9).mean():20.2f}")
