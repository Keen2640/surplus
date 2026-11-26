"""Economic scenario generator: Hull-White one-factor short rate fitted exactly to today's Treasury curve, plus a
correlated equity index.

    dr = (theta(t) - a r) dt + sigma dW

theta(t) is chosen so the model reproduces the initial discount curve. Writing r(t) = x(t) + alpha(t) with
x an Ornstein-Uhlenbeck process (x(0) = 0) and alpha(t) = f(0,t) + sigma^2 / (2 a^2) (1 - e^{-a t})^2, the pair
(x, integral of x) is Gaussian, so each time step is sampled *exactly* (no Euler discretisation error), and the
bank-account discount factor D(0,t) = exp(-integral of r) is an exact martingale against P(0,t).

Zero-coupon bond prices at any future node have the closed form
    P(t,T) = P(0,T)/P(0,t) * exp(B f(0,t) - sigma^2/(4a) (1 - e^{-2at}) B^2 - B r(t)),   B = (1 - e^{-a(T-t)})/a
which is what lets the simulator revalue the whole liability and every bond in every scenario at every step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curve import ZeroCurve


@dataclass
class HullWhite:
    curve: ZeroCurve
    a: float = 0.05  # mean reversion
    sigma: float = 0.01  # short-rate vol (about 100bp a year)

    def B(self, t, T):
        return (1 - np.exp(-self.a * (np.asarray(T) - np.asarray(t)))) / self.a

    def G(self, t):
        a = self.a
        return t - 2 * (1 - np.exp(-a * t)) / a + (1 - np.exp(-2 * a * t)) / (2 * a)

    def alpha(self, t):
        a, s = self.a, self.sigma
        return self.curve.fwd(t) + s * s / (2 * a * a) * (1 - np.exp(-a * np.asarray(t))) ** 2

    def int_alpha(self, s, t):
        """Integral of alpha(u) du from s to t."""
        P = self.curve.df
        lp = np.log(P(s)) if s > 0 else 0.0
        return lp - np.log(P(t)) + self.sigma ** 2 / (2 * self.a ** 2) * (self.G(t) - self.G(s))

    def zcb(self, t: float, T, r: np.ndarray) -> np.ndarray:
        """P(t, T) for every scenario. r: (n,), T: (m,) -> (n, m)."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if t <= 0:
            return np.broadcast_to(self.curve.df(T), (len(r), len(T))).copy()
        a, s = self.a, self.sigma
        B = self.B(t, T)
        P0t, P0T = self.curve.df(t), self.curve.df(T)
        lnA = np.log(P0T / P0t) + B * self.curve.fwd(t) - s * s / (4 * a) * (1 - np.exp(-2 * a * t)) * B * B
        return np.exp(lnA[None, :] - np.outer(r, B))


@dataclass
class Scenarios:
    times: np.ndarray  # (nt,)
    r: np.ndarray  # (n, nt) short rate at each node
    int_r: np.ndarray  # (n, nt-1) integral of r over each step
    equity_logret: np.ndarray  # (n, nt-1)
    model: HullWhite

    @property
    def n(self) -> int:
        return self.r.shape[0]

    def discount(self) -> np.ndarray:
        """Bank-account discount factor D(0, t_k) for every node, (n, nt)."""
        return np.exp(-np.concatenate([np.zeros((self.n, 1)), np.cumsum(self.int_r, axis=1)], axis=1))


def simulate(model: HullWhite, horizon: float, dt: float = 0.5, n: int = 10_000, seed: int = 0,
             equity_premium: float = 0.04, equity_vol: float = 0.16, rho: float = -0.2,
             antithetic: bool = True) -> Scenarios:
    """Simulate short rates, step integrals and equity log-returns on the grid 0, dt, ..., horizon."""
    rng = np.random.default_rng(seed)
    a, s = model.a, model.sigma
    times = np.round(np.arange(0, horizon + 1e-9, dt), 10)
    nt = len(times)
    e = np.exp(-a * dt)
    v1 = s * s * (1 - e * e) / (2 * a)
    v2 = s * s / (a * a) * (dt - 2 * (1 - e) / a + (1 - e * e) / (2 * a))
    c12 = s * s / (2 * a * a) * (1 - e) ** 2
    L = np.linalg.cholesky(np.array([[v1, c12], [c12, v2]]))

    half = (n + 1) // 2 if antithetic else n
    Z = rng.standard_normal((half, nt - 1, 3))
    if antithetic:
        Z = np.concatenate([Z, -Z])[:n]

    x = np.zeros((n, nt))
    int_r = np.empty((n, nt - 1))
    eq = np.empty((n, nt - 1))
    for k in range(nt - 1):
        eps = Z[:, k, :2] @ L.T  # (eps1, eps2) jointly Gaussian
        x[:, k + 1] = x[:, k] * e + eps[:, 0]
        int_x = x[:, k] * (1 - e) / a + eps[:, 1]
        int_r[:, k] = model.int_alpha(times[k], times[k + 1]) + int_x
        z_rate = Z[:, k, 0]  # same shock that drives eps1, so corr(equity, rate move) = rho
        z_eq = rho * z_rate + np.sqrt(1 - rho * rho) * Z[:, k, 2]
        # equities earn the realised short rate plus a premium (rates up -> equities down when rho < 0)
        eq[:, k] = int_r[:, k] + (equity_premium - equity_vol ** 2 / 2) * dt + equity_vol * np.sqrt(dt) * z_eq
    r = x + model.alpha(times)[None, :]
    return Scenarios(times, r, int_r, eq, model)
