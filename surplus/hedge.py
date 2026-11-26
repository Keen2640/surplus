"""LDI hedge construction: choose Treasury STRIPS, Treasury bonds and receive-fixed swaps so the hedge portfolio's
key-rate DV01s match a target fraction (the hedge ratio) of the liability's key-rate DV01s, spending exactly the
physical-asset budget. Swaps cost nothing up front, so they extend the hedge beyond what the budget can buy.

Swaps are priced off the Treasury curve (a SOFR-curve stand-in; real swap spreads are ignored).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .curve import ZeroCurve
from .liability import KEY_TENORS, risk

STRIPS = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
BONDS = [2, 5, 10, 20, 30]
SWAPS = [10, 20, 30]


@dataclass
class Instrument:
    name: str
    kind: str  # "strip" | "bond" | "swap"
    times: np.ndarray
    amounts: np.ndarray  # cash flows per 1 unit (MV 1 for strips/bonds, notional 1 for swaps)
    kr_dv01: np.ndarray  # per unit
    funded: bool  # uses physical budget


def build_instruments(curve: ZeroCurve, spread_free: ZeroCurve | None = None) -> list[Instrument]:
    out = []
    for T in STRIPS:
        t, a = np.array([float(T)]), np.array([1.0 / float(curve.df(T))])  # $1 market value
        out.append(Instrument((f"STRIPS {T}y" if T >= 1 else "T-bill 6m"), "strip", t, a, risk(t, a, curve).kr_dv01, True))
    for T in BONDS:
        c = curve.par_yield(T)
        t = np.arange(1, 2 * T + 1) / 2
        a = np.full(len(t), c / 2)
        a[-1] += 1  # par bond: $1 MV
        out.append(Instrument(f"UST {T}y {c * 100:.2f}%", "bond", t, a, risk(t, a, curve).kr_dv01, True))
    for T in SWAPS:
        c = curve.par_yield(T)
        t = np.arange(1, 2 * T + 1) / 2
        a = np.full(len(t), c / 2)
        a[-1] += 1
        # receive fixed, pay floating: the floating leg is a note worth 1 whose next coupon is already fixed,
        # i.e. a single cash flow of 1/D(0.5) at t = 0.5
        t = np.concatenate([t, [0.5]])
        a = np.concatenate([a, [-1 / float(curve.df(0.5))]])
        rk = risk(t[:-1], a[:-1], curve).kr_dv01 - risk(t[-1:], a[-1:], curve).kr_dv01
        out.append(Instrument(f"Rec-fixed swap {T}y {c * 100:.2f}%", "swap", t, a, rk, False))
    return out


@dataclass
class HedgeResult:
    holdings: list[tuple[str, str, float]]  # (name, kind, MV for funded / notional for swaps)
    target_ratio: float
    achieved_ratio: float  # total hedge DV01 / liability DV01
    liability_kr_dv01: np.ndarray
    hedge_kr_dv01: np.ndarray
    physical_budget: float
    swap_notional: float
    times: np.ndarray  # aggregated physical cash flows (for the simulator)
    amounts: np.ndarray

    @property
    def mismatch_pct(self) -> float:
        """Key-rate mismatch: ||hedge - target|| / ||target||, in %."""
        tgt = self.target_ratio * self.liability_kr_dv01
        return float(np.linalg.norm(self.hedge_kr_dv01 - tgt) / np.linalg.norm(tgt) * 100)

    def as_dict(self) -> dict:
        return {"target_ratio": self.target_ratio, "achieved_ratio": self.achieved_ratio,
                "key_rate_mismatch_pct": self.mismatch_pct, "physical_budget": self.physical_budget,
                "swap_notional": self.swap_notional,
                "holdings": [{"instrument": n, "kind": k, "amount": v} for n, k, v in self.holdings if abs(v) > 1],
                "key_rates": [{"tenor": float(T), "liability_dv01": float(l), "hedge_dv01": float(h),
                               "residual_dv01": float(h - self.target_ratio * l)}
                              for T, l, h in zip(KEY_TENORS, self.liability_kr_dv01, self.hedge_kr_dv01)]}


def optimize_hedge(liability_kr_dv01: np.ndarray, curve: ZeroCurve, target_ratio: float, physical_budget: float,
                   allow_swaps: bool = True, swap_penalty: float = 1e-3) -> HedgeResult:
    """min ||A w - h L||^2 / ||h L||^2 + penalty * (swap notional / budget)^2
       s.t. sum(physical MV) = budget, physical MV >= 0, swap notional >= 0."""
    inst = build_instruments(curve)
    if not allow_swaps:
        inst = [i for i in inst if i.funded]
    A = np.stack([i.kr_dv01 for i in inst], axis=1)  # (keys, m)
    funded = np.array([i.funded for i in inst])
    tgt = target_ratio * liability_kr_dv01
    scale = physical_budget  # optimise in units of the budget for conditioning
    tn = np.linalg.norm(tgt)

    def obj(u):
        r = (A @ (u * scale) - tgt) / tn
        return r @ r + swap_penalty * np.sum(u[~funded] ** 2)

    def grad(u):
        r = (A @ (u * scale) - tgt) / tn
        g = 2 * (A.T @ r) * scale / tn
        g[~funded] += 2 * swap_penalty * u[~funded]
        return g

    m = len(inst)
    u0 = np.where(funded, 1 / funded.sum(), 0.0)
    cons = [{"type": "eq", "fun": lambda u: u[funded].sum() - 1, "jac": lambda u: funded.astype(float)}]
    res = minimize(obj, u0, jac=grad, bounds=[(0, None)] * m, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-14})
    w = np.maximum(res.x, 0) * scale
    hedge = A @ w
    times = np.unique(np.concatenate([i.times for i in inst if i.funded]))
    amounts = np.zeros(len(times))
    for wi, i in zip(w, inst):
        if i.funded and wi > 0:
            amounts[np.searchsorted(times, i.times)] += wi * i.amounts
    return HedgeResult([(i.name, i.kind, float(wi)) for i, wi in zip(inst, w)], target_ratio,
                       float(hedge.sum() / liability_kr_dv01.sum()), liability_kr_dv01, hedge, physical_budget,
                       float(w[~funded].sum()), times, amounts)
