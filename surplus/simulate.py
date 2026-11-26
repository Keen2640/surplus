"""Funded-status projection: run a pension plan's assets and liabilities through Hull-White rate scenarios.

At every half-year node, in every scenario:
- equities earn their simulated return, cash earns the realised short rate;
- bonds, swaps and the liability are revalued with closed-form Hull-White bond prices;
- coupons, swap net payments and benefit payments move through cash.
Once a year the plan rebalances: the sponsor contributes 1/7 of any deficit (a simplified PPA amortisation),
equities and physical bonds are reset to the strategy's weights, and the swap overlay is re-struck at par so the
total rate delta equals hedge_ratio x liability delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .esg import HullWhite, Scenarios


@dataclass
class Strategy:
    name: str
    equity_weight: float
    hedge_ratio: float | None  # None: no swap overlay, the hedge is whatever the physical bonds give
    bond_times: np.ndarray = field(default=None, repr=False)  # physical bond cash-flow shape, per $1 MV at t=0
    bond_amounts: np.ndarray = field(default=None, repr=False)
    roll: str = "hold"  # "hold": keep the dated bonds (LDI); "constant": re-buy the same maturities (bond index)
    glide: list[tuple[float, float, float]] | None = None  # (funded ratio, equity weight, hedge ratio) triggers
    swap_tenor: float = 20.0

    def targets(self, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.glide:
            h = None if self.hedge_ratio is None else np.full_like(fr, self.hedge_ratio)
            return np.full_like(fr, self.equity_weight), h
        f, w, h = map(np.array, zip(*sorted(self.glide)))
        return np.interp(fr, f, w), np.interp(fr, f, h)


@dataclass
class Projection:
    strategy: str
    years: np.ndarray
    funded_ratio: np.ndarray  # (n, years+1), before contributions
    assets: np.ndarray
    liabilities: np.ndarray
    hedge_ratio: np.ndarray  # (n, years) realised at each rebalance
    contributions_pv: np.ndarray  # (n,)
    surplus_change_y1: np.ndarray  # (n,) as a fraction of the starting liability

    def metrics(self) -> dict:
        loss = -self.surplus_change_y1
        var = float(np.quantile(loss, 0.95))
        fr_end = self.funded_ratio[:, -1]
        pct = [5, 25, 50, 75, 95]
        return {
            "strategy": self.strategy,
            "funded_ratio_start": float(self.funded_ratio[:, 0].mean()),
            "funded_ratio_end_percentiles": dict(zip(pct, map(float, np.percentile(fr_end, pct)))),
            "prob_end_below_80pct": float((fr_end < 0.8).mean()),
            "prob_end_fully_funded": float((fr_end >= 1.0).mean()),
            "surplus_var95_1y_pct_of_liability": var * 100,
            "surplus_cvar95_1y_pct_of_liability": float(loss[loss >= var].mean()) * 100,
            "contributions_pv_mean": float(self.contributions_pv.mean()),
            "contributions_pv_p95": float(np.quantile(self.contributions_pv, 0.95)),
            "avg_hedge_ratio": float(np.nanmean(self.hedge_ratio)),
        }

    def fan(self) -> dict:
        p = np.percentile(self.funded_ratio, [5, 25, 50, 75, 95], axis=0)
        return {"years": self.years.tolist(), **{f"p{q}": p[i].tolist() for i, q in enumerate([5, 25, 50, 75, 95])}}


def _value(model: HullWhite, t: float, r: np.ndarray, times: np.ndarray, amounts: np.ndarray, offset, spread: float = 0.0):
    """Value and rate delta (dV/dr) of fixed cash flows at absolute times `times + offset` still in the future.

    offset may be a scalar or per-scenario array of equal values (strategies rebalance on common dates)."""
    T = times + float(np.max(offset))
    live = T > t + 1e-9
    if not live.any():
        z = np.zeros_like(r)
        return z, z
    T, a = T[live], amounts[live]
    P = model.zcb(t, T, r) * np.exp(-spread * (T - t))[None, :]
    B = model.B(t, T)
    return P @ a, -(P * B[None, :]) @ a


def _received(times, amounts, offset, t0, t1):
    T = times + offset
    m = (T > t0 + 1e-9) & (T <= t1 + 1e-9)
    return float(amounts[m].sum())


def project(sc: Scenarios, strategy: Strategy, liab_times: np.ndarray, liab_amounts: np.ndarray, assets0: float,
            spread: float = 0.009, amortization_years: float = 7.0, contributions: bool = True) -> Projection:
    model, times, n = sc.model, sc.times, sc.n
    D = sc.discount()
    H = int(round(times[-1]))
    E = np.zeros(n); C = np.zeros(n)
    k_bonds = np.zeros(n); bond_offset = 0.0
    N = np.zeros(n); swap_c = np.zeros(n); swap_t0 = 0.0; p_fix = np.ones(n)
    fr = np.zeros((n, H + 1)); A_rec = np.zeros((n, H + 1)); L_rec = np.zeros((n, H + 1))
    h_rec = np.full((n, H), np.nan); contrib_pv = np.zeros(n); s0 = None; s1 = None
    sw_times = np.arange(1, int(strategy.swap_tenor * 2) + 1) / 2
    bt, ba = strategy.bond_times, strategy.bond_amounts

    for k, t in enumerate(times):
        r = sc.r[:, k]
        if k > 0:
            tp = times[k - 1]
            E *= np.exp(sc.equity_logret[:, k - 1])
            C *= np.exp(sc.int_r[:, k - 1])
            C += k_bonds * _received(bt, ba, bond_offset, tp, t)
            if t - swap_t0 <= strategy.swap_tenor + 1e-9:  # swap net payment: fixed coupon - floating fixed at tp
                C += N * (swap_c * 0.5 - (1 / p_fix - 1))
            C -= _received(liab_times, liab_amounts, 0.0, tp, t)
        bond_v, bond_d = _value(model, t, r, bt, ba, bond_offset)
        bond_v, bond_d = k_bonds * bond_v, k_bonds * bond_d
        rem = sw_times[sw_times + swap_t0 > t + 1e-9] + swap_t0
        if len(rem):
            P = model.zcb(t, rem, r)
            swap_v = N * (swap_c * 0.5 * P.sum(axis=1) + P[:, -1] - 1)
        else:
            swap_v = np.zeros(n)
        L, L_d = _value(model, t, r, liab_times, liab_amounts, 0.0, spread)
        A = E + C + bond_v + swap_v

        if abs(t - round(t)) < 1e-9:  # annual node: record, contribute, rebalance
            y = int(round(t))
            fr[:, y], A_rec[:, y], L_rec[:, y] = A / L, A, L
            if y == 0:
                A = np.full(n, float(assets0)); fr[:, 0] = A / L; A_rec[:, 0] = A; s0 = (A - L) / L
            if y == 1:
                s1 = (A - L) / L_rec[:, 0]
            if y < H:
                if y > 0 and contributions:
                    c = np.maximum(L - A, 0) / amortization_years
                    A = A + c
                    contrib_pv += c * D[:, k]
                we, h = strategy.targets(A / L)
                E = we * A
                C = np.zeros(n)
                if strategy.roll == "constant":
                    bond_offset = t
                unit_v, unit_d = _value(model, t, r, bt, ba, bond_offset)
                k_bonds = (1 - we) * A / unit_v
                swap_t0 = t
                P = model.zcb(t, sw_times + t, r)
                swap_c = (1 - P[:, -1]) / (0.5 * P.sum(axis=1))
                B = model.B(t, sw_times + t)
                swap_unit_d = -(swap_c * 0.5 * (P * B[None, :]).sum(axis=1) + P[:, -1] * B[-1])
                if h is None:
                    N = np.zeros(n)
                else:
                    N = (h * L_d - k_bonds * unit_d) / swap_unit_d
                h_rec[:, y] = (k_bonds * unit_d + N * swap_unit_d) / L_d
        p_fix = model.zcb(t, [t + 0.5], r)[:, 0]

    return Projection(strategy.name, np.arange(H + 1), fr, A_rec, L_rec, h_rec, contrib_pv,
                      (s1 - s0) if s1 is not None else np.zeros(n))
