"""Liability valuation and interest-rate risk: PV, effective duration/convexity, and key-rate durations.

The pension discount curve is the Treasury zero curve plus a flat credit spread (a stand-in for the AA corporate
curve US GAAP pension accounting uses). All rate sensitivities are with respect to Treasury zero rates, which is
what a Treasury/swap hedge portfolio can offset.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curve import ZeroCurve

KEY_TENORS = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
BP = 1e-4


def pv(times: np.ndarray, amounts: np.ndarray, curve: ZeroCurve, spread: float = 0.0) -> float:
    return float(np.sum(amounts * curve.df(times) * np.exp(-spread * times)))


def tent(k: int, keys: np.ndarray = KEY_TENORS):
    """Triangular bump for key rate k: 1 at keys[k], 0 at neighbouring keys, flat beyond the first/last key.

    The tents sum to 1 at every maturity, so the key-rate durations add up to the parallel (effective) duration.
    """
    def f(t):
        t = np.asarray(t, dtype=float)
        y = np.zeros(len(keys))
        y[k] = 1.0
        return np.interp(t, keys, y)
    return f


@dataclass
class RiskReport:
    pv: float
    duration: float  # effective (modified) duration, years
    convexity: float
    dv01: float  # $ change for a 1bp parallel fall in rates
    krd: np.ndarray  # key-rate durations (years), sum ~= duration
    kr_dv01: np.ndarray  # $ per 1bp at each key tenor

    def as_dict(self) -> dict:
        return {"pv": self.pv, "duration": self.duration, "convexity": self.convexity, "dv01": self.dv01,
                "krd": dict(zip(map(float, KEY_TENORS), map(float, self.krd))),
                "kr_dv01": dict(zip(map(float, KEY_TENORS), map(float, self.kr_dv01)))}


def risk(times: np.ndarray, amounts: np.ndarray, curve: ZeroCurve, spread: float = 0.0, h: float = 10 * BP) -> RiskReport:
    p0 = pv(times, amounts, curve, spread)
    up = pv(times, amounts, curve.shifted(h), spread)
    dn = pv(times, amounts, curve.shifted(-h), spread)
    dur = (dn - up) / (2 * p0 * h)
    conv = (up + dn - 2 * p0) / (p0 * h * h)
    krd = np.empty(len(KEY_TENORS))
    for k in range(len(KEY_TENORS)):
        b = tent(k)
        u = pv(times, amounts, curve.shifted(lambda t, b=b: h * b(t)), spread)
        d = pv(times, amounts, curve.shifted(lambda t, b=b: -h * b(t)), spread)
        krd[k] = (d - u) / (2 * p0 * h)
    return RiskReport(p0, dur, conv, p0 * dur * BP, krd, p0 * krd * BP)


def longevity_sensitivity(census, mortality, curve: ZeroCurve, spread: float, delta: float = 0.005,
                          valuation_year: int = 2026) -> dict:
    """PV change if annual mortality improvement is `delta` higher (people live longer)."""
    from .plan import benefit_cashflows
    t, a0 = benefit_cashflows(census, mortality, valuation_year)
    _, a1 = benefit_cashflows(census, mortality.with_improvement(mortality.improvement + delta), valuation_year)
    p0, p1 = pv(t, a0, curve, spread), pv(t, a1, curve, spread)
    return {"base_pv": p0, "shocked_pv": p1, "change": p1 - p0, "change_pct": (p1 / p0 - 1) * 100,
            "improvement_base": mortality.improvement, "improvement_shocked": mortality.improvement + delta}
