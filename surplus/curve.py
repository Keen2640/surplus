"""Treasury yield curve: load par yields, bootstrap a zero curve, and fit a smooth Nelson-Siegel-Svensson curve.

Conventions
- Treasury par yields (CMT) are bond-equivalent yields: semiannual compounding, semiannual coupons.
- Internally every curve is a continuously compounded zero curve z(t), with discount factor D(t) = exp(-z(t) t).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

DATA = Path(__file__).resolve().parent.parent / "data"

COLUMN_TENORS = {  # Treasury CSV header -> maturity in years
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 3 / 12, "4 Mo": 4 / 12, "6 Mo": 0.5,
    "1 Yr": 1, "2 Yr": 2, "3 Yr": 3, "5 Yr": 5, "7 Yr": 7, "10 Yr": 10, "20 Yr": 20, "30 Yr": 30,
}


@dataclass
class ParCurve:
    date: str
    tenors: np.ndarray  # years
    yields: np.ndarray  # decimals, bond-equivalent (semiannual)


def load_par_curve(date: str | None = None, path: Path | None = None) -> ParCurve:
    """Read a day of the Treasury daily par yield curve CSV (latest day if `date` is None, format MM/DD/YYYY)."""
    path = path or DATA / "ust_par_yields_2026.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: datetime.strptime(r["Date"], "%m/%d/%Y"), reverse=True)
    row = rows[0] if date is None else next(r for r in rows if r["Date"] == date)
    pairs = sorted((COLUMN_TENORS[k], float(v) / 100) for k, v in row.items() if k in COLUMN_TENORS and v not in ("", "N/A"))
    t, y = map(np.array, zip(*pairs))
    return ParCurve(row["Date"], t, y)


def load_history(path: Path | None = None) -> list[ParCurve]:
    path = path or DATA / "ust_par_yields_2026.csv"
    with open(path) as f:
        dates = [r["Date"] for r in csv.DictReader(f)]
    return [load_par_curve(d, path) for d in sorted(dates, key=lambda d: datetime.strptime(d, "%m/%d/%Y"))]


# ---------------------------------------------------------------------------------------------------------------------
# Zero curve container
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class ZeroCurve:
    """Continuously compounded zero curve given as a function of maturity (years)."""

    zero_fn: Callable[[np.ndarray], np.ndarray]
    name: str = "curve"
    meta: dict = field(default_factory=dict)

    def zero(self, t) -> np.ndarray:
        return self.zero_fn(np.asarray(t, dtype=float))

    def df(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(np.maximum(t, 1e-9)) * t)

    def fwd(self, t, h: float = 1e-4) -> np.ndarray:
        """Instantaneous forward rate f(0,t) = -d ln D / dt (central difference)."""
        t = np.maximum(np.asarray(t, dtype=float), h)
        return -(np.log(self.df(t + h)) - np.log(self.df(t - h))) / (2 * h)

    def shifted(self, bump: Callable[[np.ndarray], np.ndarray] | float, name: str | None = None) -> "ZeroCurve":
        """New curve with z(t) + bump(t). A float is a parallel shift."""
        b = (lambda t: np.full_like(t, bump, dtype=float)) if np.isscalar(bump) else bump
        return ZeroCurve(lambda t: self.zero_fn(t) + b(t), name or f"{self.name}+bump", self.meta)

    def par_yield(self, T: float, freq: int = 2) -> float:
        """Par coupon (compounded `freq` times a year) of a bond maturing at T."""
        n = max(int(round(T * freq)), 1)
        times = np.arange(1, n + 1) / freq
        d = self.df(times)
        return float(freq * (1 - d[-1]) / d.sum())


# ---------------------------------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------------------------------
def bootstrap(par: ParCurve) -> ZeroCurve:
    """Bootstrap discount factors on a semiannual grid from par yields.

    Bills (< 6 months) are zero-coupon: D = (1 + y/2)^(-2t). From 6 months on, par yields are interpolated with a
    shape-preserving cubic (PCHIP) onto every coupon date and solved one date at a time:
        1 = c/2 * sum_{i<n} D_i + (1 + c/2) D_n.
    The result reprices every input par bond exactly. Zero rates between nodes are interpolated linearly in z(t).
    """
    interp = PchipInterpolator(par.tenors, par.yields, extrapolate=True)
    grid = np.arange(1, int(round(par.tenors.max() * 2)) + 1) / 2  # 0.5 .. 30
    D = np.empty_like(grid)
    for n, T in enumerate(grid):
        c = float(interp(T))
        if T <= 0.5:
            D[n] = (1 + c / 2) ** (-2 * T)
        else:
            D[n] = (1 - c / 2 * D[:n].sum()) / (1 + c / 2)
    short = par.tenors[par.tenors < 0.5]
    t_nodes = np.concatenate([short, grid])
    d_nodes = np.concatenate([(1 + par.yields[par.tenors < 0.5] / 2) ** (-2 * short), D])
    z_nodes = -np.log(d_nodes) / t_nodes

    def z(t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, t_nodes, z_nodes)  # flat extrapolation at both ends

    return ZeroCurve(z, f"UST bootstrap {par.date}", {"t": t_nodes, "z": z_nodes, "date": par.date})


# ---------------------------------------------------------------------------------------------------------------------
# Nelson-Siegel-Svensson
# ---------------------------------------------------------------------------------------------------------------------
def nss_zero(t, b0, b1, b2, b3, tau1, tau2):
    t = np.maximum(np.asarray(t, dtype=float), 1e-6)
    x1, x2 = t / tau1, t / tau2
    l1 = (1 - np.exp(-x1)) / x1
    l2 = l1 - np.exp(-x1)
    l3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return b0 + b1 * l1 + b2 * l2 + b3 * l3


def fit_nss(par: ParCurve) -> tuple[ZeroCurve, dict]:
    """Fit NSS zero-curve parameters so the curve reprices the observed par yields (least squares on yield error).

    Returns the curve and fit diagnostics (RMSE / max error in basis points).
    """
    boot = bootstrap(par)

    def model_par(p, T):
        c = ZeroCurve(lambda t: nss_zero(t, *p))
        if T < 0.5:  # bills: zero-coupon bond-equivalent yield
            return 2 * (c.df(T) ** (-1 / (2 * T)) - 1)
        return c.par_yield(T)

    def resid(p):
        return np.array([model_par(p, T) - y for T, y in zip(par.tenors, par.yields)]) * 1e4  # bp

    z_long, z_short = boot.zero(30.0), boot.zero(par.tenors.min())
    best = None
    for tau1, tau2 in [(1.0, 8.0), (2.0, 12.0), (0.5, 5.0), (3.0, 15.0)]:  # multi-start: NSS is not convex in tau
        x0 = [float(z_long), float(z_short - z_long), 0.0, 0.0, tau1, tau2]
        r = least_squares(resid, x0, bounds=([-0.05, -0.2, -0.3, -0.3, 0.05, 0.5], [0.2, 0.2, 0.3, 0.3, 10, 30]))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    err = resid(p)
    curve = ZeroCurve(lambda t, p=p: nss_zero(t, *p), f"UST NSS {par.date}",
                      {"params": dict(zip(["b0", "b1", "b2", "b3", "tau1", "tau2"], map(float, p))), "date": par.date})
    return curve, {"rmse_bp": float(np.sqrt(np.mean(err ** 2))), "max_abs_bp": float(np.abs(err).max()),
                   "errors_bp": dict(zip(map(float, par.tenors), map(float, err)))}
