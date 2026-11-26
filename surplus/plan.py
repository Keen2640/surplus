"""A frozen corporate defined-benefit pension plan: participant census and expected benefit cash flows.

Benefit model (stated assumptions):
- The plan is frozen: every participant's accrued annual benefit is fixed (no new accruals, no salary growth).
  Most corporate plans that buy LDI programs are frozen, so this is the common case.
- Benefits are single-life annuities payable from age 65 (retirees are already in pay).
- Payments are annual, paid mid-year (t = k + 0.5) as an approximation of monthly payments; survival to mid-year
  uses the uniform-distribution-of-deaths assumption.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mortality import MAX_AGE, MortalityTable

RETIREMENT_AGE = 65


@dataclass
class Census:
    age: np.ndarray
    sex: np.ndarray  # "M" / "F"
    status: np.ndarray  # "active" (frozen), "deferred", "retiree"
    benefit: np.ndarray  # annual accrued benefit, $

    def __len__(self) -> int:
        return len(self.age)

    @classmethod
    def synthetic(cls, n: int = 20_000, seed: int = 7) -> "Census":
        """A realistic mid-size frozen plan: ~40% retirees, ~25% deferred vested, ~35% frozen actives."""
        rng = np.random.default_rng(seed)
        status = rng.choice(["retiree", "deferred", "active"], size=n, p=[0.40, 0.25, 0.35])
        age = np.empty(n, dtype=int)
        r, d, a = status == "retiree", status == "deferred", status == "active"
        age[r] = np.clip(65 + rng.gamma(2.2, 5.5, r.sum()).round().astype(int), 65, 104)
        age[d] = rng.integers(35, 65, d.sum())
        age[a] = np.clip(rng.normal(50, 8, a.sum()).round().astype(int), 25, 64)
        sex = np.where(rng.random(n) < 0.55, "M", "F")
        mean = np.select([r, d, a], [18_000, 9_000, 14_000])
        benefit = mean * rng.lognormal(-0.18, 0.6, n)  # E[lognormal(-0.18, 0.6)] = 1.0
        return cls(age, sex, status, benefit.round(2))

    @classmethod
    def from_csv(cls, path: str | Path) -> "Census":
        with open(path) as f:
            rows = list(csv.DictReader(f))
        return cls(np.array([int(r["age"]) for r in rows]), np.array([r["sex"] for r in rows]),
                   np.array([r["status"] for r in rows]), np.array([float(r["benefit"]) for r in rows]))

    def summary(self) -> dict:
        out = {"participants": len(self)}
        for s in ("retiree", "deferred", "active"):
            m = self.status == s
            out[s] = {"count": int(m.sum()), "avg_age": round(float(self.age[m].mean()), 1),
                      "annual_benefits": round(float(self.benefit[m].sum()), 0)}
        return out


def benefit_cashflows(census: Census, mortality: MortalityTable, valuation_year: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Expected benefit payments by year: returns (times, amounts) with times = 0.5, 1.5, ...

    Participants are grouped into (age, sex) cohorts so the survival curve is computed once per cohort; the
    benefit for each cohort is the sum of its members' benefits.
    """
    horizon = MAX_AGE - int(census.age.min())
    amounts = np.zeros(horizon)
    k = np.arange(horizon)
    for sex in ("M", "F"):
        for age in np.unique(census.age[census.sex == sex]):
            m = (census.sex == sex) & (census.age == age)
            total = census.benefit[m].sum()
            n = MAX_AGE - age
            s = mortality.survival(sex, int(age), valuation_year, n)  # kp_x, k = 0..n
            q = 1 - s[1:] / np.where(s[:-1] > 0, s[:-1], 1)
            mid = s[:-1] * (1 - 0.5 * q)  # survival to k + 0.5 under UDD
            in_pay = (age + k[:n]) >= RETIREMENT_AGE
            amounts[:n] += total * mid * in_pay
    return k + 0.5, amounts
