"""Mortality: SSA 2023 period life table with generational mortality improvement.

q(x, year) = q_2023(x) * (1 - improvement) ** (year - 2023)

A flat annual improvement rate is a simplification of the SOA MP-2021 scale (which varies by age and year);
it is the single knob the longevity stress test moves.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
MAX_AGE = 120


@dataclass
class MortalityTable:
    qx: dict[str, np.ndarray]  # sex -> q at ages 0..119 for the base year
    base_year: int = 2023
    improvement: float = 0.01

    @classmethod
    def ssa_2023(cls, improvement: float = 0.01, path: Path | None = None) -> "MortalityTable":
        path = path or DATA / "ssa_2023_period_life_table.csv"
        with open(path) as f:
            rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
        m = np.array([float(r["male_qx"]) for r in rows])
        w = np.array([float(r["female_qx"]) for r in rows])
        return cls({"M": m, "F": w}, 2023, improvement)

    def with_improvement(self, improvement: float) -> "MortalityTable":
        return MortalityTable(self.qx, self.base_year, improvement)

    def q(self, sex: str, ages: np.ndarray, years: np.ndarray) -> np.ndarray:
        ages = np.asarray(ages)
        base = np.where(ages >= MAX_AGE, 1.0, self.qx[sex][np.minimum(ages, MAX_AGE - 1)])
        factor = (1 - self.improvement) ** (np.asarray(years) - self.base_year)
        return np.clip(np.where(ages >= MAX_AGE - 1, base, base * factor), 0.0, 1.0)

    def survival(self, sex: str, age: int, year: int, n_years: int) -> np.ndarray:
        """kp_x for k = 0..n_years, following the cohort (age and calendar year both advance)."""
        k = np.arange(n_years)
        q = self.q(sex, age + k, year + k)
        return np.concatenate([[1.0], np.cumprod(1 - q)])

    def life_expectancy(self, sex: str, age: int, year: int) -> float:
        """Curtate + 0.5 complete expectation of life for a cohort member."""
        s = self.survival(sex, age, year, MAX_AGE - age)
        return float(s[1:].sum() + 0.5)
