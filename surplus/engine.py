"""One object that wires the pieces together; the API, the CLI report and the AI analyst all call this."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .curve import bootstrap, fit_nss, load_par_curve
from .esg import HullWhite, simulate
from .hedge import optimize_hedge
from .liability import KEY_TENORS, longevity_sensitivity, pv, risk, tent
from .mortality import MortalityTable
from .plan import Census, benefit_cashflows
from .simulate import Strategy, project


@dataclass
class Config:
    curve_date: str | None = None  # latest available if None
    spread: float = 0.009  # liability discount spread over Treasuries (AA corporate proxy)
    improvement: float = 0.01
    funded_ratio: float = 0.88  # starting assets / liability
    participants: int = 20_000
    seed: int = 7
    hw_a: float = 0.05
    hw_sigma: float = 0.01
    valuation_year: int = 2026


@dataclass
class Engine:
    cfg: Config = field(default_factory=Config)

    @cached_property
    def par(self):
        return load_par_curve(self.cfg.curve_date)

    @cached_property
    def curve(self):
        return fit_nss(self.par)[0]

    @cached_property
    def curve_fit(self):
        return fit_nss(self.par)[1]

    @cached_property
    def mortality(self):
        return MortalityTable.ssa_2023(self.cfg.improvement)

    @cached_property
    def census(self):
        return Census.synthetic(self.cfg.participants, self.cfg.seed)

    @cached_property
    def cashflows(self):
        return benefit_cashflows(self.census, self.mortality, self.cfg.valuation_year)

    @cached_property
    def risk(self):
        return risk(*self.cashflows, self.curve, self.cfg.spread)

    @property
    def assets0(self) -> float:
        return self.cfg.funded_ratio * self.risk.pv

    # ---- questions the API and the analyst ask ---------------------------------------------------------------
    def liability_summary(self) -> dict:
        r = self.risk
        t, a = self.cashflows
        return {"curve_date": self.par.date, "liability_pv": r.pv, "assets": self.assets0,
                "funded_ratio": self.cfg.funded_ratio, "surplus": self.assets0 - r.pv,
                "effective_duration": r.duration, "convexity": r.convexity, "dv01": r.dv01,
                "key_rate_durations": dict(zip(map(float, KEY_TENORS), map(float, r.krd))),
                "discount_spread_bp": self.cfg.spread * 1e4, "mortality_improvement": self.cfg.improvement,
                "next_year_benefits": float(a[0]), "census": self.census.summary(),
                "curve_fit_rmse_bp": self.curve_fit["rmse_bp"]}

    def rate_shock(self, shift_bp: float, key_tenor: float | None = None) -> dict:
        """Full revaluation of liability, assets-held-in-cash and funded ratio for a parallel or key-rate shock."""
        h = shift_bp * 1e-4
        if key_tenor is None:
            curve = self.curve.shifted(h)
        else:
            k = int(np.argmin(np.abs(KEY_TENORS - key_tenor)))
            b = tent(k)
            curve = self.curve.shifted(lambda t: h * b(t))
        t, a = self.cashflows
        p0, p1 = self.risk.pv, pv(t, a, curve, self.cfg.spread)
        return {"shift_bp": shift_bp, "key_tenor": None if key_tenor is None else float(KEY_TENORS[k]),
                "liability_before": p0, "liability_after": p1, "change": p1 - p0, "change_pct": (p1 / p0 - 1) * 100,
                "funded_ratio_after_if_unhedged": self.assets0 / p1}

    def hedge(self, target_ratio: float = 0.9, physical_fraction: float = 0.5, allow_swaps: bool = True):
        """physical_fraction: share of plan assets in hedge bonds (the rest is return-seeking)."""
        return optimize_hedge(self.risk.kr_dv01, self.curve, target_ratio, physical_fraction * self.assets0, allow_swaps)

    def longevity(self, delta: float = 0.005) -> dict:
        return longevity_sensitivity(self.census, self.mortality, self.curve, self.cfg.spread, delta, self.cfg.valuation_year)

    def strategies(self) -> list[Strategy]:
        c = self.curve.par_yield(7)
        agg_t = np.arange(1, 15) / 2
        agg_a = np.full(14, c / 2); agg_a[-1] += 1  # 7-year par Treasury as an aggregate-bond proxy
        h50 = self.hedge(0.5, 0.4, allow_swaps=False)
        h90 = self.hedge(0.9, 0.4)
        hg = self.hedge(0.8, 0.5)
        norm = lambda hr: (hr.times, hr.amounts / hr.physical_budget)
        return [
            Strategy("Traditional 60/40", 0.6, None, agg_t, agg_a, roll="constant"),
            Strategy("LDI 60/40, long bonds", 0.6, None, *norm(h50)),
            Strategy("LDI 60/40 + swaps, 90% hedge", 0.6, 0.9, *norm(h90)),
            Strategy("Glide path (de-risk as funding improves)", 0.5, 0.8, *norm(hg),
                     glide=[(0.80, 0.60, 0.50), (0.90, 0.50, 0.70), (1.00, 0.35, 0.85), (1.10, 0.20, 1.00)]),
        ]

    def project_all(self, years: int = 10, n: int = 10_000, seed: int = 1, names: list[str] | None = None,
                    equity_vol: float = 0.16) -> list:
        """equity_vol=0 isolates interest-rate risk (equities then earn the short rate plus the premium, riskless)."""
        hw = HullWhite(self.curve, self.cfg.hw_a, self.cfg.hw_sigma)
        sc = simulate(hw, years, 0.5, n, seed, equity_vol=equity_vol)
        out = []
        for s in self.strategies():
            if names and s.name not in names:
                continue
            out.append(project(sc, s, *self.cashflows, self.assets0, self.cfg.spread))
        return out
