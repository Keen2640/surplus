"""Correctness checks against closed-form results. Each test pins down a property a finance reviewer would ask about."""
import numpy as np
import pytest

from surplus.curve import ZeroCurve, bootstrap, fit_nss, load_par_curve
from surplus.esg import HullWhite, simulate
from surplus.hedge import optimize_hedge
from surplus.liability import KEY_TENORS, pv, risk
from surplus.mortality import MortalityTable
from surplus.plan import Census, benefit_cashflows
from surplus.simulate import Strategy, project


@pytest.fixture(scope="module")
def par():
    return load_par_curve("11/25/2026")


@pytest.fixture(scope="module")
def curve(par):
    return fit_nss(par)[0]


@pytest.fixture(scope="module")
def plan():
    m = MortalityTable.ssa_2023()
    c = Census.synthetic(4000, seed=3)
    return c, m, benefit_cashflows(c, m)


# ---- curve -----------------------------------------------------------------------------------------------------
def test_bootstrap_reprices_every_par_bond(par):
    z = bootstrap(par)
    for T, y in zip(par.tenors, par.yields):
        if T >= 0.5:
            assert abs(z.par_yield(T) - y) < 1e-10


def test_nss_fit_is_within_a_few_bp(par):
    _, fit = fit_nss(par)
    assert fit["rmse_bp"] < 3 and fit["max_abs_bp"] < 6


def test_forward_rate_integrates_to_discount_factor(curve):
    t = np.linspace(1e-4, 30, 30001)
    integral = np.trapezoid(curve.fwd(t), t)
    assert abs(np.exp(-integral) - curve.df(30)) < 1e-5


# ---- liability risk ------------------------------------------------------------------------------------------------
def test_zero_coupon_duration_equals_maturity(curve):
    r = risk(np.array([12.0]), np.array([1.0]), curve)
    # central difference with a 10bp bump: exact answer is T * sinh(T h) / (T h)
    h = 10e-4
    assert r.duration == pytest.approx(12.0 * np.sinh(12 * h) / (12 * h), rel=1e-9)


def test_key_rate_durations_sum_to_effective_duration(curve, plan):
    _, _, (t, a) = plan
    r = risk(t, a, curve, 0.009)
    assert r.krd.sum() == pytest.approx(r.duration, rel=2e-3)


def test_flat_curve_flat_mortality_annuity_matches_closed_form():
    q, i = 0.02, 0.04
    curve = ZeroCurve(lambda t: np.full_like(t, np.log(1 + i)))
    k = np.arange(500)
    t, a = k + 1.0, (1 - q) ** (k + 1)  # end-of-year annuity, constant survival
    v = (1 - q) / (1 + i)
    assert pv(t, a, curve) == pytest.approx(v / (1 - v), rel=1e-6)


def test_survival_is_monotone_and_improvement_lengthens_life():
    m = MortalityTable.ssa_2023(0.0)
    s = m.survival("M", 65, 2026, 55)
    assert np.all(np.diff(s) <= 0) and s[0] == 1
    assert m.with_improvement(0.02).life_expectancy("F", 65, 2026) > m.life_expectancy("F", 65, 2026)


def test_deferred_members_pay_nothing_before_65(plan):
    m = MortalityTable.ssa_2023()
    c = Census(np.array([40]), np.array(["F"]), np.array(["deferred"]), np.array([10_000.0]))
    t, a = benefit_cashflows(c, m)
    assert np.all(a[:25] == 0) and a[25] > 0


# ---- Hull-White ----------------------------------------------------------------------------------------------------
def test_hull_white_discount_is_a_martingale(curve):
    sc = simulate(HullWhite(curve), 30, 0.5, 40_000, seed=11, antithetic=False)
    D = sc.discount()
    for T in (5, 10, 20, 30):
        k = int(T / 0.5)
        se = D[:, k].std() / np.sqrt(sc.n)
        assert abs(D[:, k].mean() - curve.df(T)) < 4 * se


def test_closed_form_bond_price_is_consistent_with_simulation(curve):
    hw = HullWhite(curve)
    sc = simulate(hw, 5, 0.5, 40_000, seed=5, antithetic=False)
    k = 10  # t = 5
    est = sc.discount()[:, k] * hw.zcb(5.0, [20.0], sc.r[:, k])[:, 0]
    assert abs(est.mean() - curve.df(20)) < 4 * est.std() / np.sqrt(sc.n)


def test_zcb_at_time_zero_is_todays_curve(curve):
    hw = HullWhite(curve)
    P = hw.zcb(0.0, [1, 10, 30], np.zeros(3))
    np.testing.assert_allclose(P[0], curve.df([1, 10, 30]))


# ---- hedging -------------------------------------------------------------------------------------------------------
def test_hedge_hits_target_ratio_and_key_rates(curve, plan):
    _, _, (t, a) = plan
    r = risk(t, a, curve, 0.009)
    h = optimize_hedge(r.kr_dv01, curve, 1.0, 0.6 * r.pv)
    assert h.achieved_ratio == pytest.approx(1.0, abs=0.01)
    assert h.mismatch_pct < 3
    assert sum(v for _, k, v in h.holdings if k != "swap") == pytest.approx(0.6 * r.pv, rel=1e-6)


def test_cash_flow_matched_plan_stays_exactly_funded(curve, plan):
    """Perfect immunisation: assets = liability cash flows, no spread -> funded ratio is 100% in every scenario."""
    _, _, (t, a) = plan
    sc = simulate(HullWhite(curve), 8, 0.5, 2000, seed=2)
    L0 = pv(t, a, curve)
    p = project(sc, Strategy("cfm", 0.0, None, t, a / L0), t, a, L0, spread=0.0)
    np.testing.assert_allclose(p.funded_ratio, 1.0, atol=1e-9)


def test_overlay_delivers_requested_hedge_ratio(curve, plan):
    _, _, (t, a) = plan
    r = risk(t, a, curve, 0.009)
    h = optimize_hedge(r.kr_dv01, curve, 0.9, 0.4 * r.pv)
    sc = simulate(HullWhite(curve), 5, 0.5, 2000, seed=4)
    p = project(sc, Strategy("ldi", 0.6, 0.9, h.times, h.amounts / h.physical_budget), t, a, 0.9 * r.pv)
    np.testing.assert_allclose(p.hedge_ratio, 0.9, atol=1e-9)


def test_hedging_cuts_rate_driven_surplus_risk(curve, plan):
    _, _, (t, a) = plan
    r = risk(t, a, curve, 0.009)
    sc = simulate(HullWhite(curve), 3, 0.5, 4000, seed=9, equity_vol=0.0)
    h = optimize_hedge(r.kr_dv01, curve, 0.9, 0.4 * r.pv)
    base = project(sc, Strategy("none", 0.6, 0.0, h.times, h.amounts / h.physical_budget), t, a, 0.9 * r.pv)
    hedged = project(sc, Strategy("ldi", 0.6, 0.9, h.times, h.amounts / h.physical_budget), t, a, 0.9 * r.pv)
    assert hedged.surplus_change_y1.std() < 0.25 * base.surplus_change_y1.std()
