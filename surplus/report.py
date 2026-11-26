"""Print the headline numbers used in the README:   python -m surplus.report"""
from __future__ import annotations

import time

from .engine import Engine


def main():
    e = Engine()
    s = e.liability_summary()
    print(f"Curve {s['curve_date']}  NSS fit RMSE {e.curve_fit['rmse_bp']:.2f}bp  max {e.curve_fit['max_abs_bp']:.2f}bp")
    print(f"Liability PV ${s['liability_pv'] / 1e6:,.1f}M  assets ${s['assets'] / 1e6:,.1f}M  funded {s['funded_ratio']:.1%}")
    print(f"Duration {s['effective_duration']:.2f}y  convexity {s['convexity']:.1f}  DV01 ${s['dv01']:,.0f}")
    print("KRDs", {k: round(v, 2) for k, v in s["key_rate_durations"].items()})
    sh = e.rate_shock(-100)
    print(f"-100bp: liability +${sh['change'] / 1e6:,.1f}M ({sh['change_pct']:.1f}%), unhedged funded ratio {sh['funded_ratio_after_if_unhedged']:.1%}")
    lo = e.longevity(0.005)
    print(f"Longevity +0.5%/yr improvement: +${lo['change'] / 1e6:,.1f}M ({lo['change_pct']:.1f}%)")
    h = e.hedge(0.9, 0.4)
    print(f"Hedge 90% w/ 40% bonds: achieved {h.achieved_ratio:.1%}, key-rate mismatch {h.mismatch_pct:.1f}%, swaps ${h.swap_notional / 1e6:,.0f}M")
    for label, vol in (("full risk", 0.16), ("rates only", 0.0)):
        t0 = time.time()
        res = e.project_all(10, 10_000, equity_vol=vol)
        print(f"\n10-year projection, 10,000 scenarios, {label} ({time.time() - t0:.1f}s for 4 strategies)")
        print(f"{'strategy':45s} {'medianFR10':>10s} {'p5FR10':>7s} {'P<80%':>6s} {'VaR95':>6s} {'CVaR95':>7s} {'PVcontrib':>10s} {'hedge':>6s}")
        for r in res:
            m = r.metrics()
            p = m["funded_ratio_end_percentiles"]
            print(f"{m['strategy']:45s} {p[50]:10.1%} {p[5]:7.1%} {m['prob_end_below_80pct']:6.1%} "
                  f"{m['surplus_var95_1y_pct_of_liability']:5.1f}% {m['surplus_cvar95_1y_pct_of_liability']:6.1f}% "
                  f"${m['contributions_pv_mean'] / 1e6:8,.1f}M {m['avg_hedge_ratio']:6.0%}")


if __name__ == "__main__":
    main()
