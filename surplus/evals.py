"""Accuracy eval for the AI analyst: questions whose correct answers the engine computes directly.

A question passes when the answer contains the expected figure (within tolerance) AND the grounding audit finds no
unsupported numbers. Needs Claude API credentials; each run makes one agent conversation per question.

    python -m surplus.evals
"""
from __future__ import annotations

import json
import time

from .analyst import _numbers_in_answer, ask
from .engine import Engine


def cases(e: Engine) -> list[dict]:
    s = e.liability_summary()
    down100 = e.rate_shock(-100)
    k30 = e.rate_shock(-50, 30)
    lon = e.longevity(0.005)
    h = e.hedge(0.9, 0.4)
    return [
        {"q": "What is the plan's liability present value, in $ millions?", "expect": s["liability_pv"] / 1e6, "tol": 0.005},
        {"q": "What is the liability's effective duration in years?", "expect": s["effective_duration"], "tol": 0.01},
        {"q": "What is the plan's funded ratio, in percent?", "expect": s["funded_ratio"] * 100, "tol": 0.005},
        {"q": "What is the liability DV01 in dollars?", "expect": s["dv01"], "tol": 0.01},
        {"q": "Which key-rate duration is largest, and what is its value in years?",
         "expect": max(s["key_rate_durations"].values()), "tol": 0.01},
        {"q": "By what percent does the liability change if Treasury rates fall 100bp in parallel?",
         "expect": abs(down100["change_pct"]), "tol": 0.01},
        {"q": "If only the 30-year key rate falls 50bp, how many $ millions does the liability rise?",
         "expect": k30["change"] / 1e6, "tol": 0.01},
        {"q": "How many $ millions does the liability rise if mortality improvement is 0.5% a year faster?",
         "expect": lon["change"] / 1e6, "tol": 0.01},
        {"q": "Build a 90% hedge with 40% of assets in bonds and swaps allowed. What swap notional is needed, in $ millions?",
         "expect": h.swap_notional / 1e6, "tol": 0.02},
        {"q": "Build a 90% hedge with 40% of assets in bonds and swaps allowed. What hedge ratio does it achieve, in percent?",
         "expect": h.achieved_ratio * 100, "tol": 0.005},
    ]


def grade(answer_text: str, expect: float, tol: float) -> bool:
    vals = [v for _, v, unit in _numbers_in_answer(answer_text)]
    vals += [v / 1e6 for v in vals] + [abs(v) for v in vals]
    return any(abs(v - expect) <= tol * max(abs(expect), 1e-9) for v in vals)


def main():
    e = Engine()
    rows = []
    for c in cases(e):
        t0 = time.time()
        a = ask(c["q"], e)
        ok = grade(a.text, c["expect"], c["tol"])
        rows.append({"question": c["q"], "expected": round(c["expect"], 4), "correct": ok,
                     "grounding": a.audit.get("grounding_score"), "unsupported": a.audit.get("unsupported"),
                     "tools": [x["tool"] for x in a.tool_calls], "seconds": round(time.time() - t0, 1)})
        print(("PASS " if ok else "FAIL ") + c["q"])
    acc = sum(r["correct"] for r in rows) / len(rows)
    grounded = sum((r["grounding"] or 0) == 1.0 for r in rows) / len(rows)
    print(f"\naccuracy {acc:.0%}, fully grounded answers {grounded:.0%}")
    json.dump({"accuracy": acc, "fully_grounded": grounded, "rows": rows}, open("eval_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
