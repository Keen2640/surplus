"""AI investment analyst: Claude answers questions about the plan by calling the engine through tools, and every
number in its answer is checked against the tool results it actually received (a grounding audit).

Design choices
- Claude never does arithmetic on its own: every figure has to come from a tool. The grounding audit flags any
  number in the final answer that does not trace back to a tool result, which is how an investment team can
  evaluate AI-generated analysis before it reaches a client.
- The agent loop runs here (Messages API, manual loop) because the tools are local numerical code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .engine import Engine

MODEL = "claude-opus-5-5"

SYSTEM = """You are an analyst on a pension Investment Solutions team (liability-driven investing, OCIO, strategic \
asset allocation). You answer questions about one corporate defined-benefit plan using the tools provided.

Rules:
- Every number you state must come from a tool result in this conversation. Never estimate or compute figures \
yourself; call a tool instead. Round money to $0.1M and percentages to 0.1 point.
- Say which assumptions drive a result when they matter (discount spread, mortality improvement, Hull-White \
volatility, equity risk premium), and name model limitations plainly.
- Be concise: lead with the answer, then at most a few supporting points."""

TOOLS = [
    {"name": "liability_summary", "strict": True,
     "description": "Plan snapshot: liability PV, assets, funded ratio, effective duration, convexity, DV01, key-rate "
                    "durations, census and assumptions.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "rate_shock", "strict": True,
     "description": "Fully revalue the liability after a Treasury curve shock. Parallel if key_tenor is null, otherwise "
                    "a triangular shock centred on that key tenor (1, 2, 3, 5, 7, 10, 20 or 30 years). Negative "
                    "shift_bp means rates fall.",
     "input_schema": {"type": "object", "properties": {
         "shift_bp": {"type": "number"},
         "key_tenor": {"type": ["number", "null"]}}, "required": ["shift_bp", "key_tenor"], "additionalProperties": False}},
    {"name": "build_hedge", "strict": True,
     "description": "Optimise an LDI hedge of Treasury STRIPS, Treasury bonds and (optionally) receive-fixed swaps that "
                    "matches target_ratio of the liability's key-rate DV01s while investing physical_fraction of plan "
                    "assets in bonds. Returns holdings, achieved hedge ratio and residual key-rate exposure.",
     "input_schema": {"type": "object", "properties": {
         "target_ratio": {"type": "number"}, "physical_fraction": {"type": "number"},
         "allow_swaps": {"type": "boolean"}},
         "required": ["target_ratio", "physical_fraction", "allow_swaps"], "additionalProperties": False}},
    {"name": "longevity_shock", "strict": True,
     "description": "Liability change if annual mortality improvement rises by improvement_delta (e.g. 0.005 = +0.5% a "
                    "year, people live longer).",
     "input_schema": {"type": "object", "properties": {"improvement_delta": {"type": "number"}},
                      "required": ["improvement_delta"], "additionalProperties": False}},
    {"name": "compare_strategies", "strict": True,
     "description": "Monte Carlo funded-status projection (Hull-White rates + correlated equities) of four investment "
                    "strategies: traditional 60/40, LDI long bonds, LDI with a 90% swap-overlay hedge, and a funded-"
                    "ratio glide path. Returns funded-ratio percentiles at the horizon, probability of ending below 80%, "
                    "1-year surplus VaR/CVaR and the present value of sponsor contributions. rates_only=true switches "
                    "off equity volatility to isolate interest-rate risk.",
     "input_schema": {"type": "object", "properties": {
         "years": {"type": "integer"}, "scenarios": {"type": "integer"}, "rates_only": {"type": "boolean"}},
         "required": ["years", "scenarios", "rates_only"], "additionalProperties": False}},
]


def _round(x):
    if isinstance(x, float):
        return round(x, 6) if abs(x) < 100 else round(x, 0)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def run_tool(engine: Engine, name: str, args: dict) -> dict:
    if name == "liability_summary":
        return engine.liability_summary()
    if name == "rate_shock":
        return engine.rate_shock(float(args["shift_bp"]), args.get("key_tenor"))
    if name == "build_hedge":
        t, p = float(args["target_ratio"]), float(args["physical_fraction"])
        if not (0 < t <= 1.5 and 0 < p <= 1):
            raise ValueError("target_ratio must be in (0, 1.5] and physical_fraction in (0, 1]")
        return engine.hedge(t, p, bool(args["allow_swaps"])).as_dict()
    if name == "longevity_shock":
        return engine.longevity(float(args["improvement_delta"]))
    if name == "compare_strategies":
        years = int(min(max(args["years"], 1), 30))
        n = int(min(max(args["scenarios"], 500), 20_000))
        res = engine.project_all(years, n, equity_vol=0.0 if args["rates_only"] else 0.16)
        return {"years": years, "scenarios": n, "rates_only": bool(args["rates_only"]),
                "strategies": [r.metrics() for r in res]}
    raise ValueError(f"unknown tool {name}")


# ---- grounding audit -------------------------------------------------------------------------------------------
NUM = re.compile(r"(?<![\w.])[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:%|bp|[MBK]\b|million|billion)?", re.I)


def _numbers_in_answer(text: str) -> list[tuple[str, float, str]]:
    out = []
    for m in NUM.finditer(text):
        raw = m.group(0).strip()
        unit = re.sub(r"[\d,.\s$+-]", "", raw).lower()
        try:
            v = float(re.sub(r"[^\d.+-]", "", raw.replace(",", "")))
        except ValueError:
            continue
        v *= {"m": 1e6, "million": 1e6, "b": 1e9, "billion": 1e9, "k": 1e3}.get(unit, 1)
        out.append((raw, v, unit))
    return out


def _flatten(x, acc):
    if isinstance(x, dict):
        for v in x.values():
            _flatten(v, acc)
        for k in x.keys():  # numeric keys such as key tenors are facts too
            try:
                acc.append(float(k))
            except (TypeError, ValueError):
                pass
    elif isinstance(x, list):
        for v in x:
            _flatten(v, acc)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        acc.append(float(x))
    return acc


def grounding_audit(answer: str, tool_results: list[dict], rel_tol: float = 0.006) -> dict:
    """Check each number in the answer against every tool-result number and the forms it is usually written in
    (x100 for percentages and ratios, /1e6 or /1e9 for money). Small integers (<= 10) such as "4 strategies" or
    "3 points" are not checked."""
    facts = _flatten(tool_results, [])
    forms = set()
    for f in facts:
        for g in (f, f * 100, f * 1e4, abs(f), abs(f) * 100):
            forms.add(g)
    checked, unsupported = [], []
    for raw, v, unit in _numbers_in_answer(answer):
        if abs(v) <= 10 and unit == "" and float(v).is_integer():
            continue
        ok = any(abs(v - g) <= max(rel_tol * abs(g), 0.051 if abs(g) < 100 else 0) for g in forms)
        (checked if ok else unsupported).append(raw)
    total = len(checked) + len(unsupported)
    return {"numbers_checked": total, "supported": len(checked), "unsupported": unsupported,
            "grounding_score": 1.0 if total == 0 else len(checked) / total}


# ---- agent loop ----------------------------------------------------------------------------------------------------
@dataclass
class Answer:
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    stop_reason: str = ""


def ask(question: str, engine: Engine | None = None, client=None, max_turns: int = 8, effort: str = "medium") -> Answer:
    import anthropic

    engine = engine or Engine()
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    calls: list[dict] = []
    for _ in range(max_turns):
        resp = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # re-run on a fallback model if declined
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            return Answer("The model declined this request.", calls, {}, "refusal")
        if resp.stop_reason == "pause_turn":
            continue
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason != "tool_use" or not uses:
            text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
            return Answer(text, calls, grounding_audit(text, [c["result"] for c in calls if "result" in c]),
                          resp.stop_reason)
        results = []
        for u in uses:  # all results go back in one user message
            try:
                out = _round(run_tool(engine, u.name, dict(u.input)))
                calls.append({"tool": u.name, "input": dict(u.input), "result": out})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out)})
            except Exception as e:  # report the error to the model instead of crashing the loop
                calls.append({"tool": u.name, "input": dict(u.input), "error": str(e)})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": str(e), "is_error": True})
        messages.append({"role": "user", "content": results})
    return Answer("Stopped: too many tool rounds.", calls, {}, "max_turns")


if __name__ == "__main__":
    import sys

    a = ask(" ".join(sys.argv[1:]) or "How exposed is the plan to falling rates, and what would a 90% hedge cost?")
    print(a.text)
    print("\nTools:", [c["tool"] for c in a.tool_calls])
    print("Grounding:", a.audit)
