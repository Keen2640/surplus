"""Offline tests for the AI analyst: tool layer, grounding audit, and the agent loop driven by a scripted client."""
import json
from types import SimpleNamespace as NS

import pytest

from surplus.analyst import TOOLS, ask, grounding_audit, run_tool
from surplus.engine import Config, Engine


@pytest.fixture(scope="module")
def engine():
    return Engine(Config(participants=3000))


def test_every_tool_runs_and_returns_json(engine):
    args = {"liability_summary": {}, "rate_shock": {"shift_bp": -50, "key_tenor": None},
            "build_hedge": {"target_ratio": 0.9, "physical_fraction": 0.4, "allow_swaps": True},
            "longevity_shock": {"improvement_delta": 0.005},
            "compare_strategies": {"years": 2, "scenarios": 500, "rates_only": True}}
    assert {t["name"] for t in TOOLS} == set(args)
    for name, a in args.items():
        json.dumps(run_tool(engine, name, a))


def test_tool_schemas_are_strict():
    for t in TOOLS:
        assert t["strict"] and t["input_schema"]["additionalProperties"] is False
        assert set(t["input_schema"]["required"]) == set(t["input_schema"]["properties"])


def test_grounding_accepts_real_numbers_and_flags_invented_ones():
    results = [{"liability_pv": 1_788_649_868.8, "effective_duration": 11.99, "funded_ratio": 0.88}]
    good = grounding_audit("Liability is $1,788.6M, duration 12.0 years, funded ratio 88.0%.", results)
    assert good["grounding_score"] == 1.0
    bad = grounding_audit("Liability is $1,788.6M and a hedge would cost $412.5M.", results)
    assert bad["unsupported"] == ["$412.5M"]


def _block(**kw):
    return NS(**kw)


class ScriptedClient:
    """Plays the model: first asks for liability_summary, then answers using the numbers it got back."""

    def __init__(self):
        self.calls = []
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return NS(stop_reason="tool_use", content=[_block(type="tool_use", id="t1", name="liability_summary", input={})])
        result = json.loads(kw["messages"][-1]["content"][0]["content"])
        text = f"Duration is {result['effective_duration']:.1f} years and the plan is {result['funded_ratio'] * 100:.1f}% funded."
        return NS(stop_reason="end_turn", content=[_block(type="text", text=text)])


def test_agent_loop_runs_tools_and_audits_answer(engine):
    client = ScriptedClient()
    a = ask("How long is the liability?", engine, client=client)
    assert [c["tool"] for c in a.tool_calls] == ["liability_summary"]
    assert a.audit["grounding_score"] == 1.0 and a.stop_reason == "end_turn"
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    # the tool result went back in a single user message keyed to the tool_use id
    msgs = client.calls[1]["messages"]  # same list the loop keeps appending to
    assert msgs[2]["role"] == "user" and msgs[2]["content"][0]["tool_use_id"] == "t1"


def test_bad_tool_input_is_returned_as_error_not_raised(engine):
    class C(ScriptedClient):
        def create(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                return NS(stop_reason="tool_use", content=[_block(type="tool_use", id="t1", name="build_hedge",
                                                                  input={"target_ratio": 5, "physical_fraction": 0.4,
                                                                         "allow_swaps": True})])
            assert kw["messages"][-1]["content"][0]["is_error"] is True
            return NS(stop_reason="end_turn", content=[_block(type="text", text="That hedge ratio is out of range.")])

    a = ask("Hedge 500%", engine, client=C())
    assert "error" in a.tool_calls[0]
