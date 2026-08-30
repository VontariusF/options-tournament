"""Offline tests for the Featherless specialist roster — scripted client, no network."""
from __future__ import annotations

import json

from options_tournament import agents as A
from options_tournament.agents import (
    _coerce_cards,
    _role_call,
    _tool_check_novelty,
    clamp_card,
    default_ctx,
    run_specialists,
)
from options_tournament.execute import StrategyCard
from options_tournament.featherless import ChatResponse, LLMError

CTX = default_ctx(universe=["AAPL", "MSFT"], book=[
    {"underlying": "AAPL", "structure": "long_call", "dte": 7, "delta": 0.35},
])

_CARD = {
    "underlying": "MSFT",
    "structure": "long_call",
    "dte": 14,
    "delta": 0.35,
    "wing_width": None,
    "thesis": "book is call-heavy on AAPL; rotate to MSFT",
}


def test_clamp_card_snaps_buckets_and_builds_strategy_card():
    raw = {
        "structure_template": "credit_put_spread",
        "underlying_selector": ["nvda"],
        "dte_bucket": 16,
        "target_delta_bucket": 0.31,
        "wing_width_bucket": 12,
        "thesis": "defined-risk put credit",
    }
    out = clamp_card(raw)
    assert out is not None
    card = StrategyCard.from_dict(out)
    assert card.underlying == "NVDA"
    assert card.structure == "credit_put_spread"
    assert card.dte == 14
    assert card.delta == 0.35
    assert card.wing_width == 10.0
    assert out["thesis"]


def test_clamp_card_rejects_unknown_structure():
    assert clamp_card({"underlying": "AAPL", "structure": "iron_condor"}) is None


def test_check_novelty_detects_collision_and_novelty():
    hit = _tool_check_novelty(CTX, underlying="AAPL", structure="long_call")
    assert hit["novel"] is False
    miss = _tool_check_novelty(CTX, underlying="MSFT", structure="long_call")
    assert miss["novel"] is True


def test_coerce_cards_shapes():
    assert _coerce_cards({"cards": [_CARD]}) == [_CARD]
    assert _coerce_cards(_CARD) == [_CARD]
    assert _coerce_cards([_CARD, "junk"]) == [_CARD]
    assert _coerce_cards("nope") == []


class FakeClient:
    def __init__(self, *, fail_reasoning=False):
        self.fail_reasoning = fail_reasoning
        self.proposer_turns = 0

    def chat(self, messages, *, model, tools=None, tool_choice=None, response_format=None,
             temperature=0.0, max_tokens=700):
        sys = (messages[0].get("content") or "") if messages else ""
        if tools:
            self.proposer_turns += 1
            if self.proposer_turns == 1:
                return ChatResponse(content="", model=model, tool_calls=[{
                    "id": "t1", "function": {"name": "check_novelty",
                    "arguments": json.dumps({"underlying": "MSFT", "structure": "long_call"})}}])
            return ChatResponse(content="", model=model, tool_calls=[{
                "id": "t2", "function": {"name": "submit_cards",
                "arguments": json.dumps({"cards": [_CARD]})}}])
        if "researcher" in sys or "strategist" in sys:
            if self.fail_reasoning:
                raise LLMError("reasoning model down")
            return ChatResponse(content="- MSFT long call dte 14", model=model)
        if "critic" in sys:
            return ChatResponse(content='{"flags": []}', model=model)
        return ChatResponse(content="{}", model=model)


def test_full_loop_returns_clamped_card():
    fake = FakeClient()
    out = run_specialists(CTX, n_target=1, client=fake)
    assert len(out) == 1
    card, model_id = out[0]
    assert model_id == A.PROPOSER_MODEL
    assert fake.proposer_turns == 2
    parsed = StrategyCard.from_dict(card)
    assert parsed.underlying == "MSFT" and parsed.structure == "long_call"


def test_run_specialists_populates_trace_sink():
    fake = FakeClient()
    sink: list = []
    out = run_specialists(CTX, 1, client=fake, trace_sink=sink)
    assert out and sink
    roles = [t["role"] for t in sink]
    assert "hypothesis" in roles and "strategist" in roles and "proposer" in roles
    prop = next(t for t in sink if t["role"] == "proposer")
    assert "long_call" in prop["content"]


def test_degrades_when_reasoning_models_fail():
    fake = FakeClient(fail_reasoning=True)
    out = run_specialists(CTX, n_target=1, client=fake)
    assert len(out) == 1 and out[0][1] == A.PROPOSER_MODEL


def test_propose_forces_submit_when_tool_loop_never_submits():
    class NoSubmitThenForced:
        def chat(self, messages, *, model, tools=None, tool_choice=None, **kw):
            if not tools:
                return ChatResponse(content="notes", model=model)
            if tool_choice == "auto":
                return ChatResponse(content="here is prose, no tool call", model=model)
            if isinstance(tool_choice, dict):
                return ChatResponse(content="", model=model, tool_calls=[{
                    "id": "f", "function": {"name": "submit_cards",
                    "arguments": json.dumps({"cards": [_CARD]})}}])
            return ChatResponse(content="{}", model=model)
    out = run_specialists(CTX, 1, client=NoSubmitThenForced())
    assert len(out) == 1 and out[0][1] == A.PROPOSER_MODEL


def test_role_call_falls_back_when_primary_returns_empty():
    class C:
        def chat(self, messages, *, model, **kw):
            return ChatResponse(content=("" if model == "primary" else "notes"), model=model)
    assert _role_call(C(), "primary", "s", "u", fallback="backup") == "notes"
    assert _role_call(C(), "primary", "s", "u") == ""


def test_proposer_total_failure_returns_empty():
    class DeadProposer(FakeClient):
        def chat(self, messages, *, model, tools=None, **kw):
            if tools:
                raise LLMError("proposer down")
            return ChatResponse(content="notes", model=model)
    assert run_specialists(CTX, n_target=2, client=DeadProposer()) == []
