"""Offline tests for the Featherless transport — fake ``_post``, no network."""
from __future__ import annotations

import pytest

from options_tournament import featherless as FL
from options_tournament.featherless import (
    ChatResponse,
    FeatherlessClient,
    LLMBudgetExceeded,
    LLMError,
    extract_json_object,
    strip_reasoning,
)


def _client(**kw):
    kw.setdefault("api_key", "test-key")
    kw.setdefault("sleep", lambda *_: None)
    kw.setdefault("daily_call_budget", 0)
    return FeatherlessClient(**kw)


def _ok(content="", tool_calls=None, model="m"):
    msg = {"content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return 200, {"model": model, "choices": [{"message": msg, "finish_reason": "stop"}]}


def test_strip_reasoning_removes_think_and_fences():
    assert strip_reasoning("<think>secret plan</think>  {\"a\":1}") == '{"a":1}'
    assert strip_reasoning("```json\n{\"a\":1}\n```").strip() == '{"a":1}'


def test_extract_json_object_widest_brace():
    assert extract_json_object("noise {\"a\": 1, \"b\": 2} tail") == {"a": 1, "b": 2}
    assert "_error" in extract_json_object("no json here")


def test_chatresponse_json_prefers_tool_args():
    r = ChatResponse(content='{"from":"content"}',
                     tool_calls=[{"function": {"name": "f", "arguments": '{"from":"tool"}'}}])
    assert r.json() == {"from": "tool"}
    r2 = ChatResponse(content='{"from":"content"}')
    assert r2.json() == {"from": "content"}


def test_429_backs_off_then_downshifts_to_fallback():
    c = _client(fallback_model="small-model", max_retries=3)
    seen_models = []

    def fake_post(path, body):
        seen_models.append(body["model"])
        if body["model"] != "small-model":
            return 429, {"error": "over concurrency budget"}
        return _ok(content="ok", model="small-model")

    c._post = fake_post
    r = c.chat([{"role": "user", "content": "hi"}], model="big-model")
    assert r.content == "ok"
    assert seen_models[0] == "big-model" and "small-model" in seen_models


def test_persistent_5xx_raises_llmerror():
    c = _client(max_retries=2)
    c._post = lambda path, body: (503, {"error": "upstream"})
    with pytest.raises(LLMError):
        c.chat([{"role": "user", "content": "hi"}], model="m")


def test_success_returns_normalized_and_strips_reasoning():
    c = _client()
    c._post = lambda path, body: _ok(content="<think>x</think>hello", model="m")
    r = c.chat([{"role": "user", "content": "hi"}], model="m")
    assert r.content == "hello" and r.model == "m" and r.finish_reason == "stop"


def test_tools_and_response_format_threaded_into_body():
    c = _client()
    captured = {}

    def fake_post(path, body):
        captured.update(body)
        return _ok(content="{}")

    c._post = fake_post
    c.chat([{"role": "user", "content": "x"}], model="m",
           tools=[{"type": "function", "function": {"name": "t"}}],
           tool_choice="auto", response_format={"type": "json_object"})
    assert captured["tools"] and captured["tool_choice"] == "auto"
    assert captured["response_format"] == {"type": "json_object"}


def test_daily_call_budget_trips():
    FL._BUDGET._n = 0
    FL._BUDGET._day = ""
    c = _client(daily_call_budget=1)
    c._post = lambda path, body: _ok(content="ok")
    assert c.chat([{"role": "user", "content": "1"}], model="m").content == "ok"
    with pytest.raises(LLMBudgetExceeded):
        c.chat([{"role": "user", "content": "2"}], model="m")
