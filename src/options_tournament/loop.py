"""One-shot agent turn: LLM plus options tools. Optional; CLI/API work without it."""

from __future__ import annotations

import json
import os

import httpx

from options_tournament.tools.options_chain_tool import OptionsChainTool
from options_tournament.tools.options_pricing_tool import OptionsPricingTool

_TOOLS = [OptionsPricingTool(), OptionsChainTool()]


def _openai_tools() -> list[dict]:
    out = []
    for t in _TOOLS:
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        })
    return out


def _dispatch(name: str, arguments: dict) -> str:
    for t in _TOOLS:
        if t.name == name:
            return t.execute(**arguments)
    return json.dumps({"error": f"unknown tool {name}"})


def run_loop(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return "Set OPENAI_API_KEY to use chat. account / chain / execute work without an LLM."
    messages = [
        {"role": "system", "content": "You size defined-risk US equity options. Use tools for chains and pricing. Never invent fills."},
        {"role": "user", "content": prompt},
    ]
    with httpx.Client(timeout=60.0) as client:
        for _ in range(6):
            resp = client.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "tools": _openai_tools()},
            )
            resp.raise_for_status()
            choice = (resp.json().get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content") or ""
            for call in calls:
                fn = call.get("function") or {}
                args = json.loads(fn.get("arguments") or "{}")
                result = _dispatch(fn.get("name") or "", args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": result,
                })
    return "stopped after tool-call budget"
