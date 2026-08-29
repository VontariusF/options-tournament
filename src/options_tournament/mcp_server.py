"""stdio JSON-RPC MCP server: account, chain, price, execute."""

from __future__ import annotations

import json
import sys
from typing import Any

from options_tournament.execute import StrategyCard, execute_card
from options_tournament.tools.options_pricing_tool import OptionsPricingTool


TOOLS = [
    {
        "name": "account",
        "description": "Read the Alpaca paper account snapshot (equity, cash, buying power).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_options_chain",
        "description": "Fetch an Alpaca paper options chain for a US ticker.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "expiration": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "options_pricing",
        "description": "Black-Scholes price and Greeks.",
        "inputSchema": OptionsPricingTool.parameters,
    },
    {
        "name": "execute_card",
        "description": "Plan (or submit, if armed) a defined-risk strategy card.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "underlying": {"type": "string"},
                "structure": {"type": "string", "enum": ["long_call", "long_put", "credit_put_spread"]},
                "dte": {"type": "integer"},
                "delta": {"type": "number"},
                "wing_width": {"type": "number"},
                "dry_run": {"type": "boolean"},
                "arm": {"type": "boolean"},
            },
            "required": ["underlying", "structure"],
        },
    },
]


def _call(name: str, arguments: dict[str, Any]) -> str:
    if name == "account":
        from pma_brokers.alpaca.broker import AlpacaPaperBroker
        return json.dumps(AlpacaPaperBroker().nav(), default=str)
    if name == "get_options_chain":
        from options_tournament.tools.options_chain_tool import OptionsChainTool
        return OptionsChainTool().execute(**arguments)
    if name == "options_pricing":
        return OptionsPricingTool().execute(**arguments)
    if name == "execute_card":
        card = StrategyCard.from_dict(arguments)
        dry = arguments.get("dry_run", True)
        arm = bool(arguments.get("arm"))
        return json.dumps(execute_card(card, dry_run=dry or not arm, arm=arm and not dry), default=str)
    raise ValueError(f"unknown tool {name}")


def _handle(msg: dict) -> dict | None:
    mid = msg.get("id")
    method = msg.get("method")
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "options-tournament", "version": "0.1.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        try:
            text = _call(name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "error": {"code": -32000, "message": str(exc)[:300]},
            }
    if mid is None:
        return None
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
