"""Read-only options chain via Alpaca paper market data."""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from options_tournament.agent.tools import BaseTool

_MAX_CONTRACTS_PER_SIDE = 60


class OptionsChainTool(BaseTool):
    name = "get_options_chain"
    description = (
        "Fetch the US-listed options chain (calls and puts) for one underlying "
        "from Alpaca paper market data. Optional expiration as YYYY-MM-DD."
    )
    parameters = {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "US underlying, e.g. AAPL"},
            "expiration": {
                "type": "string",
                "description": "Optional expiration YYYY-MM-DD. Omit for a near-dated window.",
            },
        },
        "required": ["ticker"],
    }

    def execute(self, **kwargs: Any) -> str:
        ticker = str(kwargs.get("ticker") or "").strip().upper()
        if ticker.endswith(".US"):
            ticker = ticker[:-3]
        if not ticker:
            return json.dumps({"ok": False, "error": "ticker is required"})
        expiration = kwargs.get("expiration")
        try:
            from pma_brokers.alpaca.broker import AlpacaPaperBroker
            from pma_brokers.alpaca.options import parse_occ_symbol

            broker = AlpacaPaperBroker()
            as_of = dt.date.today()
            if expiration:
                exp = dt.date.fromisoformat(str(expiration))
                chain = broker.option_chain(
                    ticker,
                    expiration_gte=exp.isoformat(),
                    expiration_lte=exp.isoformat(),
                    feed="indicative",
                )
            else:
                lo = (as_of + dt.timedelta(days=1)).isoformat()
                hi = (as_of + dt.timedelta(days=45)).isoformat()
                chain = broker.option_chain(
                    ticker, expiration_gte=lo, expiration_lte=hi, feed="indicative",
                )
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"ok": False, "error": str(exc)[:300]})

        calls, puts = [], []
        for row in chain or []:
            try:
                meta = parse_occ_symbol(row.get("symbol", ""))
            except ValueError:
                continue
            bucket = calls if meta["option_type"] == "C" else puts
            if len(bucket) < _MAX_CONTRACTS_PER_SIDE:
                bucket.append(row)
        return json.dumps({
            "ok": True,
            "market": "us",
            "source": "alpaca_paper",
            "data": {
                "ticker": ticker,
                "calls_count": len(calls),
                "puts_count": len(puts),
                "calls": calls,
                "puts": puts,
            },
        }, default=str)
