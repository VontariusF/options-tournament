"""Offline tests for the strategy-card executor — no network."""

import datetime as dt

from pma_brokers.alpaca.broker import AlpacaPaperBroker, ArmingGate, OrderIntent, OrderRefused
from pma_brokers.alpaca.client import AlpacaClient
from pma_brokers.alpaca.options import build_occ_symbol

from options_tournament.execute import StrategyCard, execute_card, plan_card


def _client():
    return AlpacaClient(
        key="k", secret="s", endpoint="https://paper-api.alpaca.markets/v2",
        min_spacing_s=0.0, sleep=lambda *_: None,
    )


def test_card_rejects_unknown_structure():
    try:
        StrategyCard.from_dict({"underlying": "AAPL", "structure": "iron_condor"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "structure" in str(e)


def test_card_requires_underlying():
    try:
        StrategyCard.from_dict({"structure": "long_call"})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "underlying" in str(e)


def test_plan_long_call_offline(monkeypatch):
    exp = dt.date.today() + dt.timedelta(days=7)
    occ = build_occ_symbol("AAPL", exp, "C", 190.0)
    chain = [{"symbol": occ, "bid": 2.0, "ask": 2.1, "delta": 0.55}]

    class _B(AlpacaPaperBroker):
        def option_chain(self, *a, **k):
            return chain

        def nav(self):
            return {"equity": 100_000.0}

    broker = _B(client=_client(), gate=ArmingGate(armed=False))
    monkeypatch.setattr("options_tournament.execute._spot", lambda *_: 190.0)
    card = StrategyCard.from_dict({"underlying": "AAPL", "structure": "long_call", "dte": 7, "delta": 0.55})
    planned = plan_card(card, broker=broker)
    assert planned["ok"] is True
    intent: OrderIntent = planned["intent"]
    assert intent.side == "buy" and intent.asset_class == "option"
    assert intent.legs is None
    out = execute_card(card, dry_run=True, arm=False, broker=broker)
    assert out["submitted"] is False and out["dry_run"] is True


def test_execute_refuses_submit_when_disarmed_env(monkeypatch):
    exp = dt.date.today() + dt.timedelta(days=7)
    occ = build_occ_symbol("AAPL", exp, "C", 190.0)
    chain = [{"symbol": occ, "bid": 2.0, "ask": 2.1, "delta": 0.55}]

    class _B(AlpacaPaperBroker):
        def option_chain(self, *a, **k):
            return chain

        def nav(self):
            return {"equity": 100_000.0}

    broker = _B(client=_client(), gate=ArmingGate(armed=False))
    monkeypatch.setattr("options_tournament.execute._spot", lambda *_: 190.0)
    monkeypatch.delenv("OPTIONS_PAPER_ARMED", raising=False)
    card = StrategyCard.from_dict({"underlying": "AAPL", "structure": "long_call"})
    try:
        execute_card(card, dry_run=False, arm=True, broker=broker)
        assert False, "expected OrderRefused"
    except OrderRefused as e:
        assert "OPTIONS_PAPER_ARMED" in str(e)


def test_pricing_tool_ok():
    from options_tournament.tools.options_pricing_tool import OptionsPricingTool
    import json
    raw = OptionsPricingTool().execute(
        spot=100, strike=100, expiry_days=30, volatility=0.2, option_type="call",
    )
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert data["price"] > 0
