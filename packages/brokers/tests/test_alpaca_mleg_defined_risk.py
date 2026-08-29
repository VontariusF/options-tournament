"""PHASE-4b — multi-leg (mleg) defined-risk order path + the risk-layer primitives (offline, no network).

The load-bearing money-path invariant: broker.submit() REFUSES any mleg whose net position is naked-short
(a short leg with no covering long of >= equal qty at a bounding strike), independent of upstream sizing,
and NEVER lets a market mleg through. It ACCEPTS a proper defined-risk spread. The single-leg body stays
byte-unchanged. Also covers max-loss sizing (qty=0 skip), the per-leg liquidity gate, and the credit
structure gate.
"""
import datetime as dt
import json

import pytest

from pma_brokers.alpaca.broker import (AlpacaPaperBroker, ArmingGate, OrderIntent, OrderRefused,
                                       assert_defined_risk)
from pma_brokers.alpaca.client import AlpacaClient
from pma_brokers.alpaca.options import (build_occ_symbol, size_by_max_loss, leg_liquidity_ok,
                                        credit_structure_ok, CONTRACT_MULTIPLIER)


def _client(**kw):
    kw.setdefault("min_spacing_s", 0.0)
    kw.setdefault("sleep", lambda *_: None)
    return AlpacaClient(key="k", secret="s", endpoint="https://paper-api.alpaca.markets/v2", **kw)


class _Resp:
    def __init__(self, body=""): self._b = body.encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


EXP = dt.date(2026, 9, 18)
SHORT_PUT = build_occ_symbol("NVDA", EXP, "P", 220.0)     # SELL (anchor)
WING_PUT = build_occ_symbol("NVDA", EXP, "P", 210.0)      # BUY (wing, $10 lower)


def _spread_legs():
    return [{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
            {"symbol": WING_PUT, "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}]


def _mleg_intent(legs, *, limit_price=1.20, order_type="limit", qty=2):
    return OrderIntent(strategy="tournament", symbol="NVDA", side="sell", asset_class="option",
                       qty=qty, order_type=order_type, limit_price=limit_price, time_in_force="day",
                       as_of="2026-08-16", order_class="mleg", legs=legs)


# --------------------------------------------------------------------------- #
# defined-risk assertion — the last-line guarantee
# --------------------------------------------------------------------------- #
def test_assert_defined_risk_accepts_proper_put_spread():
    assert_defined_risk(_spread_legs())                  # no raise = covered


def test_assert_defined_risk_rejects_naked_short_put():
    # a lone short put with NO covering long → naked, undefined risk.
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        assert_defined_risk([{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 1,
                              "position_intent": "sell_to_open"}])


def test_assert_defined_risk_rejects_wing_on_wrong_side():
    # a "wing" at a HIGHER strike than the short put does NOT bound the downside → still naked-risk.
    higher = build_occ_symbol("NVDA", EXP, "P", 230.0)
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        assert_defined_risk([{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 1,
                              "position_intent": "sell_to_open"},
                             {"symbol": higher, "side": "buy", "ratio_qty": 1,
                              "position_intent": "buy_to_open"}])


def test_assert_defined_risk_rejects_short_qty_exceeding_long():
    # 2 short puts covered by only 1 long put → uncovered net short.
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        assert_defined_risk([{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 2,
                              "position_intent": "sell_to_open"},
                             {"symbol": WING_PUT, "side": "buy", "ratio_qty": 1,
                              "position_intent": "buy_to_open"}])


def test_assert_defined_risk_rejects_wrong_expiry_cover():
    # a long put at a lower strike but a DIFFERENT expiration does not define the risk (calendar).
    other = build_occ_symbol("NVDA", dt.date(2026, 10, 16), "P", 210.0)
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        assert_defined_risk([{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 1,
                              "position_intent": "sell_to_open"},
                             {"symbol": other, "side": "buy", "ratio_qty": 1,
                              "position_intent": "buy_to_open"}])


def test_assert_defined_risk_call_credit_spread_ok_and_naked_rejected():
    sc = build_occ_symbol("NVDA", EXP, "C", 230.0)       # SELL lower-strike call (anchor)
    lc = build_occ_symbol("NVDA", EXP, "C", 240.0)       # BUY higher-strike call (wing)
    assert_defined_risk([{"symbol": sc, "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
                         {"symbol": lc, "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"}])
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        # long call at a LOWER strike does not cap the upside of the short call
        low = build_occ_symbol("NVDA", EXP, "C", 220.0)
        assert_defined_risk([{"symbol": sc, "side": "sell", "ratio_qty": 1,
                              "position_intent": "sell_to_open"},
                             {"symbol": low, "side": "buy", "ratio_qty": 1,
                              "position_intent": "buy_to_open"}])


# --------------------------------------------------------------------------- #
# broker.submit() — mleg body + refusals
# --------------------------------------------------------------------------- #
def test_submit_mleg_emits_defined_risk_body(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp('{"id": "m1", "status": "accepted"}')
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    b.submit(_mleg_intent(_spread_legs()))
    body = captured["body"]
    assert body["order_class"] == "mleg"
    assert body["qty"] == "2" and body["type"] == "limit" and body["limit_price"] == "1.2"
    assert body["time_in_force"] == "day"
    assert "symbol" not in body                           # mleg carries no top-level symbol
    legs = body["legs"]
    assert {l["symbol"] for l in legs} == {SHORT_PUT, WING_PUT}
    sell = next(l for l in legs if l["side"] == "sell")
    assert sell["position_intent"] == "sell_to_open" and sell["ratio_qty"] == "1"
    assert body["client_order_id"].startswith("pma-tournament-")


def test_submit_mleg_refuses_naked_short_at_broker(monkeypatch):
    # even if a caller hands submit() a naked mleg, the broker refuses BEFORE any POST.
    posted = {"n": 0}
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: posted.__setitem__("n", posted["n"] + 1))
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    naked = [{"symbol": SHORT_PUT, "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
             {"symbol": build_occ_symbol("NVDA", EXP, "P", 230.0), "side": "buy", "ratio_qty": 1,
              "position_intent": "buy_to_open"}]
    with pytest.raises(OrderRefused, match="NAKED SHORT"):
        b.submit(_mleg_intent(naked))
    assert posted["n"] == 0                               # nothing was posted


def test_submit_mleg_refuses_market_order(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("must not POST"))
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    with pytest.raises(OrderRefused, match="LIMIT"):
        b.submit(_mleg_intent(_spread_legs(), order_type="market", limit_price=None))


def test_submit_mleg_refused_when_disarmed():
    b = AlpacaPaperBroker(client=_client())              # DISARMED
    with pytest.raises(OrderRefused, match="DISARMED"):
        b.submit(_mleg_intent(_spread_legs()))


def test_single_leg_body_byte_unchanged(monkeypatch):
    # a legs=None option order is BYTE-IDENTICAL to the pre-mleg single-leg body (no new keys leak in).
    captured = {}
    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: (captured.__setitem__("body", json.loads(req.data.decode()))
                                                   or _Resp('{"id":"o1","status":"accepted"}')))
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    b.submit(OrderIntent(strategy="pead_opt_earn", symbol=SHORT_PUT, side="buy", asset_class="option",
                         qty=2, order_type="limit", limit_price=1.1, time_in_force="day",
                         as_of="2026-08-16"))
    assert set(captured["body"]) == {"symbol", "side", "type", "time_in_force", "client_order_id",
                                     "qty", "limit_price"}
    assert "order_class" not in captured["body"] and "legs" not in captured["body"]


def test_single_leg_client_order_id_unchanged():
    # the deterministic single-leg id must NOT shift because the mleg fields exist (default None).
    intent = OrderIntent(strategy="pead_opt_earn", symbol="AAPL260731C00190000", side="buy",
                         asset_class="option", qty=2, order_type="limit", limit_price=1.1,
                         time_in_force="day", as_of="2026-07-23")
    assert intent.client_order_id().startswith("pma-pead_opt_earn-")
    # a distinct mleg leg-set yields a DIFFERENT id (legs folded into the seed only when present)
    mleg = _mleg_intent(_spread_legs())
    assert mleg.client_order_id() != intent.client_order_id()


# --------------------------------------------------------------------------- #
# risk-layer primitives — max-loss sizing + liquidity gate + credit structure gate
# --------------------------------------------------------------------------- #
def test_size_by_max_loss_1pct_nav_and_zero_skip():
    # $10 width − $2 credit = $8/share = $800 max loss/contract; 1% of $200k = $2000 → 2 contracts.
    s = size_by_max_loss(8.0, 200_000, risk_frac=0.01)
    assert s["qty"] == 2 and s["est_max_loss"] == round(2 * 8.0 * CONTRACT_MULTIPLIER, 2)
    assert s["est_max_loss"] <= s["risk_cap"]
    # a max loss that exceeds the whole 1% budget for even one contract → qty 0 (SKIP, never round up).
    assert size_by_max_loss(30.0, 200_000, risk_frac=0.01)["qty"] == 0
    assert size_by_max_loss(8.0, None)["qty"] == 0        # no NAV → skip
    assert size_by_max_loss(0.0, 200_000)["qty"] == 0     # degenerate max-loss → skip


def test_leg_liquidity_gate():
    good = {"bid": 2.00, "ask": 2.10, "mid": 2.05, "open_interest": 1200, "volume": 400}
    assert leg_liquidity_ok(good)[0] is True
    assert leg_liquidity_ok({**good, "open_interest": 100})[0] is False   # OI < 500
    assert leg_liquidity_ok({**good, "volume": 50})[0] is False           # vol < 100
    wide = {"bid": 1.0, "ask": 1.5, "mid": 1.25, "open_interest": 1200, "volume": 400}
    assert leg_liquidity_ok(wide)[0] is False            # 40% spread > 10%
    # fail-closed on missing OI/volume (the RH bridge must forward them before Phase 5)
    assert leg_liquidity_ok({"bid": 2.0, "ask": 2.1, "mid": 2.05})[0] is False


def test_credit_structure_gate():
    assert credit_structure_ok(2.0, 10.0)[0] is True      # 20% of width, > $0.10
    assert credit_structure_ok(1.5, 10.0)[0] is False     # 15% of width < 20%
    assert credit_structure_ok(0.05, 0.20)[0] is False    # < $0.10 absolute
    assert credit_structure_ok(-1.0, 10.0)[0] is False    # non-positive credit


def test_assert_defined_risk_rejects_side_intent_disagreement():
    # QA MEDIUM: a leg whose side contradicts its position_intent is ambiguous (which field the venue
    # honors decides naked-vs-covered) → must fail-closed, not be classified by side alone.
    legs = _spread_legs()
    legs[0] = {**legs[0], "side": "buy"}          # short-put leg now claims side=buy vs sell_to_open
    with pytest.raises(OrderRefused, match="disagrees with|position_intent"):
        assert_defined_risk(legs)


def test_assert_defined_risk_rejects_bad_position_intent():
    legs = _spread_legs()
    legs[0] = {**legs[0], "position_intent": "open"}   # not a valid Alpaca intent
    with pytest.raises(OrderRefused, match="position_intent"):
        assert_defined_risk(legs)
