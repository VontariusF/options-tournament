"""Offline tests for the Alpaca single-leg LONG options core — no network.

Covers: OCC symbol build/parse, the P0 contract-selection rules, the premium-cap sizing, the
options-chain flatten (via a fake urlopen), and the broker's select_long_call wiring.
"""
import datetime as dt
import json

import pytest

from pma_brokers.alpaca.client import AlpacaClient
from pma_brokers.alpaca.broker import AlpacaPaperBroker, ArmingGate, OrderIntent
from pma_brokers.alpaca.options import (build_occ_symbol, parse_occ_symbol, select_call_contract,
                                        size_by_premium_cap, merge_chain_alpaca_primary,
                                        CONTRACT_MULTIPLIER)


# ── close-side chain merge: require_tradable + canonical-OCC RH overlay ─────────────────────────────
def test_merge_open_side_drops_untradable_rows():
    # default require_tradable=True: a one-sided (deep-ITM short near assignment) row is dropped
    alp = [{"symbol": "AAPL261218P00300000", "bid": 0, "ask": 12.0, "delta": None}]
    assert merge_chain_alpaca_primary(alp, []) == []


def test_merge_close_side_keeps_untradable_and_overlays_delta_despite_format_mismatch():
    # deep-ITM short put, one-sided Alpaca quote, no Alpaca delta; RH carries delta but under a
    # DIFFERENTLY FORMATTED OCC symbol (lowercase root / no zero-pad). The canonical-OCC join must
    # still overlay the delta, and require_tradable=False must keep the row.
    alp = [{"symbol": "AAPL261218P00300000", "bid": 0, "ask": 12.0, "delta": None}]
    rh = [{"symbol": "aapl261218P00300000", "delta": -0.72}]
    out = merge_chain_alpaca_primary(alp, rh, require_tradable=False)
    assert len(out) == 1
    assert out[0]["delta"] == -0.72        # trigger stays alive for the assignment guard


def test_merge_never_overwrites_present_alpaca_greeks():
    alp = [{"symbol": "AAPL261218C00150000", "bid": 1.0, "ask": 1.1, "delta": 0.4}]
    rh = [{"symbol": "AAPL261218C00150000", "delta": 0.9}]
    out = merge_chain_alpaca_primary(alp, rh)
    assert out[0]["delta"] == 0.4          # Alpaca wins; RH only fills None


def _client(**kw):
    kw.setdefault("min_spacing_s", 0.0)
    kw.setdefault("sleep", lambda *_: None)
    return AlpacaClient(key="k", secret="s", endpoint="https://paper-api.alpaca.markets/v2", **kw)


class _Resp:
    def __init__(self, body=""): self._b = body.encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ── OCC / OSI symbol build + parse ────────────────────────────────────────────────────────────────
def test_occ_build_roundtrip():
    sym = build_occ_symbol("AAPL", dt.date(2024, 8, 16), "C", 190.0)
    assert sym == "AAPL240816C00190000"
    meta = parse_occ_symbol(sym)
    assert meta == {"underlying": "AAPL", "expiration": dt.date(2024, 8, 16),
                    "option_type": "C", "strike": 190.0}


def test_occ_build_put_and_fractional_strike():
    # $7.50 strike, put, single-letter root
    sym = build_occ_symbol("f", dt.date(2025, 1, 17), "put", 7.5)
    assert sym == "F250117P00007500"
    assert parse_occ_symbol(sym)["strike"] == 7.5
    assert parse_occ_symbol(sym)["option_type"] == "P"


def test_occ_build_rejects_bad_inputs():
    with pytest.raises(ValueError):
        build_occ_symbol("TOOLONGROOT", dt.date(2025, 1, 17), "C", 10.0)   # >6 char root
    with pytest.raises(ValueError):
        build_occ_symbol("AAPL", dt.date(2025, 1, 17), "X", 10.0)          # bad type
    with pytest.raises(ValueError):
        build_occ_symbol("AAPL", dt.date(2025, 1, 17), "C", 0.0)           # non-positive strike


def test_occ_parse_rejects_non_option():
    with pytest.raises(ValueError):
        parse_occ_symbol("AAPL")            # a plain equity ticker is not an OCC symbol
    with pytest.raises(ValueError):
        parse_occ_symbol("")
    # tolerates surrounding whitespace / lowercase
    assert parse_occ_symbol(" aapl240816c00190000 ")["underlying"] == "AAPL"


# ── contract selection (P0 rules) ─────────────────────────────────────────────────────────────────
def _row(sym, bid, ask, delta=None):
    return {"symbol": sym, "bid": bid, "ask": ask,
            "mid": ((bid + ask) / 2 if bid and ask else None), "delta": delta}


AS_OF = dt.date(2026, 7, 23)


def _call(strike, exp_days, bid, ask, delta):
    exp = AS_OF + dt.timedelta(days=exp_days)
    return _row(build_occ_symbol("AAPL", exp, "C", strike), bid, ask, delta)


def test_select_picks_delta_nearest_on_target_expiry():
    chain = [
        _call(190, 7, 1.0, 1.1, 0.60),    # dte 7 (== target), delta 0.60
        _call(195, 7, 0.5, 0.6, 0.45),    # dte 7, delta 0.45
        _call(185, 7, 2.0, 2.1, 0.75),    # dte 7, delta 0.75
        _call(190, 3, 0.8, 0.9, 0.55),    # dte 3 (wrong expiry — target is 7)
    ]
    got = select_call_contract(chain, AS_OF, target_delta=0.55, target_dte=7)
    # target-dte expiry is the 7d one; nearest |delta| to 0.55 there is the 0.60 (|.05|) vs 0.45 (|.10|)
    assert parse_occ_symbol(got["symbol"])["strike"] == 190.0
    assert got["dte"] == 7 and got["cost"] == 1.1     # cost = ask (marketable-buy worst case)


def test_select_filters_expiry_window_and_one_sided_and_type():
    chain = [
        _call(190, 0, 1.0, 1.1, 0.55),        # dte 0 < min_dte → excluded (no same-day expiry)
        _call(190, 30, 1.0, 1.1, 0.55),       # dte 30 > max_dte → excluded
        _call(190, 5, 0.0, 1.1, 0.55),        # no bid (one-sided) → excluded (can't exit)
        _row(build_occ_symbol("AAPL", AS_OF + dt.timedelta(days=5), "P", 190), 1.0, 1.1, -0.5),  # put
        _call(200, 5, 0.9, 1.0, 0.50),        # the only survivor
    ]
    got = select_call_contract(chain, AS_OF, target_delta=0.55, min_dte=1, max_dte=10, target_dte=7)
    assert parse_occ_symbol(got["symbol"])["strike"] == 200.0


def test_select_returns_none_when_nothing_qualifies():
    assert select_call_contract([], AS_OF) is None
    # all outside the DTE window
    assert select_call_contract([_call(190, 60, 1.0, 1.1, 0.55)], AS_OF, max_dte=10) is None


def test_select_atm_fallback_when_no_greeks():
    # no deltas anywhere → fall back to strike nearest spot
    chain = [_call(180, 7, 1.0, 1.1, None), _call(190, 7, 1.0, 1.1, None),
             _call(210, 7, 1.0, 1.1, None)]
    got = select_call_contract(chain, AS_OF, spot=192.0, target_dte=7)
    assert parse_occ_symbol(got["symbol"])["strike"] == 190.0
    # no delta AND no spot → refuse to size blind
    assert select_call_contract(chain, AS_OF, spot=None, target_dte=7) is None


# ── premium-cap sizing (convex → tight) ──────────────────────────────────────────────────────────
def test_size_absolute_cap_binds():
    # $1.10 ask * 100 = $110/contract; abs cap $250 (no equity) → 2 contracts, est $220 <= cap
    s = size_by_premium_cap(1.10, equity=None, max_premium_abs=250.0)
    assert s["qty"] == 2 and s["premium_cap"] == 250.0 and s["est_cost"] == 220.0
    assert s["est_cost"] <= s["premium_cap"]          # max loss never exceeds the cap


def test_size_equity_fraction_binds_when_smaller():
    # equity $20k * 0.5% = $100 cap < $250 abs → only $100 to spend; $1.10*100=$110/contract → 0 → skip
    s = size_by_premium_cap(1.10, equity=20_000, max_premium_abs=250.0, max_premium_frac=0.005)
    assert s["premium_cap"] == 100.0 and s["qty"] == 0 and s["est_cost"] == 0.0


def test_size_zero_on_bad_price():
    assert size_by_premium_cap(0.0, equity=100_000)["qty"] == 0
    assert size_by_premium_cap(None, equity=100_000)["qty"] == 0
    # a cheap contract under a big cap → many contracts, still <= cap
    s = size_by_premium_cap(0.20, equity=None, max_premium_abs=250.0)
    assert s["qty"] == 12 and s["est_cost"] == round(12 * 0.20 * CONTRACT_MULTIPLIER, 2)


# ── client.options_chain flatten (fake urlopen; no network) ───────────────────────────────────────
def test_options_chain_flatten(monkeypatch):
    payload = {"snapshots": {
        "AAPL260731C00190000": {
            "latestQuote": {"bp": 1.0, "ap": 1.2, "bs": 10, "as": 8},
            "latestTrade": {"p": 1.1},
            "greeks": {"delta": 0.55, "gamma": 0.02, "theta": -0.05, "vega": 0.10, "rho": 0.01},
            "impliedVolatility": 0.42,
        },
        "AAPL260731C00200000": {"latestQuote": {}, "greeks": {}},   # sparse → Nones, no crash
    }, "next_page_token": None}

    def fake_urlopen(req, timeout=None):
        assert req.method == "GET"
        assert "data.alpaca.markets/v1beta1/options/snapshots/AAPL" in req.full_url
        return _Resp(json.dumps(payload))
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    res = _client().options_chain("AAPL", {"type": "call"})
    rows = {r["symbol"]: r for r in res["contracts"]}
    a = rows["AAPL260731C00190000"]
    assert a["bid"] == 1.0 and a["ask"] == 1.2 and a["mid"] == 1.1
    assert a["delta"] == 0.55 and a["iv"] == 0.42 and a["last"] == 1.1
    b = rows["AAPL260731C00200000"]
    assert b["bid"] is None and b["ask"] is None and b["mid"] is None and b["delta"] is None


def test_broker_select_long_call_endtoend(monkeypatch):
    # broker.select_long_call: fetch chain (fake) → apply P0 rules → return chosen contract
    exp = AS_OF + dt.timedelta(days=7)
    esym = f"AAPL{exp:%y%m%d}C00190000"
    payload = {"snapshots": {
        esym: {"latestQuote": {"bp": 1.0, "ap": 1.1}, "greeks": {"delta": 0.55}},
    }, "next_page_token": None}
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Resp(json.dumps(payload)))
    b = AlpacaPaperBroker(client=_client())
    got = b.select_long_call("AAPL", AS_OF, target_delta=0.55)
    assert got["symbol"] == esym and got["cost"] == 1.1 and got["dte"] == 7


def test_option_order_intent_is_single_leg_long_limit(monkeypatch):
    # the intent the executor builds: option / buy / qty (contracts) / limit — never a spread/short
    intent = OrderIntent(strategy="pead_opt_earn", symbol="AAPL260731C00190000", side="buy",
                         asset_class="option", qty=2, order_type="limit", limit_price=1.1,
                         time_in_force="day", as_of="2026-07-23")
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _Resp('{"id": "o1", "status": "accepted"}')
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b.submit(intent)
    body = captured["body"]
    assert body["symbol"] == "AAPL260731C00190000" and body["side"] == "buy"
    assert body["type"] == "limit" and body["qty"] == "2" and body["limit_price"] == "1.1"
    assert body["time_in_force"] == "day"
    assert body["client_order_id"].startswith("pma-pead_opt_earn-")


def test_merge_alpaca_primary_keeps_alpaca_quotes_overlays_rh_delta():
    from pma_brokers.alpaca.options import merge_chain_alpaca_primary
    alpaca = [_row("AAPL260730C00190000", 1.0, 1.2, delta=None)]
    rh = [_row("AAPL260730C00190000", 9.0, 9.5, delta=0.55)]  # different quotes — must not win
    merged = merge_chain_alpaca_primary(alpaca, rh)
    assert len(merged) == 1
    assert merged[0]["bid"] == 1.0 and merged[0]["ask"] == 1.2
    assert merged[0]["delta"] == 0.55


def test_merge_alpaca_primary_empty_alpaca_does_not_use_rh_quotes():
    from pma_brokers.alpaca.options import merge_chain_alpaca_primary
    rh = [_row("AAPL260730C00190000", 1.0, 1.2, delta=0.4)]
    assert merge_chain_alpaca_primary([], rh) == []
    assert merge_chain_alpaca_primary(None, rh) == []


def test_merge_alpaca_primary_untradeable_alpaca_does_not_use_rh_quotes():
    from pma_brokers.alpaca.options import merge_chain_alpaca_primary
    alpaca = [_row("AAPL260730C00190000", None, None, delta=None)]
    rh = [_row("AAPL260730C00190000", 1.0, 1.2, delta=0.4)]
    merged = merge_chain_alpaca_primary(alpaca, rh)
    assert merged == []
    assert all(r.get("ask") != 1.2 for r in merged)


def test_fetch_chain_alpaca_primary_calls_alpaca_then_rh():
    from pma_brokers.alpaca.options import fetch_chain_alpaca_primary
    calls = []

    class _B:
        def option_chain(self, underlying, **kw):
            calls.append(("alpaca", underlying, kw.get("feed"), kw.get("type")))
            return [_row("AAPL260730C00190000", 1.0, 1.2, delta=None)]

    def rh_fetch(underlying, **kw):
        calls.append(("rh", underlying, kw.get("option_type")))
        return [_row("AAPL260730C00190000", 9.0, 9.5, delta=0.55)]

    rows = fetch_chain_alpaca_primary(_B(), "AAPL", option_type="call", as_of=AS_OF,
                                      min_dte=1, max_dte=10, rh_fetch=rh_fetch)
    assert calls[0][0] == "alpaca" and calls[0][2] == "indicative"
    assert calls[1][0] == "rh"
    assert rows[0]["ask"] == 1.2 and rows[0]["delta"] == 0.55
