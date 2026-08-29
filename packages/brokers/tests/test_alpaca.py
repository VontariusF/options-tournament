"""Offline tests for the Alpaca paper adapter — fail-closed guards, no network."""
import io
import json
import urllib.error

import pytest

from pma_brokers.alpaca import client as alpaca_client
from pma_brokers.alpaca.client import AlpacaClient, AlpacaError, NotPaperEndpoint, _resolve_creds
from pma_brokers.alpaca.broker import (AlpacaPaperBroker, ArmingGate, OrderIntent, OrderRefused,
                                       strategy_from_client_order_id)


def _client(**kw):
    # construct against the paper host with injected creds (no network until a request is made);
    # zero spacing + no-op sleep so retry/backoff tests don't actually wait.
    kw.setdefault("min_spacing_s", 0.0)
    kw.setdefault("sleep", lambda *_: None)
    return AlpacaClient(key="k", secret="s", endpoint="https://paper-api.alpaca.markets/v2", **kw)


class _Resp:
    """Minimal urlopen response context manager."""
    def __init__(self, body=""): self._b = body.encode()
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _http_error(url, code, detail="err"):
    return urllib.error.HTTPError(url, code, "msg", {}, io.BytesIO(detail.encode()))


def test_creds_options_sleeve_scopes_to_own_keys(monkeypatch):
    """The options sleeve resolves ALPACA_OPTIONS_PAPER_* first and can NEVER fall through to the
    equity account's ALPACA_PAPER_KEY when its own key is present."""
    monkeypatch.setenv("ALPACA_BROKER_TAG", "alpaca_options")
    monkeypatch.delenv("OPTIONS_ONLY", raising=False)
    store = {"ALPACA_OPTIONS_PAPER_KEY": "opt-key", "ALPACA_OPTIONS_PAPER_SECRET": "opt-sec",
             "ALPACA_PAPER_KEY": "EQUITY-key", "ALPACA_PAPER_SECRET": "EQUITY-sec"}
    # emulate get_secret env-then-store lookup across aliases in order
    def fake_get_secret(*keys, default=""):
        import os
        for k in keys:
            if os.getenv(k):
                return os.getenv(k)
        for k in keys:
            if store.get(k):
                return store[k]
        return default
    monkeypatch.setattr("pma_common.secrets.get_secret", fake_get_secret)
    key, secret, _ = _resolve_creds()
    assert key == "opt-key" and secret == "opt-sec"     # NEVER the equity account


def test_creds_options_sleeve_falls_back_to_shared_only_when_scoped_absent(monkeypatch):
    monkeypatch.setenv("OPTIONS_ONLY", "1")
    store = {"ALPACA_PAPER_KEY": "shared-key", "ALPACA_PAPER_SECRET": "shared-sec"}
    monkeypatch.setattr("pma_common.secrets.get_secret",
                        lambda *keys, default="": next((store[k] for k in keys if store.get(k)), default))
    key, secret, _ = _resolve_creds()
    assert key == "shared-key" and secret == "shared-sec"


def test_creds_equity_book_never_reads_options_scoped_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_BROKER_TAG", raising=False)
    monkeypatch.delenv("OPTIONS_ONLY", raising=False)
    seen = []
    def fake_get_secret(*keys, default=""):
        seen.extend(keys)
        return {"ALPACA_PAPER_KEY": "eq", "ALPACA_PAPER_SECRET": "eq"}.get(keys[0], default)
    monkeypatch.setattr("pma_common.secrets.get_secret", fake_get_secret)
    _resolve_creds()
    assert not any("OPTIONS" in k for k in seen)        # equity path never touches options names


def test_client_refuses_non_paper_endpoint():
    with pytest.raises(NotPaperEndpoint):
        AlpacaClient(key="k", secret="s", endpoint="https://api.alpaca.markets/v2")  # live host
    # paper host is accepted
    c = _client()
    assert "paper-api.alpaca.markets" in c.trading_base


def test_submit_refused_when_disarmed():
    b = AlpacaPaperBroker(client=_client())          # gate DISARMED by default
    intent = OrderIntent(strategy="pead", symbol="AAPL", side="buy", asset_class="us_equity",
                         qty=1, as_of="2026-07-23")
    with pytest.raises(OrderRefused):
        b.submit(intent)


def test_submit_validates_asset_class_and_qty_when_armed():
    gate = ArmingGate(armed=True)
    b = AlpacaPaperBroker(client=_client(), gate=gate)
    with pytest.raises(OrderRefused):
        b.submit(OrderIntent("pead", "AAPL", "buy", "futures", qty=1))          # bad asset class
    with pytest.raises(OrderRefused):
        b.submit(OrderIntent("pead", "AAPL", "buy", "us_equity"))               # neither qty nor notional
    with pytest.raises(OrderRefused):
        b.submit(OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, notional=100))  # both


def test_client_order_id_is_deterministic_and_idempotent():
    a = OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-23")
    a2 = OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-23")  # identical intent
    b = OrderIntent("pead", "AAPL", "buy", "us_equity", qty=5, as_of="2026-07-23")  # qty differs
    c = OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-24")  # date differs
    assert a.client_order_id() == a2.client_order_id()  # identical intent -> same (retry-idempotent) id
    assert a.client_order_id() != b.client_order_id()   # qty now distinguishes (audit P2 collision fix)
    assert a.client_order_id() != c.client_order_id()   # different date -> different id
    assert a.client_order_id().startswith("pma-pead-")  # strategy carried un-hashed for attribution


def test_client_order_id_carries_strategy_for_attribution():
    cid = OrderIntent("pead_sued", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-23").client_order_id()
    assert strategy_from_client_order_id(cid) == "pead_sued"
    assert strategy_from_client_order_id("pma-deadbeef") == "unknown"   # legacy hash-only id degrades
    assert strategy_from_client_order_id("") == "unknown"


def test_crypto_tif_coerced_and_validated():
    gate = ArmingGate(armed=True)

    class _Rec:  # capture the posted body without network
        def __init__(self): self.body = None
        def post(self, path, body): self.body = body; return {"id": "x", **body}
        def get(self, *a, **k): return {}

    rec = _Rec()
    b = AlpacaPaperBroker(client=rec, gate=gate)
    # crypto default 'day' TIF is coerced to gtc (Alpaca 422s on 'day' for crypto)
    b.submit(OrderIntent("hf", "BTC/USD", "buy", "crypto", qty=0.1, as_of="d"))
    assert rec.body["time_in_force"] == "gtc"
    # an explicitly invalid crypto TIF is refused
    with pytest.raises(OrderRefused):
        b.submit(OrderIntent("hf", "BTC/USD", "buy", "crypto", qty=0.1, time_in_force="opg", as_of="d"))
    # a limit order without a price is refused (silent no-op at the API otherwise)
    with pytest.raises(OrderRefused):
        b.submit(OrderIntent("hf", "AAPL", "buy", "us_equity", qty=1, order_type="limit", as_of="d"))


# ── client _request: retry classification, backoff, error surfacing (was untested — audit gap) ──
def test_request_retries_5xx_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(req.full_url, 503)          # transient -> retried
        return _Resp('{"ok": true}')
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert _client().get("account") == {"ok": True}
    assert calls["n"] == 2


def test_request_raises_on_4xx_without_retry(monkeypatch):
    calls = {"n": 0}
    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(req.full_url, 400, "bad request")   # client error -> surfaced immediately
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(AlpacaError):
        _client().get("account")
    assert calls["n"] == 1


def test_submit_payload_shape_when_armed(monkeypatch):
    captured = {}
    def fake_urlopen(req, timeout=None):
        captured["method"] = req.method
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        return _Resp('{"id": "o1"}')
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    b.submit(OrderIntent("pead", "AAPL", "buy", "us_equity", qty=3, as_of="2026-07-23"))
    assert captured["method"] == "POST" and captured["url"].endswith("/orders")
    body = captured["body"]
    assert body["symbol"] == "AAPL" and body["side"] == "buy" and body["type"] == "market"
    assert body["qty"] == "3" and body["time_in_force"] == "day"
    assert body["client_order_id"].startswith("pma-pead-")


def test_submit_reconciles_duplicate_after_failed_post(monkeypatch):
    # a POST that 422s (duplicate) must reconcile via GET-by-client_order_id, not surface a failure
    def fake_urlopen(req, timeout=None):
        if req.method == "POST":
            raise _http_error(req.full_url, 422, "order already exists")
        return _Resp('{"id": "o1", "status": "accepted"}')   # the reconcile GET finds it
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    res = b.submit(OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-23"))
    assert res["id"] == "o1" and res["status"] == "accepted"


def test_submit_raises_when_reconcile_finds_nothing(monkeypatch):
    # POST fails AND the order truly isn't there (GET 404) -> surface the failure
    def fake_urlopen(req, timeout=None):
        if req.method == "POST":
            raise _http_error(req.full_url, 500, "server error")
        raise _http_error(req.full_url, 404, "not found")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    b = AlpacaPaperBroker(client=_client(), gate=ArmingGate(armed=True))
    with pytest.raises(AlpacaError):
        b.submit(OrderIntent("pead", "AAPL", "buy", "us_equity", qty=1, as_of="2026-07-23"))
