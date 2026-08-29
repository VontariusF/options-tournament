"""Alpaca REST client — stdlib urllib only (matches the fleet's dependency-light broker style).

Hard paper guard: the trading base URL MUST be the Alpaca *paper* host, else construction raises
``NotPaperEndpoint``. There is no live host default and no way to reach one through this client.
Retries with exponential backoff on 429/5xx; a small spacing between requests respects the
200 req/min limit.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

PAPER_HOST = "paper-api.alpaca.markets"
DATA_HOST = "data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


class NotPaperEndpoint(RuntimeError):
    """Raised if the trading endpoint is not the Alpaca paper host — this client is paper-only."""


def _is_options_sleeve() -> bool:
    """True when this process is the OPTIONS sleeve (its own paper account), not the equity book.

    The options sleeve sets ``ALPACA_BROKER_TAG=alpaca_options`` + ``OPTIONS_ONLY=1``; the
    equity book sets neither. Used to scope credential resolution so the options sleeve resolves
    its OWN keys and can NEVER fall through to the equity account's ``ALPACA_PAPER_KEY``."""
    import os
    return (os.getenv("ALPACA_BROKER_TAG", "").strip() == "alpaca_options"
            or os.getenv("OPTIONS_ONLY", "").strip() in ("1", "true", "yes", "on"))


def _resolve_creds():
    """(key, secret, endpoint) from pma_common.secrets (env -> Supabase). Endpoint defaults to the
    paper trading base; never a live host.

    SLEEVE SCOPING (2026-08-28): the options sleeve resolves ``ALPACA_OPTIONS_PAPER_KEY/SECRET``
    FIRST (its own Supabase-stored paper account), falling back to the shared ``ALPACA_PAPER_KEY``
    only if the options-scoped name is unset. This is the safety fix behind arming: a missing
    options env can no longer route options orders to the EQUITY account via the Supabase fallback.
    The equity book is untouched — it never sets the options tag, so it resolves ``ALPACA_PAPER_KEY``
    exactly as before."""
    try:
        from pma_common.secrets import format_secret_miss, get_secret
        if _is_options_sleeve():
            key = get_secret("ALPACA_OPTIONS_PAPER_KEY", "ALPACA_PAPER_KEY")
            secret = get_secret("ALPACA_OPTIONS_PAPER_SECRET", "ALPACA_PAPER_SECRET")
            miss = ("ALPACA_OPTIONS_PAPER_KEY", "ALPACA_OPTIONS_PAPER_SECRET")
        else:
            key = get_secret("ALPACA_PAPER_KEY")
            secret = get_secret("ALPACA_PAPER_SECRET")
            miss = ("ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET")
        endpoint = get_secret("ALPACA_PAPER_ENDPOINT", default="https://paper-api.alpaca.markets/v2")
    except Exception as e:  # noqa: BLE001
        raise AlpacaError(f"could not resolve Alpaca creds: {e}")
    if not key or not secret:
        raise AlpacaError(format_secret_miss(*miss))
    return key, secret, endpoint


class AlpacaClient:
    def __init__(self, key: Optional[str] = None, secret: Optional[str] = None,
                 endpoint: Optional[str] = None, timeout: float = 20.0,
                 min_spacing_s: float = 0.34, max_retries: int = 4, sleep=time.sleep):
        if key is None or secret is None or endpoint is None:
            rk, rs, re_ = _resolve_creds()
            key = key or rk; secret = secret or rs; endpoint = endpoint or re_
        host = urllib.parse.urlparse(endpoint).hostname or ""
        if host != PAPER_HOST:
            raise NotPaperEndpoint(f"refusing non-paper trading endpoint: {host!r} (must be {PAPER_HOST})")
        self._key = key
        self._secret = secret
        self.trading_base = endpoint.rstrip("/")
        self.data_base = f"https://{DATA_HOST}"
        self.timeout = timeout
        self._min_spacing = min_spacing_s
        self._max_retries = max_retries
        self._sleep = sleep
        self._last_req = 0.0

    # ── low level ──────────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return {"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret,
                "Content-Type": "application/json"}

    def _request(self, method: str, url: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(self._max_retries + 1):
            # simple client-side spacing for the rate limit
            gap = self._min_spacing - (time.monotonic() - self._last_req)
            if gap > 0:
                self._sleep(gap)
            req = urllib.request.Request(url, data=data, method=method, headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self._last_req = time.monotonic()
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                self._last_req = time.monotonic()
                if e.code in (429, 500, 502, 503, 504) and attempt < self._max_retries:
                    self._sleep(min(2 ** attempt, 8))
                    continue
                detail = e.read().decode("utf-8", "ignore")[:300]
                raise AlpacaError(f"{method} {url} -> HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self._max_retries:
                    self._sleep(min(2 ** attempt, 8))
                    continue
                raise AlpacaError(f"{method} {url} -> {type(e).__name__}: {e}")
        raise AlpacaError(f"{method} {url} -> exhausted retries")

    # ── trading plane ─────────────────────────────────────────────────────────
    def get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.trading_base}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        return self._request("GET", url)

    def post(self, path: str, body: dict) -> Any:
        return self._request("POST", f"{self.trading_base}/{path.lstrip('/')}", body)

    def get_order_by_client_id(self, client_order_id: str) -> Any:
        """Reconcile lookup: the order with this client_order_id, or None if none exists. Lets order
        submission stay idempotent when a POST times out or 422s on a duplicate — the order may have
        already been accepted, and the deterministic client_order_id lets us find out for sure."""
        try:
            return self.get("orders:by_client_order_id", {"client_order_id": client_order_id})
        except AlpacaError as e:
            if "HTTP 404" in str(e):
                return None
            raise

    def delete(self, path: str) -> Any:
        return self._request("DELETE", f"{self.trading_base}/{path.lstrip('/')}")

    # ── data plane ──────────────────────────────────────────────────────────────
    def data_get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.data_base}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        return self._request("GET", url)

    # ── options data plane (read-only; paper-safe) ───────────────────────────────
    # Options chains + greeks live under the market-DATA host at v1beta1/options
    # (integration plan §1). These are READ-ONLY — no order path here. Auth/retry/backoff
    # ride the same _request as everything else. The trading endpoint stays paper-pinned.
    OPTIONS_BASE = "v1beta1/options"

    def options_snapshots(self, underlying: str, params: Optional[dict] = None) -> Any:
        """Option-chain SNAPSHOTS for one underlying (quotes + greeks + IV in one call).

        GET https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}
        Useful params (all optional, passed through): ``feed`` ('indicative'|'opra'),
        ``type`` ('call'|'put'), ``strike_price_gte``/``strike_price_lte``,
        ``expiration_date`` (YYYY-MM-DD), ``expiration_date_gte``/``expiration_date_lte``,
        ``limit`` (<=1000), ``page_token``.

        ASSUMED response shape (Alpaca v1beta1, documented for the offline tests)::

            {"snapshots": {
                "<OCC symbol>": {
                    "latestQuote": {"ap": <ask px>, "as": <ask sz>,
                                     "bp": <bid px>, "bs": <bid sz>, "t": <ts>},
                    "latestTrade": {"p": <last px>, "s": <sz>, "t": <ts>},
                    "greeks": {"delta":.., "gamma":.., "theta":.., "vega":.., "rho":..},
                    "impliedVolatility": <iv>}},
             "next_page_token": <str|null>}
        """
        return self.data_get(f"{self.OPTIONS_BASE}/snapshots/{urllib.parse.quote(underlying)}",
                             params)

    def options_chain(self, underlying: str, params: Optional[dict] = None) -> dict:
        """Flatten ``options_snapshots`` to a list of per-contract rows (one page).

        Returns {"contracts": [ {symbol, bid, ask, mid, last, delta, gamma, theta, vega, rho, iv},
        ... ], "next_page_token": <str|None>}. Missing quote/greek fields degrade to None so
        selection can filter fail-closed rather than KeyError. Pagination is left to the caller
        (pass the returned token back as ``page_token``) — the P0 earnings sleeve needs only the
        near-dated slice, which fits one page under an expiration_date filter.
        """
        raw = self.options_snapshots(underlying, params) or {}
        snaps = raw.get("snapshots") or {}
        out = []
        for sym, snap in snaps.items():
            out.append(_flatten_option_snapshot(sym, snap or {}))
        return {"contracts": out, "next_page_token": raw.get("next_page_token")}


def _q(quote: Optional[dict], *keys) -> Optional[float]:
    """First present numeric among ``keys`` in a quote/trade dict (tolerates Alpaca's short
    keys 'ap'/'bp'/'p' and any longhand aliases), else None."""
    if not quote:
        return None
    for k in keys:
        v = quote.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _flatten_option_snapshot(symbol: str, snap: dict) -> dict:
    """One snapshot -> a flat, selection-friendly row. Bid/ask from latestQuote (short keys
    'bp'/'ap' or longhand), mid = midpoint when both sides quoted, greeks from the greeks block."""
    quote = snap.get("latestQuote") or {}
    trade = snap.get("latestTrade") or {}
    greeks = snap.get("greeks") or {}
    bid = _q(quote, "bp", "bidPrice", "bid")
    ask = _q(quote, "ap", "askPrice", "ask")
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None and bid > 0 and ask > 0) else None
    return {
        "symbol": symbol,
        "bid": bid, "ask": ask, "mid": mid,
        "last": _q(trade, "p", "price", "last"),
        "delta": _q(greeks, "delta"), "gamma": _q(greeks, "gamma"),
        "theta": _q(greeks, "theta"), "vega": _q(greeks, "vega"), "rho": _q(greeks, "rho"),
        "iv": _q(snap, "impliedVolatility", "iv"),
    }
