"""AlpacaPaperBroker — read account/positions/orders, and a FAIL-CLOSED order path.

Order submission is guarded three ways: (1) the client is paper-only by construction
(NotPaperEndpoint), (2) an ArmingGate that is DISARMED by default — a submit on a disarmed gate
raises OrderRefused, and (3) a deterministic client_order_id so a retry can never double-submit.
Read paths (account/positions/orders) are always allowed. There is no real-capital path.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import datetime as _dt
from typing import Optional as _Optional

from .client import AlpacaClient
from .options import parse_occ_symbol, select_call_contract

# Alpaca time-in-force validity by asset class. Crypto rejects 'day'/'opg'/'cls' (422); equities and
# options accept 'day' (and gtc/ioc/fok). We validate against this so a bad TIF fails locally, loudly.
_TIF_BY_CLASS = {
    "us_equity": {"day", "gtc", "opg", "cls", "ioc", "fok"},
    "option": {"day", "gtc", "ioc", "fok"},
    "crypto": {"gtc", "ioc", "fok"},
}
# When the dataclass default 'day' is used for crypto (which forbids it), coerce to a safe equivalent.
_CRYPTO_DEFAULT_TIF = "gtc"


class OrderRefused(RuntimeError):
    """Raised when an order is blocked (disarmed gate, bad asset class, missing fields)."""


# ── strategy-tag sanitize seam (THE one rule; single source of truth) ─────────────────────────────
# Every strategy tag that reaches the Alpaca ledger travels through client_order_id, which strips
# everything outside this charset (so 'book:abc' is tagged 'book_abc'). Any consumer that joins
# or filters the fills/orders ledger by strategy MUST compare in this sanitized space — import
# sanitize_strategy_tag (Python) or SQL_SANITIZE_EXPR (SQL literal) from here; never re-type the rule
# (a drifted copy silently orphans/blinds every key containing ':' — observed 2026-08).
TAG_SANITIZE_RE = r"[^A-Za-z0-9_]"
# SQL twin of sanitize_strategy_tag for ledger-side joins: format with the column expression, e.g.
# SQL_SANITIZE_EXPR.format(col="r.strategy_key"). Keep the character class identical to
# TAG_SANITIZE_RE forever (tests pin all three — regex, Python fn, SQL literal — to each other).
SQL_SANITIZE_EXPR = "regexp_replace({col}, '[^A-Za-z0-9_]', '_', 'g')"


def sanitize_strategy_tag(tag: Optional[str]) -> str:
    """Raw strategy/registry key → the tag it appears as in client_order_id (and thus in the
    alpaca_fills/alpaca_orders ledger). Empty/None degrades to 'unknown', mirroring the id path."""
    return re.sub(TAG_SANITIZE_RE, "_", tag or "") or "unknown"


# ── defined-risk guarantee (the LAST line of defense, independent of upstream sizing) ──────────────
def assert_defined_risk(legs: List[dict]) -> None:
    """Raise :class:`OrderRefused` unless every SHORT option leg is COVERED by a long option leg of at
    least equal quantity at a BOUNDING strike on the SAME expiration — i.e. the net position is NOT
    naked-short. This is the money-path invariant for the first live short-vol fill: even if upstream
    sizing/selection has a bug, a naked short can NEVER reach the broker.

    Coverage rule (per right, per expiration):
      • PUT  short is covered by a long PUT  at a strike ``<=`` the short's strike (a lower-strike long
        put caps the downside — the credit-put-spread shape);
      • CALL short is covered by a long CALL at a strike ``>=`` the short's strike (a higher-strike long
        call caps the upside);
      • quantities are matched greedily: total long qty (of that right) must be ``>=`` total short qty
        AND each short unit must find an unused bounding long unit at the same expiration.
    Any leg whose OCC symbol can't be parsed, any short with no bounding long, or any coverage shortfall
    → OrderRefused (fail-closed; never assume risk is defined)."""
    # Expand each leg into (right, strike, expiration, signed_units) — sign by side (buy=+, sell=-).
    shorts: dict = {}                                   # right -> list[(strike, exp)] each = one short unit
    longs: dict = {}                                    # right -> list[(strike, exp)] each = one long unit
    for leg in legs or []:
        try:
            meta = parse_occ_symbol(leg.get("symbol", ""))
        except ValueError as e:
            raise OrderRefused(f"defined-risk check: leg symbol {leg.get('symbol')!r} is not an OCC "
                               f"option symbol ({e}) — refusing (cannot prove risk is defined)")
        side = (leg.get("side") or "").lower()
        if side not in ("buy", "sell"):
            raise OrderRefused(f"defined-risk check: leg side {leg.get('side')!r} invalid")
        # side↔position_intent coherence (fail-closed on ambiguity): the coverage math classifies a
        # leg long/short by `side`, but the emitted body ALSO carries position_intent. If they disagree
        # (e.g. side='buy' with 'sell_to_open'), the leg's true direction is ambiguous — which field the
        # venue honors would decide naked-vs-covered. Refuse rather than assume. (Only checked when
        # position_intent is present; single-leg legs omit it.)
        pi = (leg.get("position_intent") or "").lower()
        if pi:
            if pi not in ("buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"):
                raise OrderRefused(f"defined-risk check: leg position_intent {leg.get('position_intent')!r} invalid")
            if pi.split("_")[0] != side:
                raise OrderRefused(f"defined-risk check: leg side={side!r} disagrees with "
                                   f"position_intent={pi!r} — ambiguous direction, refusing")
        try:
            units = int(leg.get("ratio_qty") or 0)
        except (TypeError, ValueError):
            raise OrderRefused(f"defined-risk check: leg ratio_qty {leg.get('ratio_qty')!r} invalid")
        if units <= 0:
            raise OrderRefused("defined-risk check: leg ratio_qty must be a positive integer")
        bucket = shorts if side == "sell" else longs
        bucket.setdefault(meta["option_type"], []).extend(
            [(meta["strike"], meta["expiration"])] * units)
    for right, short_units in shorts.items():
        avail = list(longs.get(right, []))
        if len(avail) < len(short_units):
            raise OrderRefused(
                f"NAKED SHORT refused: {len(short_units)} short {right}-unit(s) but only {len(avail)} "
                f"covering long {right}-unit(s) — net short is undefined-risk")
        # PUT covered by a LOWER-or-equal long strike; CALL by a HIGHER-or-equal long strike. Match the
        # HARDEST-to-cover shorts first (deepest strike) against the best-bounding long available.
        if right == "P":
            short_units = sorted(short_units, key=lambda su: su[0])          # lowest strike first
            avail.sort(key=lambda lu: lu[0])                                 # lowest long strike first
            def bounds(lu, su):  # long put strike must be <= short put strike, same expiry
                return lu[1] == su[1] and lu[0] <= su[0]
        else:                                                               # calls
            short_units = sorted(short_units, key=lambda su: -su[0])         # highest strike first
            avail.sort(key=lambda lu: -lu[0])
            def bounds(lu, su):  # long call strike must be >= short call strike, same expiry
                return lu[1] == su[1] and lu[0] >= su[0]
        for su in short_units:
            match = next((i for i, lu in enumerate(avail) if bounds(lu, su)), None)
            if match is None:
                raise OrderRefused(
                    f"NAKED SHORT refused: short {right} at strike {su[0]} exp {su[1]} has no covering "
                    f"long {right} at a bounding strike/expiry — undefined-risk")
            avail.pop(match)


def strategy_from_client_order_id(cid: Optional[str]) -> str:
    """Inverse of OrderIntent.client_order_id — recover the strategy tag for ledger attribution.
    New ids are `pma-<tag>-<hash>` with `tag` sanitized to contain no '-'; legacy `pma-<hash>` ids
    (or anything unrecognized) degrade to 'unknown' rather than a hash fragment (audit P0-1)."""
    if not cid or not cid.startswith("pma-"):
        return "unknown"
    body = cid[4:]
    if "-" not in body:            # legacy hash-only id — no strategy was carried
        return "unknown"
    return body.rsplit("-", 1)[0] or "unknown"


@dataclass
class ArmingGate:
    """Explicit arm/disarm. Default DISARMED — nothing submits until an operator arms it."""
    armed: bool = False

    def arm(self) -> None:
        self.armed = True

    def disarm(self) -> None:
        self.armed = False


@dataclass
class OrderIntent:
    strategy: str
    symbol: str                     # 'AAPL' | 'BTC/USD' | OCC option symbol (mleg: the underlier, informational)
    side: str                       # 'buy' | 'sell'
    asset_class: str                # 'us_equity' | 'crypto' | 'option'
    qty: Optional[float] = None
    notional: Optional[float] = None
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    as_of: str = ""                 # feeds the deterministic client_order_id
    # MULTI-LEG (defined-risk options spreads). When ``legs`` is set, submit() emits an Alpaca
    # ``order_class='mleg'`` body instead of the single-leg body — the single-leg path (legs=None) is
    # BYTE-UNCHANGED. Each leg is a dict: {symbol(OCC), side('buy'|'sell'), ratio_qty(int),
    # position_intent('buy_to_open'|'sell_to_open'|'buy_to_close'|'sell_to_close')}.
    order_class: Optional[str] = None
    legs: Optional[List[dict]] = None

    def client_order_id(self) -> str:
        # Carry the strategy UN-HASHED as a readable prefix so the paper-ledger sync can attribute
        # per-strategy P&L; a fully-hashed seed erased attribution (audit P0-1). The hash suffix keeps
        # the id deterministic (retry-safe) AND now folds in qty/notional/type so a second legitimate
        # same-symbol/side/day order with a different size gets a distinct id (audit P2 collision).
        # Sanitize the tag to a charset that survives Alpaca's client_order_id and contains no '-'
        # (our delimiter), so the sync can split it back out unambiguously. The rule lives ONCE, in
        # sanitize_strategy_tag above — ledger consumers (orphan_flatten, strategy_oos_ingest,
        # panel_signal_execute's dead-man scope, the watchdog exit probe) join in the same space.
        tag = sanitize_strategy_tag(self.strategy)
        seed = (f"{self.strategy}|{self.symbol}|{self.side}|{self.asset_class}|{self.qty}|"
                f"{self.notional}|{self.order_type}|{self.time_in_force}|{self.limit_price}|{self.as_of}")
        # For a multi-leg order the (symbol,side) above are the underlier placeholders, so fold the leg
        # legs into the seed too — two spreads on the same underlier/day with different strikes get
        # distinct ids. Appended ONLY when legs are present, so the single-leg seed is BYTE-UNCHANGED.
        if self.legs:
            leg_sig = ";".join(f"{(l.get('symbol') or '')}:{l.get('side')}:{l.get('ratio_qty')}:"
                               f"{l.get('position_intent')}" for l in self.legs)
            seed = f"{seed}|mleg={self.order_class}|legs={leg_sig}"
        return f"pma-{tag}-" + hashlib.sha256(seed.encode()).hexdigest()[:24]


class AlpacaPaperBroker:
    def __init__(self, client: Optional[AlpacaClient] = None, gate: Optional[ArmingGate] = None):
        self.client = client or AlpacaClient()
        self.gate = gate or ArmingGate()   # DISARMED by default

    # ── read (always allowed) ────────────────────────────────────────────────
    def account(self) -> dict:
        return self.client.get("account")

    def positions(self) -> List[dict]:
        return self.client.get("positions") or []

    def orders(self, status: str = "all", limit: int = 200, until: Optional[str] = None,
               after: Optional[str] = None, direction: Optional[str] = None) -> List[dict]:
        params = {"status": status, "limit": limit, "until": until, "after": after,
                  "direction": direction}
        return self.client.get("orders", {k: v for k, v in params.items() if v is not None}) or []

    def fills(self, limit: int = 500, page_token: Optional[str] = None) -> List[dict]:
        """Best-effort execution activity rows from Alpaca.

        Alpaca's activity surface has appeared under both ``account/activities/FILL``
        and ``account/activities?activity_types=FILL`` shapes. Try the dedicated
        path first, then the query-param variant. Returns [] on a 404-style shape
        miss so the caller can degrade cleanly.
        """
        params = {"page_size": limit}
        if page_token:
            params["page_token"] = page_token
        try:
            return self.client.get("account/activities/FILL", params) or []
        except Exception:
            try:
                fallback = {"activity_types": "FILL", "page_size": limit}
                if page_token:
                    fallback["page_token"] = page_token
                return self.client.get("account/activities", fallback) or []
            except Exception:
                return []

    def clock(self) -> dict:
        """Market clock (is_open + next_open/next_close). Used by the equity executor's RTH gate —
        notional market orders are RTH-only, so we don't submit into a closed session."""
        return self.client.get("clock") or {}

    def nav(self) -> dict:
        """Normalized account snapshot for the paper_book_nav ledger."""
        a = self.account()
        f = lambda k: float(a[k]) if a.get(k) not in (None, "") else None
        return {"account_number": a.get("account_number"), "equity": f("equity"),
                "last_equity": f("last_equity"), "cash": f("cash"),
                "buying_power": f("buying_power"), "long_market_value": f("long_market_value"),
                "short_market_value": f("short_market_value"), "status": a.get("status")}

    # ── options data (read-only; paper-safe) ─────────────────────────────────
    def option_chain(self, underlying: str, *, option_type: _Optional[str] = None,
                     expiration_gte: _Optional[str] = None, expiration_lte: _Optional[str] = None,
                     feed: _Optional[str] = None, limit: _Optional[int] = None) -> list:
        """Flattened option-chain rows (quotes+greeks) for one underlying — READ-ONLY. Thin wrapper
        over AlpacaClient.options_chain; narrow the pull with the near-dated expiration window and a
        call/put ``type`` so the P0 sleeve fetches only the slice it needs (one page). ``option_type``
        is 'call'/'put' (Alpaca's data-plane spelling)."""
        params = {"type": option_type, "expiration_date_gte": expiration_gte,
                  "expiration_date_lte": expiration_lte, "feed": feed, "limit": limit}
        return (self.client.options_chain(underlying,
                {k: v for k, v in params.items() if v is not None}) or {}).get("contracts", [])

    def select_long_call(self, underlying: str, as_of: _dt.date, *, spot: _Optional[float] = None,
                         target_delta: float = 0.55, min_dte: int = 1, max_dte: int = 10,
                         target_dte: int = 7) -> _Optional[dict]:
        """Fetch the near-dated call slice and apply the documented P0 selection rules
        (options.select_call_contract). Returns the chosen contract row (with OCC symbol, ask/mid,
        greeks, dte) or None if nothing qualifies — the executor then sizes + submits a single LONG
        buy. Read-only; never places an order."""
        lo = (as_of + _dt.timedelta(days=min_dte)).isoformat()
        hi = (as_of + _dt.timedelta(days=max_dte)).isoformat()
        chain = self.option_chain(underlying, option_type="call",
                                  expiration_gte=lo, expiration_lte=hi,
                                  feed="indicative")
        return select_call_contract(chain, as_of, option_type="C", target_delta=target_delta,
                                    min_dte=min_dte, max_dte=max_dte, target_dte=target_dte,
                                    spot=spot)

    # ── submit (FAIL-CLOSED) ─────────────────────────────────────────────────
    def submit(self, intent: OrderIntent) -> dict:
        if not self.gate.armed:
            raise OrderRefused("ArmingGate is DISARMED — refusing to submit (arm explicitly to trade paper)")
        if intent.asset_class not in ("us_equity", "crypto", "option"):
            raise OrderRefused(f"unknown asset_class {intent.asset_class!r}")
        if intent.side not in ("buy", "sell"):
            raise OrderRefused(f"bad side {intent.side!r}")
        # MULTI-LEG (defined-risk spread) path. Kept separate so the single-leg body below is
        # BYTE-UNCHANGED (legs=None never reaches here).
        if intent.legs:
            return self._submit_mleg(intent)
        if (intent.qty is None) == (intent.notional is None):
            raise OrderRefused("exactly one of qty / notional must be set")
        # a 'limit' order without a price is a silent no-op at the API — refuse it here (audit P1)
        if intent.order_type == "limit" and intent.limit_price is None:
            raise OrderRefused("order_type='limit' requires limit_price")
        tif = self._resolve_tif(intent)
        body = {
            "symbol": intent.symbol, "side": intent.side, "type": intent.order_type,
            "time_in_force": tif, "client_order_id": intent.client_order_id(),
        }
        if intent.qty is not None:
            body["qty"] = str(intent.qty)
        else:
            body["notional"] = str(intent.notional)
        if intent.order_type == "limit":
            body["limit_price"] = str(intent.limit_price)
        try:
            return self.client.post("orders", body)
        except Exception:
            # A POST that times out or 422s on a duplicate may mean the order was ALREADY accepted —
            # the retry can't tell. Reconcile by the deterministic client_order_id before surfacing a
            # failure, so a flaky network never double-submits or falsely reports a miss (audit P2).
            try:
                existing = self.client.get_order_by_client_id(body["client_order_id"])
            except Exception:
                existing = None
            if existing:
                return existing
            raise

    def _submit_mleg(self, intent: OrderIntent) -> dict:
        """Emit an Alpaca ``order_class='mleg'`` order for a defined-risk options spread.

        Fail-closed, defined-risk-only. Validations (any failure → OrderRefused, nothing posted):
          • options-only (mleg is an options construct here); ``qty`` (# spreads) set, ``notional`` unset;
          • LIMIT only with a ``limit_price`` — NEVER a market mleg (a market spread has unbounded fill
            slippage; the plan mandates limit-at-net-credit-or-better);
          • ``assert_defined_risk`` — the last-line guarantee: no naked short can pass.

        ASSUMED Alpaca body shape (v2 /orders, order_class='mleg'; UNVERIFIED against a live dry-run —
        flagged for the Phase-5 dry-run gate): top-level ``qty`` = number of spread packages, no top-level
        ``symbol``; each leg carries ``symbol``/``side``/``ratio_qty``/``position_intent``; ``limit_price``
        is the net price of the package (positive number; for a CREDIT the account is paid — the
        credit/debit SIGN CONVENTION is the specific open question a dry-run must confirm)."""
        if intent.asset_class != "option":
            raise OrderRefused(f"mleg is options-only, got asset_class {intent.asset_class!r}")
        if intent.qty is None or intent.notional is not None:
            raise OrderRefused("mleg requires qty (# spreads) and no notional")
        try:
            n_spreads = int(intent.qty)
        except (TypeError, ValueError):
            raise OrderRefused(f"mleg qty {intent.qty!r} must be an integer number of spreads")
        if n_spreads <= 0:
            raise OrderRefused("mleg qty must be a positive integer number of spreads")
        if intent.order_type != "limit" or intent.limit_price is None:
            raise OrderRefused("mleg orders must be LIMIT with a limit_price (net-credit-or-better; "
                               "never a market spread)")
        if len(intent.legs) < 2:
            raise OrderRefused("mleg requires >= 2 legs")
        # THE defined-risk guarantee — independent of upstream sizing. Naked short can never pass.
        assert_defined_risk(intent.legs)
        tif = self._resolve_tif(intent)
        legs_body = []
        for leg in intent.legs:
            if (leg.get("side") or "").lower() not in ("buy", "sell"):
                raise OrderRefused(f"mleg leg bad side {leg.get('side')!r}")
            pi = (leg.get("position_intent") or "").lower()
            if pi not in ("buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close"):
                raise OrderRefused(f"mleg leg bad position_intent {leg.get('position_intent')!r}")
            legs_body.append({"symbol": leg["symbol"], "side": leg["side"].lower(),
                              "ratio_qty": str(int(leg["ratio_qty"])), "position_intent": pi})
        body = {
            "order_class": "mleg",
            "qty": str(n_spreads),
            "type": intent.order_type,
            "time_in_force": tif,
            "limit_price": str(intent.limit_price),
            "client_order_id": intent.client_order_id(),
            "legs": legs_body,
        }
        try:
            return self.client.post("orders", body)
        except Exception:
            try:
                existing = self.client.get_order_by_client_id(body["client_order_id"])
            except Exception:
                existing = None
            if existing:
                return existing
            raise

    @staticmethod
    def _resolve_tif(intent: OrderIntent) -> str:
        """Validate/repair time_in_force for the asset class. Crypto rejects 'day' (422); if the
        dataclass default 'day' is used for crypto, coerce to gtc; any other invalid TIF is refused."""
        allowed = _TIF_BY_CLASS[intent.asset_class]
        tif = intent.time_in_force
        if intent.asset_class == "crypto" and tif == "day":
            return _CRYPTO_DEFAULT_TIF
        if tif not in allowed:
            raise OrderRefused(
                f"time_in_force {tif!r} invalid for {intent.asset_class} (allowed: {sorted(allowed)})")
        return tif
