"""Pure, offline options core for the single-leg LONG earnings-move sleeve (integration plan §4 P3).

Everything here is deterministic and network-free so it unit-tests without touching Alpaca:
  * OCC (OSI) option-symbol build/parse — the symbol AlpacaPaperBroker.submit() routes.
  * select_call_contract(...) — the documented P0 contract-selection rules.
  * size_by_premium_cap(...) — the conservative premium cap (options are convex → cap tightly).

HARD CONSTRAINT baked in here: single-leg LONG only. This module never constructs a short leg or a
spread; it only ever picks ONE long contract and sizes a BUY. (Mirrors the RH cash-account rule.)

── The P0 play (documented, defensible-simplest) ──────────────────────────────────────────────
Long a near-dated, ~ATM/slightly-ITM CALL on a positive-earnings-surprise name, held through the
earnings move. Rationale:
  * LONG option only → max loss is the premium paid (fully known, fully capped). No short gamma,
    no assignment, no margin — the only sleeve shape allowed on a cash-style account.
  * CALL (not a straddle) → a straddle is two legs; single-leg-only forbids it. The natural universe
    is the PEAD/SUE LONG names (positive surprise → upward post-earnings drift), so a long call is
    the directional, convex expression of that same signal.
  * NEAR-DATED → concentrate the premium on the earnings move (high gamma/vega per $), not on time.
  * ~ATM/slightly-ITM (target delta ~0.55) → real directional exposure with a live delta, avoiding
    far-OTM lottery tickets whose expected decay dominates.
Puts are the symmetric expression for negative-surprise names; P0 ships calls and leaves puts as a
one-line flip (option_type='P') once the sleeve earns its keep.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Optional

# US equity options are 100 shares/contract. Premium at risk = price * this * qty.
CONTRACT_MULTIPLIER = 100

# ── OCC / OSI symbol ─────────────────────────────────────────────────────────────────────────────
# Format (21 chars in the padded spec; Alpaca uses the UNPADDED root form):
#   <ROOT><YYMMDD><C|P><STRIKE*1000 zero-padded to 8 digits>
#   e.g.  AAPL  2024-08-16  Call  $190.00  ->  "AAPL240816C00190000"
# We emit the unpadded-root form Alpaca expects (no space padding). Parsing tolerates trailing/leading
# whitespace and lowercase.
_OCC_RE = re.compile(r"^\s*([A-Za-z]{1,6})(\d{6})([CP])(\d{8})\s*$", re.IGNORECASE)


def build_occ_symbol(underlying: str, expiration: _dt.date, option_type: str, strike: float) -> str:
    """Build an OCC option symbol. ``option_type`` is 'C'/'call' or 'P'/'put' (case-insensitive)."""
    root = (underlying or "").strip().upper()
    if not root or not root.isalpha() or len(root) > 6:
        raise ValueError(f"bad underlying root {underlying!r} (expect 1-6 letters)")
    ot = option_type.strip().upper()[:1]
    if ot not in ("C", "P"):
        raise ValueError(f"bad option_type {option_type!r} (expect C/P)")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    # strike * 1000, rounded to the nearest mill (half-cent strikes exist; OCC encodes tenths of a
    # cent). Round-half-up to avoid float truncation eating a legal strike.
    mills = int(math.floor(strike * 1000 + 0.5))
    if mills > 99_999_999:
        raise ValueError(f"strike {strike} too large to encode")
    return f"{root}{expiration:%y%m%d}{ot}{mills:08d}"


def parse_occ_symbol(symbol: str) -> dict:
    """Inverse of build_occ_symbol → {underlying, expiration(date), option_type('C'|'P'), strike}."""
    m = _OCC_RE.match(symbol or "")
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, ymd, ot, mills = m.groups()
    exp = _dt.datetime.strptime(ymd, "%y%m%d").date()
    return {"underlying": root.upper(), "expiration": exp,
            "option_type": ot.upper(), "strike": int(mills) / 1000.0}


# ── contract selection (P0 rules; pure) ────────────────────────────────────────────────────────────
def select_call_contract(contracts, as_of: _dt.date, *, option_type: str = "C",
                         target_delta: float = 0.55, min_dte: int = 1, max_dte: int = 10,
                         target_dte: int = 7, spot: Optional[float] = None,
                         require_two_sided: bool = True) -> Optional[dict]:
    """Pick ONE long contract from a flattened chain (see AlpacaClient.options_chain rows).

    Selection rules (documented; all fail-closed — an empty/degenerate chain returns None, never a
    guess):
      1. TYPE: keep only ``option_type`` (P0 = 'C' calls). Parsed from each row's OCC symbol so the
         chain need not pre-filter.
      2. TRADABILITY: require a positive ASK (can't buy without an offer); if ``require_two_sided``
         (default) also require a positive BID — a one-sided market is untradeable to EXIT, and this
         is a long book that must be able to sell to close.
      3. EXPIRY: DTE (calendar days from ``as_of`` to expiration) must lie in [min_dte, max_dte].
         Among survivors, prefer the expiration whose DTE is closest to ``target_dte`` (near-dated,
         concentrate on the move). min_dte>=1 forbids a same-day expiry.
      4. STRIKE/DELTA: within the chosen expiration, pick the contract whose |delta| is closest to
         ``target_delta`` (~ATM/slightly-ITM). If NO contract on that expiry has a delta, fall back
         to the strike nearest ``spot`` (ATM by moneyness); if ``spot`` is also unknown, return None
         (never size blind).
    Returns the winning row augmented with {expiration, strike, dte, cost} (cost = ASK, the
    marketable-buy worst case used by sizing), or None if nothing qualifies.
    """
    want = option_type.strip().upper()[:1]
    rows = []
    for c in contracts or []:
        try:
            meta = parse_occ_symbol(c.get("symbol", ""))
        except ValueError:
            continue                                   # not an option symbol — skip, don't crash
        if meta["option_type"] != want:
            continue
        ask = c.get("ask")
        bid = c.get("bid")
        if ask is None or ask <= 0:                    # rule 2: need an offer to buy
            continue
        if require_two_sided and (bid is None or bid <= 0):
            continue                                    # need a bid to be able to exit
        dte = (meta["expiration"] - as_of).days
        if dte < min_dte or dte > max_dte:             # rule 3: expiry window
            continue
        rows.append({**c, "expiration": meta["expiration"], "strike": meta["strike"],
                     "dte": dte, "cost": ask})
    if not rows:
        return None
    # rule 3b: choose the single expiration closest to target_dte
    best_exp = min({r["expiration"] for r in rows},
                   key=lambda e: (abs((e - as_of).days - target_dte), (e - as_of).days))
    on_exp = [r for r in rows if r["expiration"] == best_exp]
    # rule 4: delta-nearest within that expiry; fall back to ATM-by-strike if no deltas
    with_delta = [r for r in on_exp if r.get("delta") is not None]
    if with_delta:
        return min(with_delta, key=lambda r: abs(abs(r["delta"]) - target_delta))
    if spot is not None:
        return min(on_exp, key=lambda r: abs(r["strike"] - spot))
    return None                                        # no delta and no spot → don't size blind


# select_call_contract is option_type-generic (it filters on the ``option_type`` arg and matches on
# |delta|), so it selects a PUT just as well when called with option_type='P'. Expose a
# structure-neutral name for the multi-leg adapter (the anchor put of a credit_put_spread), while
# keeping select_call_contract for the existing single-leg-call callers (byte-unchanged).
select_option_contract = select_call_contract


# Fields RH may fill in when the Alpaca paper feed has quotes but no OPRA greeks/OI.
_RH_OVERLAY_KEYS = (
    "delta", "gamma", "theta", "vega", "rho", "iv",
    "open_interest", "oi", "volume", "vol",
)


def _chain_row_tradable(row: dict) -> bool:
    """True iff the row has a two-sided quote we can buy and later sell (same rule as select)."""
    ask = row.get("ask") if isinstance(row, dict) else None
    bid = row.get("bid") if isinstance(row, dict) else None
    try:
        ask_f = float(ask) if ask is not None else None
        bid_f = float(bid) if bid is not None else None
    except (TypeError, ValueError):
        return False
    return ask_f is not None and ask_f > 0 and bid_f is not None and bid_f > 0


def _canon_occ(sym) -> str:
    """Canonical OCC key for a cross-venue join: parse→rebuild so Alpaca-vs-RH format/case/padding
    differences collapse to ONE key. Falls back to the upper-cased raw symbol when unparseable — so
    the RH greeks overlay matches even when the two venues encode the same contract differently
    (the logged 'RH vs Alpaca OCC mismatch' that silently killed the |delta| assignment trigger)."""
    try:
        p = parse_occ_symbol(str(sym))
        return build_occ_symbol(p["underlying"], p["expiration"], p["option_type"], p["strike"])
    except Exception:  # noqa: BLE001
        return str(sym or "").strip().upper()


def merge_chain_alpaca_primary(alpaca_rows, rh_rows=None, *, require_tradable=True):
    """Alpaca venue quotes win; RH only overlays fields Alpaca left None (greeks/OI).

    Hackathon constraint: the strategy's tradable chain is Alpaca's. Robinhood is a mixture
    overlay for OPRA greeks the paper feed often omits — never the price we size or submit.
    If Alpaca has no tradable two-sided quotes, return [] so selection fails closed (skip the
    name) rather than rest a LIMIT on a Robinhood ask and send it to Alpaca.

    ``require_tradable`` (default True) is the OPEN-side rule: drop rows without a two-sided quote.
    The CLOSE side reads greeks to DECIDE (never to trade), so it passes ``require_tradable=False``
    — otherwise a deep-ITM short near assignment (frequently one-sided on the indicative feed) is
    dropped and its RH delta is lost, silently disabling the |delta|≥0.45 assignment-close trigger
    for exactly the contract it guards. The RH overlay is joined on the CANONICAL OCC key so a
    venue format mismatch can't zero out the join.
    """
    rh_rows = list(rh_rows or [])
    by_sym: dict = {}
    for r in rh_rows:
        s = r.get("symbol")
        if s:
            by_sym.setdefault(_canon_occ(s), r)
    out = []
    for r in (alpaca_rows or []):
        if require_tradable and not _chain_row_tradable(r):
            continue
        merged = dict(r)
        extra = by_sym.get(_canon_occ(merged.get("symbol") or ""))
        if extra:
            for k in _RH_OVERLAY_KEYS:
                if merged.get(k) is None and extra.get(k) is not None:
                    merged[k] = extra[k]
        out.append(merged)
    return out


def fetch_chain_alpaca_primary(broker, underlying, *, option_type="call", as_of,
                               min_dte=1, max_dte=30, rh_fetch=None,
                               feed="indicative", require_tradable=True):
    """Read-only chain: Alpaca snapshots first (``feed=indicative`` on paper), RH greeks overlay.

    ``rh_fetch`` is the Robinhood bridge callable (same kwargs as ``rh_option_chain``). Pass
    None to skip the overlay. Per-source exceptions yield an empty side, never a crash.
    Merge drops Alpaca rows without a two-sided quote and never copies RH bid/ask.

    ``require_tradable`` (default True) is the OPEN-side tradability filter. Greeks-READ callers
    (the assignment-close sweeper) pass False so illiquid deep-ITM contracts keep their delta.
    """
    ot = "put" if str(option_type).strip().lower()[:1] == "p" else "call"
    alpaca = []
    try:
        lo = (as_of + _dt.timedelta(days=int(min_dte))).isoformat()
        hi = (as_of + _dt.timedelta(days=int(max_dte))).isoformat()
        alpaca = list(broker.option_chain(underlying, option_type=ot,
                                          expiration_gte=lo, expiration_lte=hi,
                                          feed=feed) or [])
    except Exception:  # noqa: BLE001 — read-only; empty Alpaca → merge returns [] (fail closed)
        alpaca = []
    rh = []
    if rh_fetch is not None:
        try:
            rh = list(rh_fetch(underlying, option_type=ot, min_dte=int(min_dte),
                               max_dte=int(max_dte), as_of=str(as_of)) or [])
        except Exception:  # noqa: BLE001
            rh = []
    return merge_chain_alpaca_primary(alpaca, rh, require_tradable=require_tradable)


# ── sizing: conservative premium cap (options are convex) ──────────────────────────────────────────
def size_by_premium_cap(cost_per_contract: float, equity: Optional[float] = None, *,
                        max_premium_abs: float = 250.0, max_premium_frac: float = 0.005,
                        multiplier: int = CONTRACT_MULTIPLIER) -> dict:
    """How many LONG contracts fit under a tight premium cap.

    A long option can expire worthless, so the premium PAID *is* the max loss — we cap it hard and
    small. The cap is the MIN of an absolute per-trade ceiling and a fraction of account equity::

        premium_cap = min(max_premium_abs, equity * max_premium_frac)   # equity optional
        qty         = floor( premium_cap / (cost_per_contract * multiplier) )

    ``cost_per_contract`` is the per-share option price (use the ASK — marketable-buy worst case).
    Returns {qty, premium_cap, est_cost} where est_cost = qty*cost*multiplier (<= premium_cap by
    construction, and = the max loss on the long). qty==0 means even one contract breaches the cap →
    the caller SKIPS (fail-closed; never round up to 1).
    """
    if cost_per_contract is None or cost_per_contract <= 0 or multiplier <= 0:
        return {"qty": 0, "premium_cap": 0.0, "est_cost": 0.0}
    caps = [max_premium_abs]
    if equity is not None and equity > 0 and max_premium_frac > 0:
        caps.append(equity * max_premium_frac)
    premium_cap = min(caps)
    per_contract = cost_per_contract * multiplier
    qty = int(math.floor(premium_cap / per_contract))
    if qty < 1:
        return {"qty": 0, "premium_cap": round(premium_cap, 2), "est_cost": 0.0}
    return {"qty": qty, "premium_cap": round(premium_cap, 2),
            "est_cost": round(qty * per_contract, 2)}


# ── defined-risk spread sizing + gates (pure; the Phase-4b credit_put_spread risk layer) ────────────
def size_by_max_loss(per_contract_max_loss: float, nav: Optional[float], *,
                     risk_frac: float = 0.01, multiplier: int = CONTRACT_MULTIPLIER) -> dict:
    """How many DEFINED-RISK spread contracts fit under a per-trade max-loss budget (the ratified
    quant rule)::

        risk_cap = risk_frac * NAV                                   # 1.0% of NAV per trade
        qty      = floor( risk_cap / (per_contract_max_loss * multiplier) )

    ``per_contract_max_loss`` is the PER-SHARE max loss of one spread contract (``width − net_credit``
    for a credit vertical); ``× multiplier`` = the dollar max loss of one contract. Returns
    {qty, risk_cap, est_max_loss} where est_max_loss = qty × per_contract_max_loss × multiplier
    (≤ risk_cap by construction). ``qty == 0`` (even ONE contract breaches the budget, or a
    non-positive max-loss / NAV) → the caller SKIPS. NEVER rounds up (fail-closed)."""
    if (per_contract_max_loss is None or per_contract_max_loss <= 0 or multiplier <= 0
            or nav is None or nav <= 0 or risk_frac <= 0):
        return {"qty": 0, "risk_cap": 0.0, "est_max_loss": 0.0}
    risk_cap = float(nav) * float(risk_frac)
    per_contract = float(per_contract_max_loss) * multiplier
    qty = int(math.floor(risk_cap / per_contract))
    if qty < 1:
        return {"qty": 0, "risk_cap": round(risk_cap, 2), "est_max_loss": 0.0}
    return {"qty": qty, "risk_cap": round(risk_cap, 2),
            "est_max_loss": round(qty * per_contract, 2)}


def _num(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def leg_liquidity_ok(row: dict, *, max_rel_spread: float = 0.10, min_oi: int = 500,
                     min_vol: int = 100) -> tuple:
    """Fail-closed per-leg liquidity gate. Returns ``(ok, reason)``.

    Rejects the leg (``ok=False``) when: the relative bid/ask spread ``(ask − bid)/mid > max_rel_spread``,
    open interest ``< min_oi``, or volume ``< min_vol``. Reads ``open_interest``/``oi`` and
    ``volume``/``vol`` (whichever the chain row carries). A MISSING or non-positive bid/ask/mid, or a
    MISSING OI/volume, is a REJECT (fail-closed — we never trade a leg whose liquidity we can't prove;
    the RH bridge must forward open_interest/volume before this gate can pass at Phase 5)."""
    bid, ask, mid = _num(row.get("bid")), _num(row.get("ask")), _num(row.get("mid"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return False, "missing/one-sided quote"
    if mid is None or mid <= 0:
        mid = (bid + ask) / 2.0
    rel = (ask - bid) / mid
    if rel > max_rel_spread:
        return False, f"wide spread {rel:.1%} > {max_rel_spread:.0%}"
    oi = _num(row.get("open_interest"))
    if oi is None:
        oi = _num(row.get("oi"))
    if oi is None or oi < min_oi:
        return False, f"open_interest {oi} < {min_oi}" if oi is not None else "open_interest missing"
    vol = _num(row.get("volume"))
    if vol is None:
        vol = _num(row.get("vol"))
    if vol is None or vol < min_vol:
        return False, f"volume {vol} < {min_vol}" if vol is not None else "volume missing"
    return True, "ok"


def credit_structure_ok(net_credit: float, width: float, *, min_credit_to_width: float = 0.20,
                        min_credit_abs: float = 0.10) -> tuple:
    """Fail-closed structure gate for a credit spread. Returns ``(ok, reason)``.

    Rejects when the net credit is too thin to be worth the pinned risk: ``net_credit < min_credit_abs``
    ($0.10) or ``net_credit < min_credit_to_width × width`` (20% of the strike width). ``net_credit`` and
    ``width`` are per-share dollars. Non-positive width/credit → reject."""
    if width is None or width <= 0 or net_credit is None or net_credit <= 0:
        return False, "non-positive width/credit"
    if net_credit < min_credit_abs:
        return False, f"credit ${net_credit:.2f} < ${min_credit_abs:.2f}"
    if net_credit < min_credit_to_width * width:
        return False, f"credit ${net_credit:.2f} < {min_credit_to_width:.0%} of width ${width:.2f}"
    return True, "ok"
