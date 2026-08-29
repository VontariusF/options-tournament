"""Position-aware reconciliation — turn a target basket into a DELTA against what's already held.

Read current positions and only act on the difference: buy names not yet held,
optionally close names that dropped out of the target. Pure and network-free;
the executor supplies positions from broker.positions().
"""
from __future__ import annotations

from typing import List, Optional


def _norm(symbol: Optional[str]) -> str:
    """Canonicalize a symbol for held-vs-target matching across venues. Alpaca reports crypto
    positions as e.g. 'BTCUSD' while orders use 'BTC/USD'; equities/options are already consistent.
    Upper-case and strip '/' so the two representations compare equal."""
    return (symbol or "").upper().replace("/", "")


def held_map(positions: List[dict]) -> dict:
    """{normalized_symbol: {"symbol": raw, "qty": float, "market_value": float}} from broker.positions().
    A zero-qty row (fully closed) is treated as not held."""
    out: dict = {}
    for p in positions or []:
        raw = p.get("symbol")
        if not raw:
            continue
        try:
            qty = float(p.get("qty") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if qty == 0.0:
            continue
        try:
            mv = float(p.get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        out[_norm(raw)] = {"symbol": raw, "qty": qty, "market_value": mv}
    return out


def reconcile(plan: List[dict], positions: List[dict], *, close_dropped: bool = False) -> dict:
    """Position-aware diff of a target `plan` (list of {"symbol", ...}) against current `positions`.

    Returns {"to_place": [...plan entries for names NOT already held...],
             "skipped_held": [...symbols already held, so we don't re-buy and accumulate...],
             "to_close": [{"symbol","qty"} ...held names no longer in the target (only if close_dropped)]}.

    The minimal reconciliation is "buy only what we don't hold" — it eliminates the accumulation bug.
    `close_dropped=True` additionally returns the held names (with their qty) that fell out of the
    target so the caller can FLATTEN them — a full rebalance. `positions` MUST be this sleeve's OWN
    book (per-strategy net fills), never the shared account, or one sleeve would close another's shared
    symbol. Left off by default so a sleeve never sells unless it opts in."""
    held = held_map(positions)
    to_place = [o for o in plan if _norm(o.get("symbol")) not in held]
    skipped = [o.get("symbol") for o in plan if _norm(o.get("symbol")) in held]
    to_close = []
    if close_dropped:
        target = {_norm(o.get("symbol")) for o in plan}
        to_close = [{"symbol": v["symbol"], "qty": v["qty"]}
                    for k, v in held.items() if k not in target]
    return {"to_place": to_place, "skipped_held": skipped, "to_close": to_close}
