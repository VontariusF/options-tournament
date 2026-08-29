"""Thin Alpaca paper executor driven by an explicit strategy card.

Structures: long_call, long_put, credit_put_spread.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Optional

from pma_brokers.alpaca.broker import AlpacaPaperBroker, ArmingGate, OrderIntent, OrderRefused
from pma_brokers.alpaca.options import (
    CONTRACT_MULTIPLIER,
    credit_structure_ok,
    parse_occ_symbol,
    select_option_contract,
    size_by_max_loss,
    size_by_premium_cap,
)

STRUCTURES = ("long_call", "long_put", "credit_put_spread")


@dataclass
class StrategyCard:
    underlying: str
    structure: str
    dte: int = 7
    delta: float = 0.55
    wing_width: float = 10.0
    as_of: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "StrategyCard":
        structure = str(raw.get("structure") or "").strip()
        if structure not in STRUCTURES:
            raise ValueError(f"structure must be one of {STRUCTURES}, got {structure!r}")
        underlying = str(raw.get("underlying") or "").strip().upper()
        if not underlying:
            raise ValueError("underlying is required")
        return cls(
            underlying=underlying,
            structure=structure,
            dte=int(raw.get("dte") or 7),
            delta=float(raw.get("delta") or 0.55),
            wing_width=float(raw.get("wing_width") or 10.0),
            as_of=str(raw["as_of"]) if raw.get("as_of") else None,
        )


def paper_armed() -> bool:
    v = os.environ.get("OPTIONS_PAPER_ARMED", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _as_of(card: StrategyCard) -> dt.date:
    if card.as_of:
        return dt.date.fromisoformat(card.as_of)
    return dt.date.today()


def _spot(broker: AlpacaPaperBroker, symbol: str) -> Optional[float]:
    try:
        quote = broker.client.get(f"stocks/{symbol}/quotes/latest")
        q = (quote or {}).get("quote") or quote or {}
        for k in ("ap", "bp", "p"):
            v = q.get(k)
            if v is not None:
                return float(v)
    except Exception:  # noqa: BLE001
        return None
    return None


def _nav(broker: AlpacaPaperBroker) -> Optional[float]:
    try:
        return float(broker.nav().get("equity") or 0) or None
    except Exception:  # noqa: BLE001
        return None


def plan_card(card: StrategyCard, broker: Optional[AlpacaPaperBroker] = None) -> dict[str, Any]:
    """Build an order intent from a card. Does not submit."""
    broker = broker or AlpacaPaperBroker()
    as_of = _as_of(card)
    spot = _spot(broker, card.underlying)
    nav = _nav(broker)
    min_dte = max(1, card.dte - 4)
    max_dte = card.dte + 7
    lo = (as_of + dt.timedelta(days=min_dte)).isoformat()
    hi = (as_of + dt.timedelta(days=max_dte)).isoformat()

    if card.structure in ("long_call", "long_put"):
        right = "call" if card.structure == "long_call" else "put"
        occ_right = "C" if right == "call" else "P"
        chain = broker.option_chain(
            card.underlying, option_type=right,
            expiration_gte=lo, expiration_lte=hi, feed="indicative",
        )
        chosen = select_option_contract(
            chain, as_of, option_type=occ_right, target_delta=card.delta,
            min_dte=min_dte, max_dte=max_dte, target_dte=card.dte, spot=spot,
        )
        if not chosen:
            return {"ok": False, "error": "no tradable contract matched delta/DTE window"}
        sized = size_by_premium_cap(chosen["cost"], nav)
        if sized["qty"] < 1:
            return {"ok": False, "error": "premium cap sized qty to 0", "chosen": chosen, "sized": sized}
        intent = OrderIntent(
            strategy="tournament",
            symbol=chosen["symbol"],
            side="buy",
            asset_class="option",
            qty=sized["qty"],
            order_type="limit",
            limit_price=chosen["cost"],
            time_in_force="day",
            as_of=as_of.isoformat(),
        )
        return {"ok": True, "structure": card.structure, "chosen": chosen, "sized": sized, "intent": intent}

    # credit_put_spread: sell higher-strike put, buy lower-strike put
    chain = broker.option_chain(
        card.underlying, option_type="put",
        expiration_gte=lo, expiration_lte=hi, feed="indicative",
    )
    short = select_option_contract(
        chain, as_of, option_type="P", target_delta=card.delta,
        min_dte=min_dte, max_dte=max_dte, target_dte=card.dte, spot=spot,
    )
    if not short:
        return {"ok": False, "error": "no tradable short put matched delta/DTE window"}
    try:
        meta = parse_occ_symbol(short["symbol"])
    except ValueError as e:
        return {"ok": False, "error": f"short put OCC parse failed: {e}"}
    wing_strike = meta["strike"] - card.wing_width
    if wing_strike <= 0:
        return {"ok": False, "error": "wing_width produces a non-positive long-put strike"}
    on_exp = []
    for row in chain or []:
        try:
            m = parse_occ_symbol(row.get("symbol", ""))
        except ValueError:
            continue
        if m["expiration"] == meta["expiration"] and m["option_type"] == "P":
            on_exp.append({**row, **m})
    long_leg = min(on_exp, key=lambda r: abs(r["strike"] - wing_strike), default=None)
    if not long_leg or long_leg.get("ask") in (None, 0) or (long_leg.get("bid") or 0) <= 0:
        return {"ok": False, "error": "no tradable long put wing on the same expiry"}
    short_bid = short.get("bid") or 0
    long_ask = long_leg.get("ask") or 0
    net_credit = float(short_bid) - float(long_ask)
    width = meta["strike"] - float(long_leg["strike"])
    ok_struct, reason = credit_structure_ok(net_credit, width)
    if not ok_struct:
        return {"ok": False, "error": f"credit structure refused: {reason}"}
    per_share_max_loss = width - net_credit
    sized = size_by_max_loss(per_share_max_loss, nav)
    if sized["qty"] < 1:
        return {"ok": False, "error": "max-loss cap sized qty to 0", "sized": sized}
    legs = [
        {"symbol": short["symbol"], "side": "sell", "ratio_qty": 1, "position_intent": "sell_to_open"},
        {"symbol": long_leg["symbol"], "side": "buy", "ratio_qty": 1, "position_intent": "buy_to_open"},
    ]
    intent = OrderIntent(
        strategy="tournament",
        symbol=card.underlying,
        side="sell",
        asset_class="option",
        qty=sized["qty"],
        order_type="limit",
        limit_price=round(net_credit, 2),
        time_in_force="day",
        as_of=as_of.isoformat(),
        order_class="mleg",
        legs=legs,
    )
    return {
        "ok": True,
        "structure": card.structure,
        "short": short,
        "long": long_leg,
        "net_credit": net_credit,
        "width": width,
        "max_loss": per_share_max_loss * CONTRACT_MULTIPLIER * sized["qty"],
        "sized": sized,
        "intent": intent,
    }


def execute_card(
    card: StrategyCard,
    *,
    dry_run: bool = True,
    arm: bool = False,
    broker: Optional[AlpacaPaperBroker] = None,
) -> dict[str, Any]:
    broker = broker or AlpacaPaperBroker(gate=ArmingGate(armed=False))
    planned = plan_card(card, broker=broker)
    if not planned.get("ok"):
        return planned
    intent: OrderIntent = planned["intent"]
    planned_out = {k: v for k, v in planned.items() if k != "intent"}
    planned_out["client_order_id"] = intent.client_order_id()
    planned_out["symbol"] = intent.symbol
    planned_out["qty"] = intent.qty
    planned_out["limit_price"] = intent.limit_price
    if dry_run or not arm:
        planned_out["submitted"] = False
        planned_out["dry_run"] = True
        return planned_out
    if not paper_armed():
        raise OrderRefused("OPTIONS_PAPER_ARMED is not set — refusing submit")
    broker.gate.arm()
    result = broker.submit(intent)
    planned_out["submitted"] = True
    planned_out["dry_run"] = False
    planned_out["order"] = result
    return planned_out
