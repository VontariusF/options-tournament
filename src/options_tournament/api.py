"""Local FastAPI: health, account, chain, execute."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from options_tournament.execute import StrategyCard, execute_card, paper_armed

app = FastAPI(title="options-tournament", version="0.1.0")


class CardBody(BaseModel):
    underlying: str
    structure: str
    dte: int = 7
    delta: float = 0.55
    wing_width: float = 10.0
    as_of: Optional[str] = None
    dry_run: bool = True
    arm: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "paper_armed": paper_armed()}


@app.get("/account")
def account() -> dict[str, Any]:
    from pma_brokers.alpaca.broker import AlpacaPaperBroker

    try:
        return AlpacaPaperBroker().nav()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)[:300]) from exc


@app.get("/chain/{ticker}")
def chain(ticker: str) -> dict[str, Any]:
    import datetime as dt
    from pma_brokers.alpaca.broker import AlpacaPaperBroker

    as_of = dt.date.today()
    lo = (as_of + dt.timedelta(days=1)).isoformat()
    hi = (as_of + dt.timedelta(days=45)).isoformat()
    try:
        rows = AlpacaPaperBroker().option_chain(
            ticker.upper(), expiration_gte=lo, expiration_lte=hi, feed="indicative",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)[:300]) from exc
    return {"ticker": ticker.upper(), "n": len(rows), "contracts": rows[:80]}


@app.post("/execute")
def execute(body: CardBody) -> dict[str, Any]:
    try:
        card = StrategyCard.from_dict(body.model_dump())
        return execute_card(card, dry_run=body.dry_run or not body.arm, arm=body.arm and not body.dry_run)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
