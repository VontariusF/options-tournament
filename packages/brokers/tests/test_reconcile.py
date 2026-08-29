"""Position-aware reconciliation — buy only the delta, never re-buy held names (no accumulation)."""
from pma_brokers.alpaca import reconcile, held_map


def _pos(symbol, qty, mv=100.0):
    return {"symbol": symbol, "qty": qty, "market_value": mv}


def test_held_map_normalizes_and_drops_zero_qty():
    h = held_map([_pos("AAPL", 3), _pos("BTC/USD", 0.5), _pos("GONE", 0)])
    assert set(h) == {"AAPL", "BTCUSD"}          # '/' stripped, upper; zero-qty dropped
    assert h["BTCUSD"]["symbol"] == "BTC/USD"    # raw symbol preserved for order placement


def test_reconcile_skips_held_buys_only_new():
    plan = [{"symbol": "AAPL", "notional": 100}, {"symbol": "MSFT", "notional": 100},
            {"symbol": "BTC/USD", "notional": 100}]
    positions = [_pos("AAPL", 3), _pos("BTCUSD", 0.5)]   # hold AAPL + BTC (Alpaca crypto no-slash)
    r = reconcile(plan, positions)
    assert [o["symbol"] for o in r["to_place"]] == ["MSFT"]      # only the un-held name
    assert set(r["skipped_held"]) == {"AAPL", "BTC/USD"}         # held names not re-bought
    assert r["to_close"] == []                                   # default: never sell


def test_reconcile_close_dropped_lists_orphans_with_qty():
    plan = [{"symbol": "AAPL", "notional": 100}]
    positions = [_pos("AAPL", 3), _pos("OLDCO", 10)]   # OLDCO held but dropped from target
    r = reconcile(plan, positions, close_dropped=True)
    assert r["to_place"] == []                          # AAPL already held
    assert r["to_close"] == [{"symbol": "OLDCO", "qty": 10.0}]   # flagged for flattening w/ qty to sell


def test_reconcile_empty_positions_places_all():
    plan = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
    r = reconcile(plan, [])
    assert [o["symbol"] for o in r["to_place"]] == ["AAPL", "MSFT"] and r["skipped_held"] == []
