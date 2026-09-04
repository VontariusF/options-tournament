"""Alpaca PAPER-trading adapter (docs/ALPACA_PAPER_INTEGRATION_PLAN.md).

PAPER ONLY by construction: the client refuses any endpoint that is not the Alpaca *paper* host,
and the broker's order path is fail-closed behind an ArmingGate (DISARMED by default). There is no
real-capital code path here. Credentials resolve via ``pma_common.secrets`` (env -> Supabase).
"""
from .client import AlpacaClient, AlpacaError, AccountFingerprintMismatch, NotPaperEndpoint
from .broker import (AlpacaPaperBroker, ArmingGate, FillsUnavailable, OrderRefused, OrderIntent,
                     strategy_from_client_order_id, assert_defined_risk,
                     TAG_SANITIZE_RE, SQL_SANITIZE_EXPR, sanitize_strategy_tag)
from .options import (build_occ_symbol, parse_occ_symbol, select_call_contract,
                      select_option_contract, merge_chain_alpaca_primary,
                      fetch_chain_alpaca_primary, size_by_premium_cap, size_by_max_loss,
                      leg_liquidity_ok, credit_structure_ok, CONTRACT_MULTIPLIER)
from .reconcile import reconcile, held_map

__all__ = [
    "AlpacaClient", "AlpacaError", "AccountFingerprintMismatch", "NotPaperEndpoint",
    "AlpacaPaperBroker", "ArmingGate", "FillsUnavailable", "OrderRefused", "OrderIntent",
    "strategy_from_client_order_id", "assert_defined_risk",
    "TAG_SANITIZE_RE", "SQL_SANITIZE_EXPR", "sanitize_strategy_tag",
    "build_occ_symbol", "parse_occ_symbol", "select_call_contract",
    "select_option_contract", "merge_chain_alpaca_primary",
    "fetch_chain_alpaca_primary", "size_by_premium_cap", "size_by_max_loss",
    "leg_liquidity_ok", "credit_structure_ok", "CONTRACT_MULTIPLIER",
    "reconcile", "held_map",
]
