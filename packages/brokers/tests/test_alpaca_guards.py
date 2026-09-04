"""Account and fill guards: strict options creds, fingerprint on POST, fill outage vs empty book.

Flags default off: with every env unset, behavior matches the prior path. Each test covers both
the default and the flipped state.
"""
import logging

import pytest

from pma_brokers.alpaca import client as client_mod
from pma_brokers.alpaca.broker import AlpacaPaperBroker, ArmingGate, FillsUnavailable
from pma_brokers.alpaca.client import AccountFingerprintMismatch, AlpacaClient, AlpacaError


PAPER_EP = "https://paper-api.alpaca.markets/v2"


def _options_env(monkeypatch, **extra):
    monkeypatch.setenv("ALPACA_BROKER_TAG", "alpaca_options")
    for k in ("OPTIONS_STRICT_CREDS", "ALPACA_OPTIONS_EXPECTED_ACCOUNT", "BROKER_FILLS_FAIL_LOUD",
              "ALPACA_OPTIONS_PAPER_KEY", "ALPACA_OPTIONS_PAPER_SECRET",
              "ALPACA_PAPER_KEY", "ALPACA_PAPER_SECRET"):
        monkeypatch.delenv(k, raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def _no_supabase(monkeypatch):
    import pma_common.secrets as S
    monkeypatch.setattr(S, "_load_supabase", lambda: {})


def test_dark_default_fallback_still_works_but_warns(monkeypatch, caplog):
    _options_env(monkeypatch, ALPACA_PAPER_KEY="eqk", ALPACA_PAPER_SECRET="eqs")
    _no_supabase(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="pma_brokers.alpaca"):
        key, secret, _ = client_mod._resolve_creds()
    assert (key, secret) == ("eqk", "eqs")
    assert any("falling back" in r.message for r in caplog.records)


def test_options_keys_present_no_fallback_no_warning(monkeypatch, caplog):
    _options_env(monkeypatch, ALPACA_OPTIONS_PAPER_KEY="ok", ALPACA_OPTIONS_PAPER_SECRET="os",
                 ALPACA_PAPER_KEY="eqk", ALPACA_PAPER_SECRET="eqs")
    _no_supabase(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="pma_brokers.alpaca"):
        key, secret, _ = client_mod._resolve_creds()
    assert (key, secret) == ("ok", "os")
    assert not any("falling back" in r.message for r in caplog.records)


def test_half_configured_pair_fails_closed_never_mixes(monkeypatch):
    _options_env(monkeypatch, ALPACA_OPTIONS_PAPER_KEY="ok",
                 ALPACA_PAPER_KEY="eqk", ALPACA_PAPER_SECRET="eqs")
    _no_supabase(monkeypatch)
    with pytest.raises(AlpacaError, match="ALPACA_OPTIONS_PAPER"):
        client_mod._resolve_creds()


def test_strict_creds_removes_fallback(monkeypatch):
    _options_env(monkeypatch, OPTIONS_STRICT_CREDS="1",
                 ALPACA_PAPER_KEY="eqk", ALPACA_PAPER_SECRET="eqs")
    _no_supabase(monkeypatch)
    with pytest.raises(AlpacaError):
        client_mod._resolve_creds()


def test_non_options_process_unaffected_by_strict_flag(monkeypatch):
    _options_env(monkeypatch, OPTIONS_STRICT_CREDS="1",
                 ALPACA_PAPER_KEY="eqk", ALPACA_PAPER_SECRET="eqs")
    monkeypatch.delenv("ALPACA_BROKER_TAG", raising=False)
    monkeypatch.delenv("OPTIONS_ONLY", raising=False)
    _no_supabase(monkeypatch)
    key, secret, _ = client_mod._resolve_creds()
    assert (key, secret) == ("eqk", "eqs")


def test_construction_never_fetches_account_even_on_mismatch(monkeypatch):
    _options_env(monkeypatch, ALPACA_OPTIONS_EXPECTED_ACCOUNT="PA3OPT")
    calls = []
    monkeypatch.setattr(AlpacaClient, "get",
                        lambda self, *a, **k: calls.append(a) or {"account_number": "PA9EQU"})
    AlpacaClient(key="k", secret="s", endpoint=PAPER_EP)
    assert calls == []


def test_fingerprint_unset_skips_check_on_post(monkeypatch):
    _options_env(monkeypatch)
    calls = []
    monkeypatch.setattr(AlpacaClient, "get", lambda self, *a, **k: calls.append(a) or {})
    monkeypatch.setattr(AlpacaClient, "_request", lambda self, *a, **k: {"id": "ok"})
    c = AlpacaClient(key="k", secret="s", endpoint=PAPER_EP)
    c.post("orders", {})
    assert calls == []


def test_fingerprint_match_passes_on_post(monkeypatch):
    _options_env(monkeypatch, ALPACA_OPTIONS_EXPECTED_ACCOUNT="PA3OPT")
    monkeypatch.setattr(AlpacaClient, "get", lambda self, path, *a, **k: {"account_number": "PA3OPT"})
    posted = []
    monkeypatch.setattr(AlpacaClient, "_request", lambda self, *a, **k: posted.append(a) or {"id": "ok"})
    c = AlpacaClient(key="k", secret="s", endpoint=PAPER_EP)
    c.post("orders", {})
    c.post("orders", {})
    assert len(posted) == 2


def test_fingerprint_mismatch_blocks_the_order(monkeypatch):
    _options_env(monkeypatch, ALPACA_OPTIONS_EXPECTED_ACCOUNT="PA3OPT")
    monkeypatch.setattr(AlpacaClient, "get", lambda self, path, *a, **k: {"account_number": "PA9EQU"})
    posted = []
    monkeypatch.setattr(AlpacaClient, "_request", lambda self, *a, **k: posted.append(a) or {})
    c = AlpacaClient(key="k", secret="s", endpoint=PAPER_EP)
    with pytest.raises(AccountFingerprintMismatch):
        c.post("orders", {})
    assert posted == []


def test_fingerprint_never_judges_a_non_options_client(monkeypatch):
    _options_env(monkeypatch, ALPACA_OPTIONS_EXPECTED_ACCOUNT="PA3OPT")
    monkeypatch.delenv("ALPACA_BROKER_TAG", raising=False)
    monkeypatch.delenv("OPTIONS_ONLY", raising=False)
    calls = []
    monkeypatch.setattr(AlpacaClient, "get",
                        lambda self, *a, **k: calls.append(a) or {"account_number": "PA9EQU"})
    AlpacaClient(key="k", secret="s", endpoint=PAPER_EP)
    assert calls == []


class _DownClient:
    def get(self, *a, **k):
        raise RuntimeError("simulated outage")


def _broker():
    return AlpacaPaperBroker(client=_DownClient(), gate=ArmingGate(armed=False))


def test_fills_dark_default_returns_empty_but_warns(monkeypatch, caplog):
    monkeypatch.delenv("BROKER_FILLS_FAIL_LOUD", raising=False)
    with caplog.at_level(logging.WARNING):
        assert _broker().fills() == []
    assert any("fills unavailable" in r.message for r in caplog.records)


def test_fills_fail_loud_raises_typed(monkeypatch):
    monkeypatch.setenv("BROKER_FILLS_FAIL_LOUD", "1")
    with pytest.raises(FillsUnavailable):
        _broker().fills()
