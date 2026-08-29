"""Tests for pma_common.secrets — env-first resolution and Supabase fetch behavior."""
import io
import json
import urllib.error

import pytest

from pma_common import secrets as S


@pytest.fixture(autouse=True)
def _clean():
    S.reset_cache()
    yield
    S.reset_cache()


def test_env_wins_over_supabase(monkeypatch):
    monkeypatch.setenv("TIINGO_API_KEY", "from-env")
    assert S.get_secret("TIINGO_API_KEY") == "from-env"


def test_supabase_fetch_success(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key")
    body = json.dumps([{"key": "ALPACA_PAPER_KEY", "value": "abc"}]).encode()

    class _Resp:
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    assert S.get_secret("ALPACA_PAPER_KEY") == "abc"
    assert S.supabase_fetch_error() is None


def test_supabase_fetch_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key")
    body = json.dumps([{"key": "TIINGO_API_KEY", "value": "tok"}]).encode()
    calls = {"n": 0}

    class _Resp:
        def read(self):
            return body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _urlopen(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.URLError("connection reset")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    assert S.get_secret("TIINGO_API_KEY") == "tok"
    assert calls["n"] == 2


def test_supabase_fetch_failure_not_cached(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "svc-key")

    def _fail(*a, **k):
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    monkeypatch.setattr(S.time, "sleep", lambda *_: None)
    assert S.get_secret("TIINGO_API_KEY") == ""
    assert S.supabase_fetch_error() is not None
    assert "fetch failed" in S.supabase_fetch_error()
    assert "transient Supabase outage" in S.format_secret_miss("TIINGO_API_KEY")


def test_missing_supabase_creds_cached(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    assert S.get_secret("TIINGO_API_KEY") == ""
    assert "SUPABASE_URL" in (S.supabase_fetch_error() or "")
