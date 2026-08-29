"""Secret resolution: env first, then optional Supabase ``app_secrets``, then default.

An explicit env var always wins. Supabase is an optional fallback so a host
without the key in its environment can still resolve it. This extract does not
require Supabase.

Stdlib only (urllib). Supabase creds come from env: ``SUPABASE_URL`` +
``SUPABASE_SERVICE_KEY``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

_cache: dict[str, str] | None = None
_fetch_error: str | None = None

_MAX_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_S = (0.5, 1.0, 2.0)


def _supabase_creds() -> tuple[str, str]:
    return (
        os.getenv("SUPABASE_URL", "").strip().rstrip("/"),
        os.getenv("SUPABASE_SERVICE_KEY", "").strip(),
    )


def supabase_fetch_error() -> str | None:
    """Last Supabase fetch failure, if the most recent load did not succeed."""
    return _fetch_error


def format_secret_miss(*keys: str) -> str:
    """Human-readable miss reason — distinguishes fetch failure from absent keys."""
    labels = " / ".join(keys)
    err = supabase_fetch_error()
    if err:
        return (f"{labels} unavailable — {err} "
                f"(transient Supabase outage; not necessarily missing from app_secrets)")
    return f"{labels} not set (env or Supabase app_secrets)"


def _load_supabase(timeout: float = 10.0) -> dict[str, str]:
    """Fetch all app_secrets rows once and cache on success.

    On network/API failure: retry with backoff and do NOT cache an empty dict so
    the next ``get_secret`` call can retry (avoids false "not configured" pages
    when Supabase blips). Missing SUPABASE_URL/SERVICE_KEY is cached as empty —
    that is a static config gap, not transient."""
    global _cache, _fetch_error
    if _cache is not None:
        return _cache
    url, key = _supabase_creds()
    if not url or not key:
        _fetch_error = "SUPABASE_URL or SUPABASE_SERVICE_KEY not set in env"
        _cache = {}
        return _cache
    req = urllib.request.Request(
        f"{url}/rest/v1/app_secrets?select=key,value",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    last_err: Exception | None = None
    for attempt in range(_MAX_FETCH_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            _cache = {
                str(r.get("key", "")).strip(): str(r.get("value", ""))
                for r in rows
                if r.get("key")
            }
            _fetch_error = None
            return _cache
        except Exception as e:  # noqa: BLE001 — retry then surface via format_secret_miss
            last_err = e
            if attempt + 1 < _MAX_FETCH_ATTEMPTS:
                time.sleep(_FETCH_BACKOFF_S[attempt])
    _fetch_error = (
        f"Supabase app_secrets fetch failed after {_MAX_FETCH_ATTEMPTS} attempts: "
        f"{last_err!s}"[:280]
    )
    return {}


def get_secret(*keys: str, default: str = "") -> str:
    """Resolve the first of ``keys`` found in env, then Supabase; else ``default``.

    Accepts multiple aliases (e.g. TIINGO_API_TOKEN, TIINGO_API_KEY) and returns
    the first non-empty hit across env (all keys) then Supabase (all keys).
    """
    for k in keys:
        v = os.getenv(k, "").strip()
        if v:
            return v
    sb = _load_supabase()
    for k in keys:
        v = sb.get(k, "").strip()
        if v:
            return v
    return default


def reset_cache() -> None:
    """Test hook: drop the Supabase cache and last fetch error."""
    global _cache, _fetch_error
    _cache = None
    _fetch_error = None
