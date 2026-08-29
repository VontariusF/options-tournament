"""Alpaca CLI / MCP preflight (hackathon: MCP server OR CLI required).

Never installs the CLI or an MCP server. If ``alpaca`` is already on PATH, run a
read-only account identity command. Order routing stays on the Trading API client.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional


def alpaca_cli_path() -> Optional[str]:
    return shutil.which("alpaca") or shutil.which("alpaca-cli")


def run_alpaca_cli_preflight(*, timeout: float = 15.0) -> dict:
    """Best-effort identity check via Alpaca CLI. Returns a JSON-serializable dict.

    ``ok=True`` means a CLI binary answered. ``ok=False`` with ``reason`` is informational
    unless the caller treats ``ALPACA_CLI_REQUIRED=1`` as fatal.
    """
    extra = (os.environ.get("ALPACA_CLI_COMMAND") or "").strip()
    exe = alpaca_cli_path()
    if extra:
        argv = extra.split()
    elif exe:
        argv = [exe, "account", "get"]
    else:
        return {"ok": False, "reason": "alpaca CLI not on PATH — using Trading API",
                "mcp_hint": "set ALPACA_CLI_COMMAND or install the Alpaca CLI yourself; "
                            "do not have this process install it"}
    try:
        proc = subprocess.run(  # noqa: S603 — argv is a local binary the operator provided
            argv, capture_output=True, text=True, timeout=timeout, check=False,
            env={**os.environ,
             "ALPACA_API_KEY": os.environ.get("ALPACA_OPTIONS_PAPER_KEY")
             or os.environ.get("ALPACA_PAPER_KEY", ""),
             "ALPACA_API_SECRET": os.environ.get("ALPACA_OPTIONS_PAPER_SECRET")
             or os.environ.get("ALPACA_PAPER_SECRET", ""),
             "APCA_API_KEY_ID": os.environ.get("ALPACA_OPTIONS_PAPER_KEY")
             or os.environ.get("ALPACA_PAPER_KEY", ""),
             "APCA_API_SECRET_KEY": os.environ.get("ALPACA_OPTIONS_PAPER_SECRET")
             or os.environ.get("ALPACA_PAPER_SECRET", ""),
             "APCA_API_BASE_URL": os.environ.get("ALPACA_PAPER_ENDPOINT",
                                                 "https://paper-api.alpaca.markets/v2")},
        )
    except FileNotFoundError:
        return {"ok": False, "reason": "CLI binary disappeared"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"CLI timed out after {timeout}s: {' '.join(argv[:4])}"}
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    parsed = None
    if out:
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            parsed = None
    account_id = None
    if isinstance(parsed, dict):
        account_id = parsed.get("id") or parsed.get("account_number") or parsed.get("account_id")
    return {
        "ok": proc.returncode == 0,
        "argv": argv[:4],
        "returncode": proc.returncode,
        "account_id": account_id,
        "stdout_head": out[:240],
        "stderr_head": err[:240],
        "reason": None if proc.returncode == 0 else (err[:160] or f"exit {proc.returncode}"),
    }
