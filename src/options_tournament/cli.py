"""CLI: account, chain, execute, serve, mcp."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from options_tournament import __version__


def _load_env() -> None:
    load_dotenv(Path.cwd() / ".env")
    os.environ.setdefault("ALPACA_BROKER_TAG", "alpaca_options")
    os.environ.setdefault("OPTIONS_ONLY", "1")


def cmd_account(_args: argparse.Namespace) -> int:
    from pma_brokers.alpaca.broker import AlpacaPaperBroker
    from pma_brokers.alpaca.cli_preflight import run_alpaca_cli_preflight

    pre = run_alpaca_cli_preflight()
    nav = AlpacaPaperBroker().nav()
    print(json.dumps({"cli": pre, "account": nav}, default=str, indent=2))
    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    import datetime as dt
    from pma_brokers.alpaca.broker import AlpacaPaperBroker

    as_of = dt.date.today()
    lo = (as_of + dt.timedelta(days=1)).isoformat()
    hi = (as_of + dt.timedelta(days=int(args.window or 45))).isoformat()
    rows = AlpacaPaperBroker().option_chain(
        args.ticker.upper(), expiration_gte=lo, expiration_lte=hi, feed="indicative",
    )
    print(json.dumps({"ticker": args.ticker.upper(), "n": len(rows), "contracts": rows[:80]}, default=str, indent=2))
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    from options_tournament.execute import StrategyCard, execute_card

    raw = json.loads(Path(args.card).read_text())
    card = StrategyCard.from_dict(raw)
    out = execute_card(card, dry_run=not args.arm or args.dry_run, arm=bool(args.arm) and not args.dry_run)
    print(json.dumps(out, default=str, indent=2))
    return 0 if out.get("ok", True) else 2


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from options_tournament.api import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_mcp(_args: argparse.Namespace) -> int:
    from options_tournament.mcp_server import main as mcp_main

    mcp_main()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    from options_tournament.loop import run_loop

    print(run_loop(args.prompt))
    return 0


def main(argv: list[str] | None = None) -> int:
    _load_env()
    p = argparse.ArgumentParser(prog="options-tournament", description="Alpaca paper options tournament agent")
    p.add_argument("--version", action="version", version=f"options-tournament {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("account", help="Read paper account snapshot")

    ch = sub.add_parser("chain", help="Fetch paper options chain")
    ch.add_argument("ticker")
    ch.add_argument("--window", type=int, default=45)

    ex = sub.add_parser("execute", help="Plan or submit a strategy card")
    ex.add_argument("card", help="Path to strategy card JSON")
    ex.add_argument("--dry-run", action="store_true", help="Plan only (default unless --arm)")
    ex.add_argument("--arm", action="store_true", help="Submit if OPTIONS_PAPER_ARMED=1")

    sv = sub.add_parser("serve", help="Local HTTP API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8899)

    sub.add_parser("mcp", help="stdio MCP server")

    chat = sub.add_parser("chat", help="One-shot agent turn with options tools")
    chat.add_argument("prompt")

    args = p.parse_args(argv)
    if args.cmd == "account":
        return cmd_account(args)
    if args.cmd == "chain":
        return cmd_chain(args)
    if args.cmd == "execute":
        return cmd_execute(args)
    if args.cmd == "serve":
        return cmd_serve(args)
    if args.cmd == "mcp":
        return cmd_mcp(args)
    if args.cmd == "chat":
        return cmd_chat(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
