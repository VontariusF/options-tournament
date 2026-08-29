# options-tournament

Paper-traded US equity options on Alpaca. The agent sizes and submits defined-risk structures — long calls, long puts, and credit put spreads — against a dedicated paper account.

Use it to run a tournament book: read the chain, pick a contract by delta and days-to-expiry, cap loss in Python, and send limit orders to the paper host. Nothing here routes to live trading.

## Install

Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill `.env` with an Alpaca **paper** keypair. Keep `OPTIONS_PAPER_ARMED=0` until you intend to submit.

## Commands

```bash
options-tournament account
options-tournament chain AAPL
options-tournament execute card.json --dry-run
options-tournament execute card.json --arm
options-tournament serve
options-tournament mcp
```

`serve` binds `127.0.0.1:8899`. `mcp` speaks stdio for MCP clients. If the Alpaca CLI is on PATH it is used for account identity; orders still go through the Trading API.

## Strategy card

```json
{
  "underlying": "AAPL",
  "structure": "long_call",
  "dte": 7,
  "delta": 0.55
}
```

`credit_put_spread` also accepts `wing_width` (strike dollars). Selection, sizing, and max-loss stay in Python.

## Safety

- The trading client refuses any host that is not `paper-api.alpaca.markets`
- Submits require `OPTIONS_PAPER_ARMED=1` and `--arm`
- A credit spread with an uncovered short leg is refused
- Keys live in `.env`, which is gitignored

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
