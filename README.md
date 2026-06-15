# Solana Memecoin Trading Bot

A capital-preservation–focused trading bot for Solana memecoins, built around
**Jupiter** (swap aggregation) and on-chain safety screening.

> ## ⚠️ Read this first — there is no such thing as a bot that "doesn't lose"
>
> Memecoins are the single highest-risk asset class in crypto. They are
> dominated by rug pulls, honeypots, sniper bots, and pump-and-dumps. Most
> memecoins go to zero. **You can and will lose money. You can lose all of it.**
>
> This bot's entire design goal is *capital preservation*: keeping losses
> small, capped, and survivable so that the occasional winner can pay for the
> many losers. It does **not** guarantee profit. Anyone who promises a
> no-loss trading bot is lying to you.
>
> Only ever trade with money you are fully prepared to lose.

## What it actually does

- **Screens tokens before buying** — rejects honeypots, unlocked liquidity,
  active mint authority, and concentrated holders (rug indicators).
- **Sizes positions defensively** — never risks more than a configured % of
  the bankroll on a single trade.
- **Always sets a stop-loss** — every position has a hard downside cap.
- **Takes profit in ladders** — sells portions on the way up so gains get
  banked instead of round-tripping to zero.
- **Enforces a daily loss limit + kill switch** — the bot stops itself for the
  day after losing a configured amount.
- **Defaults to a dry-run safety lock** — real trades require you to
  explicitly set `LIVE_TRADING=true` *and* provide a funded wallet key.

## Safety model (please understand this before going live)

The bot ships **locked**. Out of the box it runs in `DRY_RUN` mode: it fetches
live prices, evaluates the strategy, and logs the trades it *would* make,
without spending a cent. To trade real funds you must do **both**:

1. Set `LIVE_TRADING=true` in your `.env`.
2. Provide a real `WALLET_PRIVATE_KEY` for a funded wallet.

This is intentional friction. Run in dry-run for as long as it takes to trust
the behavior.

## Quick start

```bash
cd trading-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.env .env      # then edit .env
python -m bot.main              # dry-run by default
```

### Run the tests

```bash
pip install -r requirements.txt
pytest
```

## Backtesting

Validate the risk rules against historical price data *before* risking funds.
The backtester replays price series through the **same** `RiskManager` the live
bot uses, so what you test is what you run.

```bash
python -m bot.backtest --demo                     # synthetic memecoin data
python -m bot.backtest data/*.csv                 # your own CSVs
python -m bot.backtest --demo --csv equity.csv    # export the equity curve
python -m bot.backtest --demo --slippage-pct 3    # model nastier slippage
```

Each CSV is one "trade opportunity" — a price path the bot rides after entry.
The report shows win rate, total PnL, max drawdown, and exits by reason. A
typical run looks like *low* win rate but *positive* PnL: most trades stop out
small, a few winners ladder out big. That asymmetry is the whole point.

### Cost model (slippage + fees)

The backtest applies realistic execution costs **per leg** — once on entry and
once on every sell:

- `--fee-pct` (default `0.3`): DEX + priority fee each swap.
- `--slippage-pct` (default `1.0`): how far above/below the quote you fill.
  Memecoins are routinely worse; try `3`–`5` for thin tokens.

Costs only bite realized PnL — risk *decisions* still trigger on the raw market
price, exactly like the live bot. This matters: on the demo data, costs cut the
result from roughly +3% (frictionless) to under +1%. Friction is often the
difference between an "edge" and a slow bleed, so test with honest numbers.

### Equity-curve export

`--csv PATH` writes one row per trade — `trade_index, symbol, exit_reason,
realized_pnl, return_pct, equity` — so you can chart the equity curve in a
spreadsheet or notebook.

### Walk-forward consistency check

`--walk-forward [FRAC]` (default `0.5`) splits the trade opportunities into an
earlier *in-sample* segment and a held-out *out-of-sample* segment, then
reports both side by side. If the result only works in-sample and falls apart
out-of-sample, you're looking at overfitting or non-stationary data — the bot
prints a loud warning. (This strategy has no fitted parameters, so it's a
consistency check rather than classic parameter walk-forward.)

```bash
python -m bot.backtest data/*.csv --walk-forward 0.6
```

## Wallet reconciliation

On startup, **when armed for live trading**, the bot reads your wallet's actual
SPL token balances from the chain and compares them to the positions it has
been tracking. It flags three kinds of drift before trading on a stale picture:

- **phantom** — the bot thinks it holds a token the wallet no longer has
  (sold elsewhere? failed record?).
- **untracked** — the wallet holds a token the bot isn't managing.
- **drift** — both agree the token is held, but the amounts differ beyond a
  small tolerance.

Mismatches are logged loudly and, if alerts are configured, pushed to
Telegram/Discord. In dry-run there's no wallet, so this step is skipped.

> A positive backtest is **not** a promise of future profit. Memecoins are
> adversarial and historical paths don't repeat.

## Alerts

Optional Telegram and/or Discord notifications on every buy, sell, stop-loss,
and daily-limit halt. Both are off unless configured (see `config.example.env`).
Alerts are best-effort and never block or crash a trade.

## Persistence

Open positions, realized PnL, and the daily circuit-breaker state are saved to
`STATE_FILE` (default `state.json`) after every trade and on shutdown, using
atomic writes. On restart the bot restores them — so a crash or restart won't
orphan a live position or reset your daily loss limit. (The state file is
git-ignored.)

## Configuration

See `config.example.env` for every knob, but the important ones:

| Variable | Meaning | Sane default |
|---|---|---|
| `LIVE_TRADING` | Master switch for real money | `false` |
| `BANKROLL_USD` | Total capital the bot may manage | `100` |
| `MAX_POSITION_PCT` | Max % of bankroll per trade | `2` |
| `STOP_LOSS_PCT` | Hard stop-loss per position | `15` |
| `DAILY_LOSS_LIMIT_PCT` | Stop trading for the day after this drawdown | `10` |
| `TAKE_PROFIT_LADDER` | Profit targets / sell fractions | `50:0.5,100:0.25,300:0.25` |
| `MAX_OPEN_POSITIONS` | Concurrent positions allowed | `3` |

## Architecture

```
bot/
  config.py      # loads + validates settings, enforces the safety lock
  safety.py      # honeypot / rug screening (the most important file)
  risk.py        # position sizing, stop-loss, take-profit, daily limit
  jupiter.py     # price quotes + swap execution via Jupiter
  portfolio.py   # tracks positions and realized/unrealized PnL
  strategy.py    # the (replaceable) entry signal
  alerts.py      # optional Telegram/Discord trade notifications
  state.py       # atomic save/restore of positions + daily state
  reconcile.py   # startup check: tracked positions vs on-chain wallet
  backtest.py    # replay prices through the real risk rules (+ walk-forward)
  main.py        # the loop that ties it together
```

The strategy in `strategy.py` is deliberately simple and meant to be replaced.
The parts that protect you — `safety.py` and `risk.py` — are the parts worth
trusting.

## Legal / disclaimer

This software is provided for educational purposes, as-is, with no warranty.
It is not financial advice. Trading cryptocurrency is risky and may be
restricted where you live. You are solely responsible for your funds, your
keys, your taxes, and your losses.
