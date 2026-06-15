"""Backtest harness — replay historical prices through the REAL risk rules.

The point of this harness is to validate the protective machinery
(stop-loss, take-profit ladder, position sizing, daily loss limit) against
historical data *before* a cent is at risk. It deliberately reuses the same
``RiskManager`` and ``Portfolio`` classes the live bot uses, so what you test
is what you run.

Input: one CSV per token "trade opportunity", each a price series the bot
would have ridden after entering. Columns: ``timestamp,price`` (timestamp is
optional and only used for ordering/labels). Entry is assumed at the first
row; exits are decided entirely by the risk rules.

Usage:
    python -m bot.backtest data/*.csv
    python -m bot.backtest --demo          # run on generated synthetic data
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field

from .config import Config, load_config
from .portfolio import Portfolio
from .risk import RiskManager

log = logging.getLogger("backtest")


@dataclass
class TradeResult:
    symbol: str
    entry_price: float
    exit_reason: str
    realized_pnl: float
    return_pct: float


@dataclass
class BacktestReport:
    trades: list[TradeResult] = field(default_factory=list)
    starting_bankroll: float = 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl > 0)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl < 0)

    @property
    def win_rate(self) -> float:
        return (self.wins / len(self.trades) * 100.0) if self.trades else 0.0

    def equity_curve(self) -> list[float]:
        eq = self.starting_bankroll
        curve = [eq]
        for t in self.trades:
            eq += t.realized_pnl
            curve.append(eq)
        return curve

    def max_drawdown_pct(self) -> float:
        curve = self.equity_curve()
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            peak = max(peak, v)
            if peak > 0:
                max_dd = max(max_dd, (peak - v) / peak * 100.0)
        return max_dd


class Backtester:
    """Simulates one position lifecycle per price series, using live risk rules.

    ``fee_pct`` and ``slippage_pct`` are applied *per leg* (once on entry, once
    on every sell) as a percentage. They model the harsh reality of memecoin
    execution: you buy a little above the quote and sell a little below it, and
    pay DEX + priority fees each way. Risk *decisions* still trigger on the raw
    market price (that's what the live bot watches); the costs only bite the
    realized PnL — exactly as they do in real life.
    """

    def __init__(self, cfg: Config, fee_pct: float = 0.0, slippage_pct: float = 0.0):
        self.cfg = cfg
        self.cost_per_leg = (fee_pct + slippage_pct) / 100.0

    def run(self, series: list[tuple[str, list[float]]]) -> BacktestReport:
        risk = RiskManager(self.cfg)
        report = BacktestReport(starting_bankroll=self.cfg.bankroll_usd)

        for symbol, prices in series:
            if not prices:
                continue
            if not risk.can_open_new_position(0):
                log.info("Daily loss limit reached — skipping remaining trades.")
                break
            result = self._simulate_one(symbol, prices, risk)
            if result is not None:
                report.trades.append(result)
                risk.record_realized_pnl(result.realized_pnl)
        return report

    def _simulate_one(self, symbol: str, prices: list[float],
                      risk: RiskManager) -> TradeResult | None:
        entry = prices[0]
        if entry <= 0:
            return None
        pf = Portfolio()
        size_usd = risk.position_size_usd()
        # Buy-side cost: you effectively pay above the market price per token.
        eff_entry = entry * (1.0 + self.cost_per_leg)
        tokens = size_usd / eff_entry
        # Portfolio's cost basis uses eff_entry so PnL reflects what you paid;
        # risk decisions below use the raw market `entry`/`price`.
        pf.open(symbol, symbol, eff_entry, size_usd, tokens)
        pos = pf.positions[symbol]

        last_reason = "end_of_series"
        for price in prices[1:]:
            actions = risk.evaluate_exit(entry, price, pos.ladder_filled)
            for action in actions:
                if action.reason.startswith("take_profit:"):
                    pos.ladder_filled.add(float(action.reason.split(":")[1]))
                last_reason = action.reason
                pf.sell_fraction(symbol, action.fraction,
                                 self._sell_price(price), action.reason)
                if symbol not in pf.positions:  # fully closed (stop-loss)
                    break
            if symbol not in pf.positions:
                break

        # Liquidate any remainder at the final price (end of the window).
        if symbol in pf.positions and pos.tokens > 1e-9:
            pf.sell_fraction(symbol, pos.tokens / pos.original_tokens,
                             self._sell_price(prices[-1]), "end_of_series")

        realized = pf.realized_pnl
        return TradeResult(
            symbol=symbol,
            entry_price=entry,
            exit_reason=last_reason,
            realized_pnl=realized,
            return_pct=(realized / size_usd * 100.0) if size_usd else 0.0,
        )

    def _sell_price(self, market_price: float) -> float:
        """Sell-side cost: you effectively receive below the market price."""
        return market_price * (1.0 - self.cost_per_leg)


# -- IO + reporting ---------------------------------------------------------

def load_series_from_csv(path: str) -> tuple[str, list[float]]:
    symbol = os.path.splitext(os.path.basename(path))[0]
    prices: list[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_price_col = reader.fieldnames and "price" in reader.fieldnames
        f.seek(0)
        if has_price_col:
            for row in csv.DictReader(f):
                try:
                    prices.append(float(row["price"]))
                except (ValueError, KeyError):
                    continue
        else:  # headerless single-column of prices
            for row in csv.reader(f):
                if not row:
                    continue
                try:
                    prices.append(float(row[-1]))
                except ValueError:
                    continue
    return symbol, prices


def generate_demo_series(n_trades: int = 40, seed: int = 7) -> list[tuple[str, list[float]]]:
    """Synthetic memecoin-like paths: mostly losers, a few big winners — the
    realistic shape. Lets you sanity-check that risk rules survive it."""
    rng = random.Random(seed)
    series: list[tuple[str, list[float]]] = []
    for i in range(n_trades):
        price = 1.0
        path = [price]
        # Heavy-tailed drift: most tokens bleed out, a minority moon.
        moon = rng.random() < 0.18
        drift = rng.uniform(0.01, 0.05) if moon else rng.uniform(-0.06, -0.01)
        vol = rng.uniform(0.04, 0.12)
        for _ in range(rng.randint(20, 60)):
            shock = rng.gauss(drift, vol)
            price = max(0.0001, price * math.exp(shock))
            path.append(price)
        series.append((f"MEME{i:02d}", path))
    return series


def split_series(series, in_sample_frac: float):
    """Split the trade opportunities into an in-sample (earlier) and
    out-of-sample (later) segment by order."""
    n = len(series)
    cut = max(1, min(n - 1, int(round(n * in_sample_frac)))) if n > 1 else n
    return series[:cut], series[cut:]


def run_walk_forward(backtester: "Backtester", series, in_sample_frac: float = 0.5):
    """Run the backtest on the first segment, then the held-out second segment.

    NOTE: this strategy has no fitted parameters, so this is a *consistency*
    check, not classic parameter walk-forward. It answers a blunt question: did
    the result hold up on data the run didn't get to 'see' first? If the
    out-of-sample segment falls apart, be very suspicious of the in-sample one.
    """
    in_series, out_series = split_series(series, in_sample_frac)
    in_report = backtester.run(in_series)
    out_report = backtester.run(out_series)
    return in_report, out_report


def print_walk_forward(in_report: BacktestReport, out_report: BacktestReport) -> None:
    def roi(r: BacktestReport) -> float:
        return (r.total_pnl / r.starting_bankroll * 100.0) if r.starting_bankroll else 0.0

    print("\n" + "=" * 60)
    print("WALK-FORWARD (consistency check)")
    print("=" * 60)
    print(f"{'':18s}{'in-sample':>14s}{'out-of-sample':>16s}")
    print(f"{'trades':18s}{len(in_report.trades):>14d}{len(out_report.trades):>16d}")
    print(f"{'win rate %':18s}{in_report.win_rate:>14.1f}{out_report.win_rate:>16.1f}")
    print(f"{'ROI %':18s}{roi(in_report):>14.1f}{roi(out_report):>16.1f}")
    print(f"{'max drawdown %':18s}{in_report.max_drawdown_pct():>14.1f}"
          f"{out_report.max_drawdown_pct():>16.1f}")
    print("-" * 60)
    in_roi, out_roi = roi(in_report), roi(out_report)
    if in_roi > 0 and out_roi < 0:
        print("⚠️  Out-of-sample is a LOSS while in-sample profited — strong "
              "overfitting / non-stationarity signal. Do not trust the result.")
    elif in_roi > 0 and out_roi < in_roi * 0.5:
        print("⚠️  Out-of-sample profit is less than half of in-sample — the "
              "edge may be fragile.")
    else:
        print("Out-of-sample is broadly consistent with in-sample (still no "
              "guarantee of future results).")
    print("=" * 60 + "\n")


def export_equity_csv(report: BacktestReport, path: str) -> None:
    """Write the per-trade equity curve so you can chart it elsewhere.

    Columns: trade_index, symbol, exit_reason, realized_pnl, return_pct,
    equity. Row 0 is the starting bankroll before any trade.
    """
    curve = report.equity_curve()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trade_index", "symbol", "exit_reason",
                    "realized_pnl", "return_pct", "equity"])
        w.writerow([0, "", "start", 0.0, 0.0, f"{curve[0]:.4f}"])
        for i, t in enumerate(report.trades, start=1):
            w.writerow([i, t.symbol, t.exit_reason,
                        f"{t.realized_pnl:.4f}", f"{t.return_pct:.4f}",
                        f"{curve[i]:.4f}"])


def print_report(report: BacktestReport, cost_per_leg_pct: float = 0.0) -> None:
    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)
    if cost_per_leg_pct > 0:
        print(f"Cost model:        {cost_per_leg_pct:.2f}% per leg "
              f"(~{cost_per_leg_pct * 2:.2f}% round trip)")
    print(f"Trades:            {len(report.trades)}")
    print(f"Wins / Losses:     {report.wins} / {report.losses}")
    print(f"Win rate:          {report.win_rate:.1f}%")
    print(f"Starting bankroll: ${report.starting_bankroll:.2f}")
    print(f"Total PnL:         ${report.total_pnl:+.2f}")
    final = report.starting_bankroll + report.total_pnl
    roi = (report.total_pnl / report.starting_bankroll * 100.0) if report.starting_bankroll else 0
    print(f"Final bankroll:    ${final:.2f}  ({roi:+.1f}%)")
    print(f"Max drawdown:      {report.max_drawdown_pct():.1f}%")
    print("-" * 60)
    by_reason: dict[str, int] = {}
    for t in report.trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
    print("Exits by reason:")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason:20s} {count}")
    print("=" * 60)
    print("Reminder: a positive backtest is NOT a promise of future profit. "
          "Memecoins are adversarial and historical paths don't repeat.\n")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description="Backtest the risk rules on price series.")
    parser.add_argument("paths", nargs="*", help="CSV files (glob ok) with a 'price' column")
    parser.add_argument("--demo", action="store_true", help="use generated synthetic data")
    parser.add_argument("--fee-pct", type=float, default=0.3,
                        help="swap fee per leg, percent (default 0.3)")
    parser.add_argument("--slippage-pct", type=float, default=1.0,
                        help="slippage per leg, percent (default 1.0; memecoins are worse)")
    parser.add_argument("--csv", metavar="PATH",
                        help="write the per-trade equity curve to this CSV")
    parser.add_argument("--walk-forward", type=float, nargs="?", const=0.5,
                        metavar="FRAC",
                        help="split trades into in-sample/out-of-sample at FRAC "
                             "(default 0.5) and report both")
    args = parser.parse_args(argv)

    cfg = load_config()

    if args.demo or not args.paths:
        if not args.demo and not args.paths:
            print("No CSVs given — running --demo on synthetic data.\n")
        series = generate_demo_series()
    else:
        files: list[str] = []
        for p in args.paths:
            files.extend(sorted(glob.glob(p)))
        if not files:
            print(f"No files matched: {args.paths}", file=sys.stderr)
            return 1
        series = [load_series_from_csv(f) for f in files]

    backtester = Backtester(cfg, fee_pct=args.fee_pct, slippage_pct=args.slippage_pct)
    report = backtester.run(series)
    print_report(report, cost_per_leg_pct=args.fee_pct + args.slippage_pct)
    if args.csv:
        export_equity_csv(report, args.csv)
        print(f"Equity curve written to {args.csv}")
    if args.walk_forward is not None:
        in_report, out_report = run_walk_forward(backtester, series, args.walk_forward)
        print_walk_forward(in_report, out_report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
