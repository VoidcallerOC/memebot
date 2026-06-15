"""Tests for the add-ons: persistence round-trip, backtest mechanics, and
that alerts stay disabled unless configured."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.alerts import Notifier
from bot.backtest import (
    Backtester,
    export_equity_csv,
    generate_demo_series,
    run_walk_forward,
    split_series,
)
from bot.reconcile import compare_holdings
from bot.config import Config
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.state import load_state, save_state


def _cfg(**kw) -> Config:
    base = dict(bankroll_usd=1000.0, max_position_pct=2.0, stop_loss_pct=15.0,
                daily_loss_limit_pct=10.0,
                take_profit_ladder=[(50.0, 0.5), (100.0, 0.25), (300.0, 0.25)])
    base.update(kw)
    return Config(**base)


# -- persistence ------------------------------------------------------------

def test_state_roundtrip_preserves_positions_and_pnl():
    pf = Portfolio()
    pf.open("mint1", "PEPE", entry_price=1.0, size_usd=100.0, tokens=100.0)
    pf.sell_fraction("mint1", 0.5, current_price=2.0, reason="take_profit:100")
    pf.positions["mint1"].ladder_filled.add(100.0)
    risk = RiskManager(_cfg())
    risk.record_realized_pnl(50.0)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "state.json")
        save_state(path, pf, risk)

        pf2, risk2 = Portfolio(), RiskManager(_cfg())
        assert load_state(path, pf2, risk2) is True

    assert pf2.realized_pnl == pf.realized_pnl
    assert "mint1" in pf2.positions
    restored = pf2.positions["mint1"]
    assert restored.tokens == 50.0
    assert restored.ladder_filled == {100.0}
    # original token count must survive so take-profit fractions stay anchored
    assert restored.original_tokens == 100.0
    assert risk2.realized_pnl_today == 50.0


def test_load_state_missing_file_is_false():
    pf, risk = Portfolio(), RiskManager(_cfg())
    assert load_state("/nonexistent/path/state.json", pf, risk) is False


def test_restored_take_profit_does_not_refire():
    """A rung sold before restart must not sell again after restart."""
    pf = Portfolio()
    pf.open("m", "X", entry_price=1.0, size_usd=100.0, tokens=100.0)
    pf.sell_fraction("m", 0.5, 1.6, "take_profit:50")
    pf.positions["m"].ladder_filled.add(50.0)
    risk = RiskManager(_cfg())

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.json")
        save_state(path, pf, risk)
        pf2, risk2 = Portfolio(), RiskManager(_cfg())
        load_state(path, pf2, risk2)

    pos = pf2.positions["m"]
    actions = risk2.evaluate_exit(pos.entry_price, 1.6, pos.ladder_filled)
    assert all(a.reason != "take_profit:50" for a in actions)


# -- backtest ---------------------------------------------------------------

def test_backtest_stop_loss_caps_loss():
    cfg = _cfg(stop_loss_pct=15)
    # price crashes straight down -> stop-loss must fire, loss ~= size * 15%
    series = [("CRASH", [1.0, 0.9, 0.8, 0.5, 0.1])]
    report = Backtester(cfg).run(series)
    assert len(report.trades) == 1
    t = report.trades[0]
    assert t.exit_reason == "stop_loss"
    # loss should be bounded near the stop, not the full -90% move
    assert t.return_pct <= -15.0
    assert t.return_pct > -25.0  # not anywhere near total wipeout


def test_backtest_take_profit_banks_gains():
    cfg = _cfg()
    series = [("MOON", [1.0, 1.5, 2.0, 4.0, 0.5])]  # pumps then dumps
    report = Backtester(cfg).run(series)
    t = report.trades[0]
    # ladder should have banked gains on the way up, so net positive despite
    # the final dump back below entry.
    assert t.realized_pnl > 0


def test_costs_reduce_pnl_vs_frictionless():
    cfg = _cfg()
    series = [("MOON", [1.0, 1.5, 2.0, 4.0, 4.0])]
    free = Backtester(cfg, fee_pct=0.0, slippage_pct=0.0).run(series).total_pnl
    costed = Backtester(cfg, fee_pct=0.3, slippage_pct=1.0).run(series).total_pnl
    # Costs must eat into PnL, but a real winner should still be profitable.
    assert costed < free
    assert costed > 0


def test_costs_make_a_marginal_winner_a_loser():
    cfg = _cfg(stop_loss_pct=15)
    # Tiny grind up that barely clears nothing: frictionless ~flat, costs bite.
    series = [("FLAT", [1.0, 1.02, 1.01, 1.0])]
    free = Backtester(cfg, fee_pct=0.0, slippage_pct=0.0).run(series).total_pnl
    costed = Backtester(cfg, fee_pct=0.3, slippage_pct=1.0).run(series).total_pnl
    assert costed < free
    assert costed < 0  # round-trip friction turns a flat trade into a loss


def test_equity_curve_export(tmp_path=None):
    import tempfile
    cfg = _cfg()
    report = Backtester(cfg).run(generate_demo_series(n_trades=10))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "equity.csv")
        export_equity_csv(report, path)
        with open(path) as f:
            lines = f.read().strip().splitlines()
    # header + start row + one row per trade
    assert lines[0].startswith("trade_index,")
    assert lines[1].split(",")[2] == "start"
    assert len(lines) == 1 + 1 + len(report.trades)


def test_backtest_demo_runs_and_reports():
    cfg = _cfg()
    report = Backtester(cfg).run(generate_demo_series(n_trades=30))
    assert len(report.trades) > 0
    assert report.max_drawdown_pct() >= 0.0
    # win_rate is a valid percentage
    assert 0.0 <= report.win_rate <= 100.0


# -- alerts -----------------------------------------------------------------

def test_notifier_disabled_without_config():
    assert Notifier(_cfg()).enabled is False


def test_notifier_enabled_with_telegram():
    cfg = _cfg(telegram_bot_token="t", telegram_chat_id="c")
    assert Notifier(cfg).enabled is True


def test_notifier_enabled_with_discord():
    cfg = _cfg(discord_webhook_url="https://discord/webhook")
    assert Notifier(cfg).enabled is True


def test_notifier_send_is_noop_when_disabled():
    # should not raise even though no channel is configured
    Notifier(_cfg()).send("hello")


# -- reconciliation ---------------------------------------------------------

def test_reconcile_clean_when_matching():
    res = compare_holdings({"a": 100.0, "b": 5.0}, {"a": 100.0, "b": 5.0})
    assert res.clean


def test_reconcile_tolerates_small_drift():
    res = compare_holdings({"a": 100.0}, {"a": 101.0})  # 1% < 2% tol
    assert res.clean


def test_reconcile_flags_phantom_position():
    res = compare_holdings({"a": 100.0}, {})  # bot thinks it holds, chain empty
    assert "a" in res.phantom
    assert not res.clean


def test_reconcile_flags_untracked_holding():
    res = compare_holdings({}, {"b": 50.0})  # chain holds, bot unaware
    assert "b" in res.untracked
    assert not res.clean


def test_reconcile_flags_amount_drift():
    res = compare_holdings({"a": 100.0}, {"a": 130.0})  # 30% off
    assert any(m == "a" for m, _, _ in res.drifted)
    assert not res.clean


# -- walk-forward -----------------------------------------------------------

def test_split_series_partitions_by_fraction():
    series = [(f"s{i}", [1.0]) for i in range(10)]
    a, b = split_series(series, 0.7)
    assert len(a) == 7 and len(b) == 3


def test_split_series_never_empty_for_two_plus():
    series = [("a", [1.0]), ("b", [1.0])]
    a, b = split_series(series, 0.99)
    assert len(a) >= 1 and len(b) >= 1


def test_walk_forward_returns_two_reports():
    cfg = _cfg()
    bt = Backtester(cfg)
    in_r, out_r = run_walk_forward(bt, generate_demo_series(n_trades=20), 0.5)
    assert len(in_r.trades) > 0
    assert len(out_r.trades) > 0
