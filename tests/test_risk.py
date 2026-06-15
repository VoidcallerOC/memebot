"""Tests for the protective machinery: risk rules, sizing, daily limit,
portfolio accounting, and the safety lock. These are the parts that must be
correct — a bug here is a bug that loses money."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import Config
from bot.portfolio import Portfolio
from bot.risk import RiskManager


def _cfg(**kw) -> Config:
    base = dict(
        bankroll_usd=1000.0,
        max_position_pct=2.0,
        max_open_positions=3,
        stop_loss_pct=15.0,
        daily_loss_limit_pct=10.0,
        take_profit_ladder=[(50.0, 0.5), (100.0, 0.25), (300.0, 0.25)],
    )
    base.update(kw)
    return Config(**base)


# -- sizing -----------------------------------------------------------------

def test_position_size_is_fixed_fraction():
    rm = RiskManager(_cfg())
    assert rm.position_size_usd() == 20.0  # 2% of 1000


def test_stop_loss_price():
    rm = RiskManager(_cfg(stop_loss_pct=15))
    assert rm.stop_loss_price(1.0) == 0.85


# -- exits ------------------------------------------------------------------

def test_stop_loss_dumps_entire_position():
    rm = RiskManager(_cfg())
    actions = rm.evaluate_exit(entry_price=1.0, current_price=0.80, ladder_filled=set())
    assert len(actions) == 1
    assert actions[0].fraction == 1.0
    assert actions[0].reason == "stop_loss"


def test_take_profit_fires_only_once_per_rung():
    rm = RiskManager(_cfg())
    filled: set[float] = set()
    # up 60% -> first rung (50%) should fire
    actions = rm.evaluate_exit(1.0, 1.6, filled)
    assert any(a.reason == "take_profit:50" for a in actions)
    filled.add(50.0)
    # still up 60% but rung already filled -> nothing
    actions2 = rm.evaluate_exit(1.0, 1.6, filled)
    assert actions2 == []


def test_take_profit_multiple_rungs_at_once():
    rm = RiskManager(_cfg())
    actions = rm.evaluate_exit(1.0, 2.5, set())  # +150% -> 50 and 100 rungs
    reasons = {a.reason for a in actions}
    assert "take_profit:50" in reasons
    assert "take_profit:100" in reasons
    assert "take_profit:300" not in reasons


def test_no_exit_in_normal_range():
    rm = RiskManager(_cfg())
    assert rm.evaluate_exit(1.0, 1.05, set()) == []


# -- daily circuit breaker --------------------------------------------------

def test_daily_loss_limit_halts_trading():
    rm = RiskManager(_cfg(bankroll_usd=1000, daily_loss_limit_pct=10))
    assert not rm.trading_halted()
    rm.record_realized_pnl(-50)
    assert not rm.trading_halted()      # -5%, still ok
    rm.record_realized_pnl(-60)         # cumulative -11% > 10%
    assert rm.trading_halted()
    assert not rm.can_open_new_position(0)


def test_max_open_positions_enforced():
    rm = RiskManager(_cfg(max_open_positions=2))
    assert rm.can_open_new_position(1)
    assert not rm.can_open_new_position(2)


# -- portfolio accounting ---------------------------------------------------

def test_partial_sell_pnl_and_remaining_tokens():
    pf = Portfolio()
    pf.open("mint", "PEPE", entry_price=1.0, size_usd=100.0, tokens=100.0)
    pnl = pf.sell_fraction("mint", 0.5, current_price=2.0, reason="take_profit:100")
    # sold 50 tokens, cost 50, proceeds 100 -> pnl 50
    assert pnl == 50.0
    assert pf.positions["mint"].tokens == 50.0
    assert pf.realized_pnl == 50.0


def test_stop_loss_closes_position_fully():
    pf = Portfolio()
    pf.open("mint", "DOGE", entry_price=1.0, size_usd=100.0, tokens=100.0)
    pf.sell_fraction("mint", 1.0, current_price=0.8, reason="stop_loss")
    assert "mint" not in pf.positions
    assert pf.realized_pnl == -20.0


# -- safety lock ------------------------------------------------------------

def test_safety_lock_blocks_live_without_key():
    cfg = _cfg(live_trading=True, wallet_private_key="")
    assert cfg.is_armed is False


def test_safety_lock_armed_with_key_and_flag():
    cfg = _cfg(live_trading=True, wallet_private_key="somekey")
    assert cfg.is_armed is True


def test_safety_lock_off_by_default():
    assert _cfg().is_armed is False
