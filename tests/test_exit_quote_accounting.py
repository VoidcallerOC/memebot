"""Missing exit quotes must not book a zero-price sale.

A failed fetch is ``None`` from ``_price_usd``. A successful quote can still
return a numeric 0, and that mark stays on the existing exit path.
"""
from __future__ import annotations

import json
import types
from datetime import date
from pathlib import Path

import pytest

import bot.main as main_module
import bot.risk as risk_module
from bot.config import MAX_EXPERIMENT_USD, Config
from bot.jupiter import USDC_MINT
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.state import load_state, save_state

MINT = "Mint111111111111111111111111111111111111111"
WORKSPACE_STATE = Path("/workspace/state.json")


def _cfg(**overrides) -> Config:
    values = dict(
        live_trading=False,
        wallet_private_key="",
        bankroll_usd=100.0,
        max_position_pct=2.0,
        daily_loss_limit_pct=10.0,
        stop_loss_pct=15.0,
        take_profit_ladder=[(50.0, 0.5), (100.0, 0.25), (300.0, 0.25)],
    )
    values.update(overrides)
    return Config(**values)


def _bot(cfg: Config | None = None):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg or _cfg()
    bot.portfolio = Portfolio()
    bot.risk = RiskManager(bot.cfg)
    bot.jup = object()
    bot.notifier = types.SimpleNamespace(
        sell=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("sell alert")),
        halt=lambda *args, **kwargs: None,
        buy=lambda *args, **kwargs: None,
    )
    bot._save = lambda: None
    bot._reconciliation_required = False
    bot.executor = types.SimpleNamespace(
        swap=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("swap")),
    )
    return bot


def _open(bot, *, entry_price=1.0, size_usd=100.0, tokens=100.0):
    bot.portfolio.open(MINT, "PEPE", entry_price, size_usd, tokens)
    return bot.portfolio.positions[MINT]


def _stub_price(monkeypatch, value):
    def _price(_jup, mint):
        assert mint == MINT
        return value() if callable(value) else value

    monkeypatch.setattr(main_module, "_price_usd", _price)


def test_price_usd_distinguishes_missing_quote_from_numeric_zero():
    missing = types.SimpleNamespace(quote=lambda *_a, **_k: None)
    assert main_module._price_usd(missing, MINT) is None

    failed = types.SimpleNamespace(quote=lambda *_a, **_k: {"error": "no route"})
    assert main_module._price_usd(failed, MINT) is None

    zero = types.SimpleNamespace(quote=lambda *_a, **_k: {"outAmount": "0"})
    assert main_module._price_usd(zero, MINT) == 0.0

    priced = types.SimpleNamespace(quote=lambda *_a, **_k: {"outAmount": "1500000"})
    assert main_module._price_usd(priced, MINT) == 1.5


def test_missing_exit_quote_does_not_sell_at_zero(monkeypatch):
    bot = _bot()
    pos = _open(bot)
    _stub_price(monkeypatch, None)

    bot._manage_positions()

    assert MINT in bot.portfolio.positions
    held = bot.portfolio.positions[MINT]
    assert held is pos
    assert held.tokens == 100.0
    assert held.size_usd == 100.0
    assert held.entry_price == 1.0
    assert held.realized_pnl == 0.0
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.risk.realized_pnl_today == 0.0
    assert bot.portfolio.pending_intents == []


def test_direct_none_price_does_not_book_no_route_exit():
    bot = _bot()
    _open(bot)

    bot._sell(MINT, 1.0, None, "no_route_exit")

    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 100.0
    assert pos.size_usd == 100.0
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.portfolio.pending_intents == []


def test_partial_exit_then_missing_quote_keeps_remainder(monkeypatch):
    bot = _bot()
    alerts = []
    bot.notifier.sell = lambda *args: alerts.append(args)
    _open(bot)
    quotes = iter([1.6, None])

    def _price(_jup, _mint):
        return next(quotes)

    monkeypatch.setattr(main_module, "_price_usd", _price)

    bot._manage_positions()
    after_partial = bot.portfolio.positions[MINT]
    assert after_partial.tokens == 50.0
    assert after_partial.size_usd == 50.0
    assert after_partial.realized_pnl == pytest.approx(30.0)
    assert bot.portfolio.realized_pnl == pytest.approx(30.0)
    booked = bot.portfolio.realized_pnl

    bot._manage_positions()

    held = bot.portfolio.positions[MINT]
    assert held.tokens == 50.0
    assert held.size_usd == 50.0
    assert held.entry_price == 1.0
    assert held.realized_pnl == pytest.approx(booked)
    assert bot.portfolio.realized_pnl == pytest.approx(booked)
    assert bot.risk.realized_pnl_today == pytest.approx(booked)
    assert len(alerts) == 1


def test_repeated_missing_quotes_do_not_double_book(monkeypatch):
    bot = _bot()
    _open(bot)
    _stub_price(monkeypatch, None)

    bot._manage_positions()
    bot._manage_positions()
    bot._manage_positions()

    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 100.0
    assert pos.size_usd == 100.0
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.risk.realized_pnl_today == 0.0
    assert bot.portfolio.pending_intents == []


def test_unresolved_position_survives_save_and_load(monkeypatch, tmp_path):
    bot = _bot()
    _open(bot)
    _stub_price(monkeypatch, None)
    bot._manage_positions()

    path = tmp_path / "state.json"
    assert path.resolve() != WORKSPACE_STATE.resolve()
    save_state(str(path), bot.portfolio, bot.risk)

    raw = json.loads(path.read_text())
    saved = raw["portfolio"]["positions"]
    assert len(saved) == 1
    assert saved[0]["mint"] == MINT
    assert saved[0]["size_usd"] == 100.0
    assert saved[0]["tokens"] == 100.0
    assert saved[0]["entry_price"] == 1.0
    assert raw["portfolio"]["realized_pnl"] == 0.0

    restored_pf, restored_risk = Portfolio(), RiskManager(_cfg())
    assert load_state(str(path), restored_pf, restored_risk) is True
    held = restored_pf.positions[MINT]
    assert held.tokens == 100.0
    assert held.size_usd == 100.0
    assert held.entry_price == 1.0
    assert restored_pf.realized_pnl == 0.0


def test_numeric_zero_quote_still_follows_exit_rules(monkeypatch):
    """A successful quote of 0 is a mark, distinct from a missing quote."""
    bot = _bot()
    alerts = []
    bot.notifier.sell = lambda *args: alerts.append(args)
    _open(bot)
    _stub_price(monkeypatch, 0.0)

    bot._manage_positions()

    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(-100.0)
    assert alerts and alerts[0][2] == 0.0
    assert alerts[0][3] == "stop_loss"


def test_armed_missing_quote_does_not_settle_or_book(monkeypatch):
    bot = _bot(_cfg(live_trading=True, wallet_private_key="test-only-sentinel"))
    assert bot.cfg.is_armed is True
    _open(bot, size_usd=5.0, tokens=5.0)
    _stub_price(monkeypatch, None)

    bot._manage_positions()

    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 5.0
    assert pos.size_usd == 5.0
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.portfolio.pending_intents == []
    assert bot._reconciliation_required is False
    assert MAX_EXPERIMENT_USD == 20.0


class _FrozenDate(date):
    current = date(2026, 10, 9)

    @classmethod
    def today(cls):
        return cls.current


def test_daily_halt_persists_across_load_and_resets_on_utc_date_roll(monkeypatch, tmp_path):
    monkeypatch.setattr(risk_module, "date", _FrozenDate)
    _FrozenDate.current = date(2026, 10, 9)

    cfg = _cfg()
    risk = RiskManager(cfg)
    risk.record_realized_pnl(-cfg.daily_loss_limit_usd())
    assert risk.trading_halted()
    assert risk.can_open_new_position(0) is False

    path = tmp_path / "halt-state.json"
    assert path.resolve() != WORKSPACE_STATE.resolve()
    save_state(str(path), Portfolio(), risk)

    same_day = RiskManager(cfg)
    assert load_state(str(path), Portfolio(), same_day) is True
    assert same_day.trading_halted()
    assert same_day.realized_pnl_today == pytest.approx(-10.0)
    assert same_day.can_open_new_position(0) is False

    _FrozenDate.current = date(2026, 10, 10)
    assert same_day.trading_halted() is False
    assert same_day.realized_pnl_today == 0.0
    assert same_day.can_open_new_position(0) is True

    on_disk = json.loads(path.read_text())
    assert on_disk["risk"]["day"] == "2026-10-09"
    assert on_disk["risk"]["halted_today"] is True
    assert on_disk["risk"]["realized_pnl_today"] == pytest.approx(-10.0)

    rolled_load = RiskManager(cfg)
    assert load_state(str(path), Portfolio(), rolled_load) is True
    assert rolled_load.trading_halted() is False
    assert rolled_load.realized_pnl_today == 0.0
    assert json.loads(path.read_text())["risk"]["halted_today"] is True


def test_quote_request_uses_usdc_not_a_zero_fill():
    """``_price_usd`` asks Jupiter for USDC out of 1 token unit; it does not invent 0."""
    seen = {}

    def quote(input_mint, output_mint, amount):
        seen["args"] = (input_mint, output_mint, amount)
        return None

    assert main_module._price_usd(types.SimpleNamespace(quote=quote), MINT) is None
    assert seen["args"] == (MINT, USDC_MINT, 1_000_000)
