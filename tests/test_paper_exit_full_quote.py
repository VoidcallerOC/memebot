"""Paper (DRY-RUN) exits request the same full-size direct token->SOL quote a
live sell would, record it next to the trigger-price proceeds, and book PnL at
the quoted outAmount (USD via SOL/USD). Any quote failure falls back to the
trigger price with a warning; the exit still happens. Live never takes this path.
"""
from __future__ import annotations

import logging
import types

import pytest

import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, USDC_MINT
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.state import load_state, save_state

MINT = "Mint111111111111111111111111111111111111111"
SLIP = 150
pytestmark = pytest.mark.real_swap_policy


def _cfg(**kw) -> Config:
    v = dict(live_trading=False, wallet_private_key="", bankroll_usd=100.0,
             max_position_pct=2.0, daily_loss_limit_pct=10.0, stop_loss_pct=15.0,
             max_slippage_bps=SLIP,
             take_profit_ladder=[(50.0, 0.5), (100.0, 0.25), (300.0, 0.25)])
    v.update(kw)
    return Config(**v)


def sell_quote(raw, out, min_out=None, inp=MINT, outp=SOL_MINT):
    out_s = str(out)
    return {
        "inputMint": inp, "outputMint": outp, "inAmount": str(raw), "outAmount": out_s,
        "otherAmountThreshold": str(min_out if min_out is not None else out * 9850 // 10000),
        "swapMode": "ExactIn", "slippageBps": SLIP,
        "routePlan": [{"percent": 100, "swapInfo": {
            "ammKey": "Amm1", "inputMint": inp, "outputMint": outp,
            "inAmount": str(raw), "outAmount": out_s}}],
    }


SOL_USD = {"inputMint": SOL_MINT, "outputMint": USDC_MINT,
           "inAmount": str(10**9), "outAmount": str(200 * 10**6)}  # $200/SOL


class FakeJup:
    def __init__(self, sell=lambda raw: None, sol=SOL_USD):
        self.calls, self._sell, self._sol = [], sell, sol

    def quote(self, i, o, amount, *, direct_only=False):
        self.calls.append((i, o, amount, direct_only))
        if i == MINT and o == SOL_MINT:
            r = self._sell(amount)
            if isinstance(r, Exception):
                raise r
            return r
        if i == SOL_MINT and o == USDC_MINT:
            return self._sol
        raise AssertionError((i, o))


def _bot(jup, cfg=None):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg or _cfg()
    bot.portfolio, bot.risk, bot.jup = Portfolio(), RiskManager(bot.cfg), jup
    bot.sells = []
    bot.notifier = types.SimpleNamespace(sell=lambda *a, **k: bot.sells.append(a),
                                         halt=lambda *a, **k: None)
    bot._save = lambda: True
    bot._reconciliation_required = False
    bot.executor = types.SimpleNamespace(
        swap=lambda *a, **k: (_ for _ in ()).throw(AssertionError("swap")))
    bot.portfolio.open(MINT, "PEPE", 1.0, 100.0, 100.0, token_decimals=6)
    return bot


def test_stop_loss_books_quoted_proceeds_and_records(caplog):
    # 100 tokens = 100_000_000 raw; quote 0.4 SOL = $80 vs trigger 0.85*100 = $85
    jup = FakeJup(sell=lambda raw: sell_quote(raw, 400_000_000, 394_000_000))
    bot = _bot(jup)
    with caplog.at_level(logging.INFO, logger="bot"):
        bot._sell(MINT, 1.0, 0.85, "stop_loss")
    assert jup.calls[0] == (MINT, SOL_MINT, 100_000_000, True)
    assert len([c for c in jup.calls if c[0] == MINT]) == 1
    assert MINT not in bot.portfolio.positions
    assert abs(bot.portfolio.realized_pnl - (-20.0)) < 1e-9  # quoted, not -15
    rec = bot.portfolio.paper_exit_quotes[-1]
    assert rec["booked"] == "quote" and rec["error"] is None
    assert rec["raw_amount"] == 100_000_000
    assert abs(rec["trigger_proceeds_usd"] - 85.0) < 1e-9
    assert rec["quoted_proceeds_sol"] == 0.4
    assert abs(rec["quoted_proceeds_usd"] - 80.0) < 1e-9
    assert abs(rec["min_out_proceeds_usd"] - 78.8) < 1e-9
    assert abs(rec["gap_pct"] - (-5 / 85 * 100)) < 1e-9
    assert "booked at quoted proceeds" in caplog.text


def _fallback(sell, sol=SOL_USD, caplog=None):
    jup = FakeJup(sell=sell, sol=sol)
    bot = _bot(jup)
    with caplog.at_level(logging.WARNING, logger="bot"):
        bot._sell(MINT, 1.0, 0.85, "stop_loss")
    assert MINT not in bot.portfolio.positions          # still closes
    assert abs(bot.portfolio.realized_pnl - (-15.0)) < 1e-9  # trigger price
    rec = bot.portfolio.paper_exit_quotes[-1]
    assert rec["booked"] == "trigger" and rec["error"]
    assert "exit quote unavailable, booked at trigger price" in caplog.text
    assert len([c for c in jup.calls if c[0] == MINT]) == 1  # no retries
    return rec


def test_fallback_none(caplog):
    _fallback(lambda raw: None, caplog=caplog)


def test_fallback_429_exception(caplog):
    _fallback(lambda raw: RuntimeError("429 Too Many Requests"), caplog=caplog)


def test_fallback_no_direct_route(caplog):
    def two_hop(raw):
        q = sell_quote(raw, 400_000_000)
        q["routePlan"] = q["routePlan"] * 2
        return q
    assert "route" in _fallback(two_hop, caplog=caplog)["error"]


def test_fallback_wrong_amount(caplog):
    _fallback(lambda raw: sell_quote(raw + 1, 400_000_000), caplog=caplog)


def test_fallback_no_sol_price(caplog):
    rec = _fallback(lambda raw: sell_quote(raw, 400_000_000), sol=None, caplog=caplog)
    assert rec["quoted_proceeds_sol"] == 0.4 and rec["quoted_proceeds_usd"] is None


def test_partial_tp_rung_uses_rung_raw_amount():
    jup = FakeJup(sell=lambda raw: sell_quote(raw, 300_000_000))  # $60 for 50 tok
    bot = _bot(jup)
    bot._sell(MINT, 0.5, 1.5, "take_profit_50%", rung=50.0)
    assert jup.calls[0] == (MINT, SOL_MINT, 50_000_000, True)
    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 50.0 and 50.0 in pos.ladder_filled
    assert abs(bot.portfolio.realized_pnl - 10.0) < 1e-9  # 60 - 50, not 75 - 50


def test_live_mode_no_extra_quote(monkeypatch):
    jup = FakeJup(sell=lambda raw: (_ for _ in ()).throw(AssertionError("quoted")))
    bot = _bot(jup)
    monkeypatch.setattr(type(bot.cfg), "is_armed", property(lambda self: True))
    called = []
    monkeypatch.setattr(bot, "_sell_frozen", lambda m: False)
    monkeypatch.setattr(bot, "_settle_live_sell", lambda *a, **k: called.append(a) or True)
    monkeypatch.setattr(bot, "_paper_exit_quote",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("paper")))
    bot._sell(MINT, 1.0, 0.85, "stop_loss")
    assert called and jup.calls == [] and bot.portfolio.paper_exit_quotes == []


def test_state_round_trip_with_and_without_field(tmp_path):
    p = str(tmp_path / "state.json")
    bot = _bot(FakeJup(sell=lambda raw: sell_quote(raw, 300_000_000)))
    bot._sell(MINT, 0.5, 1.5, "take_profit_50%", rung=50.0)
    assert save_state(p, bot.portfolio, bot.risk) is True
    pf, rk = Portfolio(), RiskManager(bot.cfg)
    assert load_state(p, pf, rk)
    assert pf.paper_exit_quotes == bot.portfolio.paper_exit_quotes
    legacy = bot.portfolio.to_dict()
    legacy.pop("paper_exit_quotes")
    pf2 = Portfolio()
    pf2.load_dict(legacy)
    assert pf2.paper_exit_quotes == [] and MINT in pf2.positions and not pf2.corrupt
    legacy["paper_exit_quotes"] = "junk"
    pf2.load_dict(legacy)
    assert pf2.paper_exit_quotes == [] and not pf2.corrupt
