"""Paper entries book the simulated swap's raw outAmount, not size/screener price."""
from __future__ import annotations

import types

import pytest

from bot.jupiter import SOL_MINT, SwapExecutor, SwapResult, TransactionStatus
from bot.safety import TokenSafety
from tests.test_p0_sizing_decimals import TOKEN9, FakeJup, make_bot


def _bot(jup, screen_price, decimals, symbol="T"):
    bot = make_bot(jup=jup)
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=TOKEN9)])
    bot._candidate_market_snapshot = lambda _mint: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, price_usd=screen_price, symbol=symbol, decimals=decimals))
    return bot


def test_paper_tokens_equal_quote_out_not_screener_price():
    # Quote prices the token at 0.0021 while the screener says 0.002 (~4.8% gap, like MAYA).
    jup = FakeJup(prices={TOKEN9: "0.0021"}, decimals={TOKEN9: 9})
    bot = _bot(jup, 0.002, 9)
    bot.executor = SwapExecutor(bot.cfg, jup=jup)
    bot._seek_entry()
    pos = bot.portfolio.positions[TOKEN9]
    out_raw = int(jup.quote(SOL_MINT, TOKEN9, 13_333_333)["outAmount"])
    assert pos.tokens == out_raw / 10 ** 9
    assert pos.original_tokens == pos.tokens
    assert pos.size_usd == 2.0
    assert pos.entry_price == pytest.approx(2.0 / pos.tokens)
    assert pos.tokens < 2.0 / 0.002  # old formula over-booked


def test_paper_round_trip_pnl_uses_quote_quantity():
    jup = FakeJup(prices={TOKEN9: "0.0021"}, decimals={TOKEN9: 9})
    bot = _bot(jup, 0.002, 9)
    bot.executor = SwapExecutor(bot.cfg, jup=jup)
    bot._seek_entry()
    pos = bot.portfolio.positions[TOKEN9]
    tokens = pos.tokens
    pnl = bot.portfolio.sell_fraction(TOKEN9, 1.0, 0.0021, "manual")
    assert pnl == pytest.approx(tokens * 0.0021 - 2.0, abs=1e-6)
    assert abs(pnl) < 1e-3  # flat at the quote price, not a phantom +5%


@pytest.mark.parametrize("out", [0, None, -1, 2 ** 64, 1.5, True])
def test_unbookable_paper_output_opens_nothing(out):
    jup = FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    bot = _bot(jup, 0.002, 9)
    bot.executor = types.SimpleNamespace(swap=lambda *a, **k: SwapResult(
        ok=True, simulated=True, in_amount=13_333_333, out_amount=out,
        tx_signature="DRYRUN", status=TransactionStatus.CONFIRMED_SUCCESS))
    bot._seek_entry()
    assert TOKEN9 not in bot.portfolio.positions


def test_unverified_decimals_open_nothing():
    jup = FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    bot = _bot(jup, 0.002, None)
    bot.executor = SwapExecutor(bot.cfg, jup=jup)
    bot._seek_entry()
    assert TOKEN9 not in bot.portfolio.positions
