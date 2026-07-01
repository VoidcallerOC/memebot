"""Tests for the smart-money candidate source.

The RPC fetch is stubbed so the accumulation-diff logic is exercised without a
network. These guard the core promise: only *newly* accumulated, non-quote
mints are surfaced, and an RPC failure never fabricates a signal.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import Config, _parse_wallets, load_config
from bot.jupiter import SOL_MINT, USDC_MINT
from bot.strategy import SmartMoneyStrategy, Strategy, build_strategy


def _strat(wallets, **kw) -> SmartMoneyStrategy:
    return SmartMoneyStrategy("http://rpc.test", wallets, **kw)


def _mints(strat) -> set:
    return {c.mint for c in strat.find_candidates()}


def test_first_sight_is_baseline_then_new_holdings_surface():
    strat = _strat(["W1"])
    strat._wallet_mints = lambda w: {"AAA", "BBB"}
    # first tick: establish baseline, surface nothing
    assert _mints(strat) == set()
    # wallet buys a new token -> only that one surfaces
    strat._wallet_mints = lambda w: {"AAA", "BBB", "CCC"}
    assert _mints(strat) == {"CCC"}


def test_holding_is_surfaced_once_not_every_tick():
    strat = _strat(["W1"])
    strat._wallet_mints = lambda w: {"AAA"}
    _mints(strat)  # baseline
    strat._wallet_mints = lambda w: {"AAA", "CCC"}
    assert _mints(strat) == {"CCC"}
    # still holding CCC next tick -> must not re-surface
    assert _mints(strat) == set()


def test_quote_and_stable_mints_are_never_candidates():
    strat = _strat(["W1"], surface_existing=True)
    strat._wallet_mints = lambda w: {SOL_MINT, USDC_MINT, "MEME"}
    assert _mints(strat) == {"MEME"}


def test_surface_existing_emits_baseline_bag():
    strat = _strat(["W1"], surface_existing=True)
    strat._wallet_mints = lambda w: {"AAA", "BBB"}
    assert _mints(strat) == {"AAA", "BBB"}


def test_rpc_failure_does_not_fabricate_or_poison_baseline():
    strat = _strat(["W1"])
    strat._wallet_mints = lambda w: {"AAA"}
    _mints(strat)  # baseline {AAA}
    # RPC down this tick -> no candidates, baseline untouched
    strat._wallet_mints = lambda w: None
    assert _mints(strat) == set()
    # recovery shows a genuinely new token -> only the new one, not AAA again
    strat._wallet_mints = lambda w: {"AAA", "CCC"}
    assert _mints(strat) == {"CCC"}


def test_dedupes_new_mint_across_multiple_wallets():
    strat = _strat(["W1", "W2"])
    strat._wallet_mints = lambda w: {"SHARED"} if w == "W1" else set()
    _mints(strat)  # baselines
    strat._wallet_mints = lambda w: {"SHARED", "NEW"}
    assert _mints(strat) == {"NEW", "SHARED"}  # SHARED new for W2, NEW for both


def test_empty_watchlist_yields_nothing():
    assert _strat([]).find_candidates() == []


# -- config parsing + factory ----------------------------------------------

def test_parse_wallets_merges_env_and_dedupes(tmp_path):
    f = tmp_path / "wallets.txt"
    f.write_text("# a comment\nWB\nWC\n")
    out = _parse_wallets("WA, WB  WA", str(f))
    assert out == ["WA", "WB", "WC"]  # order preserved, deduped, comment skipped


def test_parse_wallets_missing_file_is_tolerated():
    assert _parse_wallets("WA", "/nope/nada.txt") == ["WA"]


def test_build_strategy_selects_smart_money(monkeypatch):
    monkeypatch.setenv("STRATEGY", "smart_money")
    monkeypatch.setenv("SMART_MONEY_WALLETS", "WA WB")
    strat = build_strategy(load_config())
    assert isinstance(strat, SmartMoneyStrategy)
    assert strat.wallets == ["WA", "WB"]


def test_build_strategy_defaults_to_boosted():
    assert isinstance(build_strategy(Config()), Strategy)


def test_build_strategy_unknown_falls_back_to_boosted():
    assert isinstance(build_strategy(Config(strategy="banana")), Strategy)
