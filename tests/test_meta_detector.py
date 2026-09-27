"""META DETECTOR component tests. No wallet, no live RPC, no execution."""
from __future__ import annotations

from bot.meta.detector import score_snapshot
from bot.meta.model import OK, UNVERIFIED, MarketWindow, SocialWindow, TokenSnapshot, WalletWindow


def snap(**overrides) -> TokenSnapshot:
    values = dict(
        mint="MintA",
        symbol="AGENT",
        name="AI Trading Agent",
        observed_at=1_000_000.0,
        liquidity_usd=80_000.0,
        market_cap_usd=400_000.0,
        texts=["autonomous ai agent on solana"],
    )
    values.update(overrides)
    return TokenSnapshot(**values)


def test_missing_provider_data_is_unverified():
    signal = score_snapshot(TokenSnapshot(mint="X", symbol="X"))
    assert signal.attention_velocity.status == UNVERIFIED
    assert signal.trading_velocity.status == UNVERIFIED
    assert signal.wallet_velocity.status == UNVERIFIED
    assert signal.attention_velocity.value is None


def test_liquidity_removal_flags():
    signal = score_snapshot(snap(liquidity_usd=8_000, liquidity_removed=True, lp_change_pct=-80))
    assert "LIQUIDITY_REMOVAL" in signal.reason_codes
    assert "LIQUIDITY_SHOCK" in signal.reason_codes


def test_volume_without_wallet_alignment():
    signal = score_snapshot(snap(
        market={
            "5m": MarketWindow(volume_usd=80_000, tx_buys=200, tx_sells=40, source="dexscreener"),
            "1h": MarketWindow(volume_usd=30_000, tx_buys=80, tx_sells=70, source="dexscreener"),
        },
        wallets={
            "5m": WalletWindow(new_holders=1, unique_buyers=2, active_wallets=2),
            "1h": WalletWindow(new_holders=12, unique_buyers=20, active_wallets=20),
        },
    ))
    assert signal.flow_alignment == "VOLUME_SPIKE_WITHOUT_WALLET_GROWTH"


def test_social_concentration():
    signal = score_snapshot(snap(social={
        "5m": SocialWindow(mentions=200, unique_contributors=3, engagement=20, top_account_share=0.85),
        "1h": SocialWindow(mentions=80, unique_contributors=20, engagement=40, top_account_share=0.2),
    }))
    assert "SOCIAL_CONCENTRATION" in signal.reason_codes


def test_stale_and_conflicting_flags():
    stale = score_snapshot(snap(stale=True, social={
        "5m": SocialWindow(mentions=10, unique_contributors=10),
        "1h": SocialWindow(mentions=10, unique_contributors=10),
    }))
    assert stale.provider_freshness == "STALE"
    conflicting = score_snapshot(snap(conflicting=True))
    assert conflicting.provider_freshness == "CONFLICTING"


def test_creator_concentration():
    signal = score_snapshot(snap(creator_pct=62.0, top_holder_pct=62.0))
    assert signal.creator_concentration.status == OK
    assert "HIGH_CREATOR_CONCENTRATION" in signal.reason_codes


# -- F1: dust-volume floor ----------------------------------------------------

from bot.meta.scoring import DUST_VOLUME, MIN_VOLUME_FLOOR_USD


def _volume_snap(vol_5m: float, vol_1h: float) -> TokenSnapshot:
    return snap(market={
        "5m": MarketWindow(volume_usd=vol_5m, source="dexscreener"),
        "1h": MarketWindow(volume_usd=vol_1h, source="dexscreener"),
    })


def test_dust_volume_is_not_extreme_acceleration():
    # Run C shape: 5m == 1h volume gives ratio 12 regardless of size.
    signal = score_snapshot(_volume_snap(0.05, 0.05))
    assert signal.trading_velocity.status == UNVERIFIED
    assert signal.trading_velocity.value is None
    assert signal.trading_velocity.details.get("state") != "EXTREME_ACCELERATION"
    assert "VOLUME_EXTREME" not in signal.reason_codes


def test_volume_below_floor_is_unverified_with_dust_code():
    signal = score_snapshot(_volume_snap(50.0, 100.0))
    assert signal.trading_velocity.status == UNVERIFIED
    assert DUST_VOLUME in signal.trading_velocity.reason_codes
    assert DUST_VOLUME in signal.reason_codes
    assert signal.trading_velocity.details["volume_floor"] == MIN_VOLUME_FLOOR_USD


def test_normal_volume_formula_unchanged():
    extreme = score_snapshot(_volume_snap(9_620.0, 9_620.0)).trading_velocity
    assert extreme.status == OK
    assert extreme.value == 100.0
    assert extreme.details["state"] == "EXTREME_ACCELERATION"
    assert extreme.details["acceleration"] == 12.0
    # 5m=1000 over 5 min vs 1h=6000 over 60 min -> ratio 2.0 -> 50 + 25*log2(2) = 75
    doubled = score_snapshot(_volume_snap(1_000.0, 6_000.0)).trading_velocity
    assert doubled.status == OK
    assert abs(doubled.value - 75.0) < 1e-9
    assert doubled.details["state"] == "ACCELERATING"
    assert DUST_VOLUME not in doubled.reason_codes


def test_volume_exactly_at_floor_is_scored():
    at_floor = score_snapshot(_volume_snap(MIN_VOLUME_FLOOR_USD, MIN_VOLUME_FLOOR_USD)).trading_velocity
    assert at_floor.status == OK
    assert DUST_VOLUME not in at_floor.reason_codes
    assert at_floor.details["long_value"] == MIN_VOLUME_FLOOR_USD


def test_volume_just_below_floor_is_dust():
    just_below = MIN_VOLUME_FLOOR_USD - 0.01
    score = score_snapshot(_volume_snap(just_below, just_below)).trading_velocity
    assert score.status == UNVERIFIED
    assert DUST_VOLUME in score.reason_codes
