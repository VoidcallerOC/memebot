"""P0: SOL/USD sizing, decimal-aware marks, and raw-amount exposure caps.

All quote/balance/mint clients are deterministic fakes; there is no network,
wallet, RPC, signing, or swap. ``_usd_to_lamports`` and ``_price_usd`` are
exercised for real (never stubbed). New helpers are looked up at call time
so this file still imports (and fails meaningfully) against the pre-fix code.
"""
from __future__ import annotations

import math
import types
from fractions import Fraction

import pytest

import bot.jupiter as jupiter_module
import bot.main as main_module
from bot.config import MAX_EXPERIMENT_USD, Config
from bot.jupiter import SOL_MINT, USDC_MINT, SwapExecutor, SwapResult, TransactionStatus
from bot.portfolio import Portfolio
from bot.preflight import run_preflight
from bot.risk import RiskManager
from bot.safety import SafetyScreener, TokenSafety

SENTINEL = "test-only-sentinel"
TOKEN6 = "Token6Mint1111111111111111111111111111111111"
TOKEN9 = "Token9Mint1111111111111111111111111111111111"
SOL_USD = 150


_UNSET = object()


class FakeJup:
    """Jupiter-shaped quotes priced from fixed whole-token USD values.

    ``amount``/``inAmount``/``outAmount`` are raw smallest units, as on Jupiter.
    """

    quote_url = "https://jupiter.test/quote"
    swap_url = "https://jupiter.test/swap"

    def __init__(self, prices=None, decimals=None, sol_override=_UNSET):
        self.prices = {SOL_MINT: Fraction(SOL_USD), USDC_MINT: Fraction(1)}
        self.decimals = {SOL_MINT: 9, USDC_MINT: 6}
        self.prices.update({k: Fraction(v) for k, v in (prices or {}).items()})
        self.decimals.update(decimals or {})
        self.sol_override = sol_override
        self.calls = []

    def quote(self, input_mint, output_mint, amount):
        self.calls.append((input_mint, output_mint, amount))
        if input_mint == SOL_MINT and output_mint == USDC_MINT and self.sol_override is not _UNSET:
            value = self.sol_override
            if isinstance(value, Exception):
                raise value
            return value
        usd = Fraction(amount, 10 ** self.decimals[input_mint]) * self.prices[input_mint]
        out_raw = math.floor(usd / self.prices[output_mint] * 10 ** self.decimals[output_mint])
        return {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "inAmount": str(amount),
            "outAmount": str(out_raw),
            "otherAmountThreshold": str(out_raw),  # zero-slippage fake
            "swapMode": "ExactIn",
        }


def sol_quote(**overrides):
    quote = {
        "inputMint": SOL_MINT,
        "outputMint": USDC_MINT,
        "inAmount": str(10 ** 9),
        "outAmount": str(SOL_USD * 10 ** 6),
    }
    for key, value in overrides.items():
        if value is _DROP:
            quote.pop(key)
        else:
            quote[key] = value
    return quote


_DROP = object()

BAD_SOL_QUOTES = [
    pytest.param(None, id="missing-quote"),
    pytest.param({}, id="empty"),
    pytest.param("150000000", id="not-a-dict"),
    pytest.param(sol_quote(outAmount=_DROP), id="no-outAmount"),
    pytest.param(sol_quote(inAmount=_DROP), id="no-inAmount"),
    pytest.param(sol_quote(inputMint=_DROP), id="no-inputMint"),
    pytest.param(sol_quote(outAmount="0"), id="zero"),
    pytest.param(sol_quote(outAmount=0), id="int-zero"),
    pytest.param(sol_quote(outAmount="-150000000"), id="negative"),
    pytest.param(sol_quote(outAmount="nan"), id="nan-str"),
    pytest.param(sol_quote(outAmount=float("nan")), id="nan"),
    pytest.param(sol_quote(outAmount="inf"), id="inf-str"),
    pytest.param(sol_quote(outAmount=float("inf")), id="inf"),
    pytest.param(sol_quote(outAmount=150_000_000.0), id="float"),
    pytest.param(sol_quote(outAmount="1.5e8"), id="exponent"),
    pytest.param(sol_quote(outAmount=True), id="bool"),
    pytest.param(sol_quote(outAmount=str(2 ** 64)), id="over-u64"),
    pytest.param(sol_quote(inAmount=str(10 ** 6)), id="inconsistent-0.001-SOL-units"),
    pytest.param(sol_quote(inAmount="1000000000.0"), id="inAmount-not-int"),
    pytest.param(sol_quote(inputMint=TOKEN6), id="wrong-input-mint"),
    pytest.param(sol_quote(outputMint=SOL_MINT), id="wrong-output-mint"),
    pytest.param(RuntimeError("quote transport down"), id="client-raises"),
]


def paper_cfg(**overrides):
    values = dict(live_trading=False, wallet_private_key="", bankroll_usd=100.0,
                  max_position_pct=2.0, stop_loss_pct=15.0)
    values.update(overrides)
    return Config(**values)


def armed_cfg(**overrides):
    values = dict(live_trading=True, wallet_private_key=SENTINEL,
                  burner_wallet_pubkey="PUBKEY", bankroll_usd=20.0,
                  process_lock_file=".test-lock", state_file=".test-state")
    values.update(overrides)
    return Config(**values)


def make_bot(cfg=None, jup=None):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg or paper_cfg()
    bot.jup = jup or FakeJup()
    bot.portfolio = Portfolio()
    bot.risk = RiskManager(bot.cfg)
    bot.notifier = types.SimpleNamespace(
        sell=lambda *a, **k: None, halt=lambda *a, **k: None, buy=lambda *a, **k: None,
    )
    bot._save = lambda: None
    bot._reconciliation_required = False
    return bot


class SignerSpy:
    """Records any attempt to sign/send; the guard must stop before these."""

    def __init__(self, executor, monkeypatch):
        self.events = []
        executor._load_keypair = lambda *_a: self.events.append("load_keypair") or (_ for _ in ()).throw(
            AssertionError("signer must not be loaded"))
        monkeypatch.setattr(jupiter_module, "send_raw_transaction",
                            lambda *a, **k: self.events.append("send") or "sig")
        original = executor._execute_live

        def _execute_live(*args, **kwargs):
            self.events.append("execute_live")
            return original(*args, **kwargs)

        executor._execute_live = _execute_live


def lamport_usd(lamports, sol_usd=SOL_USD):
    return Fraction(lamports, 10 ** 9) * sol_usd


# 1 ---------------------------------------------------------------- SOL price

def test_sol_price_quotes_exactly_one_sol_and_scales_usdc():
    jup = FakeJup()
    sol_price_usd = getattr(jupiter_module, "sol_price_usd", None)
    assert sol_price_usd is not None, "bot.jupiter.sol_price_usd missing"
    assert sol_price_usd(jup) == 150.0
    assert jup.calls == [(SOL_MINT, USDC_MINT, 10 ** 9)]
    assert jupiter_module.sol_usdc_raw_per_sol(FakeJup()) == 150 * 10 ** 6


def test_sizing_requests_a_one_sol_quote_not_a_milli_sol_quote():
    jup = FakeJup()
    make_bot(jup=jup)._usd_to_lamports(2.0)
    assert jup.calls == [(SOL_MINT, USDC_MINT, 10 ** 9)]


# 2 ---------------------------------------------------------------- sizing

def test_two_dollars_at_150_is_13_333_333_lamports():
    lamports = make_bot()._usd_to_lamports(2.0)
    assert lamports == 13_333_333  # floor(2 / 150 * 1e9)
    assert lamports == math.floor(Fraction(2) / 150 * 10 ** 9)
    assert lamports < 10 ** 8  # << 1 SOL (pre-fix: ~13.33 SOL)
    assert lamport_usd(lamports) <= 2


@pytest.mark.parametrize("usd", [0.01, 0.1, 1.0, 2.0, 3.3, 7.77, 19.99, 20.0])
@pytest.mark.parametrize("sol_usd", [1, 37, 149.99, 150, 233.17, 1000])
def test_floor_never_exceeds_allocation(usd, sol_usd):
    jup = FakeJup(prices={SOL_MINT: Fraction(str(sol_usd))})
    lamports = make_bot(jup=jup)._usd_to_lamports(usd)
    assert isinstance(lamports, int) and lamports > 0
    assert lamport_usd(lamports, Fraction(str(sol_usd))) <= Fraction(usd)
    assert lamport_usd(lamports + 1, Fraction(str(sol_usd))) > Fraction(usd)


@pytest.mark.parametrize("usd", [None, 0, 0.0, -2.0, float("nan"), float("inf"),
                                 float("-inf"), True, "2.0", 1e-12])
def test_invalid_usd_input_is_rejected(usd):
    assert make_bot()._usd_to_lamports(usd) is None


# 3/4 ------------------------------------------------- swap raw-amount cap

def _armed_executor(monkeypatch, jup=None):
    jup = jup or FakeJup(prices={TOKEN6: "0.002"}, decimals={TOKEN6: 6})
    executor = SwapExecutor(armed_cfg(), jup=jup, session=object())
    return executor, SignerSpy(executor, monkeypatch)


def test_13_sol_with_two_dollar_allocation_is_rejected_before_signing(monkeypatch):
    jup = FakeJup(prices={TOKEN6: "0.002"}, decimals={TOKEN6: 6})
    executor, spy = _armed_executor(monkeypatch, jup)
    result = executor.swap(SOL_MINT, TOKEN6, 13_333_333_333, allocation_usd=2.0)
    assert not result.ok
    assert result.simulated is False
    assert spy.events == []
    # Only the independent SOL price quote ran; no route quote for the swap.
    assert jup.calls == [(SOL_MINT, USDC_MINT, 10 ** 9)]


@pytest.mark.parametrize(("lamports", "allocation"), [
    (166_666_667, 20.0),       # $25.00 > $20 ceiling, allocation itself legal
    (140_000_000, 20.0),       # $21.00 > $20 ceiling
    (14_666_667, 2.0),         # $2.2000 > $2.10 (5% tolerance)
    (10 ** 9, 19.99),          # 1 SOL = $150
])
def test_allocation_within_limits_but_raw_amount_over_cap_is_rejected(monkeypatch, lamports, allocation):
    executor, spy = _armed_executor(monkeypatch)
    result = executor.swap(SOL_MINT, TOKEN6, lamports, allocation_usd=allocation)
    assert not result.ok
    assert spy.events == []


@pytest.mark.parametrize("bad", BAD_SOL_QUOTES)
def test_armed_swap_fails_closed_without_valid_sol_price(monkeypatch, bad):
    executor, spy = _armed_executor(monkeypatch, FakeJup(sol_override=bad))
    result = executor.swap(SOL_MINT, TOKEN6, 13_333_333, allocation_usd=2.0)
    assert not result.ok
    assert spy.events == []


@pytest.mark.parametrize("amount", [0, -1, 13_333_333.0, True, None])
def test_armed_swap_rejects_non_integer_or_non_positive_amount(monkeypatch, amount):
    executor, spy = _armed_executor(monkeypatch)
    result = executor.swap(SOL_MINT, TOKEN6, amount, allocation_usd=2.0)
    assert not result.ok
    assert spy.events == []


@pytest.mark.parametrize(("lamports", "allocation"), [
    (13_333_333, 2.0),         # sized exactly by _usd_to_lamports
    (13_999_999, 2.0),         # $2.0999 <= $2.10 tolerance
    (133_333_333, 20.0),       # $19.99999995 <= $20 ceiling
])
def test_valid_armed_swap_within_cap_reaches_execution(monkeypatch, lamports, allocation):
    executor = SwapExecutor(armed_cfg(), jup=FakeJup(prices={TOKEN6: "0.002"}, decimals={TOKEN6: 6}),
                            session=object())
    seen = []
    executor._execute_live = lambda quote, amount, out, **kw: seen.append(amount) or SwapResult(
        ok=True, simulated=False, in_amount=amount, out_amount=out,
        status=TransactionStatus.CONFIRMED_SUCCESS)
    assert executor.swap(SOL_MINT, TOKEN6, lamports, allocation_usd=allocation).ok
    assert seen == [lamports]


def test_paper_swap_is_not_blocked_by_live_guard():
    executor = SwapExecutor(paper_cfg(), jup=FakeJup(prices={TOKEN6: "0.002"}, decimals={TOKEN6: 6}))
    result = executor.swap(SOL_MINT, TOKEN6, 13_333_333, allocation_usd=2.0)
    assert result.ok and result.simulated


def test_ceiling_constant_is_unchanged():
    assert MAX_EXPERIMENT_USD == 20.0


# 5 ---------------------------------------------- malformed SOL quote data

@pytest.mark.parametrize("bad", BAD_SOL_QUOTES)
def test_bad_sol_quote_fails_closed_for_sizing(bad):
    jup = FakeJup(sol_override=bad)
    assert make_bot(jup=jup)._usd_to_lamports(2.0) is None
    helper = getattr(jupiter_module, "sol_price_usd", None)
    assert helper is not None and helper(FakeJup(sol_override=bad)) is None


def test_entry_without_sol_price_never_swaps():
    bot = make_bot(jup=FakeJup(sol_override=None))
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=TOKEN6)])
    bot._candidate_market_snapshot = lambda _mint: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, price_usd=0.002, symbol="T6", decimals=6))
    bot.executor = types.SimpleNamespace(
        swap=lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not swap")))
    bot._seek_entry()
    assert bot.portfolio.positions == {}


# 6 ---------------------------------------------------- decimal-aware marks

@pytest.mark.parametrize(("mint", "decimals"), [(TOKEN6, 6), (TOKEN9, 9)])
def test_whole_token_mark_is_independent_of_decimals(mint, decimals):
    jup = FakeJup(prices={mint: "0.002"}, decimals={mint: decimals})
    assert main_module._price_usd(jup, mint, decimals) == pytest.approx(0.002)
    assert jup.calls == [(mint, USDC_MINT, 10 ** decimals)]


def test_sol_mark_with_nine_decimals_is_per_sol():
    assert main_module._price_usd(FakeJup(), SOL_MINT, 9) == 150.0


@pytest.mark.parametrize(("mint", "decimals"), [(TOKEN6, 6), (TOKEN9, 9)])
def test_flat_paper_position_is_not_fake_stopped_out(mint, decimals):
    """Pre-fix a 9-dec mint marked 1000x low and fired a fake -99.9% stop-loss."""
    bot = make_bot(jup=FakeJup(prices={mint: "0.002"}, decimals={mint: decimals}))
    sells = []
    bot.notifier.sell = lambda *args: sells.append(args)
    bot.portfolio.open(mint, "T", 0.002, 2.0, 1000.0, token_decimals=decimals)
    bot._manage_positions()
    assert sells == []
    assert bot.portfolio.positions[mint].tokens == 1000.0
    assert bot.portfolio.realized_pnl == 0.0


def test_take_profit_on_nine_decimal_mint_uses_whole_token_price():
    bot = make_bot(jup=FakeJup(prices={TOKEN9: "0.0032"}, decimals={TOKEN9: 9}))
    sells = []
    bot.notifier.sell = lambda *args: sells.append(args)
    bot.portfolio.open(TOKEN9, "T9", 0.002, 2.0, 1000.0, token_decimals=9)
    bot._manage_positions()
    assert sells and sells[0][2] == pytest.approx(0.0032)
    assert sells[0][3].startswith("take_profit:")


def test_mark_uses_screener_verified_decimals_when_position_has_none():
    bot = make_bot(jup=FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9}))
    fetched = []
    bot.screener = types.SimpleNamespace(
        _fetch_mint_account=lambda mint: fetched.append(mint) or {
            "mint_authority": None, "freeze_authority": None, "decimals": 9})
    bot.portfolio.open(TOKEN9, "T9", 0.002, 2.0, 1000.0)
    bot._manage_positions()
    bot._manage_positions()
    assert TOKEN9 in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == 0.0
    assert fetched == [TOKEN9]  # cached after one verified fetch
    assert (TOKEN9, USDC_MINT, 10 ** 9) in bot.jup.calls


def test_paper_entry_records_verified_decimals_and_correct_size():
    jup = FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    bot = make_bot(jup=jup)
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=TOKEN9)])
    bot._candidate_market_snapshot = lambda _mint: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, price_usd=0.002, symbol="T9", decimals=9))
    bot.executor = SwapExecutor(bot.cfg, jup=jup)
    bot._seek_entry()
    pos = bot.portfolio.positions[TOKEN9]
    assert pos.token_decimals == 9
    assert (SOL_MINT, TOKEN9, 13_333_333) in jup.calls  # $2 of SOL, not ~13 SOL


def test_screener_reads_decimals_from_verified_mint_account():
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"value": {"data": {"parsed": {"info": {
                "mintAuthority": None, "freezeAuthority": None, "decimals": 9}}}}}}

    screener = SafetyScreener(paper_cfg(), session=types.SimpleNamespace(post=lambda *a, **k: Resp()))
    assert screener._fetch_mint_account(TOKEN9)["decimals"] == 9
    result = TokenSafety(mint=TOKEN9, passed=True)
    screener._check_mint_authorities(result, TOKEN9)
    assert result.decimals == 9 and result.passed


# 7 ------------------------------------------------------ unknown decimals

@pytest.mark.parametrize("decimals", [None, True, -1, 19, 9.0, "9"])
def test_unknown_or_invalid_decimals_give_no_mark(decimals):
    jup = FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    assert main_module._price_usd(jup, TOKEN9, decimals) is None
    assert jup.calls == []


def test_position_with_unknown_decimals_is_held_without_a_mark():
    jup = FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    bot = make_bot(jup=jup)
    bot.screener = types.SimpleNamespace(_fetch_mint_account=lambda mint: {
        "mint_authority": None, "freeze_authority": None, "decimals": None})
    bot.notifier.sell = lambda *a: (_ for _ in ()).throw(AssertionError("no sell"))
    bot.portfolio.open(TOKEN9, "T9", 0.002, 2.0, 1000.0)
    bot._manage_positions()
    assert bot.portfolio.positions[TOKEN9].tokens == 1000.0
    assert jup.calls == []


def test_inconsistent_mark_quote_units_give_no_mark():
    class Lying(FakeJup):
        def quote(self, i, o, amount):
            q = super().quote(i, o, amount)
            q["inAmount"] = str(10 ** 6)
            return q

    jup = Lying(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9})
    assert main_module._price_usd(jup, TOKEN9, 9) is None


def test_live_entry_without_verified_decimals_records_nothing():
    # TOKEN9 is priced so the pre-buy sell-route gate passes and the
    # missing-decimals fill path is what this test exercises.
    bot = make_bot(cfg=armed_cfg(), jup=FakeJup(prices={TOKEN9: "0.002"}, decimals={TOKEN9: 9}))
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=TOKEN9)])
    bot._candidate_market_snapshot = lambda _mint: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, price_usd=0.002, symbol="T9"))
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 2.0)
    bot.executor = types.SimpleNamespace(swap=lambda *a, **k: SwapResult(
        ok=True, simulated=False, in_amount=1, out_amount=1, tx_signature="sig",
        status=TransactionStatus.CONFIRMED_SUCCESS, actual_in_amount=13_333_333,
        actual_out_amount=1_000_000_000_000, out_decimals=None))
    bot._seek_entry()
    assert bot.portfolio.positions == {}
    assert bot._reconciliation_required is True


# 8 --------------------------------------------------------------- preflight

class PreflightJupiter(FakeJup):
    pass


class NoSwapExecutor:
    def __init__(self, *a, **k):
        pass

    def swap(self, *a, **k):
        raise AssertionError("preflight must never swap")

    def _confirm_transaction(self, *a, **k):
        raise AssertionError("preflight must never confirm")


def preflight(tmp_path, balance, jup=None, **overrides):
    cfg = Config(live_trading=True, wallet_private_key=SENTINEL, burner_wallet_pubkey="PUBKEY",
                 bankroll_usd=20.0, max_position_pct=10.0, stop_loss_pct=15.0,
                 require_liquidity_locked=True, require_mint_revoked=True,
                 require_freeze_revoked=True, state_file=str(tmp_path / "state.json"))
    kwargs = dict(
        derive_pubkey=lambda _cfg: "PUBKEY",
        read_sol_balance=lambda *_a: balance,
        jupiter_factory=lambda _cfg, **k: jup or PreflightJupiter(),
        swap_executor_factory=NoSwapExecutor,
        safety_factory=lambda _cfg, **k: types.SimpleNamespace(screen=lambda *_a: None),
        conflict_checker=lambda: False,
        journal_path=str(tmp_path / "journal.log"),
    )
    kwargs.update(overrides)
    return run_preflight(cfg, **kwargs)


def test_wallet_within_ceiling_is_eligible(tmp_path):
    result = preflight(tmp_path, 0.13)  # $19.50
    assert result.eligible, result.reason_codes


def test_over_ceiling_wallet_balance_fails_preflight(tmp_path):
    result = preflight(tmp_path, 1.0)  # $150 at $150/SOL
    assert not result.eligible
    assert "WALLET_BALANCE_EXCEEDS_CEILING" in result.reason_codes


def test_fee_buffer_boundary(tmp_path):
    buffer = getattr(__import__("bot.preflight", fromlist=["x"]), "PREFLIGHT_FEE_BUFFER_USD", None)
    assert buffer == 1.0
    assert preflight(tmp_path, 0.1399).eligible          # $20.985: within $20 + $1 buffer
    assert not preflight(tmp_path, 0.1401).eligible      # $21.015: over


@pytest.mark.parametrize("balance", [None, float("nan"), float("inf"), -0.1, True])
def test_missing_or_unverifiable_balance_fails_preflight(tmp_path, balance):
    assert not preflight(tmp_path, balance).eligible


@pytest.mark.parametrize("bad", BAD_SOL_QUOTES)
def test_unverifiable_sol_price_fails_preflight(tmp_path, bad):
    result = preflight(tmp_path, 0.01, jup=PreflightJupiter(sol_override=bad))
    assert not result.eligible
    assert "SOL_PRICE_UNVERIFIED" in result.reason_codes
