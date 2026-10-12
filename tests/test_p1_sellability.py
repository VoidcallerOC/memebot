"""P1: fail-closed pre-buy sell-route (round-trip quote) gate.

Offline fakes only: no network, RPC, signing or swaps. ``_usd_to_lamports``
runs for real. ``_sell_route_ok`` is looked up at call time so this file still
imports (and fails meaningfully) against code without the gate.
"""
from __future__ import annotations

import math
import types
from fractions import Fraction

import pytest
import requests

import bot.jupiter as jupiter_module
import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, USDC_MINT, JupiterClient, SwapExecutor, SwapResult, TransactionStatus
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.safety import TokenSafety

SENTINEL = "test-only-sentinel"
MINT = "SellCheckMint111111111111111111111111111111"
OTHER = "OtherMint11111111111111111111111111111111111"
PLANNED = 13_333_333          # $2 at $150/SOL, floor
TOKENS_PER_LAMPORT = 1000     # raw token units per lamport on the buy leg


class RouteJup:
    """Jupiter-shaped ExactIn quotes with programmable buy/sell legs.

    ``otherAmountThreshold`` = ``outAmount`` reduced by ``slippage_bps``
    (default 0 here, so threshold == outAmount; slippage-specific cases live
    in tests/test_p1_roundtrip_min_out.py).
    """

    def __init__(self, *, sell_ratio=Fraction(98, 100), buy=None, sell=None, slippage_bps=0):
        self.sell_ratio = sell_ratio
        self.slippage_bps = slippage_bps
        self.buy = buy      # callable(default_quote, amount) -> response, or Exception
        self.sell = sell
        self.calls = []

    def _leg(self, override, default, amount):
        if override is None:
            return default
        if isinstance(override, Exception):
            raise override
        return override(default, amount)

    def quote(self, input_mint, output_mint, amount):
        self.calls.append((input_mint, output_mint, amount))
        q = {"inputMint": input_mint, "outputMint": output_mint,
             "inAmount": str(amount), "swapMode": "ExactIn"}
        if input_mint == SOL_MINT and output_mint == USDC_MINT:
            q["outAmount"] = str(150 * 10 ** 6 * amount // 10 ** 9)
            return q
        if input_mint == SOL_MINT:
            out = amount * TOKENS_PER_LAMPORT
            q["outAmount"] = str(out)
            q["otherAmountThreshold"] = str(out * (10_000 - self.slippage_bps) // 10_000)
            return self._leg(self.buy, q, amount)
        if output_mint == SOL_MINT:
            out = math.floor(Fraction(amount, TOKENS_PER_LAMPORT) * self.sell_ratio)
            q["outAmount"] = str(out)
            q["otherAmountThreshold"] = str(out * (10_000 - self.slippage_bps) // 10_000)
            return self._leg(self.sell, q, amount)
        raise AssertionError((input_mint, output_mint))

    def route_calls(self):
        return [c for c in self.calls if USDC_MINT not in c[:2]]


def with_fields(**fields):
    def apply(q, _amount):
        q = dict(q)
        for k, v in fields.items():
            if v is _DROP:
                q.pop(k, None)
            else:
                q[k] = v
        return q
    return apply


_DROP = object()


def sell_out(value):
    # Expected and slippage-minimum output both set to ``value``.
    return with_fields(outAmount=value, otherAmountThreshold=value)


def paper_cfg():
    return Config(live_trading=False, wallet_private_key="", bankroll_usd=100.0,
                  max_position_pct=2.0, stop_loss_pct=15.0)


def armed_cfg():
    return Config(live_trading=True, wallet_private_key=SENTINEL, burner_wallet_pubkey="PUBKEY",
                  bankroll_usd=20.0, max_position_pct=10.0,
                  process_lock_file=".test-lock", state_file=".test-state")


def make_bot(jup, cfg=None):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg or paper_cfg()
    bot.jup = jup
    bot.portfolio = Portfolio()
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 2.0)
    bot.notifier = types.SimpleNamespace(buy=lambda *a: None, sell=lambda *a: None, halt=lambda *a: None)
    bot._save = lambda: None
    bot._reconciliation_required = False
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=MINT)])
    bot._candidate_market_snapshot = lambda _m: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, liquidity_usd=50_000, price_usd=0.002, symbol="SELL", decimals=6))
    return bot


def gate(bot, mint=MINT, lamports=PLANNED):
    fn = getattr(bot, "_sell_route_ok", None)
    assert fn is not None, "TradingBot._sell_route_ok missing"
    return fn(mint, lamports)


class SignerSpy:
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


def run_entry(jup, mode, monkeypatch):
    """Drive _seek_entry with a real SwapExecutor; return (bot, swap_calls, spy)."""
    cfg = armed_cfg() if mode == "armed" else paper_cfg()
    bot = make_bot(jup, cfg)
    executor = SwapExecutor(cfg, jup=jup, session=object())
    spy = SignerSpy(executor, monkeypatch) if mode == "armed" else None
    swaps = []
    real_swap = executor.swap

    def swap(*args, **kwargs):
        swaps.append((args, kwargs))
        return real_swap(*args, **kwargs)

    executor.swap = swap
    bot.executor = executor
    bot._seek_entry()
    return bot, swaps, spy


MODES = ["paper", "armed"]


def assert_rejected(bot, swaps, spy):
    assert swaps == []
    assert bot.portfolio.positions == {}
    assert bot.portfolio.pending_intents == []
    if spy is not None:
        assert spy.events == []


# -- end-to-end through _seek_entry ---------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_no_sell_route_rejects_entry(mode, monkeypatch):
    jup = RouteJup(sell=lambda q, a: None)
    bot, swaps, spy = run_entry(jup, mode, monkeypatch)
    assert_rejected(bot, swaps, spy)


@pytest.mark.parametrize("mode", MODES)
def test_fifty_percent_round_trip_loss_rejects_entry(mode, monkeypatch):
    bot, swaps, spy = run_entry(RouteJup(sell_ratio=Fraction(1, 2)), mode, monkeypatch)
    assert_rejected(bot, swaps, spy)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad_leg", ["buy", "sell"])
def test_malformed_quote_rejects_entry(mode, bad_leg, monkeypatch):
    jup = RouteJup(**{bad_leg: with_fields(outAmount="1.5")})
    bot, swaps, spy = run_entry(jup, mode, monkeypatch)
    assert_rejected(bot, swaps, spy)


def test_two_percent_loss_passes_and_paper_swap_uses_planned_lamports(monkeypatch):
    jup = RouteJup(sell_ratio=Fraction(98, 100))
    bot, swaps, _ = run_entry(jup, "paper", monkeypatch)
    assert len(swaps) == 1
    args, kwargs = swaps[0]
    assert args == (SOL_MINT, MINT, PLANNED)
    assert kwargs == {"allocation_usd": 2.0}
    assert MINT in bot.portfolio.positions
    # Gate ran before the swap: buy leg, exact sell leg, then the swap's own quote.
    assert jup.route_calls() == [
        (SOL_MINT, MINT, PLANNED),
        (MINT, SOL_MINT, PLANNED * TOKENS_PER_LAMPORT),
        (SOL_MINT, MINT, PLANNED),
    ]


def test_two_percent_loss_passes_and_armed_swap_reaches_execution(monkeypatch):
    jup = RouteJup(sell_ratio=Fraction(98, 100))
    cfg = armed_cfg()
    bot = make_bot(jup, cfg)
    bot._save = lambda: True                   # armed entries need a durable buy intent
    executor = SwapExecutor(cfg, jup=jup, session=object())
    reached = []
    executor._execute_live = lambda quote, amount, out, **kw: reached.append(amount) or SwapResult(
        ok=False, simulated=False, in_amount=amount, out_amount=out,
        status=TransactionStatus.CONFIRMED_FAILURE, error="test stop")
    bot.executor = executor
    bot._seek_entry()
    assert reached == [PLANNED]
    assert bot.portfolio.positions == {}


# -- the 10% integer threshold -----------------------------------------------

@pytest.mark.parametrize(("lamports", "returned", "ok"), [
    (10_000_000, 9_000_000, True),       # exactly 10% loss passes
    (10_000_000, 8_999_999, False),      # 10% + 1 lamport fails
    (PLANNED, 12_000_000, True),         # ceil(0.9 * 13_333_333) = 12_000_000
    (PLANNED, 11_999_999, False),
    (PLANNED, PLANNED, True),            # no loss
    (PLANNED, PLANNED * 2, True),        # quoted gain is not a failure
])
def test_ten_percent_boundary_is_integer_exact(lamports, returned, ok):
    jup = RouteJup(sell=sell_out(str(returned)))
    assert gate(make_bot(jup), lamports=lamports) is ok


def test_threshold_constant_is_ten_percent():
    assert main_module.MAX_ROUNDTRIP_LOSS == Fraction(1, 10)


# -- exact request/echo validation ------------------------------------------

def test_sell_quote_receives_exact_buy_out_amount():
    # threshold == outAmount here; the threshold-vs-outAmount case is in
    # tests/test_p1_roundtrip_min_out.py.
    jup = RouteJup(buy=with_fields(outAmount="987654321", otherAmountThreshold="987654321"),
                   sell=lambda q, a: dict(q, outAmount=str(PLANNED), otherAmountThreshold=str(PLANNED)))
    assert gate(make_bot(jup)) is True
    assert jup.route_calls() == [(SOL_MINT, MINT, PLANNED), (MINT, SOL_MINT, 987654321)]


BAD_AMOUNTS = [
    pytest.param(_DROP, id="missing"), pytest.param(None, id="None"), pytest.param("", id="empty"),
    pytest.param("1.5", id="decimal-str"), pytest.param(1.5, id="float"),
    pytest.param(float(PLANNED * 1000), id="float-int-valued"),
    pytest.param("1e6", id="exponent"), pytest.param("-5", id="negative"), pytest.param(-5, id="neg-int"),
    pytest.param("0", id="zero"), pytest.param(0, id="int-zero"), pytest.param(True, id="bool"),
    pytest.param(float("nan"), id="nan"), pytest.param("inf", id="inf-str"),
    pytest.param(float("inf"), id="inf"), pytest.param("+5", id="plus"), pytest.param(" 5", id="space"),
    pytest.param(str(2 ** 64), id="over-u64"), pytest.param(["5"], id="list"),
]


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("value", BAD_AMOUNTS)
def test_bad_out_amount_rejects(leg, value):
    jup = RouteJup(**{leg: with_fields(outAmount=value)})
    assert gate(make_bot(jup)) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("value", [
    pytest.param(_DROP, id="missing"), pytest.param(None, id="None"),
    pytest.param("plus-one", id="mismatch+1"), pytest.param("float", id="float-equal"),
    pytest.param("decimal", id="decimal-str"), pytest.param(True, id="bool"),
    pytest.param("0", id="zero"),
])
def test_contradictory_in_amount_rejects(leg, value):
    def apply(q, amount):
        q = dict(q)
        if value is _DROP:
            q.pop("inAmount")
        elif value == "plus-one":
            q["inAmount"] = str(amount + 1)
        elif value == "float":
            q["inAmount"] = float(amount)
        elif value == "decimal":
            q["inAmount"] = f"{amount}.0"
        else:
            q["inAmount"] = value
        return q

    assert gate(make_bot(RouteJup(**{leg: apply}))) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("field", ["inputMint", "outputMint"])
@pytest.mark.parametrize("value", [OTHER, USDC_MINT, None, _DROP])
def test_wrong_or_missing_mint_rejects(leg, field, value):
    assert gate(make_bot(RouteJup(**{leg: with_fields(**{field: value})}))) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
def test_swapped_mints_reject(leg):
    def swap_mints(q, _a):
        return dict(q, inputMint=q["outputMint"], outputMint=q["inputMint"])
    assert gate(make_bot(RouteJup(**{leg: swap_mints}))) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
def test_exact_out_quote_rejects(leg):
    assert gate(make_bot(RouteJup(**{leg: with_fields(swapMode="ExactOut")}))) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("response", [None, {}, "quote", [], 42, {"error": "no route"}])
def test_none_or_unexpected_response_rejects(leg, response):
    assert gate(make_bot(RouteJup(**{leg: lambda q, a: response}))) is False


@pytest.mark.parametrize("leg", ["buy", "sell"])
@pytest.mark.parametrize("exc", [RuntimeError("provider down"), requests.Timeout("timeout"),
                                 TimeoutError("timeout"), KeyError("outAmount"), ValueError("json")])
def test_provider_exception_rejects(leg, exc):
    assert gate(make_bot(RouteJup(**{leg: exc}))) is False


def test_real_client_http_timeout_rejects_via_existing_policy():
    """JupiterClient.quote turns its HTTP timeout into None; the gate fails closed."""
    class TimeoutSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append(kwargs.get("timeout"))
            raise requests.Timeout("simulated")

    session = TimeoutSession()
    client = JupiterClient(paper_cfg(), session)
    assert gate(make_bot(client)) is False
    assert session.calls == [jupiter_module.HTTP_TIMEOUT]


@pytest.mark.parametrize("lamports", [0, -1, None, 1.5, float(PLANNED), True, "13333333"])
def test_invalid_planned_lamports_reject_without_quoting(lamports):
    jup = RouteJup()
    assert gate(make_bot(jup), lamports=lamports) is False
    assert jup.calls == []


@pytest.mark.parametrize("mint", ["", None, SOL_MINT])
def test_invalid_mint_rejects_without_quoting(mint):
    jup = RouteJup()
    assert gate(make_bot(jup), mint=mint) is False
    assert jup.calls == []


def test_missing_quote_client_rejects():
    bot = make_bot(RouteJup())
    del bot.jup
    assert gate(bot) is False
