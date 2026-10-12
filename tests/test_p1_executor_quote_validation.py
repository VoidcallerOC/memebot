"""P1: SwapExecutor.swap validates its own quote before booking or signing.

On main, ``int(quote["outAmount"])`` raised ValueError for malformed amounts,
escaping into callers (an armed exit left its intent "pending" and aborted
the rest of the position loop). Now every invalid quote is a clean
``ok=False`` / CONFIRMED_FAILURE result, like "no route". Offline fakes only.
"""
from __future__ import annotations

import types

import pytest

import bot.jupiter as jupiter_module
import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, USDC_MINT, SwapExecutor, SwapResult, TransactionStatus
from bot.portfolio import Portfolio
from bot.risk import RiskManager

SENTINEL = "test-only-sentinel"
MINT = "ExecQuoteMint11111111111111111111111111111"
MINT2 = "ExecQuoteMintTwo111111111111111111111111111"
OTHER = "OtherMint11111111111111111111111111111111111"
_DROP = object()


def paper_cfg():
    return Config(live_trading=False, wallet_private_key="", bankroll_usd=100.0, max_position_pct=2.0)


def armed_cfg():
    return Config(live_trading=True, wallet_private_key=SENTINEL, burner_wallet_pubkey="PUBKEY",
                  bankroll_usd=20.0, max_position_pct=10.0,
                  process_lock_file=".test-lock", state_file=".test-state")


def good_quote(i, o, amount, out=None):
    out = amount * 1000 if out is None else out
    return {"inputMint": i, "outputMint": o, "inAmount": str(amount), "outAmount": str(out),
            "otherAmountThreshold": str(out * 985 // 1000), "swapMode": "ExactIn", "slippageBps": 150}


class FakeJup:
    """SOL->USDC priced at $150 (for the armed exposure guard); swap legs programmable."""

    quote_url = "https://jupiter.test/quote"
    swap_url = "https://jupiter.test/swap"

    def __init__(self, mutate=None, marks=None):
        self.mutate = mutate
        self.marks = marks or {}
        self.calls = []

    def quote(self, i, o, amount):
        self.calls.append((i, o, amount))
        if i == SOL_MINT and o == USDC_MINT:
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount),
                    "outAmount": str(150 * 10 ** 6 * amount // 10 ** 9)}
        if o == USDC_MINT:  # exit marks: whole-token USD price, 6-dec USDC
            return {"inputMint": i, "outputMint": o, "inAmount": str(amount),
                    "outAmount": str(self.marks[i])}
        q = good_quote(i, o, amount)
        if self.mutate is None:
            return q
        if isinstance(self.mutate, Exception):
            raise self.mutate
        return self.mutate(q, amount)


def set_field(field, value):
    def apply(q, amount):
        q = dict(q)
        if value is _DROP:
            q.pop(field, None)
        elif value == "plus-one":
            q[field] = str(amount + 1)
        elif value == "above-out":
            q[field] = str(int(q["outAmount"]) + 1)
        else:
            q[field] = value
        return q
    return apply


BAD_QUOTES = [
    pytest.param(lambda q, a: "not-a-dict", id="string"),
    pytest.param(lambda q, a: [], id="list"),
    pytest.param(lambda q, a: {}, id="empty"),
    pytest.param(lambda q, a: {"outAmount": q["outAmount"]}, id="outAmount-only"),
] + [
    pytest.param(set_field("outAmount", v), id=f"out-{name}") for name, v in [
        ("missing", _DROP), ("None", None), ("empty", ""), ("decimal", "1.5"), ("float", 1.5),
        ("exponent", "1e6"), ("negative", "-5"), ("neg-int", -5), ("zero", "0"), ("int-zero", 0),
        ("bool", True), ("nan", "nan"), ("inf", float("inf")), ("space", " 5"), ("plus", "+5"),
        ("over-u64", str(2 ** 64)), ("list", ["5"]),
    ]
] + [
    pytest.param(set_field("inAmount", v), id=f"in-{name}") for name, v in [
        ("missing", _DROP), ("mismatch", "plus-one"), ("float", 5.0), ("zero", "0"), ("bool", True),
    ]
] + [
    pytest.param(set_field(f, v), id=f"{f}-{name}") for f in ("inputMint", "outputMint")
    for name, v in [("wrong", OTHER), ("missing", _DROP), ("None", None)]
] + [
    pytest.param(set_field("swapMode", "ExactOut"), id="exact-out"),
    pytest.param(set_field("otherAmountThreshold", "above-out"), id="threshold-above-out"),
    pytest.param(set_field("otherAmountThreshold", "0"), id="threshold-zero"),
    pytest.param(set_field("otherAmountThreshold", "1.5"), id="threshold-malformed"),
]


class SignerSpy:
    def __init__(self, executor, monkeypatch):
        self.events = []
        executor._load_keypair = lambda *_a: self.events.append("load_keypair") or (_ for _ in ()).throw(
            AssertionError("signer must not be loaded"))
        monkeypatch.setattr(jupiter_module, "send_raw_transaction",
                            lambda *a, **k: self.events.append("send") or "sig")
        executor._execute_live = lambda *a, **k: self.events.append("execute_live") or (_ for _ in ()).throw(
            AssertionError("must not execute"))


# -- executor contract ---------------------------------------------------------

@pytest.mark.parametrize("mutate", BAD_QUOTES)
@pytest.mark.parametrize(("direction"), ["buy", "sell"])
def test_paper_invalid_quote_is_clean_failure(mutate, direction):
    i, o, amount = (SOL_MINT, MINT, 13_333_333) if direction == "buy" else (MINT, SOL_MINT, 5_000_000)
    result = SwapExecutor(paper_cfg(), jup=FakeJup(mutate)).swap(i, o, amount, allocation_usd=2.0)
    assert result.ok is False
    assert result.simulated is True
    assert result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.out_amount == 0 and result.tx_signature is None
    assert result.error.startswith("invalid quote:")


@pytest.mark.parametrize("mutate", BAD_QUOTES)
@pytest.mark.parametrize(("direction"), ["buy", "sell"])
def test_armed_invalid_quote_never_signs_or_sends(mutate, direction, monkeypatch):
    i, o, amount = (SOL_MINT, MINT, 13_333_333) if direction == "buy" else (MINT, SOL_MINT, 5_000_000)
    executor = SwapExecutor(armed_cfg(), jup=FakeJup(mutate), session=object())
    spy = SignerSpy(executor, monkeypatch)
    result = executor.swap(i, o, amount, allocation_usd=2.0)
    assert result.ok is False
    assert result.status == TransactionStatus.CONFIRMED_FAILURE
    assert result.tx_signature is None
    assert spy.events == []


@pytest.mark.parametrize("amount", [0, -1, 1.5, None, True, "5000000"])
def test_invalid_amount_rejected_before_quoting(amount):
    jup = FakeJup()
    result = SwapExecutor(paper_cfg(), jup=jup).swap(MINT, SOL_MINT, amount, allocation_usd=2.0)
    assert result.ok is False and result.status == TransactionStatus.CONFIRMED_FAILURE
    assert jup.calls == []


def test_valid_quote_paper_still_simulates():
    result = SwapExecutor(paper_cfg(), jup=FakeJup()).swap(SOL_MINT, MINT, 13_333_333, allocation_usd=2.0)
    assert result.ok and result.simulated
    assert result.out_amount == 13_333_333_000
    assert result.status == TransactionStatus.CONFIRMED_SUCCESS


@pytest.mark.parametrize("threshold_present", [True, False])
@pytest.mark.parametrize(("direction"), ["buy", "sell"])
def test_valid_quote_armed_still_reaches_execution(direction, threshold_present):
    i, o, amount = (SOL_MINT, MINT, 13_333_333) if direction == "buy" else (MINT, SOL_MINT, 5_000_000)
    mutate = None if threshold_present else set_field("otherAmountThreshold", _DROP)
    executor = SwapExecutor(armed_cfg(), jup=FakeJup(mutate), session=object())
    seen = []
    executor._execute_live = lambda quote, amt, out, **kw: seen.append((quote, amt, out, kw)) or SwapResult(
        ok=True, simulated=False, in_amount=amt, out_amount=out,
        status=TransactionStatus.CONFIRMED_SUCCESS)
    # armed sells must carry a pre-broadcast persistence callback; buys do not (yet)
    cb = (lambda sig: True) if direction == "sell" else None
    assert executor.swap(i, o, amount, allocation_usd=2.0, before_broadcast=cb).ok
    (quote, amt, out, kw), = seen
    assert amt == amount and out == amount * 1000 and isinstance(out, int)
    assert quote["inAmount"] == str(amount)
    assert kw == {"input_mint": i, "output_mint": o, "before_broadcast": cb}


# -- callers: armed exits ----------------------------------------------------

def armed_bot(jup):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = armed_cfg()
    bot.jup = jup
    bot.portfolio = Portfolio()
    bot.risk = RiskManager(bot.cfg)
    bot.notifier = types.SimpleNamespace(
        sell=lambda *a: (_ for _ in ()).throw(AssertionError("no sell alert")),
        halt=lambda *a: None, buy=lambda *a: None)
    bot._save = lambda: None
    bot._reconciliation_required = False
    return bot


@pytest.mark.parametrize("mutate", [
    set_field("outAmount", "1.5"), set_field("outAmount", "-5"), set_field("outAmount", _DROP),
    set_field("outAmount", "0"), set_field("inAmount", "plus-one"), set_field("outputMint", OTHER),
])
def test_armed_exit_with_malformed_sell_quote_is_safe(mutate, monkeypatch):
    # Both positions are marked at a stop-loss price (entry $1, mark $0.50).
    jup = FakeJup(mutate, marks={MINT: 500_000, MINT2: 500_000})
    bot = armed_bot(jup)
    executor = SwapExecutor(bot.cfg, jup=jup, session=object())
    spy = SignerSpy(executor, monkeypatch)
    bot.executor = executor
    bot.portfolio.open(MINT, "A", 1.0, 2.0, 2.0, token_decimals=6, raw_tokens=2_000_000, raw_tokens_initial=2_000_000)
    bot.portfolio.open(MINT2, "B", 1.0, 2.0, 2.0, token_decimals=6, raw_tokens=2_000_000, raw_tokens_initial=2_000_000)

    bot._manage_positions()  # must not raise

    assert spy.events == []
    for mint in (MINT, MINT2):  # positions and cost basis untouched
        pos = bot.portfolio.positions[mint]
        assert (pos.tokens, pos.size_usd, pos.entry_price) == (2.0, 2.0, 1.0)
    assert bot.portfolio.realized_pnl == 0.0
    # Both exits were attempted in the same tick (the loop was not aborted),
    # and each intent ends as a clean "unresolved" failure, not "pending".
    sells = [c for c in jup.calls if c[1] == SOL_MINT]
    assert sells == [(MINT, SOL_MINT, 2_000_000), (MINT2, SOL_MINT, 2_000_000)]
    assert [i["status"] for i in bot.portfolio.pending_intents] == ["unresolved", "unresolved"]
    assert all(i["tx_signature"] is None for i in bot.portfolio.pending_intents)


def test_armed_exit_with_valid_quote_still_reaches_execution():
    jup = FakeJup(marks={MINT: 500_000})
    bot = armed_bot(jup)
    executor = SwapExecutor(bot.cfg, jup=jup, session=object())
    reached = []
    executor._execute_live = lambda quote, amt, out, **kw: reached.append(amt) or SwapResult(
        ok=False, simulated=False, in_amount=amt, out_amount=out,
        status=TransactionStatus.CONFIRMED_FAILURE, error="test stop")
    bot.executor = executor
    bot.portfolio.open(MINT, "A", 1.0, 2.0, 2.0, token_decimals=6, raw_tokens=2_000_000, raw_tokens_initial=2_000_000)
    bot._manage_positions()
    assert reached == [2_000_000]
    assert bot.portfolio.positions[MINT].tokens == 2.0


# -- callers: entries -----------------------------------------------------------

@pytest.mark.parametrize("armed", [False, True])
def test_entry_with_malformed_executor_quote_opens_nothing(armed, monkeypatch):
    from bot.safety import TokenSafety
    jup = FakeJup(set_field("outAmount", "1.5"))
    cfg = armed_cfg() if armed else paper_cfg()
    bot = armed_bot(jup)
    bot.cfg = cfg
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 2.0)
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=MINT)])
    bot._candidate_market_snapshot = lambda _m: None
    bot.screener = types.SimpleNamespace(screen=lambda mint, **k: TokenSafety(
        mint=mint, passed=True, liquidity_usd=50_000, price_usd=0.002, symbol="E", decimals=6))
    # Isolate the executor: the pre-buy gate is covered elsewhere.
    bot._sell_route_ok = lambda _m, _l: True
    executor = SwapExecutor(cfg, jup=jup, session=object())
    spy = SignerSpy(executor, monkeypatch) if armed else None
    bot.executor = executor
    bot._seek_entry()  # must not raise
    assert bot.portfolio.positions == {}
    assert bot._reconciliation_required is False
    if spy is not None:
        assert spy.events == []
