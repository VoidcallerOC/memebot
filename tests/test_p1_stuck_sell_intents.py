"""P1: live sell intents are resolved on evidence instead of freezing the bot.

On main, any pending/unresolved/blocked intent made every tick reconcile and
return (nothing ever cleared it), so after one failed armed sell no position
was managed again (no stop-loss / take-profit) or the bot halted. Now:

* definitive non-execution (refused before send, on-chain error, or expired
  unconfirmed with the tokens still held) -> ``failed``; exits resume/retry;
* confirmed execution with a matching on-chain fill -> ``settled``, booked once;
* anything else stays open: that mint's exits are frozen (no double sell), new
  entries are blocked, other positions keep SL/TP, with an explicit reason.

Offline fakes only: no RPC, quotes, signing or sends.
"""
from __future__ import annotations

import types

import pytest

import bot.main as main_module
from bot.config import Config
from bot.jupiter import SOL_MINT, SwapResult, TransactionStatus
from bot.portfolio import Portfolio
from bot.risk import RiskManager
from bot.state import load_state

SENTINEL = "test-only-sentinel"
WALLET = "WalletPubkey1111111111111111111111111111111"
MINT = "StuckSellMint1111111111111111111111111111111"
MINT2 = "StuckSellMintTwo11111111111111111111111111111"
SL_MARK = 0.5      # entry 1.0, stop loss 15% -> 0.5 fires a full stop-loss
OK_MARK = 1.0      # no exit action
RAW = 2_000_000    # 2.0 tokens at 6 decimals
EXPIRY = getattr(main_module, "SELL_INTENT_EXPIRY_SECONDS", 300)  # getattr: mutation runs vs main
T0 = 1_800_000_000.0


class Clock:
    def __init__(self):
        self.now = T0

    def time(self):
        return self.now


class FakeExecutor:
    """Returns queued SwapResults; default is a confirmed sell with a full fill."""

    def __init__(self, results=()):
        self.results = list(results)
        self.calls = []

    def swap(self, i, o, amount, *, allocation_usd=None, before_broadcast=None):
        self.calls.append((i, o, amount))
        if self.results:
            return self.results.pop(0)
        return confirmed(amount)


def confirmed(amount, sig="sig-ok"):
    return SwapResult(ok=True, simulated=False, in_amount=amount, out_amount=1,
                      tx_signature=sig, status=TransactionStatus.CONFIRMED_SUCCESS,
                      actual_in_amount=amount, actual_out_amount=6_000_000, fee_lamports=5000)


def failure(status, sig=None, error="test failure"):
    return SwapResult(ok=False, simulated=False, in_amount=RAW, out_amount=0,
                      tx_signature=sig, status=status, error=error)


class FakeChain:
    """Signature status, getTransaction and wallet balances, all programmable."""

    def __init__(self):
        self.status = {}        # sig -> status dict or None
        self.txs = {}           # sig -> parsed tx dict
        self.balances = {}      # mint -> ui amount (None => unavailable)
        self.calls = []

    def get_signature_status(self, session, rpc_url, sig):
        self.calls.append(("status", sig))
        value = self.status.get(sig)
        if isinstance(value, Exception):
            raise value
        return value

    def get_confirmed_transaction(self, session, rpc_url, sig, commitment):
        self.calls.append(("tx", sig))
        return self.txs.get(sig)

    def fetch_wallet_token_balances(self, cfg, owner, session=None):
        self.calls.append(("balances", owner))
        if self.balances is None:
            return None
        return types.SimpleNamespace(amounts=dict(self.balances))


def sell_tx(mint=MINT, token_pre=RAW, token_post=0, wsol_out=6_000_000, owner=WALLET):
    """Parsed token -> SOL sell (wSOL output visible in postTokenBalances)."""
    def bal(m, amount, dec):
        return {"owner": owner, "mint": m, "uiTokenAmount": {"amount": str(amount), "decimals": dec}}
    post = [bal(mint, token_post, 6), bal(SOL_MINT, wsol_out, 9)]
    return {"transaction": {"message": {"accountKeys": [owner]}},
            "meta": {"fee": 5000, "preTokenBalances": [bal(mint, token_pre, 6)],
                     "postTokenBalances": post, "preBalances": [10**9], "postBalances": [10**9]}}


class Notes:
    def __init__(self):
        self.sells, self.halts, self.sent = [], [], []
        self.enabled = True

    def sell(self, *a):
        self.sells.append(a)

    def halt(self, *a):
        self.halts.append(a)

    def send(self, text):
        self.sent.append(text)

    def startup(self, *a):
        pass

    def buy(self, *a):
        raise AssertionError("no buys in these tests")


@pytest.fixture
def env(monkeypatch, tmp_path):
    clock, chain, marks, reconciles = Clock(), FakeChain(), {}, []
    monkeypatch.setattr(main_module, "time", types.SimpleNamespace(time=clock.time, sleep=lambda s: None))
    for name in ("get_signature_status", "get_confirmed_transaction", "fetch_wallet_token_balances"):
        monkeypatch.setattr(main_module, name, getattr(chain, name), raising=False)
    monkeypatch.setattr(main_module, "_price_usd", lambda jup, mint, dec: marks.get(mint))

    def fake_reconcile(cfg, portfolio, session=None, notifier=None):
        reconciles.append({m: p.tokens for m, p in portfolio.positions.items()})
        return types.SimpleNamespace(clean=True)

    monkeypatch.setattr(main_module, "reconcile_wallet", fake_reconcile)
    return types.SimpleNamespace(clock=clock, chain=chain, marks=marks, reconciles=reconciles,
                                 state=str(tmp_path / "state.json"), lock=str(tmp_path / "lock"))


def make_bot(env, results=(), armed=True, load=False):
    cfg = Config(live_trading=armed, wallet_private_key=SENTINEL if armed else "",
                 burner_wallet_pubkey=WALLET, bankroll_usd=20.0, max_position_pct=10.0,
                 state_file=env.state, process_lock_file=env.lock)
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg
    bot.session = object()
    bot.jup = object()
    bot.portfolio = Portfolio()
    bot.risk = RiskManager(cfg)
    bot.notifier = Notes()
    bot.executor = FakeExecutor(results)
    bot._running = True
    bot._reconciliation_required = False
    bot.entry_scans = []
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: bot.entry_scans.append(1) or [])
    bot._verified_decimals = lambda mint, pos=None: pos.token_decimals
    if load:
        assert load_state(env.state, bot.portfolio, bot.risk)
    return bot


def open_pos(bot, mint=MINT, symbol="A"):
    bot.portfolio.open(mint, symbol, 1.0, 2.0, 2.0, token_decimals=6, raw_tokens=2_000_000, raw_tokens_initial=2_000_000)


def intents(bot):
    return [(i["mint"], i["status"]) for i in bot.portfolio.pending_intents]


def sells_of(bot, mint):
    return [c for c in bot.executor.calls if c[0] == mint and c[1] == SOL_MINT]


# -- definitive non-execution: clear and retry the exit -----------------------

@pytest.mark.parametrize("first", [
    failure(TransactionStatus.CONFIRMED_FAILURE, error="no route / quote unavailable"),
    failure(TransactionStatus.CONFIRMED_FAILURE, error="invalid quote: quote outAmount missing"),
])
def test_confirmed_failure_sell_clears_and_stop_loss_fires_next_tick(env, first):
    bot = make_bot(env, [first])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert bot.portfolio.positions[MINT].tokens == 2.0
    assert bot.portfolio.realized_pnl == 0.0

    bot._tick()  # resolves as not executed, then the stop-loss is retried
    assert intents(bot) == [(MINT, "failed"), (MINT, "settled")]
    assert len(sells_of(bot, MINT)) == 2
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(-1.0)
    assert len(bot.notifier.sells) == 1
    assert env.chain.calls == []  # a pre-send rejection needs no chain lookup
    assert bot.portfolio.pending_intents[0]["resolution"] == "rejected before send (no transaction)"


def test_onchain_error_confirmed_clears_and_retries(env):
    bot = make_bot(env, [failure(TransactionStatus.CONFIRMED_FAILURE, sig="sig-err",
                                 error="transaction confirmed_failure")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.status["sig-err"] = {"err": {"InstructionError": [0, "x"]}, "confirmationStatus": "confirmed"}
    bot._tick()
    assert intents(bot) == [(MINT, "failed"), (MINT, "settled")]
    assert bot.portfolio.pending_intents[0]["resolution"] == "transaction failed on chain"
    assert MINT not in bot.portfolio.positions


def test_onchain_error_only_processed_is_not_definitive(env):
    bot = make_bot(env, [failure(TransactionStatus.CONFIRMED_FAILURE, sig="sig-err")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.status["sig-err"] = {"err": {"x": 1}, "confirmationStatus": "processed"}
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert len(sells_of(bot, MINT)) == 1


def test_persisted_blocked_intent_clears_and_stop_loss_fires(env):
    bot = make_bot(env)
    open_pos(bot)
    bot.portfolio.pending_intents.append({"mint": MINT, "fraction": 1.0, "reason": "stop_loss",
                                          "requested_price": 0.5, "status": "blocked",
                                          "tx_signature": None})
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert intents(bot) == [(MINT, "failed"), (MINT, "settled")]
    assert MINT not in bot.portfolio.positions


# -- unknown outcome: frozen until evidence, then settled exactly once --------

def test_unknown_outcome_frozen_until_signature_confirms_then_settles_once(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1", error="timeout")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert bot._reconciliation_required is True

    env.chain.status["sig-1"] = None  # not found yet
    for _ in range(3):
        env.clock.now += 10
        bot._tick()
        assert len(sells_of(bot, MINT)) == 1      # no second sell while unknown
        assert bot.portfolio.positions[MINT].tokens == 2.0
        assert bot.portfolio.realized_pnl == 0.0
        assert bot.entry_scans == []              # entries blocked
        assert bot._running is True               # not halted
    assert len(bot.notifier.sent) == 1            # alerted once, loudly

    env.marks[MINT] = 0.3  # the mark moves; booking must still use the intent's price
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "confirmed"}
    env.chain.txs["sig-1"] = sell_tx()
    bot._tick()
    assert intents(bot) == [(MINT, "settled")]
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(2.0 * (0.5 - 1.0))
    assert bot.risk.realized_pnl_today == pytest.approx(-1.0)
    settled = bot.portfolio.pending_intents[0]
    assert (settled["actual_in_amount"], settled["actual_out_amount"]) == (RAW, 6_000_000)

    for _ in range(3):  # never booked or sold again
        bot._tick()
    assert len(sells_of(bot, MINT)) == 1
    assert len(bot.notifier.sells) == 1
    assert bot.portfolio.realized_pnl == pytest.approx(-1.0)


@pytest.mark.parametrize("tx", [
    None,                                   # getTransaction unavailable
    sell_tx(token_post=1_000_000),          # actual_in 1_000_000 != intent raw 2_000_000
    sell_tx(wsol_out=0),                    # no SOL received in the parsed fill
    sell_tx(owner="SomeoneElse"),           # not our wallet's balances
])
def test_confirmed_without_matching_fill_stays_frozen_no_fabricated_pnl(env, tx):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "finalized"}
    if tx is not None:
        env.chain.txs["sig-1"] = tx
    for _ in range(2):
        env.clock.now += EXPIRY + 1
        bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert "operator action required" in bot.portfolio.pending_intents[0]["resolution"]
    assert bot.portfolio.positions[MINT].tokens == 2.0
    assert bot.portfolio.realized_pnl == 0.0
    assert bot.notifier.sells == []
    assert len(sells_of(bot, MINT)) == 1
    assert bot.entry_scans == []


def test_status_lookup_error_keeps_intent_open(env):
    bot = make_bot(env, [failure(TransactionStatus.UNKNOWN, sig="sig-1")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.status["sig-1"] = RuntimeError("rpc down")
    env.clock.now += EXPIRY + 1
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert len(sells_of(bot, MINT)) == 1


@pytest.mark.parametrize("sig", [None, "sig-lost"])
def test_expired_unconfirmed_with_tokens_held_clears_and_retries(env, sig):
    bot = make_bot(env, [failure(TransactionStatus.UNKNOWN, sig=sig, error="LIVE swap failed")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    env.chain.balances = {MINT: 2.0}
    bot._tick()
    env.clock.now += EXPIRY - 1
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]   # still inside the blockhash window
    assert len(sells_of(bot, MINT)) == 1
    env.clock.now += 2
    bot._tick()
    assert intents(bot) == [(MINT, "failed"), (MINT, "settled")]
    assert bot.portfolio.pending_intents[0]["resolution"] == (
        "expired unconfirmed; wallet still holds the tokens")
    assert MINT not in bot.portfolio.positions


@pytest.mark.parametrize("balances", [{MINT: 0.0}, {}, None])
def test_expired_unconfirmed_without_held_tokens_stays_frozen(env, balances):
    bot = make_bot(env, [failure(TransactionStatus.UNKNOWN, error="LIVE swap failed")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.balances = balances
    env.clock.now += EXPIRY + 1
    bot._tick()
    assert intents(bot) == [(MINT, "unresolved")]
    assert bot.portfolio.positions[MINT].tokens == 2.0
    assert bot.portfolio.realized_pnl == 0.0
    assert len(sells_of(bot, MINT)) == 1


# -- other positions keep SL/TP; entries blocked; daily halt untouched --------

def test_other_positions_still_get_stop_loss_while_one_intent_unresolved(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    open_pos(bot, MINT, "A")
    env.marks[MINT] = SL_MARK
    bot._tick()
    open_pos(bot, MINT2, "B")
    env.marks[MINT2] = SL_MARK
    env.chain.status["sig-1"] = None
    bot._reconciliation_required = False
    bot._tick()
    assert len(sells_of(bot, MINT)) == 1           # frozen
    assert len(sells_of(bot, MINT2)) == 1          # managed
    assert MINT2 not in bot.portfolio.positions
    assert MINT in bot.portfolio.positions
    assert bot.portfolio.realized_pnl == pytest.approx(-1.0)
    assert bot.entry_scans == []


def test_take_profit_still_fires_for_other_position(env):
    bot = make_bot(env)
    open_pos(bot, MINT, "A")
    open_pos(bot, MINT2, "B")
    bot.portfolio.pending_intents.append({"mint": MINT, "fraction": 1.0, "reason": "stop_loss",
                                          "requested_price": 0.5, "status": "unresolved",
                                          "tx_signature": "sig-1", "created_at": T0,
                                          "raw_amount": RAW, "outcome": "timeout"})
    env.chain.status["sig-1"] = None
    env.marks.update({MINT: SL_MARK, MINT2: 1.6})  # +60% -> first take-profit rung
    bot._tick()
    assert sells_of(bot, MINT) == []
    assert sells_of(bot, MINT2) == [(MINT2, SOL_MINT, 1_000_000)]
    assert bot.portfolio.positions[MINT2].tokens == pytest.approx(1.0)


def test_entries_resume_once_intent_resolved(env):
    bot = make_bot(env, [failure(TransactionStatus.CONFIRMED_FAILURE)])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert bot.entry_scans == []
    bot._tick()  # failed -> retried stop-loss settles; nothing open
    assert bot.entry_scans == [1]


def test_daily_loss_halt_never_cleared_by_resolution(env):
    bot = make_bot(env, [failure(TransactionStatus.CONFIRMED_FAILURE)])
    bot.risk.record_realized_pnl(-5.0)  # beyond the $2 daily limit
    assert bot.risk.trading_halted()
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    bot._tick()
    assert intents(bot) == [(MINT, "failed"), (MINT, "settled")]
    assert bot.risk.trading_halted() is True
    assert bot.risk.realized_pnl_today == pytest.approx(-6.0)
    assert bot.entry_scans == []


def test_settlement_can_trip_daily_halt_but_resolver_never_resets_it(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    bot.risk.record_realized_pnl(-1.5)
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "confirmed"}
    env.chain.txs["sig-1"] = sell_tx()
    bot._tick()
    assert bot.risk.trading_halted() is True
    assert len(bot.notifier.halts) == 1


# -- persistence / restart ----------------------------------------------------

def test_persisted_unknown_intent_behaves_the_same_after_restart(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    bot._save()

    env.chain.status["sig-1"] = None
    bot2 = make_bot(env, load=True)
    assert intents(bot2) == [(MINT, "unresolved")]
    bot2._tick()
    assert sells_of(bot2, MINT) == []               # still frozen after restart
    assert bot2.entry_scans == []
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "confirmed"}
    env.chain.txs["sig-1"] = sell_tx()
    bot2._tick()
    bot2._tick()
    assert intents(bot2) == [(MINT, "settled")]
    assert MINT not in bot2.portfolio.positions
    assert bot2.portfolio.realized_pnl == pytest.approx(-1.0)
    assert len(bot2.notifier.sells) == 1
    assert sells_of(bot2, MINT) == []               # settled without a second sell

    bot3 = make_bot(env, load=True)                 # settled state survives too
    bot3._tick()
    assert intents(bot3) == [(MINT, "settled")]
    assert bot3.notifier.sells == [] and bot3.executor.calls == []


def test_legacy_pending_intent_waits_from_first_seen_then_clears(env):
    bot = make_bot(env)
    open_pos(bot)
    bot.portfolio.pending_intents.append({"mint": MINT, "fraction": 1.0, "reason": "stop_loss",
                                          "requested_price": 0.5, "status": "pending",
                                          "tx_signature": None})  # main-era shape, no clock
    bot._save()
    bot2 = make_bot(env, load=True)
    env.marks[MINT] = SL_MARK
    env.chain.balances = {MINT: 2.0}
    bot2._tick()
    assert intents(bot2) == [(MINT, "pending")]
    assert bot2.portfolio.pending_intents[0]["first_seen_at"] == T0
    env.clock.now += EXPIRY + 1
    bot2._tick()
    assert intents(bot2) == [(MINT, "failed"), (MINT, "settled")]


def test_startup_resolves_persisted_intent_before_reconciliation(env):
    bot = make_bot(env, [failure(TransactionStatus.TIMEOUT, sig="sig-1")])
    open_pos(bot)
    env.marks[MINT] = SL_MARK
    bot._tick()
    bot._save()
    env.chain.status["sig-1"] = {"err": None, "confirmationStatus": "confirmed"}
    env.chain.txs["sig-1"] = sell_tx()
    env.reconciles.clear()
    bot2 = make_bot(env, load=True)
    bot2._running = False
    bot2._banner = lambda: None
    bot2._summary = lambda: None
    bot2.run()
    assert env.reconciles[0] == {}                  # settled before the startup check
    assert intents(bot2) == [(MINT, "settled")]


def test_dry_run_with_persisted_open_intent_halts_loudly(env, caplog):
    bot = make_bot(env, armed=False)
    open_pos(bot)
    bot.portfolio.pending_intents.append({"mint": MINT, "status": "unresolved", "tx_signature": "s"})
    env.marks[MINT] = SL_MARK
    bot._tick()
    assert bot._running is False
    assert bot.executor.calls == [] and env.chain.calls == []
    assert bot.portfolio.positions[MINT].tokens == 2.0


def test_terminal_intents_are_pruned(env):
    bot = make_bot(env)
    bot.portfolio.pending_intents = [
        {"mint": MINT, "status": "failed", "n": n} for n in range(70)
    ] + [{"mint": MINT, "status": "blocked", "tx_signature": None}]
    bot._resolve_sell_intents()
    kept = bot.portfolio.pending_intents
    assert len(kept) == getattr(main_module, "TERMINAL_SELL_INTENTS_KEPT", 50)
    assert all(i["status"] == "failed" for i in kept)
    assert kept[0]["n"] == 21                       # oldest terminal entries dropped
    assert kept[-1]["resolution"] == "refused before send"  # newest kept
