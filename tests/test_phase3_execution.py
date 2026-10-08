"""Phase 3 fill extraction, live entry accounting, and fail-closed sells."""
from __future__ import annotations

import sys
import types

from bot.chain import extract_fill
from bot.config import Config
from bot.jupiter import JupiterClient, SwapExecutor, SwapResult, TransactionStatus
import bot.main as main_module
from bot.portfolio import Portfolio

SENTINEL = "test-only-sentinel"
WALLET = "PUBKEY"
MINT = "Mint111111111111111111111111111111111111111"
SOL_MINT = "So11111111111111111111111111111111111111112"


def cfg(**overrides):
    values = dict(
        live_trading=True,
        wallet_private_key=SENTINEL,
        burner_wallet_pubkey=WALLET,
        confirmation_timeout_seconds=0.0,
        confirmation_poll_interval_seconds=0.01,
        bankroll_usd=20.0,
        max_position_pct=100.0,
    )
    values.update(overrides)
    return Config(**values)


def parsed_swap_tx(*, owner=WALLET, out_mint=MINT, out_pre=0, out_post=25_000_000,
                   sol_pre=2_000_000_000, sol_post=1_499_995_000, fee=5_000):
    return {
        "transaction": {"message": {"accountKeys": [owner, "Other"]}},
        "meta": {
            "fee": fee,
            "preBalances": [sol_pre, 0],
            "postBalances": [sol_post, 0],
            "preTokenBalances": [{
                "owner": owner, "mint": out_mint,
                "uiTokenAmount": {"amount": str(out_pre), "decimals": 6, "uiAmount": out_pre / 1e6},
            }],
            "postTokenBalances": [{
                "owner": owner, "mint": out_mint,
                "uiTokenAmount": {"amount": str(out_post), "decimals": 6, "uiAmount": out_post / 1e6},
            }],
        },
    }


def test_extract_fill_uses_token_delta_and_native_sol_principal():
    fill = extract_fill(parsed_swap_tx(), WALLET, SOL_MINT, MINT)
    assert fill.complete
    assert fill.actual_out_amount == 25_000_000
    assert fill.out_decimals == 6
    assert fill.actual_in_amount == 500_000_000
    assert fill.in_decimals == 9
    assert fill.fee_lamports == 5_000


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self._payload


class Phase3Session:
    def __init__(self, tx):
        self.tx = tx
    def post(self, url, json=None, headers=None, timeout=None):
        method = (json or {}).get("method")
        if method == "sendTransaction":
            return FakeResponse({"result": "sig-fill"})
        if method == "getSignatureStatuses":
            return FakeResponse({"result": {"value": [{"err": None, "confirmationStatus": "confirmed"}]}})
        if method == "getTransaction":
            return FakeResponse({"result": self.tx})
        return FakeResponse({"swapTransaction": "AA=="})


def install_solders(monkeypatch):
    class FakeKeypair:
        def pubkey(self):
            return WALLET
    class RawTx:
        message = object()
        @classmethod
        def from_bytes(cls, _raw):
            return cls()
    class SignedTx:
        def __init__(self, *_args):
            pass
        def __bytes__(self):
            return b"signed"
    Versioned = type("VersionedTransaction", (), {"from_bytes": RawTx.from_bytes, "__init__": SignedTx.__init__, "__bytes__": SignedTx.__bytes__})
    modules = {
        "solders": types.ModuleType("solders"),
        "solders.keypair": types.ModuleType("solders.keypair"),
        "solders.transaction": types.ModuleType("solders.transaction"),
    }
    modules["solders.keypair"].Keypair = FakeKeypair
    modules["solders.transaction"].VersionedTransaction = Versioned
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeKeypair()


def test_live_swap_attaches_getTransaction_fill(monkeypatch):
    install_solders(monkeypatch)
    session = Phase3Session(parsed_swap_tx())
    executor = SwapExecutor(cfg(), jup=JupiterClient(cfg()), session=session)
    executor._load_keypair = lambda _keypair: types.SimpleNamespace(pubkey=lambda: WALLET)
    result = executor._execute_live({"outAmount": "1"}, 5, 1, input_mint=SOL_MINT, output_mint=MINT)
    assert result.ok
    assert result.status == TransactionStatus.CONFIRMED_SUCCESS
    assert result.tx_signature == "sig-fill"
    assert result.actual_out_amount == 25_000_000
    assert result.actual_in_amount == 500_000_000
    assert result.fee_lamports == 5_000
    assert result.fill_source == "getTransaction"


def test_live_entry_opens_from_confirmed_raw_fill_and_persists_full_record(monkeypatch):
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg()
    bot.portfolio = Portfolio()
    bot.strategy = types.SimpleNamespace(find_candidates=lambda: [types.SimpleNamespace(mint=MINT)])
    bot.screener = types.SimpleNamespace(
        screen=lambda _mint, **kwargs: types.SimpleNamespace(passed=True, price_usd=1.0, symbol="TEST")
    )
    bot.risk = types.SimpleNamespace(position_size_usd=lambda: 5.0)
    bot.executor = types.SimpleNamespace(swap=lambda *args, **kwargs: SwapResult(
        ok=True, simulated=False, in_amount=1, out_amount=1, tx_signature="sig-fill",
        status=TransactionStatus.CONFIRMED_SUCCESS, actual_in_amount=500_000_000,
        actual_out_amount=25_000_000, fee_lamports=5000, in_decimals=9, out_decimals=6,
        fill_source="getTransaction",
    ))
    bot.notifier = types.SimpleNamespace(buy=lambda *args: None)
    bot._usd_to_lamports = lambda _usd: 1
    bot._reconciliation_required = False
    saved = []
    bot._save = lambda: saved.append(True)
    bot._seek_entry()
    pos = bot.portfolio.positions[MINT]
    assert pos.tokens == 25.0
    assert pos.size_usd == 5.0
    assert pos.entry_price == 0.2
    assert pos.entry_signature == "sig-fill"
    assert pos.actual_out_amount == 25_000_000
    assert pos.actual_in_amount == 500_000_000
    assert pos.fee_lamports == 5000
    assert pos.token_decimals == 6
    assert saved == [True]


def test_live_sell_without_settlement_does_not_reduce_position():
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = cfg()
    bot.portfolio = Portfolio()
    bot.portfolio.open(MINT, "TEST", 1.0, 5.0, 25.0, token_decimals=6)
    bot.executor = types.SimpleNamespace(swap=lambda *args, **kwargs: SwapResult(
        ok=False, simulated=False, in_amount=1, out_amount=0,
        status=TransactionStatus.TIMEOUT, error="timeout",
    ))
    bot.risk = types.SimpleNamespace(
        trading_halted=lambda: False,
        record_realized_pnl=lambda pnl: (_ for _ in ()).throw(AssertionError("must not book pnl")),
    )
    bot.notifier = types.SimpleNamespace(sell=lambda *args: (_ for _ in ()).throw(AssertionError("no sell alert")))
    bot._save = lambda: None
    bot._reconciliation_required = False
    bot._sell(MINT, 1.0, 1.0, "stop_loss")
    assert MINT in bot.portfolio.positions
    assert bot.portfolio.positions[MINT].tokens == 25.0
    assert bot._reconciliation_required is True
    assert bot.portfolio.pending_intents[-1]["status"] == "unresolved"


def test_dry_run_sell_still_updates_portfolio_without_chain():
    bot = main_module.TradingBot.__new__(main_module.TradingBot)
    bot.cfg = Config(live_trading=False)
    bot.portfolio = Portfolio()
    bot.portfolio.open(MINT, "TEST", 1.0, 5.0, 25.0)
    bot.risk = types.SimpleNamespace(trading_halted=lambda: False, record_realized_pnl=lambda pnl: None)
    bot.notifier = types.SimpleNamespace(sell=lambda *args: None)
    bot._save = lambda: None
    bot._sell(MINT, 1.0, 1.0, "stop_loss")
    assert MINT not in bot.portfolio.positions
    assert bot.portfolio.pending_intents == []
