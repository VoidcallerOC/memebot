"""Balance-proof fix (independent review @ 4d5fb73).

An expired, unconfirmed buy intent is resolved ``not_executed`` (failed ->
entries unblock -> a new buy) only when the wallet provably holds none of the
mint. ``_accounts_to_amounts`` used ``float(uiAmount or 0)``: a null
``uiAmount`` read as zero and an entry without it was skipped (absent ->
zero), so an executed-but-unseen buy could be failed and bought again.

Now the exact raw ``tokenAmount.amount`` string is authoritative; any
malformed / incomplete entry or unavailable response makes the whole snapshot
UNKNOWN (None), never zero, and the buy resolver proves absence only from the
raw amounts of a fully readable snapshot.

Real here: ``TradingBot`` buy lifecycle and resolver, ``fetch_wallet_token_balances``
/ ``_accounts_to_raw`` / ``rpc_call``, ``SwapExecutor`` signing (throwaway
key). Fake: the RPC/HTTP boundary (only the getTokenAccountsByOwner response
is overridden per case).
"""
from __future__ import annotations

import pytest

import bot.main as main_module
from bot.chain import SPL_TOKEN_PROGRAM, _accounts_to_amounts, _accounts_to_raw
import tests.test_p1_buy_prebroadcast_signature as B
from tests.test_p1_buy_prebroadcast_signature import bw  # noqa: F401  (fixture)

EXPIRY = main_module.BUY_INTENT_EXPIRY_SECONDS
RAW = "4200000000"


def acct(mint=B.MINT, **token_amount):
    return {"account": {"data": {"parsed": {"info": {"mint": mint, "tokenAmount": token_amount}}}}}


def install(bw, responder):
    """Override ONLY getTokenAccountsByOwner on the fake RPC."""
    real = bw.post

    def post(url, json=None, headers=None, timeout=None):
        if url == B.RPC and (json or {}).get("method") == "getTokenAccountsByOwner":
            bw.rpc.append("getTokenAccountsByOwner")
            return responder(json["params"])
        return real(url, json=json, headers=headers, timeout=timeout)
    bw.post = post


def spl_only(value):
    """``value`` for SPL Token, an empty (valid) list for Token-2022."""
    def responder(params):
        if params[1]["programId"] == SPL_TOKEN_PROGRAM:
            return B.Resp({"result": {"value": value}})
        return B.Resp({"result": {"value": []}})
    return responder


def dropped_then_expired(bw):
    """A buy whose send never becomes visible (unknown signature status),
    then past blockhash expiry: the resolver must consult the wallet."""
    bot = B.start(bw)
    bw.plan = ["drop"]
    bot._tick()
    (sig,) = bw.sends
    (intent,) = bot.portfolio.buy_intents
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    bw.now += EXPIRY + 1
    return bot, sig


def assert_ambiguous(bw, bot, sig):
    scans = len(bot.entry_scans)
    for _ in range(3):
        bot._tick()
    state = B.persisted(bw)
    (intent,) = B.buys_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bw.sends == [sig] and bw.builds == 1            # no rebuild / rebroadcast
    assert len(bot.entry_scans) == scans                    # entries blocked
    assert bot.portfolio.positions == {} and state["portfolio"]["positions"] == []
    again = B.boot(bw)                                      # restart: same answer
    B.run(again, 3)
    state = B.persisted(bw)
    (intent,) = B.buys_of(state)
    assert intent["status"] == "unresolved" and intent["tx_signature"] == sig
    assert bw.sends == [sig] and bw.builds == 1 and again.entry_scans == []
    assert state["portfolio"]["positions"] == []


AMBIGUOUS = [
    pytest.param(spl_only([acct(amount=RAW, decimals=6, uiAmount=None)]), id="nonzero-raw-uiAmount-null"),
    pytest.param(spl_only([acct(amount=RAW, decimals=6)]), id="nonzero-raw-uiAmount-omitted"),
    pytest.param(spl_only([acct(decimals=6, uiAmount=0)]), id="raw-missing-uiAmount-zero"),
    pytest.param(spl_only([acct(amount=None, decimals=6, uiAmount=0)]), id="raw-null"),
    pytest.param(spl_only([acct(amount="abc", decimals=6, uiAmount=0)]), id="raw-not-integer"),
    pytest.param(spl_only([acct(amount="-1", decimals=6, uiAmount=0)]), id="raw-negative"),
    pytest.param(spl_only([acct(amount="1.5", decimals=6, uiAmount=0)]), id="raw-decimal-string"),
    pytest.param(spl_only([acct(amount=4200000000, decimals=6, uiAmount=0)]), id="raw-int-not-string"),
    pytest.param(spl_only([acct(amount=RAW, uiAmount=None)]), id="decimals-missing"),
    pytest.param(spl_only([{"account": {"data": "base64-blob"}}]), id="account-data-not-parsed"),
    pytest.param(spl_only([{"account": None}]), id="account-null"),
    pytest.param(spl_only([acct(mint=None, amount=RAW, decimals=6)]), id="mint-missing"),
    pytest.param(spl_only("not-a-list"), id="value-not-list"),
    pytest.param(lambda params: B.Resp({"result": {}}), id="value-missing"),
    pytest.param(lambda params: B.Resp({"result": None}), id="result-null"),
    pytest.param(lambda params: B.Resp({"error": {"code": -32005, "message": "node behind"}}), id="rpc-error"),
    pytest.param(lambda params: (_ for _ in ()).throw(ConnectionError("rpc down")), id="rpc-exception"),
    pytest.param(lambda params: B.Resp({"result": {"value": []}})
                 if params[1]["programId"] == SPL_TOKEN_PROGRAM
                 else (_ for _ in ()).throw(TimeoutError("2022 timed out")), id="partial-second-program-fails"),
]


@pytest.mark.parametrize("responder", AMBIGUOUS)
def test_expired_unknown_status_with_unprovable_balance_stays_unresolved(bw, responder):
    bot, sig = dropped_then_expired(bw)
    install(bw, responder)
    assert_ambiguous(bw, bot, sig)


@pytest.mark.parametrize("responder", [
    pytest.param(spl_only([]), id="no-accounts"),
    pytest.param(spl_only([acct(amount="0", decimals=6, uiAmount=0.0)]), id="known-zero-raw"),
    pytest.param(spl_only([acct(amount="0", decimals=6, uiAmount=None)]), id="known-zero-raw-uiAmount-null"),
    pytest.param(spl_only([acct(mint="OtherMint111", amount="5", decimals=0)]), id="only-other-mints"),
])
def test_known_zero_with_valid_evidence_resolves_not_executed(bw, responder):
    bot, sig = dropped_then_expired(bw)
    install(bw, responder)
    bot._tick()
    state = B.persisted(bw)
    first = B.buys_of(state)[0]
    assert first["status"] == "failed" and first["tx_signature"] == sig
    assert "holds none" in first["resolution"]


def test_wallet_holding_the_token_stays_unresolved(bw):
    bot, sig = dropped_then_expired(bw)
    install(bw, spl_only([acct(amount=RAW, decimals=6, uiAmount=4200.0)]))
    assert_ambiguous(bw, bot, sig)


# -- parser unit checks -----------------------------------------------------------

def test_parser_uses_exact_raw_amount_never_uiAmount():
    assert _accounts_to_raw([acct(amount=RAW, decimals=6, uiAmount=None)]) == {B.MINT: (4_200_000_000, 6)}
    assert _accounts_to_raw([acct(amount=RAW, decimals=6, uiAmount=0)]) == {B.MINT: (4_200_000_000, 6)}
    assert _accounts_to_amounts([acct(amount=RAW, decimals=6)]) == {B.MINT: 4200.0}
    assert _accounts_to_raw([acct(amount="1", decimals=6), acct(amount="2", decimals=6)]) == {B.MINT: (3, 6)}
    assert _accounts_to_raw([]) == {}


@pytest.mark.parametrize("value", [
    None, "x", [None], [acct(amount="1", decimals=6), acct(amount="1", decimals=9)],
    [acct(amount="١٢", decimals=6)], [acct(amount="1", decimals=True)], [acct(amount="1", decimals=-1)],
])
def test_parser_unreadable_is_none_not_empty(value):
    assert _accounts_to_raw(value) is None and _accounts_to_amounts(value) is None
