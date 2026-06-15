"""Configuration loading and validation.

This module is also where the *safety lock* lives: even if the operator sets
``LIVE_TRADING=true``, we refuse to arm live trading unless a wallet key is
actually present, and we surface a loud warning. The goal is that going live
is a deliberate, explicit act and never an accident.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

try:  # python-dotenv is optional at import time; nice-to-have for local dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience only
    pass

log = logging.getLogger(__name__)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r, using default %s", name, raw, default)
        return default


def _get_int(name: str, default: int) -> int:
    return int(_get_float(name, default))


def _parse_ladder(raw: str) -> list[tuple[float, float]]:
    """Parse ``"50:0.5,100:0.25"`` into ``[(50.0, 0.5), (100.0, 0.25)]``."""
    rungs: list[tuple[float, float]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        gain_str, frac_str = chunk.split(":")
        rungs.append((float(gain_str), float(frac_str)))
    rungs.sort(key=lambda r: r[0])  # ascending by gain target
    return rungs


def _parse_wallets(raw: str, path: str) -> list[str]:
    """Collect watched wallet addresses from a comma/whitespace-separated env
    value and/or a file (one address per line, ``#`` comments allowed)."""
    seen: set[str] = set()
    wallets: list[str] = []

    def _add(token: str) -> None:
        token = token.strip()
        if token and not token.startswith("#") and token not in seen:
            seen.add(token)
            wallets.append(token)

    for chunk in raw.replace(",", " ").split():
        _add(chunk)
    if path:
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    _add(line)
        except OSError as exc:
            log.warning("could not read SMART_MONEY_WALLETS_FILE %r: %s", path, exc)
    return wallets


@dataclass(frozen=True)
class Config:
    # Safety
    live_trading: bool = False
    wallet_private_key: str = ""
    rpc_url: str = "https://api.mainnet-beta.solana.com"

    # Jupiter swap API. The legacy quote-api.jup.ag/v6 host was deprecated on
    # 2025-10-01; the current free tier is lite-api.jup.ag/swap/v1 (no key).
    # Holders of a Jupiter API key should point this at https://api.jup.ag/swap/v1
    # and set jupiter_api_key.
    jupiter_base_url: str = "https://lite-api.jup.ag/swap/v1"
    jupiter_api_key: str = ""

    # Bankroll & sizing
    bankroll_usd: float = 100.0
    max_position_pct: float = 2.0
    max_open_positions: int = 3

    # Risk
    stop_loss_pct: float = 15.0
    daily_loss_limit_pct: float = 10.0
    take_profit_ladder: list[tuple[float, float]] = field(
        default_factory=lambda: [(50.0, 0.5), (100.0, 0.25), (300.0, 0.25)]
    )
    max_slippage_bps: int = 150

    # Token safety screen
    min_liquidity_usd: float = 20000.0
    require_liquidity_locked: bool = True
    require_mint_revoked: bool = True
    require_freeze_revoked: bool = True
    max_top_holder_pct: float = 25.0

    # Loop
    poll_interval_seconds: int = 15

    # Strategy / candidate source: "boosted" (DexScreener) or "smart_money".
    strategy: str = "boosted"
    smart_money_wallets: list[str] = field(default_factory=list)
    smart_money_surface_existing: bool = False

    # Alerts (all optional; off unless configured)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    discord_webhook_url: str = ""

    # Persistence
    state_file: str = "state.json"

    @property
    def is_armed(self) -> bool:
        """True only when the bot is genuinely cleared to spend real money."""
        return self.live_trading and bool(self.wallet_private_key)

    def max_position_usd(self) -> float:
        return self.bankroll_usd * (self.max_position_pct / 100.0)

    def daily_loss_limit_usd(self) -> float:
        return self.bankroll_usd * (self.daily_loss_limit_pct / 100.0)


def load_config() -> Config:
    ladder_raw = os.getenv("TAKE_PROFIT_LADDER", "50:0.5,100:0.25,300:0.25")
    cfg = Config(
        live_trading=_get_bool("LIVE_TRADING", False),
        wallet_private_key=os.getenv("WALLET_PRIVATE_KEY", "").strip(),
        rpc_url=os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com").strip(),
        jupiter_base_url=os.getenv(
            "JUPITER_BASE_URL", "https://lite-api.jup.ag/swap/v1"
        ).strip().rstrip("/"),
        jupiter_api_key=os.getenv("JUPITER_API_KEY", "").strip(),
        bankroll_usd=_get_float("BANKROLL_USD", 100.0),
        max_position_pct=_get_float("MAX_POSITION_PCT", 2.0),
        max_open_positions=_get_int("MAX_OPEN_POSITIONS", 3),
        stop_loss_pct=_get_float("STOP_LOSS_PCT", 15.0),
        daily_loss_limit_pct=_get_float("DAILY_LOSS_LIMIT_PCT", 10.0),
        take_profit_ladder=_parse_ladder(ladder_raw),
        max_slippage_bps=_get_int("MAX_SLIPPAGE_BPS", 150),
        min_liquidity_usd=_get_float("MIN_LIQUIDITY_USD", 20000.0),
        require_liquidity_locked=_get_bool("REQUIRE_LIQUIDITY_LOCKED", True),
        require_mint_revoked=_get_bool("REQUIRE_MINT_REVOKED", True),
        require_freeze_revoked=_get_bool("REQUIRE_FREEZE_REVOKED", True),
        max_top_holder_pct=_get_float("MAX_TOP_HOLDER_PCT", 25.0),
        poll_interval_seconds=_get_int("POLL_INTERVAL_SECONDS", 15),
        strategy=os.getenv("STRATEGY", "boosted").strip().lower(),
        smart_money_wallets=_parse_wallets(
            os.getenv("SMART_MONEY_WALLETS", ""),
            os.getenv("SMART_MONEY_WALLETS_FILE", "").strip(),
        ),
        smart_money_surface_existing=_get_bool("SMART_MONEY_SURFACE_EXISTING", False),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        state_file=os.getenv("STATE_FILE", "state.json").strip(),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    problems = []
    if cfg.max_position_pct <= 0 or cfg.max_position_pct > 100:
        problems.append("MAX_POSITION_PCT must be in (0, 100]")
    if cfg.stop_loss_pct <= 0 or cfg.stop_loss_pct >= 100:
        problems.append("STOP_LOSS_PCT must be in (0, 100)")
    if cfg.bankroll_usd <= 0:
        problems.append("BANKROLL_USD must be > 0")
    ladder_total = sum(frac for _, frac in cfg.take_profit_ladder)
    if ladder_total > 1.0 + 1e-9:
        problems.append(
            f"TAKE_PROFIT_LADDER sell fractions sum to {ladder_total:.2f} (> 1.0)"
        )
    if problems:
        raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(problems))

    if cfg.live_trading and not cfg.wallet_private_key:
        log.warning(
            "LIVE_TRADING=true but no WALLET_PRIVATE_KEY provided — the safety "
            "lock is keeping the bot in DRY-RUN mode. No real trades will occur."
        )
