"""Tests for the Jupiter client URL/auth wiring.

The legacy quote-api.jup.ag/v6 host was deprecated on 2025-10-01; these guard
against regressing back to it and check that the API key header is only sent
when a key is configured.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bot.config import Config, load_config
from bot.jupiter import JupiterClient


def test_default_endpoints_use_current_lite_api():
    jup = JupiterClient(Config())
    assert jup.quote_url == "https://lite-api.jup.ag/swap/v1/quote"
    assert jup.swap_url == "https://lite-api.jup.ag/swap/v1/swap"
    # the deprecated host must not reappear
    assert "quote-api.jup.ag" not in jup.quote_url
    assert "quote-api.jup.ag" not in jup.swap_url


def test_no_api_key_header_without_key():
    assert JupiterClient(Config())._headers() == {}


def test_api_key_header_when_configured():
    jup = JupiterClient(Config(jupiter_api_key="secret"))
    assert jup._headers() == {"x-api-key": "secret"}


def test_base_url_is_configurable_and_trailing_slash_trimmed(monkeypatch):
    monkeypatch.setenv("JUPITER_BASE_URL", "https://api.jup.ag/swap/v1/")
    monkeypatch.setenv("JUPITER_API_KEY", "k")
    cfg = load_config()
    jup = JupiterClient(cfg)
    assert jup.quote_url == "https://api.jup.ag/swap/v1/quote"
    assert jup.swap_url == "https://api.jup.ag/swap/v1/swap"
