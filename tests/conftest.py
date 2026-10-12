"""Shared pytest configuration.

Direct-route swap policy (``bot.swap_policy``): pre-existing tests drive
``SwapExecutor`` with SYNTHETIC quotes (no ``routePlan``) and placeholder
unsigned transactions that predate the policy and are not shaped like a real
Jupiter build. For those tests only, the two policy checks imported into
``bot.jupiter`` are replaced by pass-throughs so they keep testing what they
were written for (sizing, intents, durable signatures, settlement, ...).

Tests marked ``real_swap_policy`` (``tests/test_direct_route_policy.py`` and
the converted buy-lifecycle tests) run the REAL policy. Production code is
never patched outside a test process.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_swap_policy: run the real direct-route quote gate and pre-sign "
        "transaction validator (not the legacy-fixture pass-through)",
    )


@pytest.fixture(autouse=True)
def _legacy_fixture_swap_policy(request, monkeypatch):
    if request.node.get_closest_marker("real_swap_policy"):
        yield
        return
    import bot.jupiter as jupiter
    monkeypatch.setattr(jupiter, "route_quote_error", lambda *a, **k: None)
    monkeypatch.setattr(jupiter, "swap_transaction_error", lambda *a, **k: None)
    yield
