"""
Phase-1.8 controller / monitor lifecycle tests.

Verifies the run-control invariants the Railway deployment depends on:

  * the app starts with NO trading cycle running (explicit start required),
  * a second start while running is refused (no duplicate controller thread),
  * stop is idempotent and clears state,
  * the monitor reconciles before it can trade, and a failed reconciliation
    blocks new entries (fail-closed),
  * the run-control endpoints never submit a broker order.

Everything is driven through fakes; no thread is left running and no order is
placed.
"""
import threading
import time

import pytest

pytest.importorskip("fastapi.testclient", reason="requires the real FastAPI")
from fastapi.testclient import TestClient


@pytest.fixture
def appmod(monkeypatch, tmp_path):
    import decision_log
    monkeypatch.setenv("DB_PATH", str(tmp_path / "decisions.db"))
    import importlib
    importlib.reload(decision_log)
    decision_log.init_db()
    import app as app_module
    importlib.reload(app_module)
    return app_module


# ============================================================================
# AgentController: explicit start, no duplicates, clean stop
# ============================================================================

def test_controller_does_not_start_itself(appmod):
    assert appmod.controller.is_running() is False
    assert appmod.controller.status()["running"] is False


def test_controller_refuses_duplicate_start(appmod, monkeypatch):
    started = {"n": 0}
    release = threading.Event()

    def fake_loop():
        started["n"] += 1
        release.wait(timeout=5)

    monkeypatch.setattr(appmod.controller, "_loop", fake_loop)
    monkeypatch.setattr(appmod, "get_symbols", lambda: [])

    assert appmod.controller.start() is True
    time.sleep(0.05)
    # A second start while the first is alive must be refused.
    assert appmod.controller.start() is False
    assert started["n"] == 1

    release.set()
    appmod.controller.stop()
    for _ in range(100):
        if not appmod.controller.is_running():
            break
        time.sleep(0.02)
    assert appmod.controller.is_running() is False


def test_controller_stop_is_idempotent(appmod):
    assert appmod.controller.stop() is False      # never started
    assert appmod.controller.is_running() is False


# ============================================================================
# Monitor: does not start itself
# (startup reconciliation + fail-closed is covered by
#  test_phase2b_paper_readiness.test_monitor_start_reconciliation_failure_blocks_entries)
# ============================================================================

def test_monitor_does_not_start_itself(appmod):
    assert appmod.monitor.is_running() is False


# ============================================================================
# Run-control endpoints: explicit only, no order
# ============================================================================

def test_run_start_endpoint_is_explicit_and_order_free(appmod, monkeypatch):
    calls = {"orders": 0}

    def boom(*a, **k):
        calls["orders"] += 1
        raise AssertionError("run control must not submit an order")

    monkeypatch.setattr(appmod.controller, "start", lambda: True)
    monkeypatch.setattr(appmod.controller, "is_running", lambda: False)

    import broker
    if hasattr(broker, "execute_order"):
        monkeypatch.setattr(broker, "execute_order", boom)

    c = TestClient(appmod.app)
    body = c.post("/api/run/start").json()
    assert "running" in body
    assert calls["orders"] == 0


def test_run_toggle_stops_when_running(appmod, monkeypatch):
    monkeypatch.setattr(appmod.controller, "is_running", lambda: True)
    monkeypatch.setattr(appmod.controller, "stop", lambda: True)
    c = TestClient(appmod.app)
    assert c.post("/api/run/toggle").json() == {"running": False}
