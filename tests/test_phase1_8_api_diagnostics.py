"""
Phase-1.8 read-only API + diagnostics regression tests.

Coverage for the operator-facing read endpoints, which previously had none:

  * /api/status            — FAIL-CLOSED shape when the broker is unreachable
  * /api/decisions         — persisted decisions, newest first
  * /api/equity            — equity snapshots, oldest -> newest
  * /api/diagnostics       — read-only classification of the latest decision
                             per symbol (invalid data vs a legitimate gate)

Every test uses FastAPI's TestClient against the real ``app`` with a temporary
decision DB. No controller is started, no monitor tick runs, no order is
submitted, and no network call is made.
"""
import json

import pytest

pytest.importorskip("fastapi", reason="requires FastAPI")
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    import decision_log
    monkeypatch.setenv("DB_PATH", str(tmp_path / "decisions.db"))
    import importlib
    importlib.reload(decision_log)
    decision_log.init_db()

    import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), decision_log


def _decision(decision_log, symbol, stage, signal="HOLD", reason="r",
              final="HOLD", score=None, threshold=None):
    """Persist one decision row shaped exactly like main.run_cycle produces.

    ``log_decision`` reads the analyst fields from ``analyst_report`` (not from
    top-level keys), so the fixture must nest them there.
    """
    decision_log.log_decision({
        "symbol": symbol,
        "risk_report": {"approved": False, "veto": False, "reason": "analyst HOLD"},
        "final_decision": final,
        "consensus_reached": False,
        "order_details": None,
        "market_data": {"atr14": 1.0},
        "analyst_report": {
            "signal": signal,
            "confidence": 0.0,
            "reason": reason,
            "signal_score": score,
            "diagnostic": {
                "stage": stage, "signal_score": score, "threshold": threshold,
                "market_regime": "SIDEWAYS", "llm_called": False},
        },
    }, None)


# ============================================================================
# /api/status — FAIL-CLOSED
# ============================================================================

def test_status_is_fail_closed_without_credentials(client, monkeypatch):
    c, _ = client
    import app as app_module
    monkeypatch.setattr(app_module, "get_account_snapshot",
                        lambda: {"available": False, "reason": "missing_alpaca_credentials"})
    body = c.get("/api/status").json()
    assert body["available"] is False
    assert body["equity"] is None and body["buying_power"] is None
    assert body["open_positions"] is None and body["trades_today"] is None
    assert "missing_alpaca_credentials" in body["reason"]


def test_status_reports_controller_state(client):
    c, _ = client
    body = c.get("/api/status").json()
    assert body["running"] is False
    assert body["current_symbol"] is None


# ============================================================================
# /api/decisions, /api/equity
# ============================================================================

def test_decisions_returns_persisted_rows_newest_first(client):
    c, dl = client
    _decision(dl, "AAA", "input")
    _decision(dl, "BBB", "signal_score", score=13, threshold=70)
    rows = c.get("/api/decisions").json()
    assert [r["symbol"] for r in rows] == ["BBB", "AAA"]


def test_equity_history_is_chronological(client):
    c, dl = client
    dl.log_equity(1000.0, 0)
    dl.log_equity(1100.0, 1)
    rows = c.get("/api/equity").json()
    assert [r["equity"] for r in rows] == [1000.0, 1100.0]


# ============================================================================
# /api/diagnostics — read-only, distinguishes invalid data from a real gate
# ============================================================================

def test_diagnostics_classifies_invalid_data_hold(client):
    c, dl = client
    _decision(dl, "ASML", "input", reason="no valid indicators")
    body = c.get("/api/diagnostics").json()
    row = next(r for r in body["latest_by_symbol"] if r["symbol"] == "ASML")
    assert row["data_valid"] is False
    assert row["hold_reason"] == "invalid_data"
    assert row["diagnostic_stage"] == "input"


def test_diagnostics_classifies_threshold_hold(client):
    c, dl = client
    _decision(dl, "CRM", "signal_score", reason="signal score below threshold (57 < 70)",
              score=57, threshold=70)
    body = c.get("/api/diagnostics").json()
    row = next(r for r in body["latest_by_symbol"] if r["symbol"] == "CRM")
    assert row["data_valid"] is True
    assert row["hold_reason"] == "below_threshold"
    assert row["signal_score"] == 57 and row["threshold"] == 70


def test_diagnostics_keeps_only_the_latest_row_per_symbol(client):
    c, dl = client
    _decision(dl, "ASML", "input")                       # older
    _decision(dl, "ASML", "signal_score", score=68, threshold=70)  # newer
    body = c.get("/api/diagnostics").json()
    rows = [r for r in body["latest_by_symbol"] if r["symbol"] == "ASML"]
    assert len(rows) == 1
    assert rows[0]["diagnostic_stage"] == "signal_score"


def test_diagnostics_is_read_only_and_starts_nothing(client, monkeypatch):
    c, _ = client
    import app as app_module
    started = {"controller": 0, "monitor": 0}
    monkeypatch.setattr(app_module.controller, "start",
                        lambda: started.__setitem__("controller", started["controller"] + 1))
    monkeypatch.setattr(app_module.monitor, "start",
                        lambda: started.__setitem__("monitor", started["monitor"] + 1))
    body = c.get("/api/diagnostics").json()
    assert body["controller_running"] is False
    assert body["monitor_running"] is False
    assert started == {"controller": 0, "monitor": 0}
    assert "read-only" in body["note"]


def test_diagnostics_tolerates_malformed_diagnostic_json(client):
    c, dl = client
    _decision(dl, "AAA", "input")
    # Corrupt the stored diagnostic JSON directly.
    with dl._conn() as conn:
        conn.execute("UPDATE decisions SET analyst_diagnostic='{not json' WHERE symbol='AAA'")
        conn.commit()
    body = c.get("/api/diagnostics").json()
    row = next(r for r in body["latest_by_symbol"] if r["symbol"] == "AAA")
    assert row["diagnostic_stage"] is None
    assert row["hold_reason"] == "analyst_or_risk_gate"


def test_diagnostics_empty_database_is_safe(client):
    c, _ = client
    body = c.get("/api/diagnostics").json()
    assert body["latest_by_symbol"] == []


def test_run_endpoints_are_not_touched_by_diagnostics(client, monkeypatch):
    """Guard: the diagnostics route must not appear to be a run-control route."""
    c, _ = client
    routes = {r.path for r in c.app.routes if hasattr(r, "path")}
    assert "/api/diagnostics" in routes
    assert "/api/run/start" in routes and "/api/diagnostics" != "/api/run/start"


# ============================================================================
# Security review: the diagnostics endpoint must not widen exposure
# ============================================================================

DIAGNOSTIC_KEYS = {
    "symbol", "timestamp", "final_decision", "analyst_signal", "analyst_reason",
    "diagnostic_stage", "signal_score", "threshold", "market_regime",
    "llm_called", "data_valid", "hold_reason",
}
# Fields the pre-existing /api/decisions already returns via SELECT *; the
# diagnostics endpoint must NOT surface the heavier / more sensitive ones.
WITHHELD_FIELDS = {"market_data", "order_details", "broker_result",
                   "risk_report", "analyst_confidence"}


def test_diagnostics_exposes_exactly_the_documented_keys(client):
    c, dl = client
    _decision(dl, "AAA", "signal_score", score=1, threshold=70)
    row = c.get("/api/diagnostics").json()["latest_by_symbol"][0]
    assert set(row) == DIAGNOSTIC_KEYS, (
        f"unexpected fields: {set(row) ^ DIAGNOSTIC_KEYS}")


def test_diagnostics_withholds_order_and_market_payloads(client):
    c, dl = client
    decision_log = dl
    decision_log.log_decision({
        "symbol": "AAA",
        "risk_report": {"approved": False, "veto": False, "reason": "analyst HOLD"},
        "final_decision": "HOLD",
        "consensus_reached": False,
        "order_details": {"symbol": "AAA", "qty": 10, "stop_loss": 9.5},
        "market_data": {"atr14": 1.0, "last_close": 100.0},
        "analyst_report": {"signal": "HOLD", "confidence": 0.0, "reason": "r",
                           "diagnostic": {"stage": "input"}},
    }, {"order_id": "abc-123", "status": "accepted"})
    body = c.get("/api/diagnostics").json()
    blob = json.dumps(body)
    for field in WITHHELD_FIELDS:
        assert field not in body["latest_by_symbol"][0]
    # Values from those payloads must not appear either.
    assert "abc-123" not in blob
    assert "stop_loss" not in blob


def test_diagnostics_leaks_no_credentials_or_env_values(client, monkeypatch):
    c, dl = client
    sentinel = "SUPER_SECRET_SENTINEL_VALUE_12345"
    monkeypatch.setenv("ALPACA_API_KEY", sentinel)
    monkeypatch.setenv("ALPACA_SECRET_KEY", sentinel)
    monkeypatch.setenv("FINNHUB_API_KEY", sentinel)
    _decision(dl, "AAA", "input")
    body = c.get("/api/diagnostics").json()
    assert sentinel not in json.dumps(body)


def test_diagnostics_never_reports_internal_paths(client):
    c, dl = client
    _decision(dl, "AAA", "input")
    blob = json.dumps(c.get("/api/diagnostics").json())
    for marker in ("/workspace/", "site-packages", "Traceback", "sqlite3",
                   "ALPACA_", "SECRET"):
        assert marker not in blob, f"diagnostics leaked {marker!r}"


def test_diagnostics_is_a_get_only_route(client):
    c, _ = client
    routes = {(r.path, tuple(sorted(getattr(r, "methods", []) or [])))
              for r in c.app.routes if hasattr(r, "path")}
    assert ("/api/diagnostics", ("GET",)) in routes, routes
