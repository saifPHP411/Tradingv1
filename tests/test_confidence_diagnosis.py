"""
Confidence and signal-score representation (Phase 1.7.1, Task 2 -> fix).

The earlier revision of this file proved the defect. This revision locks in
the FIXED contract. The defect was:

  * Three analyst paths returned HOLD *before the LLM was consulted* and wrote
    ``confidence = 0.0`` as a placeholder: no valid indicators, signal score
    below threshold, and BEAR regime. The stored 0.0 was therefore not a
    measurement at all.
  * The retry loop accepted a reply only if ``confidence > 0``, so a genuine,
    well-formed LLM HOLD at confidence 0.0 was retried and recorded as
    ``invalid_response`` -- identical to a malformed reply.
  * The template rendered ``analyst_confidence`` unconditionally as a
    percentage and never read ``analyst_diagnostic``, so a placeholder showed
    as a measured "0%".
  * The technical ``signal_score`` had no column, so the UI could not show it.

The fixed contract, asserted below:

  * ``confidence`` is ``None`` when not measured, and a float in [0, 1] when it
    is. ``None`` and ``0.0`` are different: ``0.0`` is a real reading.
  * A well-formed LLM reply is accepted at any confidence, including 0.0. Only
    an unparseable reply is retried.
  * ``signal_score`` is persisted as its own column and returned as its own
    field, never conflated with ``confidence``.
  * The API distinguishes ``measured`` / ``placeholder`` / ``unavailable`` /
    ``unknown`` and never presents a missing value as 0.
  * Legacy rows with a numeric value and no evidence are ``unknown``; they are
    NOT relabelled as unavailable.

Nothing here changes a threshold, a risk rule, or an entry condition. A
BUY/SELL at zero (or missing) confidence is still vetoed downstream. No
network, no credentials, no broker orders.

Known remaining limitation (documented, NOT silently changed): the exit path in
``monitor.py`` has its own confidence contract. Its retry loop returns on the
first parseable reply and defaults a missing confidence to ``0`` in the
returned dict, and that value gates CLOSE_NOW against
``llm_exit_min_confidence``. It writes to ``exit_decisions``, never to
``analyst_confidence``, and it does not surface in the decisions API or the
dashboard, so it is outside the scope of this fix. Changing it would alter exit
behaviour, which this task must not do without authorisation.
"""
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("DB_PATH", os.path.join(tempfile.mkdtemp(), "conf_diag.db"))

import agents  # noqa: E402
import decision_log  # noqa: E402

_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "dashboard.html"

_INDICATORS = {
    "atr14": 1.0, "last_close": 100.0, "ema20": 100.0, "ema50": 100.0,
    "sma50": 101.0, "rsi14": 50.0, "adx": 10.0, "macd_hist": 0.0,
    "bb_pct_b": 0.5, "bb_bandwidth": 2.0, "stoch_k": 50.0, "stoch_d": 50.0,
    "volume_ratio": 1.0, "momentum_5d_pct": 0.0, "momentum_10d_pct": 0.0,
}


def _settings(**over):
    s = {"regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
         "min_confidence": 0.7}
    s.update(over)
    return s


def _strong_score():
    return {"score": 85, "trend": 25, "momentum": 20, "rsi": 15, "bb": 8,
            "volume": 14, "news": 3}


class _FakeLLM:
    """Stands in for the external model only; every other path is real."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, *a, **k):
        self.calls += 1

        class _R:
            content = self.payload
        return _R()


def _fresh_db(monkeypatch, name="d.db"):
    db = Path(tempfile.mkdtemp()) / name
    monkeypatch.setattr(decision_log, "DB", db)
    decision_log.init_db()
    return db


# ---------------------------------------------------------------------------
# Required case 1: an early HOLD that never consults the LLM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,state,extra,stage", [
    ("no valid indicators",
     {"symbol": "AAA", "market_data": {}, "memory_context": ""}, {}, "input"),
    ("signal score below threshold",
     {"symbol": "AAA", "market_data": dict(_INDICATORS), "memory_context": ""},
     {"score": 12}, "signal_score"),
])
def test_early_hold_without_llm_reports_unavailable_confidence(
        monkeypatch, label, state, extra, stage):
    """A HOLD decided before the LLM runs reports confidence None, not 0.0.

    ``diagnostic.llm_called`` is False, which is the evidence that the value is
    not a measurement.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": extra.get("score", 0), "trend": 0, "momentum": 0,
        "rsi": 0, "bb": 0, "volume": 0, "news": 0,
    })
    monkeypatch.setattr(agents, "llm", _FakeLLM('{"signal":"BUY","confidence":0.9}'))

    report = agents.analyst_agent(state)["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] is None
    assert report["diagnostic"]["llm_called"] is False
    assert report["diagnostic"]["stage"] == stage
    assert agents.llm.calls == 0, "the LLM must not be consulted on this path"


def test_bear_gate_hold_reports_unavailable_confidence(monkeypatch):
    """The BEAR-regime HOLD also returns before the LLM, with confidence None."""
    monkeypatch.setattr(agents, "get_all_settings",
                        lambda: _settings(regime_enabled=1))
    monkeypatch.setattr(agents, "get_market_regime", lambda: {"regime": "BEAR"})
    monkeypatch.setattr(agents, "get_news_sentiment", lambda _s: None)
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    monkeypatch.setattr(agents, "llm", _FakeLLM('{"signal":"BUY","confidence":0.9}'))

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] is None
    assert report["diagnostic"]["llm_called"] is False
    assert report["diagnostic"]["stage"] == "regime"
    assert agents.llm.calls == 0


# ---------------------------------------------------------------------------
# Required case 2: a valid LLM HOLD at zero confidence
# ---------------------------------------------------------------------------

def test_valid_llm_hold_at_zero_confidence_is_accepted(monkeypatch):
    """A well-formed HOLD at confidence 0.0 is a valid reply, accepted once.

    Zero is a measured answer meaning "no confidence". It must not be retried
    and must not be reported as an invalid response.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    llm = _FakeLLM('{"signal":"HOLD","confidence":0.0,"reason":"unclear"}')
    monkeypatch.setattr(agents, "llm", llm)

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] == 0.0, "0.0 is a measured value, not None"
    assert report["confidence"] is not None
    assert report["diagnostic"]["llm_status"] == "success"
    assert report["diagnostic"]["llm_attempts"] == 1
    assert report["diagnostic"]["confidence_available"] is True
    assert llm.calls == 1, "a valid reply must not be retried"


def test_llm_confidence_is_distinct_from_signal_score(monkeypatch):
    """confidence and signal_score stay separate quantities in the report."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    monkeypatch.setattr(agents, "llm", _FakeLLM(
        '{"signal":"BUY","confidence":0.55,"reason":"ok"}'))

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal_score"] == 85
    assert report["confidence"] == pytest.approx(0.55)
    assert report["signal_score"] != report["confidence"]


# ---------------------------------------------------------------------------
# Required case 3: a genuinely invalid LLM reply
# ---------------------------------------------------------------------------

def test_invalid_llm_reply_is_retried_and_marked_invalid(monkeypatch):
    """Unparseable output is retried, then recorded as an invalid response.

    The final confidence is None -- "not measured" -- which is now distinct
    from a valid reply at 0.0.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    llm = _FakeLLM("definitely not json")
    monkeypatch.setattr(agents, "llm", llm)

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] is None
    assert report["diagnostic"]["llm_status"] == "invalid_response"
    assert report["diagnostic"]["llm_attempts"] == 2
    assert report["diagnostic"]["confidence_available"] is False
    assert llm.calls == 2


def test_llm_exception_reports_unavailable_confidence(monkeypatch):
    """A raised exception yields confidence None and no leaked error text."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())

    class BoomLLM:
        def invoke(self, *a, **k):
            raise RuntimeError("Authorization: Bearer secret-value")

    monkeypatch.setattr(agents, "llm", BoomLLM())
    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["confidence"] is None
    assert report["diagnostic"]["llm_status"] == "error"
    assert report["diagnostic"]["confidence_available"] is False
    assert "secret-value" not in str(report)


def test_parser_distinguishes_invalid_from_zero_confidence():
    """The parser returns None for junk, and 0.0 for an explicit zero."""
    assert agents._parse_llm_json("not json") is None
    assert agents._parse_llm_json("[]") is None
    parsed = agents._parse_llm_json('{"signal":"HOLD","confidence":0.0}')
    assert parsed is not None
    assert parsed["confidence"] == 0.0
    missing = agents._parse_llm_json('{"signal":"HOLD"}')
    assert missing is not None and missing["confidence"] is None


# ---------------------------------------------------------------------------
# Required case 4: persistence and retrieval without conflation
# ---------------------------------------------------------------------------

def test_signal_score_and_confidence_persist_separately(monkeypatch):
    """Both quantities round-trip through the DB without being mixed."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "BUY", "confidence": 0.8,
                           "reason": "trend", "signal_score": 85,
                           "diagnostic": {"stage": "llm", "llm_called": True,
                                          "signal_score": 85}},
        "risk_report": {"approved": True, "veto": False, "reason": "ok"},
        "final_decision": "EXECUTE_BUY", "consensus_reached": True,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence"] == pytest.approx(0.8)
    assert row["analyst_signal_score"] == pytest.approx(85)
    assert row["analyst_confidence"] != row["analyst_signal_score"]
    assert row["analyst_confidence_source"] == "measured"


def test_missing_confidence_is_stored_as_null_not_zero(monkeypatch):
    """An unmeasured confidence is NULL in the DB, never coerced to 0.0."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": None,
                           "reason": "signal score below threshold",
                           "signal_score": 12,
                           "diagnostic": {"stage": "signal_score",
                                          "llm_called": False,
                                          "signal_score": 12}},
        "risk_report": {"approved": False, "veto": False, "reason": "analyst HOLD"},
        "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence"] is None
    assert row["analyst_confidence_source"] == "placeholder"
    assert row["analyst_confidence_display"] is None
    assert row["analyst_signal_score"] == pytest.approx(12)


def test_signal_score_falls_back_to_diagnostic(monkeypatch):
    """If signal_score is absent at the top level, the diagnostic supplies it."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": None,
                           "reason": "x",
                           "diagnostic": {"stage": "signal_score",
                                          "llm_called": False,
                                          "signal_score": 44}},
        "risk_report": {}, "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_signal_score"] == pytest.approx(44)


# ---------------------------------------------------------------------------
# Required case 5: missing values in the API and the UI
# ---------------------------------------------------------------------------

def test_api_marks_placeholder_confidence_and_exposes_score(monkeypatch):
    """The API tells a placeholder apart from a measurement, and shows score."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": None,
                           "reason": "signal score below threshold (12 < 70)",
                           "signal_score": 12,
                           "diagnostic": {"stage": "signal_score",
                                          "llm_called": False,
                                          "signal_score": 12}},
        "risk_report": {"approved": False, "veto": False, "reason": "analyst HOLD"},
        "final_decision": "HOLD", "consensus_reached": False, "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence_available"] is False
    assert row["analyst_confidence_display"] is None
    assert row["analyst_llm_called"] is False
    assert row["analyst_signal_score"] == pytest.approx(12)
    # The original keys are untouched, so existing clients keep working.
    assert row["analyst_signal"] == "HOLD"
    assert row["final_decision"] == "HOLD"


def test_measured_zero_is_reported_as_measured_zero(monkeypatch):
    """A genuine 0.0 from the LLM is 'measured' and displayed as 0, not hidden."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": 0.0, "reason": "unclear",
                           "signal_score": 85,
                           "diagnostic": {"stage": "llm", "llm_called": True,
                                          "signal_score": 85}},
        "risk_report": {}, "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence_source"] == "measured"
    assert row["analyst_confidence_display"] == 0.0
    assert row["analyst_confidence_available"] is True


def test_template_shows_confidence_and_score_separately():
    """The dashboard reads the display field and the score field, and never
    multiplies the raw column (which would show NULL as 0%)."""
    tpl = _TEMPLATE.read_text(encoding="utf-8")
    assert "d.analyst_confidence_display" in tpl
    assert "d.analyst_signal_score" in tpl
    assert "d.analyst_gate_stage" in tpl
    assert "ثقة غير متاحة" in tpl
    assert "d.analyst_confidence*100" not in tpl


def test_api_exposes_gate_stage_for_a_pre_llm_hold(monkeypatch):
    """The gate stage is derived server-side, so the UI need not parse JSON."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": None, "reason": "bear",
                           "signal_score": 80,
                           "diagnostic": {"stage": "regime", "llm_called": False,
                                          "signal_score": 80}},
        "risk_report": {}, "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_gate_stage"] == "regime"
    assert row["analyst_llm_called"] is False


def test_api_does_not_report_a_gate_stage_when_the_llm_was_called(monkeypatch):
    """A measured row has no gate stage, so the UI shows no gate label."""
    _fresh_db(monkeypatch)
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": 0.0, "reason": "unclear",
                           "signal_score": 85,
                           "diagnostic": {"stage": "llm", "llm_called": True,
                                          "signal_score": 85}},
        "risk_report": {}, "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_gate_stage"] is None
    assert row["analyst_llm_called"] is True


# ---------------------------------------------------------------------------
# Required case 6: historical rows with zero or missing values
# ---------------------------------------------------------------------------

def test_legacy_numeric_zero_is_not_assumed_unavailable(monkeypatch):
    """A legacy row with 0.0 and no llm_called evidence stays 'unknown'.

    We must not claim it was unavailable; the data does not prove that.
    """
    _fresh_db(monkeypatch)
    with decision_log._conn() as conn:
        conn.execute(
            "INSERT INTO decisions (timestamp, symbol, analyst_signal, "
            "analyst_confidence, analyst_reason, final_decision) "
            "VALUES (?,?,?,?,?,?)",
            ("2025-01-01T00:00:00+00:00", "OLD", "HOLD", 0.0, "legacy", "HOLD"))
        conn.commit()

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence"] == 0.0
    assert row["analyst_confidence_source"] == "unknown"
    assert row["analyst_confidence_available"] is True
    assert row["analyst_confidence_display"] == 0.0
    assert row["analyst_signal_score"] is None


def test_legacy_null_confidence_is_unavailable(monkeypatch):
    """A legacy row whose confidence is NULL is reported unavailable."""
    _fresh_db(monkeypatch)
    with decision_log._conn() as conn:
        conn.execute(
            "INSERT INTO decisions (timestamp, symbol, analyst_signal, "
            "analyst_confidence, analyst_reason, final_decision) "
            "VALUES (?,?,?,?,?,?)",
            ("2025-01-01T00:00:00+00:00", "OLD", "HOLD", None, "legacy", "HOLD"))
        conn.commit()

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence"] is None
    assert row["analyst_confidence_source"] == "unavailable"
    assert row["analyst_confidence_display"] is None


def test_migration_adds_signal_score_column_to_existing_db(monkeypatch):
    """init_db() adds analyst_signal_score to a pre-existing decisions table."""
    db = Path(tempfile.mkdtemp()) / "legacy.db"
    monkeypatch.setattr(decision_log, "DB", db)
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, symbol TEXT,
            analyst_signal TEXT, analyst_confidence REAL, analyst_reason TEXT,
            risk_approved INTEGER, risk_veto INTEGER, risk_reason TEXT,
            final_decision TEXT, consensus_reached INTEGER,
            order_details TEXT, broker_result TEXT, market_data TEXT)""")
        conn.commit()

    decision_log.init_db()  # must migrate, not raise

    with decision_log._conn() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
    assert "analyst_signal_score" in cols
    assert "analyst_diagnostic" in cols


def test_annotation_tolerates_corrupt_diagnostic_json(monkeypatch):
    """A row with unparseable diagnostic JSON still annotates without error."""
    _fresh_db(monkeypatch)
    with decision_log._conn() as conn:
        conn.execute(
            "INSERT INTO decisions (timestamp, symbol, analyst_confidence, "
            "analyst_diagnostic, final_decision) VALUES (?,?,?,?,?)",
            ("2025-01-01T00:00:00+00:00", "OLD", 0.5, "{not json", "HOLD"))
        conn.commit()

    row = decision_log.recent_decisions(1)[0]
    assert row["analyst_confidence_source"] == "unknown"
    assert row["analyst_confidence_display"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Required case 7: entry threshold and risk rules unchanged
# ---------------------------------------------------------------------------

def test_hold_still_returns_before_the_confidence_gate(monkeypatch):
    """HOLD remains an analyst decision, not a confidence veto."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "HOLD", "confidence": None,
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]
    assert out["approved"] is False
    assert out["veto"] is False
    assert out["reason"] == "analyst HOLD"


def test_missing_confidence_vetoes_an_actionable_signal(monkeypatch):
    """A BUY with no measured confidence fails closed."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "BUY", "confidence": None,
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]
    assert out["approved"] is False
    assert out["veto"] is True
    assert out["reason"] == "confidence unavailable"


def test_low_confidence_still_vetoed_against_the_same_threshold(monkeypatch):
    """The min_confidence gate is unchanged: 0.0 < 0.7 is still a veto."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "BUY", "confidence": 0.0,
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]
    assert out["veto"] is True
    assert "confidence 0.00 < 0.7" in out["reason"]


def test_accepting_a_zero_confidence_reply_does_not_bypass_risk_gates(monkeypatch):
    """End to end: a valid LLM BUY at 0.0 is accepted by the analyst but still
    vetoed by the risk manager. Accepting the reply admits no trade."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    monkeypatch.setattr(agents, "llm", _FakeLLM(
        '{"signal":"BUY","confidence":0.0,"reason":"weak"}'))

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]
    assert report["signal"] == "BUY"
    assert report["confidence"] == 0.0

    risk = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": report,
    })["risk_report"]
    assert risk["approved"] is False
    assert risk["veto"] is True


# ---------------------------------------------------------------------------
# Required case 8: no broker orders during these tests
# ---------------------------------------------------------------------------

def test_no_broker_order_is_submitted(monkeypatch):
    """Tripwire: the analyst flow must not reach an order-submitting broker
    entry point.

    The earlier revision of this test had no assertion and exercised only
    ``analyst_agent``, which never imports ``broker`` -- so it could not fail.
    This version guards the real entry points, self-checks that the tripwire
    actually fires, and asserts the flow produced a signal, so the test fails
    if it stops exercising anything.
    """
    import broker

    calls = []

    def _forbidden(*a, **k):
        calls.append(1)
        raise AssertionError("a broker order was submitted during the tests")

    entry_points = [
        "execute_order", "_submit_entry", "_submit_stop",
        "_attach_protection", "_emergency_close_and_verify", "move_broker_stop",
    ]
    guarded = [name for name in entry_points if hasattr(broker, name)]
    assert guarded, "no broker order entry points found to guard"
    for name in guarded:
        monkeypatch.setattr(broker, name, _forbidden)

    # Self-check: the tripwire must actually fire when a guarded point is hit,
    # otherwise this test would silently stop protecting anything.
    with pytest.raises(AssertionError):
        getattr(broker, guarded[0])()
    calls.clear()

    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: _strong_score())
    monkeypatch.setattr(agents, "llm", _FakeLLM(
        '{"signal":"BUY","confidence":0.9,"reason":"ok"}'))
    report = agents.analyst_agent({"symbol": "AAA",
                                   "market_data": dict(_INDICATORS),
                                   "memory_context": ""})["analyst_report"]

    # The flow really ran (a real signal), and it never reached the broker.
    assert report["signal"] == "BUY"
    assert calls == [], "the analyst flow reached a broker order entry point"


# ---------------------------------------------------------------------------
# End-to-end: the real FastAPI app, not just the helpers
# ---------------------------------------------------------------------------

def test_real_api_endpoint_exposes_separated_fields(monkeypatch, tmp_path):
    """The live /api/decisions endpoint returns confidence and score apart, and
    the dashboard page still renders. Exercises the real app object."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "api.db"))
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setattr(decision_log, "DB", tmp_path / "api.db")
    decision_log.init_db()
    decision_log.log_decision({
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": None,
                           "reason": "signal score below threshold (12 < 70)",
                           "signal_score": 12,
                           "diagnostic": {"stage": "signal_score",
                                          "llm_called": False, "signal_score": 12}},
        "risk_report": {"approved": False, "veto": False, "reason": "analyst HOLD"},
        "final_decision": "HOLD", "consensus_reached": False, "market_data": {},
    }, None)

    import app as appmod
    from fastapi.testclient import TestClient

    client = TestClient(appmod.app)
    resp = client.get("/api/decisions?limit=5")
    assert resp.status_code == 200
    row = resp.json()[0]
    assert row["analyst_confidence"] is None
    assert row["analyst_confidence_source"] == "placeholder"
    assert row["analyst_confidence_display"] is None
    assert row["analyst_signal_score"] == pytest.approx(12)
    assert row["analyst_gate_stage"] == "signal_score"
    # Backward compatibility: the original keys are still present.
    assert row["analyst_signal"] == "HOLD"
    assert row["final_decision"] == "HOLD"

    page = client.get("/")
    assert page.status_code == 200
    assert "analyst_confidence_display" in page.text


def test_non_numeric_confidence_fails_closed_without_crashing(monkeypatch):
    """A malformed confidence on an actionable signal is vetoed, not a crash."""
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "BUY", "confidence": "not-a-number",
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]
    assert out["approved"] is False
    assert out["veto"] is True
    assert out["reason"] == "confidence unavailable"
