"""
Confidence semantics diagnosis (Phase 1.7.1, Task 2).

These tests PROVE the root cause of "confidence = 0 persists in bot decisions"
and lock the finding in. They are characterisation tests: they assert the
current, measured behaviour and do NOT change any signal, threshold, risk rule,
or trading decision. Nothing here lowers ``min_confidence``, invents a
confidence value, or produces extra trades.

Root cause, established by tracing the value end to end:

  1. ``signal_score`` and ``confidence`` are different quantities stored in
     different places. ``signal_score`` is the deterministic 0-100 technical
     composite from ``compute_signal_score``. ``confidence`` is the LLM's own
     self-reported 0-1 number. The decisions table has an
     ``analyst_confidence`` column but no ``signal_score`` column; the score
     lives only inside the ``analyst_diagnostic`` JSON blob.

  2. Three analyst paths return HOLD *before the LLM is consulted* and hard-code
     ``confidence = 0.0``: no valid indicators, signal score below threshold,
     and BEAR regime. In those paths ``confidence`` is a placeholder, not a
     measurement -- ``diagnostic.llm_called`` is ``False``. Since the score gate
     is the normal case, this is the usual source of the stored 0.0.

  3. When the LLM *is* consulted, the retry loop accepts a response only if the
     signal is valid AND ``confidence > 0``. A genuine LLM HOLD carrying
     ``confidence = 0.00`` therefore fails the check, is retried, and ends as
     ``llm_status = "invalid_response"`` with ``confidence = 0.0`` -- identical
     to a malformed response. "The model said HOLD with zero confidence" and
     "the model never answered" collapse into the same stored value.

  4. The UI renders ``analyst_confidence`` unconditionally as a percentage and
     never reads ``analyst_diagnostic`` (0 references in the template). So a
     placeholder 0.0 is displayed as "0%" as though it were a measured
     confidence, and the diagnostic that would explain it is invisible.

Consequence: the persistent 0% is a semantics/display defect, not a stuck
trading gate. ``risk_manager_agent`` returns early for HOLD before the
confidence gate, so a 0.0 confidence never vetoes a HOLD; it only matters for
BUY/SELL, where 0.0 correctly fails closed.

No network, no credentials, no broker orders.
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


# ---------------------------------------------------------------------------
# 1) confidence and signal_score are distinct quantities
# ---------------------------------------------------------------------------

def test_signal_score_and_confidence_are_distinct_fields(monkeypatch):
    """The analyst reports a numeric signal_score alongside an LLM confidence.

    They are not the same number and do not come from the same source.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": 82, "trend": 25, "momentum": 20, "rsi": 15, "bb": 8,
        "volume": 14, "news": 0,
    })
    monkeypatch.setattr(agents, "llm", _FakeLLM(
        '{"signal":"BUY","confidence":0.55,"reason":"ok"}'))

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal_score"] == 82          # technical composite
    assert report["confidence"] == pytest.approx(0.55)  # LLM self-report
    assert report["signal_score"] != report["confidence"]


# ---------------------------------------------------------------------------
# 2) the pre-LLM HOLD paths hard-code 0.0 as a placeholder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,state,extra", [
    ("no valid indicators",
     {"symbol": "AAA", "market_data": {}, "memory_context": ""}, {}),
    ("signal score below threshold",
     {"symbol": "AAA", "market_data": dict(_INDICATORS), "memory_context": ""},
     {"score": 12}),
])
def test_pre_llm_hold_paths_report_zero_confidence_placeholder(
        monkeypatch, label, state, extra):
    """A HOLD decided before the LLM runs carries confidence 0.0 by default.

    ``diagnostic.llm_called`` is False, which is the only signal that the 0.0
    is a placeholder rather than a measured value.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": extra.get("score", 0), "trend": 0, "momentum": 0,
        "rsi": 0, "bb": 0, "volume": 0, "news": 0,
    })
    monkeypatch.setattr(agents, "llm", _FakeLLM('{"signal":"BUY","confidence":0.9}'))

    report = agents.analyst_agent(state)["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] == 0.0
    assert report["diagnostic"]["llm_called"] is False
    assert agents.llm.calls == 0, "the LLM must not be consulted on this path"


def test_bear_gate_reports_zero_confidence_placeholder(monkeypatch):
    """The BEAR-regime HOLD also returns before the LLM, with confidence 0.0."""
    monkeypatch.setattr(agents, "get_all_settings",
                        lambda: _settings(regime_enabled=1))
    monkeypatch.setattr(agents, "get_market_regime", lambda: {"regime": "BEAR"})
    monkeypatch.setattr(agents, "get_news_sentiment", lambda _s: None)
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": 90, "trend": 25, "momentum": 20, "rsi": 15, "bb": 8,
        "volume": 14, "news": 8,
    })
    monkeypatch.setattr(agents, "llm", _FakeLLM('{"signal":"BUY","confidence":0.9}'))

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] == 0.0
    assert report["diagnostic"]["llm_called"] is False
    assert report["diagnostic"]["stage"] == "regime"
    assert agents.llm.calls == 0


# ---------------------------------------------------------------------------
# 3) a genuine LLM HOLD at zero confidence is indistinguishable from failure
# ---------------------------------------------------------------------------

def test_llm_hold_with_zero_confidence_is_treated_as_invalid_response(monkeypatch):
    """Documents the conflation that produces a misleading 0.0.

    A well-formed response ``{"signal":"HOLD","confidence":0.0}`` is a valid
    answer, but the acceptance check requires ``confidence > 0``. The response
    is therefore retried and finally recorded as ``invalid_response`` with
    confidence 0.0 -- exactly what a malformed reply produces. This is the
    limitation to report; the test pins it so a future change is deliberate.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": 85, "trend": 25, "momentum": 20, "rsi": 15, "bb": 8,
        "volume": 14, "news": 8,
    })
    llm = _FakeLLM('{"signal":"HOLD","confidence":0.0,"reason":"unclear"}')
    monkeypatch.setattr(agents, "llm", llm)

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] == 0.0
    assert report["diagnostic"]["llm_called"] is True
    assert report["diagnostic"]["llm_status"] == "invalid_response"
    assert report["diagnostic"]["llm_attempts"] == 2
    assert llm.calls == 2, "the valid-but-zero-confidence reply is retried"


def test_llm_hold_with_nonzero_confidence_is_accepted(monkeypatch):
    """Control: the same HOLD with confidence > 0 is recorded as success.

    This isolates confidence (not the HOLD signal) as the discriminator.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a, **k: {
        "score": 85, "trend": 25, "momentum": 20, "rsi": 15, "bb": 8,
        "volume": 14, "news": 8,
    })
    llm = _FakeLLM('{"signal":"HOLD","confidence":0.4,"reason":"mixed"}')
    monkeypatch.setattr(agents, "llm", llm)

    report = agents.analyst_agent(
        {"symbol": "AAA", "market_data": dict(_INDICATORS),
         "memory_context": ""})["analyst_report"]

    assert report["signal"] == "HOLD"
    assert report["confidence"] == pytest.approx(0.4)
    assert report["diagnostic"]["llm_status"] == "success"
    assert llm.calls == 1


# ---------------------------------------------------------------------------
# 4) the 0.0 reaches the DB, the API and the UI unchanged
# ---------------------------------------------------------------------------

def test_placeholder_zero_is_persisted_and_rendered_as_zero_percent(monkeypatch):
    """The stored 0.0 is written verbatim and the template shows "0%".

    The decisions row has no signal_score column, so the UI cannot tell a
    score-gate HOLD from a low-confidence LLM HOLD.
    """
    db = Path(tempfile.mkdtemp()) / "ui.db"
    monkeypatch.setattr(decision_log, "DB", db)
    decision_log.init_db()

    state = {
        "symbol": "AAA",
        "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                           "reason": "signal score below threshold (12 < 70)",
                           "signal_score": 12,
                           "diagnostic": {"stage": "signal_score",
                                          "llm_called": False,
                                          "signal_score": 12, "threshold": 70}},
        "risk_report": {"approved": False, "veto": False,
                        "reason": "analyst HOLD"},
        "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {"last_close": 100.0},
    }
    decision_log.log_decision(state, None)
    row = decision_log.recent_decisions(1)[0]

    assert row["analyst_confidence"] == 0.0
    assert row["analyst_signal"] == "HOLD"
    # The technical score is not a column; it is buried in the diagnostic JSON.
    assert "signal_score" not in row
    assert '"signal_score": 12' in row["analyst_diagnostic"]

    # The template formats analyst_confidence directly and ignores the
    # diagnostic entirely, so the placeholder renders as a measured "0%".
    tpl = _TEMPLATE.read_text(encoding="utf-8")
    assert "d.analyst_confidence*100" in tpl
    assert "analyst_diagnostic" not in tpl


# ---------------------------------------------------------------------------
# 5) the zero does not itself block trading; the gates stay separate
# ---------------------------------------------------------------------------

def test_hold_returns_before_the_confidence_gate(monkeypatch):
    """A HOLD with confidence 0.0 is an analyst decision, not a risk veto.

    The confidence gate is never reached for HOLD, so the persistent 0.0 is
    not what keeps the bot out of trades.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())

    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]

    assert out["approved"] is False
    assert out["veto"] is False
    assert out["reason"] == "analyst HOLD"


def test_confidence_gate_still_fails_closed_for_non_hold(monkeypatch):
    """For a BUY, zero confidence is still vetoed. That behaviour is unchanged.

    This confirms the diagnosis targets reporting, not the entry gate: the gate
    uses the LLM's confidence for actionable signals, and low confidence must
    keep failing closed.
    """
    monkeypatch.setattr(agents, "get_all_settings", lambda: _settings())

    out = agents.risk_manager_agent({
        "symbol": "AAA", "market_data": dict(_INDICATORS),
        "analyst_report": {"signal": "BUY", "confidence": 0.0,
                           "signal_score": 95, "regime": "BULL"},
    })["risk_report"]

    assert out["approved"] is False
    assert out["veto"] is True
    assert "confidence" in out["reason"]
