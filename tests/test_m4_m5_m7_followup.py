"""Regression tests for the M4 / M5 / M7 follow-up review.

M4 - Gemini generation config (maxOutputTokens vs thinking level).
M5 - one-shot retry for transient HTTP failures, bounded by budget + deadline.
M7 - A/B fairness: the engine-based comparison vs the legacy simplified replay.

No network, no credentials, no broker, no live model call.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_context as mc  # noqa: E402
from evaluation import ab_ai_context as ab  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FEATURES = {"last_close": 150.0, "ema20": 148.0, "ema50": 145.0,
            "rsi14": 55.0, "atr14": 2.5, "adx": 30.0,
            "volume_ratio": 1.2, "bb_bandwidth": 0.04}

_GOOD_TEXT = json.dumps({"bias": "NEUTRAL", "confidence": 0.5,
                         "risk_level": "LOW", "rationale": "ok"})


def _reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _enabled() -> dict:
    return {"ai_context_enabled": 1}


def _http_error(code: int, retry_after: str | None = None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, "err", headers,
                                  io.BytesIO(b""))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Fresh budget and a fake key for every test in this module."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    mc._reset_budget_for_tests()
    yield
    mc._reset_budget_for_tests()


def _capture_body(monkeypatch) -> dict:
    """Run one successful call and return the parsed request body."""
    seen: dict = {}

    def _transport(url, body, timeout):
        seen["body"] = json.loads(body.decode("utf-8"))
        return _reply(_GOOD_TEXT)

    mc.get_market_context("AAPL", FEATURES, _enabled(), _transport=_transport)
    return seen["body"]


# ===========================================================================
# M4 - generation config
# ===========================================================================

def test_max_output_tokens_is_sent_and_is_not_the_old_tiny_value(monkeypatch):
    gen = _capture_body(monkeypatch)["generationConfig"]
    assert gen["maxOutputTokens"] == mc.DEFAULT_MAX_OUTPUT_TOKENS
    # 256 was the pre-fix value; it is too small once thought tokens are
    # counted inside the same ceiling.
    assert gen["maxOutputTokens"] > 256


def test_thinking_level_is_sent_separately_from_the_output_ceiling(monkeypatch):
    """The two budgets must be distinct fields, per Google's docs."""
    gen = _capture_body(monkeypatch)["generationConfig"]
    assert gen["thinkingConfig"] == {"thinkingLevel": mc.DEFAULT_THINKING_LEVEL}
    assert "maxOutputTokens" in gen
    # They must not be conflated into one field.
    assert "thinkingBudget" not in gen


def test_temperature_stays_deterministic(monkeypatch):
    gen = _capture_body(monkeypatch)["generationConfig"]
    assert gen["temperature"] == 0.0


def test_max_output_tokens_env_is_clamped(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "999999")
    cfg = mc._cfg(_enabled())
    assert cfg["max_output_tokens"] == mc._MAX_MAX_OUTPUT_TOKENS

    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "1")
    assert mc._cfg(_enabled())["max_output_tokens"] == mc._MIN_MAX_OUTPUT_TOKENS


def test_max_output_tokens_malformed_falls_back(monkeypatch):
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "not-a-number")
    cfg = mc._cfg(_enabled())
    assert cfg["max_output_tokens"] == mc.DEFAULT_MAX_OUTPUT_TOKENS


def test_unknown_thinking_level_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "hyper")
    assert mc._cfg(_enabled())["thinking_level"] == mc.DEFAULT_THINKING_LEVEL


def test_thinking_level_is_normalised_case_insensitively(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "  LOW  ")
    assert mc._cfg(_enabled())["thinking_level"] == "low"


def test_empty_thinking_level_omits_the_field(monkeypatch):
    """Opting out must produce a body without thinkingConfig at all."""
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", "")
    cfg = mc._cfg(_enabled())
    assert cfg["thinking_level"] == ""
    gen = mc._build_generation_config(cfg)
    assert "thinkingConfig" not in gen
    assert gen["maxOutputTokens"] == mc.DEFAULT_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize("level", mc._ALLOWED_THINKING_LEVELS)
def test_every_documented_thinking_level_is_accepted(monkeypatch, level):
    monkeypatch.setenv("GEMINI_THINKING_LEVEL", level)
    assert mc._cfg(_enabled())["thinking_level"] == level


# ===========================================================================
# M4 - offline handling of degenerate model replies
# ===========================================================================

def _run_with_text(text: str, monkeypatch) -> dict:
    mc._reset_budget_for_tests()
    return mc.get_market_context("AAPL", FEATURES, _enabled(),
                                 _transport=lambda u, b, t: _reply(text))


def test_empty_response_is_reported_as_empty_not_as_success(monkeypatch):
    out = _run_with_text("", monkeypatch)
    assert out["available"] is False
    assert out["reason"] == "empty_response"


def test_truncated_json_is_rejected(monkeypatch):
    """A reply cut off mid-object must not be accepted."""
    out = _run_with_text('{"bias": "BULL', monkeypatch)
    assert out["available"] is False
    assert out["reason"] == "unparseable_response"


def test_truncated_json_with_valid_prefix_is_still_rejected(monkeypatch):
    """Guards against a 'parse the longest valid prefix' style regression."""
    out = _run_with_text(
        '{"bias": "BULLISH", "confidence": 0.9, "risk_level": "LOW",'
        ' "rationale": "trunca', monkeypatch)
    assert out["available"] is False


def test_refusal_style_plain_text_is_rejected(monkeypatch):
    out = _run_with_text(
        "I'm sorry, I cannot help with that request.", monkeypatch)
    assert out["available"] is False
    assert out["reason"] == "unparseable_response"


def test_no_candidates_key_is_treated_as_empty(monkeypatch):
    mc._reset_budget_for_tests()
    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=lambda u, b, t: {})
    assert out["available"] is False
    assert out["reason"] == "empty_response"


def test_blocked_prompt_feedback_is_rejected(monkeypatch):
    """A safety-blocked reply has no text parts and must degrade safely."""
    payload = {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
    mc._reset_budget_for_tests()
    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=lambda u, b, t: payload)
    assert out["available"] is False
    assert out["affects_trading"] is False


def test_schema_violating_reply_is_rejected_not_coerced(monkeypatch):
    bad = json.dumps({"bias": "MOON", "confidence": 5.0,
                      "risk_level": "LOW", "rationale": "x"})
    out = _run_with_text(bad, monkeypatch)
    assert out["available"] is False
    assert out["reason"] == "schema_rejected"


# ===========================================================================
# M5 - retry policy
# ===========================================================================

def test_retryable_statuses_are_retried_once(monkeypatch):
    for code in sorted(mc._RETRYABLE_STATUS):
        mc._reset_budget_for_tests()
        calls = {"n": 0}

        def _transport(u, b, t, code=code):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(code, retry_after="0")
            return _reply(_GOOD_TEXT)

        out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                    _transport=_transport)
        assert calls["n"] == 2, f"status {code} was not retried"
        assert out["available"] is True


def test_retry_gives_up_after_exactly_one_extra_attempt(monkeypatch):
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise _http_error(503, retry_after="0")

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert calls["n"] == 1 + mc._MAX_RETRIES == 2
    assert out["available"] is False
    assert out["reason"].startswith("request_failed")


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_never_retried(monkeypatch, code):
    """A bad key or malformed request cannot succeed on a repeat."""
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise _http_error(code)

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert calls["n"] == 1, f"status {code} must not be retried"
    assert out["available"] is False


def test_retry_consumes_two_units_of_budget(monkeypatch):
    """A retry must not slip past the per-minute budget."""
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_MINUTE", "2")
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_DAY", "100")
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise _http_error(503, retry_after="0")

    first = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                  _transport=_transport)
    assert calls["n"] == 2          # attempt + retry
    assert first["available"] is False
    # Budget of 2 is now exhausted, so a fresh call is refused up front.
    second = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                   _transport=_transport)
    assert second["reason"] == "rate_limited_per_minute"
    assert calls["n"] == 2


def test_retry_respects_retry_after_cap(monkeypatch):
    """A huge Retry-After must be capped, never slept verbatim."""
    slept = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: slept.append(s))
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after="3600")
        return _reply(_GOOD_TEXT)

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert out["available"] is True
    assert slept and slept[0] <= mc._MAX_RETRY_AFTER_SECONDS


def test_retry_after_absent_uses_a_small_default(monkeypatch):
    slept = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: slept.append(s))
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)
        return _reply(_GOOD_TEXT)

    mc.get_market_context("AAPL", FEATURES, _enabled(), _transport=_transport)
    assert slept == [mc._DEFAULT_RETRY_DELAY_SECONDS]


def test_no_sleep_when_the_deadline_cannot_fit_the_delay(monkeypatch):
    """If the retry delay would exceed the remaining budget, give up."""
    slept = []
    monkeypatch.setattr(mc.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "1")
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise _http_error(429, retry_after="5")

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert slept == []                 # 5s > 1s deadline: no pointless wait
    assert calls["n"] == 1
    assert out["available"] is False


def test_retry_delay_helper_returns_none_at_zero_remaining():
    assert mc._retry_delay(_http_error(503), remaining=0.0) is None
    assert mc._retry_delay(_http_error(503), remaining=-1.0) is None


def test_timeout_errors_are_retried(monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("timed out")
        return _reply(_GOOD_TEXT)

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert calls["n"] == 2
    assert out["available"] is True


def test_non_retryable_exception_is_not_retried(monkeypatch):
    """A local programming error must not be retried."""
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise RuntimeError("boom")

    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=_transport)
    assert calls["n"] == 1
    assert out["available"] is False


def test_retry_never_loops_forever_with_a_permanently_failing_transport(
        monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    mc._reset_budget_for_tests()
    calls = {"n": 0}

    def _transport(u, b, t):
        calls["n"] += 1
        raise _http_error(503, retry_after="0")

    mc.get_market_context("AAPL", FEATURES, _enabled(), _transport=_transport)
    assert calls["n"] <= 1 + mc._MAX_RETRIES


def test_status_extraction_handles_wrapped_response_objects():
    class _Resp:
        status_code = 503

    class _Exc(Exception):
        response = _Resp()

    assert mc._http_status(_Exc()) == 503
    assert mc._is_retryable(_Exc()) is True


def test_retry_after_http_date_form_is_ignored_not_crashed():
    """The date form must not raise; it degrades to the default delay."""
    err = _http_error(503, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")
    assert mc._retry_after_seconds(err) is None


# ===========================================================================
# M7 - A/B fairness
# ===========================================================================

def _synthetic_ohlcv(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """Deterministic OHLCV with a clear uptrend so signals actually fire."""
    import numpy as np
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.4, 1.2, n).cumsum()
    close = 100.0 + steps
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close - 0.2,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)


def test_stand_in_gate_duplicates_the_engine_strong_uptrend_filter():
    """The M7 root cause: the gate tests exactly what the engine already does.

    If this equivalence ever breaks, the stand-in becomes a real (non
    redundant) filter and the engine-mode delta becomes meaningful.
    """
    assert ab.STAND_IN_IS_REDUNDANT_WITH_ENGINE is True

    allow_case = {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0,
                  "adx": 30.0}
    # Engine condition: close > ema20 > ema50 and adx > 25
    engine_allows = (allow_case["last_close"] > allow_case["ema20"]
                     > allow_case["ema50"]) and allow_case["adx"] > 25
    assert engine_allows is True
    assert ab.ai_gate_decision(allow_case) == "allow"

    block_cases = [
        {"last_close": 100.0, "ema20": 105.0, "ema50": 100.0, "adx": 30.0},
        {"last_close": 110.0, "ema20": 100.0, "ema50": 105.0, "adx": 30.0},
        {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0, "adx": 20.0},
    ]
    for ind in block_cases:
        engine_allows = (ind["last_close"] > ind["ema20"] > ind["ema50"]
                         and ind["adx"] > 25)
        assert engine_allows is False
        assert ab.ai_gate_decision(ind) == "block"


def test_engine_experiment_runs_both_arms_on_the_real_engine(monkeypatch):
    """Both arms must go through simulate_portfolio, not a private replay."""
    calls = {"n": 0}
    real = ab.engine_mod.simulate_portfolio

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ab.engine_mod, "simulate_portfolio", _counting)

    df = _synthetic_ohlcv()
    arm_base = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "none")
    arm_ai = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
    assert calls["n"] == 2
    assert "equity_curve" in arm_base and "trades" in arm_base
    assert "equity_curve" in arm_ai and "trades" in arm_ai


def test_engine_gate_restores_the_original_function_on_success():
    df = _synthetic_ohlcv()
    original = ab.engine_mod.compute_signal_score
    ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
    assert ab.engine_mod.compute_signal_score is original


def test_engine_gate_restores_the_original_function_on_error(monkeypatch):
    """Even if the engine raises, the global hook must be undone."""
    original = ab.engine_mod.compute_signal_score

    def _boom(*a, **k):
        raise RuntimeError("engine failed")

    monkeypatch.setattr(ab.engine_mod, "simulate_portfolio", _boom)
    with pytest.raises(RuntimeError):
        ab.run_engine_arm({"AAA": _synthetic_ohlcv()}, None, 10_000.0, "ai")
    assert ab.engine_mod.compute_signal_score is original


def test_engine_arm_rejects_an_unknown_gate():
    with pytest.raises(ValueError):
        ab.run_engine_arm({"AAA": _synthetic_ohlcv()}, None, 10_000.0, "wat")


def test_engine_baseline_arm_does_not_install_a_wrapper():
    original = ab.engine_mod.compute_signal_score
    ab.run_engine_arm({"AAA": _synthetic_ohlcv()}, None, 10_000.0, "none")
    assert ab.engine_mod.compute_signal_score is original


def test_engine_ai_arm_reports_blocked_entries():
    """The wrapper must count what it blocked, even when nothing changes."""
    df = _synthetic_ohlcv()
    arm = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
    assert arm["blocked_entries"] >= 0


def test_simplified_replay_diverges_from_the_real_engine():
    """The M7 defect: run_arm is NOT the strategy.

    On this fixture the simplified replay takes a different number of trades
    than the real engine on identical data, because it does not model the
    engine's ``strong_uptrend`` filter, regime gate, cooldown or sizing. This
    is the regression guard proving the two are not interchangeable.
    """
    df = _synthetic_ohlcv(n=400, seed=7)
    engine_res = ab.engine_mod.simulate_portfolio(
        {"AAA": df}, None, start_equity=10_000.0,
        settings=ab.get_all_settings())
    simplified = ab.run_arm("AAA", df, None, 10_000.0, "none")

    engine_trades = len(engine_res["trades"])
    sim_trades = simplified["num_trades"]

    assert engine_trades != sim_trades, (
        "fixture no longer demonstrates the divergence; pick another seed "
        "rather than weakening this assertion")
    # The simplified replay is the more permissive of the two: it lacks the
    # engine's extra entry filters.
    assert sim_trades > engine_trades


def test_simplified_arm_ignores_the_engine_regime_and_cooldown():
    """Structural check: run_arm never consults regime or cooldown state."""
    import inspect
    src = inspect.getsource(ab.run_arm)
    for engine_only in ("compute_regime_at", "cooldown_until",
                        "consecutive_losses", "strong_uptrend",
                        "compute_position_size"):
        assert engine_only not in src, (
            f"run_arm unexpectedly models {engine_only!r}; the M7 docstring "
            "claiming it is approximate would then be wrong")


def test_simplified_replay_is_labelled_as_approximate():
    """The docstring must not present run_arm as the real strategy."""
    doc = ab.__doc__ or ""
    assert "run_arm" in doc
    assert "approximation" in doc.lower() or "approximate" in doc.lower()
    assert "exploratory" in doc.lower()


def test_run_experiment_marks_its_mode_as_simplified():
    import inspect
    src = inspect.getsource(ab.run_experiment)
    assert '"simplified"' in src


def test_engine_experiment_marks_its_mode_as_engine():
    import inspect
    src = inspect.getsource(ab.run_engine_experiment)
    assert '"engine"' in src


def test_compare_engine_arithmetic():
    result = {
        "arms": {
            "none": {"metrics": {"total_return_pct": 2.0,
                                 "max_drawdown_pct": -5.0},
                     "num_trades": 10, "blocked_entries": 0},
            "ai": {"metrics": {"total_return_pct": 3.5,
                               "max_drawdown_pct": -4.0},
                   "num_trades": 8, "blocked_entries": 4},
        }
    }
    c = ab.compare_engine(result)
    assert c["delta_total_return_pct"] == 1.5
    assert c["delta_num_trades"] == -2
    assert c["delta_max_drawdown_pct"] == 1.0
    assert c["ai_blocked_entries"] == 4


def test_engine_arm_metrics_reduce_the_engine_result():
    df = _synthetic_ohlcv()
    arm = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "none")
    metrics = ab._engine_arm_metrics(arm)
    assert "total_return_pct" in metrics
    assert "max_drawdown_pct" in metrics
    assert metrics["end_equity"] == arm["end_equity"]


def test_engine_arm_is_deterministic():
    df = _synthetic_ohlcv()
    a = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
    b = ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
    assert a["end_equity"] == b["end_equity"]
    assert len(a["trades"]) == len(b["trades"])
    assert a["blocked_entries"] == b["blocked_entries"]


def test_no_network_is_used_by_the_evaluation_module(monkeypatch):
    """Guard: importing and running the driver must not touch the network."""
    import socket

    def _no_net(*a, **k):
        raise AssertionError("evaluation module attempted a network call")

    monkeypatch.setattr(socket, "create_connection", _no_net)
    df = _synthetic_ohlcv()
    ab.run_engine_arm({"AAA": df}, None, 10_000.0, "none")
    ab.run_engine_arm({"AAA": df}, None, 10_000.0, "ai")
