"""Shadow Mode tests: audit integrity, no leakage, and provable no-op trading.

The central claim under test is that Shadow Mode changes NOTHING about
trading. That is asserted directly against the real engine (arms A vs C,
trade-for-trade), not inferred.

No network, no credentials, no broker, no live model call: every transport is
a fake injected by the test.
"""
from __future__ import annotations

import io
import json
import socket
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_context as mc  # noqa: E402
import shadow_mode as sm  # noqa: E402
from evaluation import ab_ai_context as ab  # noqa: E402


FEATURES = {"last_close": 150.0, "ema20": 148.0, "ema50": 145.0,
            "rsi14": 55.0, "atr14": 2.5, "adx": 30.0,
            "volume_ratio": 1.2, "bb_bandwidth": 0.04}

GOOD = json.dumps({
    "trend_quality": "STRONG_UP", "volatility_regime": "NORMAL",
    "momentum_quality": "STEADY", "market_risk": "LOW",
    "confidence": 0.91, "recommendation": "UNFAVORABLE",
})


def _reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _enabled() -> dict:
    return {"ai_context_enabled": 1}


def _http_error(code: int, retry_after: str | None = None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("http://x", code, "err", headers,
                                  io.BytesIO(b""))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("GEMINI_THINKING_LEVEL", raising=False)
    mc._reset_budget_for_tests()
    yield
    mc._reset_budget_for_tests()


@pytest.fixture()
def log(tmp_path):
    return sm.ShadowLog(tmp_path / "shadow.jsonl")


def _runner(log, transport, **kw):
    return sm.ShadowRunner(log, run_id="t", transport=transport, **kw)


def _observe(runner, transport_text=GOOD, **overrides):
    kwargs = dict(
        symbol="AAPL", market_data=dict(FEATURES),
        decision_ts="2026-11-03T15:59:00Z",
        feature_asof_ts="2026-11-03T15:59:00Z",
        execution_ts="2026-11-03T16:00:00Z",
        settings=_enabled(),
    )
    kwargs.update(overrides)
    return runner.observe(**kwargs)


# ===========================================================================
# 1. Closed schema: no free text, strict enums
# ===========================================================================

def test_schema_has_no_free_text_field():
    assert "notes" not in sm.REQUIRED_FIELDS
    assert "rationale" not in sm.REQUIRED_FIELDS
    for field in sm.REQUIRED_FIELDS:
        assert field in sm.PROMPT_SCHEMA


def test_valid_payload_is_accepted():
    out = sm.validate_shadow_payload(json.loads(GOOD))
    assert out["recommendation"] == "UNFAVORABLE"
    assert out["confidence"] == 0.91
    assert set(out) == set(sm.REQUIRED_FIELDS)


def test_extra_key_is_rejected():
    bad = json.loads(GOOD)
    bad["notes"] = "free text"
    assert sm.validate_shadow_payload(bad) is None


def test_missing_key_is_rejected():
    bad = json.loads(GOOD)
    del bad["market_risk"]
    assert sm.validate_shadow_payload(bad) is None


def test_free_text_key_that_looks_like_advice_is_rejected():
    bad = json.loads(GOOD)
    bad["side"] = "BUY"
    bad["qty"] = 100
    assert sm.validate_shadow_payload(bad) is None


@pytest.mark.parametrize("field,value", [
    ("trend_quality", "MOON"),
    ("volatility_regime", "CHAOS"),
    ("momentum_quality", "SIDEWAYS"),
    ("market_risk", "APOCALYPTIC"),
    ("recommendation", "BUY"),
    ("recommendation", "SELL"),
    ("recommendation", "HOLD"),
])
def test_out_of_enum_values_are_rejected(field, value):
    bad = json.loads(GOOD)
    bad[field] = value
    assert sm.validate_shadow_payload(bad) is None


@pytest.mark.parametrize("value", [-0.1, 1.1, "0.9", None, True, float("nan")])
def test_invalid_confidence_is_rejected(value):
    bad = json.loads(GOOD)
    bad["confidence"] = value
    assert sm.validate_shadow_payload(bad) is None


def test_non_mapping_is_rejected():
    for value in (None, [], "text", 42):
        assert sm.validate_shadow_payload(value) is None


def test_recommendation_can_never_be_an_order_verb():
    """The closed enum structurally cannot express a trade order."""
    for banned in ("BUY", "SELL", "HOLD", "SHORT", "LONG", "EXECUTE"):
        assert banned not in sm.RECOMMENDATION_VALUES


# ===========================================================================
# 2. Prompt contains no regime, no size_multiplier, no free text
# ===========================================================================

def test_prompt_does_not_leak_engine_regime_or_sizing():
    """``volatility_regime`` is our own enum; the ENGINE's regime must not leak.

    So we look for the engine's regime as a JSON key/value, not the substring
    inside our own field name.
    """
    prompt = sm.build_shadow_prompt("AAPL", FEATURES)
    assert '"regime"' not in prompt          # no engine-regime JSON key
    assert "BULL" not in prompt and "BEAR" not in prompt
    assert "size_multiplier" not in prompt.lower()
    assert "position_size" not in prompt.lower()
    # Our own market-condition field is expected and allowed.
    assert "volatility_regime" in prompt


def test_prompt_lists_the_closed_schema():
    prompt = sm.build_shadow_prompt("AAPL", FEATURES)
    assert "recommendation" in prompt
    assert "INSUFFICIENT_DATA" in prompt
    assert "Never include any other key" in prompt


def test_request_body_never_contains_regime_keys(log):
    seen = {}

    def _t(url, body, timeout):
        seen["body"] = body
        return _reply(GOOD)

    runner = _runner(log, _t)
    _observe(runner, market_data={**FEATURES, "regime": "BULL",
                                  "size_multiplier": 1.5})
    text = seen["body"].decode("utf-8")
    assert "size_multiplier" not in text
    assert "BULL" not in text


def test_sanitizer_drops_non_allowlisted_keys():
    got = mc.sanitize_market_features({**FEATURES, "regime": "BULL",
                                       "size_multiplier": 2.0, "equity": 12345})
    assert "regime" not in got
    assert "size_multiplier" not in got
    assert "equity" not in got


# ===========================================================================
# 3. Veto rule (deletion-only, theoretical)
# ===========================================================================

def test_veto_candidate_true_at_or_above_c_min():
    assert sm.is_veto_candidate("UNFAVORABLE", 0.80) is True
    assert sm.is_veto_candidate("UNFAVORABLE", 0.95) is True


def test_veto_candidate_false_below_c_min():
    assert sm.is_veto_candidate("UNFAVORABLE", 0.79) is False


@pytest.mark.parametrize("rec", ["FAVORABLE", "NEUTRAL", "INSUFFICIENT_DATA"])
def test_only_unfavorable_can_be_a_veto_candidate(rec):
    assert sm.is_veto_candidate(rec, 1.0) is False


def test_veto_candidate_never_creates_anything():
    """FAVORABLE is not an entry signal; it is inert."""
    assert sm.is_veto_candidate("FAVORABLE", 1.0) is False


def test_c_min_is_preregistered_at_0_80():
    assert sm.C_MIN == 0.80


def test_c_min_is_recorded_in_every_entry(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["c_min"] == 0.80


# ===========================================================================
# 4. Record shape, hashes, no key
# ===========================================================================

def test_record_contains_all_required_audit_fields(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    for field in sm.RECORD_FIELDS:
        assert field in rec, f"missing audit field {field}"


def test_record_hashes_are_sha256_hex(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    for field in ("feature_hash", "prompt_hash", "request_body_hash",
                  "config_snapshot_hash", "entry_hash", "prev_hash"):
        assert isinstance(rec[field], str)
        assert len(rec[field]) == 64
        int(rec[field], 16)  # raises if not hex


def test_genesis_prev_hash_on_first_entry(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["prev_hash"] == sm.GENESIS_HASH
    assert rec["seq"] == 1


def test_api_key_never_written_to_the_log(log):
    _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    raw = log.path.read_text()
    assert "test-key" not in raw
    assert json.loads(raw.strip())["api_key_present"] is True


def test_api_key_absent_flag_is_false_without_a_key(log, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["api_key_present"] is False
    assert rec["parse_status"] == "no_api_key"


def test_git_sha_is_recorded(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert isinstance(rec["git_sha"], str)


def test_feature_hash_matches_the_stored_features(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["feature_hash"] == sm.feature_hash(rec["features"])


def test_request_body_hash_matches_the_actual_body(log):
    seen = {}

    def _t(url, body, timeout):
        seen["body"] = body
        return _reply(GOOD)

    rec = _observe(_runner(log, _t))
    assert rec["request_body_hash"] == sm.sha256_hex(seen["body"])


def test_config_snapshot_hash_changes_when_config_changes():
    a = sm.config_snapshot_hash(model="m1", c_min=0.8, thinking_level="minimal",
                                max_output_tokens=1024, schema_version=1)
    b = sm.config_snapshot_hash(model="m2", c_min=0.8, thinking_level="minimal",
                                max_output_tokens=1024, schema_version=1)
    c = sm.config_snapshot_hash(model="m1", c_min=0.9, thinking_level="minimal",
                                max_output_tokens=1024, schema_version=1)
    assert a != b
    assert a != c


def test_veto_applied_is_false_even_when_veto_candidate_is_true(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["veto_candidate"] is True
    assert rec["veto_applied"] is False


def test_veto_applied_is_never_configurable(log):
    """No setting, env var or argument may flip veto_applied."""
    import inspect
    src = inspect.getsource(sm)
    assert '"veto_applied": False' in src
    # It must be assigned a literal, not a parameter or setting.
    assert 'veto_applied"] = True' not in src


# ===========================================================================
# 5. Hash chain: tamper detection
# ===========================================================================

def _write_three(log):
    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    for i in range(3):
        ts = f"2026-11-0{i + 1}T15:59:00Z"
        _observe(runner, decision_ts=ts, feature_asof_ts=ts,
                 execution_ts=f"2026-11-0{i + 1}T16:00:00Z")
    return log


def test_chain_verifies_when_untouched(log):
    _write_three(log)
    ok, reason = sm.verify_chain(log.path)
    assert ok, reason


def test_chain_detects_edited_line(log):
    _write_three(log)
    lines = log.path.read_text().splitlines()
    entry = json.loads(lines[1])
    entry["recommendation"] = "FAVORABLE"          # tamper
    lines[1] = json.dumps(entry, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    ok, reason = sm.verify_chain(log.path)
    assert ok is False
    assert "entry_hash mismatch" in reason


def test_chain_detects_deleted_line(log):
    _write_three(log)
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    ok, reason = sm.verify_chain(log.path)
    assert ok is False


def test_chain_detects_reordered_lines(log):
    _write_three(log)
    lines = log.path.read_text().splitlines()
    log.path.write_text("\n".join([lines[1], lines[0], lines[2]]) + "\n")
    ok, reason = sm.verify_chain(log.path)
    assert ok is False


def test_chain_detects_inserted_forged_line(log):
    _write_three(log)
    lines = log.path.read_text().splitlines()
    forged = dict(json.loads(lines[0]))
    forged["seq"] = 4
    forged["prev_hash"] = json.loads(lines[2])["entry_hash"]
    forged["entry_hash"] = "0" * 64
    log.path.write_text("\n".join(lines + [json.dumps(forged, sort_keys=True)])
                        + "\n")
    ok, reason = sm.verify_chain(log.path)
    assert ok is False


def test_chain_continues_after_reopening_the_log(log):
    _write_three(log)
    reopened = sm.ShadowLog(log.path)
    rec = _observe(_runner(reopened, lambda u, b, t: _reply(GOOD)))
    assert rec["seq"] == 4
    ok, reason = sm.verify_chain(log.path)
    assert ok, reason


def test_log_has_no_update_or_delete_api():
    """Append-only: there must be no mutating method besides append."""
    public = [n for n in dir(sm.ShadowLog) if not n.startswith("_")]
    for banned in ("update", "delete", "remove", "truncate", "rewrite", "pop"):
        assert banned not in public


# ===========================================================================
# 6. No future-data access; feature vector is reconstructible
# ===========================================================================

def test_features_are_exactly_the_sanitized_allowlist(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)),
                   market_data={**FEATURES, "future_close": 999.0})
    assert "future_close" not in rec["features"]
    assert rec["features"] == mc.sanitize_market_features(FEATURES)


def test_shadow_runner_never_receives_a_dataframe(log):
    """A dict cannot expose later rows, so slicing past asof is impossible."""
    import inspect
    sig = inspect.signature(sm.ShadowRunner.observe)
    assert sig.parameters["market_data"].annotation in (dict, "dict")


def test_feature_rebuild_is_reported_when_a_source_is_given(tmp_path):
    """The rebuilt vector must match what was sent, when data is available."""
    def _source(asof):
        # A causal source: only values at/before asof.
        return dict(FEATURES)

    runner = sm.ShadowRunner(sm.ShadowLog(tmp_path / "s.jsonl"), run_id="r",
                             transport=lambda u, b, t: _reply(GOOD),
                             feature_source=_source)
    rec = _observe(runner)
    assert rec["feature_rebuild"]["available"] is True
    assert rec["feature_rebuild"]["rebuilt_hash"] == rec["feature_hash"]


def test_feature_rebuild_flags_a_mismatch(tmp_path):
    """A divergent source is recorded as a mismatch, not hidden."""
    def _leaky_source(asof):
        return {**FEATURES, "last_close": 999.0}

    runner = sm.ShadowRunner(sm.ShadowLog(tmp_path / "s.jsonl"), run_id="r",
                             transport=lambda u, b, t: _reply(GOOD),
                             feature_source=_leaky_source)
    rec = _observe(runner)
    assert rec["feature_rebuild"]["available"] is True
    assert rec["feature_rebuild"]["rebuilt_hash"] != rec["feature_hash"]


def test_feature_rebuild_reports_unavailable_without_a_source(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["feature_rebuild"]["available"] is False


def test_feature_rebuild_error_does_not_raise(tmp_path):
    def _boom(asof):
        raise RuntimeError("no data")

    runner = sm.ShadowRunner(sm.ShadowLog(tmp_path / "s.jsonl"), run_id="r",
                             transport=lambda u, b, t: _reply(GOOD),
                             feature_source=_boom)
    rec = _observe(runner)
    assert rec["feature_rebuild"]["available"] is False
    assert rec["feature_rebuild"]["error"] == "RuntimeError"


def test_timestamps_are_recorded_and_ordered(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["feature_asof_ts"] <= rec["decision_ts"] <= rec["execution_ts"]


# ===========================================================================
# 7. Failure handling: the original strategy is never disturbed
# ===========================================================================

def test_timeout_is_recorded_and_does_not_raise(log, monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def _t(u, b, t):
        calls["n"] += 1
        raise TimeoutError("timed out")

    rec = _observe(_runner(log, _t))
    assert rec["veto_candidate"] is False
    assert rec["veto_applied"] is False
    assert rec["recommendation"] == "INSUFFICIENT_DATA"
    assert rec["parse_status"].startswith("request_failed")
    assert calls["n"] == 2          # one retry


def test_429_is_retried_then_recorded(log, monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def _t(u, b, t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429, retry_after="0")
        return _reply(GOOD)

    rec = _observe(_runner(log, _t))
    assert calls["n"] == 2
    assert rec["parse_status"] == "ok"
    assert rec["attempts"] == 2


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_5xx_is_retried_once_then_recorded(log, monkeypatch, code):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def _t(u, b, t):
        calls["n"] += 1
        raise _http_error(code, retry_after="0")

    rec = _observe(_runner(log, _t))
    assert calls["n"] == 2
    assert rec["http_status"] == code
    assert rec["veto_applied"] is False


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(log, code):
    calls = {"n": 0}

    def _t(u, b, t):
        calls["n"] += 1
        raise _http_error(code)

    rec = _observe(_runner(log, _t))
    assert calls["n"] == 1
    assert rec["http_status"] == code
    assert rec["veto_candidate"] is False


@pytest.mark.parametrize("text", ["", "not json", '{"trend_quality": "STRONG',
                                  "I cannot help with that."])
def test_malformed_responses_degrade_safely(log, text):
    rec = _observe(_runner(log, lambda u, b, t: _reply(text)))
    assert rec["veto_candidate"] is False
    assert rec["veto_applied"] is False
    assert rec["recommendation"] == "INSUFFICIENT_DATA"
    assert rec["parse_status"] != "ok"


def test_schema_violating_reply_is_rejected(log):
    bad = json.dumps({"trend_quality": "MOON", "volatility_regime": "NORMAL",
                      "momentum_quality": "STEADY", "market_risk": "LOW",
                      "confidence": 0.99, "recommendation": "UNFAVORABLE"})
    rec = _observe(_runner(log, lambda u, b, t: _reply(bad)))
    assert rec["parse_status"] == "schema_rejected"
    assert rec["veto_candidate"] is False


def test_missing_api_key_degrades_safely(log, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    assert rec["parse_status"] == "no_api_key"
    assert rec["recommendation"] == "INSUFFICIENT_DATA"


def test_disabled_by_default(log):
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)),
                   settings={})
    assert rec["parse_status"] == "disabled"


def test_transport_that_raises_arbitrarily_is_contained(log):
    def _t(u, b, t):
        raise RuntimeError("unexpected")

    rec = _observe(_runner(log, _t))
    assert rec["veto_applied"] is False


def test_failure_entries_are_still_appended_and_chained(log, monkeypatch):
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    runner = _runner(log, lambda u, b, t: (_ for _ in ()).throw(
        TimeoutError("t")))
    _observe(runner)
    _observe(runner)
    ok, reason = sm.verify_chain(log.path)
    assert ok, reason
    assert len(sm.read_records(log.path)) == 2


def test_no_network_is_used_by_shadow_mode(log, monkeypatch):
    def _no_net(*a, **k):
        raise AssertionError("shadow mode attempted a network call")

    monkeypatch.setattr(socket, "create_connection", _no_net)
    _observe(_runner(log, lambda u, b, t: _reply(GOOD)))


# ===========================================================================
# 8. THE CENTRAL PROOF: A == C, trade-for-trade, against the real engine
# ===========================================================================

def _ohlcv(n=400, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.4, 1.2, n).cumsum()
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)


def _trade_fingerprint(result: dict):
    return [
        (t.get("symbol"), t.get("entry_time"), t.get("exit_time"),
         t.get("entry_price"), t.get("exit_price"), t.get("pnl"))
        for t in (result.get("trades") or [])
    ]


def _run_engine(df):
    return ab.engine_mod.simulate_portfolio(
        {"AAA": df}, None, start_equity=10_000.0,
        settings=ab.get_all_settings())


def test_arm_a_equals_arm_c_trade_for_trade(log, monkeypatch):
    """A (AI off) and C (AI shadowing) must be identical in every respect."""
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    df = _ohlcv()

    arm_a = _run_engine(df)

    # Arm C: hook the engine's score function with a Shadow observer whose
    # result is DISCARDED. The engine source is not modified.
    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    original = ab.engine_mod.compute_signal_score

    def _shadow_hook(ind, regime, **kwargs):
        result = original(ind, regime, **kwargs)
        runner.observe(
            symbol="AAA",
            market_data={k: ind.get(k) for k in
                         ("last_close", "ema20", "ema50", "rsi14",
                          "atr14", "adx", "volume_ratio", "bb_bandwidth")},
            decision_ts=str(ind.get("ts", "")),
            feature_asof_ts=str(ind.get("ts", "")),
            settings=_enabled(),
            engine_context={"score": result.get("score"),
                            "would_enter": result.get("score", 0) > 0},
        )
        return result          # unchanged: the observer cannot alter a decision

    ab.engine_mod.compute_signal_score = _shadow_hook
    try:
        arm_c = _run_engine(df)
    finally:
        ab.engine_mod.compute_signal_score = original

    assert _trade_fingerprint(arm_a) == _trade_fingerprint(arm_c)
    assert len(arm_a["trades"]) == len(arm_c["trades"])
    assert arm_a["end_equity"] == arm_c["end_equity"]
    assert arm_a["equity_curve"] == arm_c["equity_curve"]


def test_arm_c_actually_recorded_observations(log, monkeypatch):
    """Guard: the previous test would pass vacuously if nothing was observed."""
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    df = _ohlcv()
    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    original = ab.engine_mod.compute_signal_score

    def _hook(ind, regime, **kwargs):
        result = original(ind, regime, **kwargs)
        runner.observe(symbol="AAA", market_data={"last_close": 1.0},
                       decision_ts="t", feature_asof_ts="t",
                       settings=_enabled())
        return result

    ab.engine_mod.compute_signal_score = _hook
    try:
        _run_engine(df)
    finally:
        ab.engine_mod.compute_signal_score = original

    assert len(sm.read_records(log.path)) > 0


def test_even_a_maximal_veto_does_not_change_trading(log, monkeypatch):
    """Force every observation to be a veto candidate; trading is unaffected."""
    monkeypatch.setattr(mc.time, "sleep", lambda s: None)
    # Raise the budget so every decision point can be observed (the default
    # per-minute cap would otherwise rate-limit most of the run).
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_MINUTE", "100000")
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_DAY", "100000")
    df = _ohlcv()
    always_veto = json.dumps({
        "trend_quality": "STRONG_DOWN", "volatility_regime": "EXTREME",
        "momentum_quality": "REVERSING", "market_risk": "EXTREME",
        "confidence": 1.0, "recommendation": "UNFAVORABLE"})

    arm_a = _run_engine(df)
    runner = _runner(log, lambda u, b, t: _reply(always_veto))
    original = ab.engine_mod.compute_signal_score

    def _hook(ind, regime, **kwargs):
        result = original(ind, regime, **kwargs)
        runner.observe(symbol="AAA", market_data={"last_close": 1.0},
                       decision_ts="t", feature_asof_ts="t", settings=_enabled())
        return result

    ab.engine_mod.compute_signal_score = _hook
    try:
        arm_c = _run_engine(df)
    finally:
        ab.engine_mod.compute_signal_score = original

    records = sm.read_records(log.path)
    observed = [r for r in records if r["parse_status"] == "ok"]
    assert observed, "no successful observations to assert on"
    assert all(r["veto_candidate"] for r in observed)
    assert all(r["veto_applied"] is False for r in records)
    # A maximal, always-on veto still changes nothing.
    assert _trade_fingerprint(arm_a) == _trade_fingerprint(arm_c)
    assert arm_a["end_equity"] == arm_c["end_equity"]


def test_ai_output_cannot_change_a_decision_even_if_mutated(log):
    """The record is inert: mutating it after the fact changes nothing."""
    rec = _observe(_runner(log, lambda u, b, t: _reply(GOOD)))
    before = dict(rec)
    rec["recommendation"] = "FAVORABLE"
    rec["confidence"] = 0.0
    # The returned dict is a plain copy; the log holds the original.
    stored = sm.read_records(log.path)[0]
    assert stored["recommendation"] == before["recommendation"]
    assert stored["confidence"] == before["confidence"]


def test_shadow_mode_module_does_not_import_the_engine():
    """Structural proof that Shadow Mode cannot influence trading."""
    import inspect
    src = inspect.getsource(sm)
    assert "import backtest" not in src
    assert "from backtest" not in src
    assert "simulate_portfolio" not in src


def test_shadow_mode_has_no_order_path():
    import inspect
    src = inspect.getsource(sm).lower()
    for banned in ("submit_order", "place_order", "buy(", "sell(",
                   "execute_trade", "alpaca"):
        assert banned not in src, f"shadow mode must not reference {banned}"


# ===========================================================================
# 9. Disabling AI restores the original path exactly
# ===========================================================================

def test_disabling_ai_restores_the_engine_exactly(log, monkeypatch):
    """With AI disabled there is no observation and no behavioural change."""
    df = _ohlcv()
    arm_a = _run_engine(df)

    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    original = ab.engine_mod.compute_signal_score

    def _hook(ind, regime, **kwargs):
        result = original(ind, regime, **kwargs)
        runner.observe(symbol="AAA", market_data={"last_close": 1.0},
                       decision_ts="t", feature_asof_ts="t",
                       settings={})            # AI disabled
        return result

    ab.engine_mod.compute_signal_score = _hook
    try:
        arm_c = _run_engine(df)
    finally:
        ab.engine_mod.compute_signal_score = original

    assert _trade_fingerprint(arm_a) == _trade_fingerprint(arm_c)
    assert arm_a["end_equity"] == arm_c["end_equity"]
    assert all(r["parse_status"] == "disabled"
               for r in sm.read_records(log.path))


def test_no_hook_means_byte_identical_result():
    """Without any hook the engine result is unchanged by this module."""
    df = _ohlcv()
    a = _run_engine(df)
    b = _run_engine(df)
    assert _trade_fingerprint(a) == _trade_fingerprint(b)
    assert a["end_equity"] == b["end_equity"]


# ===========================================================================
# 10. Strategy / engine source is untouched by this feature
# ===========================================================================

def test_engine_source_does_not_reference_shadow_mode():
    import inspect
    src = inspect.getsource(ab.engine_mod)
    assert "shadow_mode" not in src
    assert "shadow" not in src.lower()


def test_market_context_public_shape_is_unchanged(log):
    """call_model is additive; get_market_context keeps its documented keys."""
    out = mc.get_market_context("AAPL", FEATURES, _enabled(),
                                _transport=lambda u, b, t: _reply(
                                    json.dumps({"bias": "NEUTRAL",
                                                "confidence": 0.5,
                                                "risk_level": "LOW",
                                                "rationale": "x"})))
    assert set(out) == {"available", "enabled", "reason", "bias", "confidence",
                        "risk_level", "rationale", "model", "advisory_only",
                        "affects_trading"}
    assert out["affects_trading"] is False


def test_call_model_is_additive_and_returns_audit_fields(log):
    out = mc.call_model("AAPL", FEATURES, _enabled(),
                        _transport=lambda u, b, t: _reply(GOOD),
                        _prompt_builder=sm.build_shadow_prompt,
                        _validator=sm.validate_shadow_payload)
    for field in ("body_bytes", "raw_text", "parsed", "http_status",
                  "attempts", "latency_ms", "api_key_present"):
        assert field in out
    assert out["parsed"]["recommendation"] == "UNFAVORABLE"


def test_call_model_never_raises_on_a_broken_transport():
    out = mc.call_model("AAPL", FEATURES, _enabled(),
                        _transport=lambda u, b, t: (_ for _ in ()).throw(
                            RuntimeError("boom")))
    assert out["available"] is False
    assert out["affects_trading"] is False


# ===========================================================================
# 11. Determinism and summary
# ===========================================================================

def test_records_are_deterministic_for_identical_input(log):
    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    a = _observe(runner)
    b = _observe(runner)
    for key in ("feature_hash", "prompt_hash", "request_body_hash",
                "config_snapshot_hash", "recommendation", "confidence"):
        assert a[key] == b[key]


def test_summarize_counts_veto_candidates_and_never_applies_them(log):
    runner = _runner(log, lambda u, b, t: _reply(GOOD))
    for _ in range(3):
        _observe(runner)
    s = sm.summarize(sm.read_records(log.path))
    assert s["total"] == 3
    assert s["veto_candidates"] == 3
    assert s["veto_applied"] == 0
    assert s["c_min"] == 0.80
