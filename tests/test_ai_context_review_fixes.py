"""
Regression tests for the independent-review fixes (priority 1-3).

Fix 1 - RSI field-name mismatch
    ``backtest.engine.compute_indicators_at`` emits ``rsi14``, but the AI
    allow-list expected ``rsi``, so RSI was silently dropped from every
    request. These tests use the REAL engine keys (not hand-made dicts) and
    prove the value reaches the request body while non-allow-listed fields
    do not leak.

Fix 2 - fail-safe configuration
    ``get_market_context`` / ``is_enabled`` documented "NEVER raises" but
    raised on malformed settings objects and malformed environment values.

Fix 3 - boolean settings round-trip
    ``update_settings({"x": True})`` stored ``"True"``, and ``int("True")``
    then crashed ``get_all_settings()`` (and therefore the monitor loop).

Deterministic, offline. No network, no credentials, no broker.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import market_context as mc  # noqa: E402


# ===========================================================================
# Fix 1 - RSI reaches the request body (real engine keys)
# ===========================================================================

def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    """Deterministic price frame with enough bars for compute_indicators_at."""
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    base = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0)) + np.arange(n) * 0.1
    return pd.DataFrame({
        "Open": base - 0.2,
        "High": base + 0.6,
        "Low": base - 0.6,
        "Close": base,
        "Volume": np.full(n, 1_500_000.0),
    }, index=idx)


def test_engine_indicator_keys_are_the_real_ones():
    """Guard: the engine really does emit ``rsi14``, not ``rsi``."""
    from backtest.engine import compute_indicators_at
    df = _synthetic_ohlcv()
    ind = compute_indicators_at(df, 100)
    assert ind is not None
    assert "rsi14" in ind, "engine key changed; update the alias map"
    assert "rsi" not in ind


def test_rsi_from_real_engine_keys_reaches_the_request_body(monkeypatch):
    """End-to-end: engine dict -> sanitize -> prompt/body contains RSI."""
    from backtest.engine import compute_indicators_at
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    df = _synthetic_ohlcv()
    ind = compute_indicators_at(df, 100)

    seen = {}

    def _transport(url, body, timeout):
        seen["body"] = body.decode("utf-8")
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(
            {"bias": "NEUTRAL", "confidence": 0.5,
             "risk_level": "LOW", "rationale": "ok"})}]}}]}

    out = mc.get_market_context("AAPL", ind, {"ai_context_enabled": 1},
                                _transport=_transport)
    assert out["available"] is True

    # The prompt text is nested JSON inside the request body; unwrap it so we
    # assert on the feature payload itself rather than on escaping artefacts.
    outer = json.loads(seen["body"])
    prompt_text = outer["contents"][0]["parts"][0]["text"]
    assert '"rsi"' in prompt_text, "RSI label missing from the request body"

    expected_rsi = round(float(ind["rsi14"]), 6)
    assert str(expected_rsi) in prompt_text, (
        f"RSI value {expected_rsi} did not reach the request body")
    assert "rsi14" not in prompt_text, "raw engine key leaked into the prompt"


def test_sanitize_accepts_both_rsi_spellings():
    """Either spelling works, and both normalise to the prompt label."""
    from_engine = mc.sanitize_market_features({"rsi14": 41.5})
    from_label = mc.sanitize_market_features({"rsi": 41.5})
    assert from_engine == {"rsi": 41.5}
    assert from_label == {"rsi": 41.5}


def test_sanitize_prefers_engine_key_when_both_present():
    out = mc.sanitize_market_features({"rsi14": 41.5, "rsi": 99.9})
    assert out == {"rsi": 41.5}


def test_rsi_still_rejected_when_non_numeric():
    assert mc.sanitize_market_features({"rsi14": "41.5"}) == {}
    assert mc.sanitize_market_features({"rsi14": None}) == {}
    assert mc.sanitize_market_features({"rsi14": float("nan")}) == {}


def test_non_allowlisted_fields_never_leak_from_engine_dict():
    """The full engine dict carries more fields than the allow-list."""
    from backtest.engine import compute_indicators_at
    df = _synthetic_ohlcv()
    ind = compute_indicators_at(df, 100)

    # Simulate a caller that enriched the dict with sensitive data.
    dirty = dict(ind)
    dirty.update({
        "account_equity": 123456.78,
        "buying_power": 500000.0,
        "position_qty": 300,
        "unrealized_pnl": -12.5,
        "api_key": "SECRET-VALUE",
    })

    kept = mc.sanitize_market_features(dirty)
    # Every kept key is on the allow-list.
    assert set(kept) <= set(mc._ALLOWED_MARKET_FIELDS)
    # RSI survived.
    assert "rsi" in kept
    # Nothing sensitive did.
    for leaked in ("account_equity", "buying_power", "position_qty",
                   "unrealized_pnl", "api_key", "SECRET-VALUE",
                   "macd_hist", "stoch_k", "momentum_5d_pct"):
        assert leaked not in kept


def test_engine_extra_fields_do_not_appear_in_request_body(monkeypatch):
    from backtest.engine import compute_indicators_at
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    df = _synthetic_ohlcv()
    ind = compute_indicators_at(df, 100)

    seen = {}

    def _transport(url, body, timeout):
        seen["body"] = body.decode("utf-8")
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(
            {"bias": "NEUTRAL", "confidence": 0.5,
             "risk_level": "LOW", "rationale": "ok"})}]}}]}

    mc.get_market_context("AAPL", ind, {"ai_context_enabled": 1},
                          _transport=_transport)
    outer = json.loads(seen["body"])
    prompt_text = outer["contents"][0]["parts"][0]["text"]
    for not_allowed in ("macd_hist", "stoch_k", "stoch_d",
                        "momentum_5d_pct", "momentum_10d_pct", "sma20", "sma50"):
        assert not_allowed not in prompt_text


# ===========================================================================
# Fix 2 - fail-safe configuration (never raises)
# ===========================================================================

@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def _blocked(*a, **k):
        raise AssertionError("test attempted a real Gemini network call")
    monkeypatch.setattr(mc, "_post_generate_content", _blocked)
    mc._reset_budget_for_tests()


@pytest.mark.parametrize("bad", [
    "true", "false", "abc", "[1]", "yes", "", "  ",
    {}, [1], [], None, -5, 10 ** 12, 3.7, True, False, object(),
])
def test_malformed_ai_context_flag_never_raises(bad):
    out = mc.get_market_context("AAPL", {"last_close": 1.0},
                                {"ai_context_enabled": bad})
    assert out["available"] is False
    assert out["affects_trading"] is False


@pytest.mark.parametrize("bad", ["true", "abc", "[1]", {}, [1], None, -5,
                                 10 ** 12, object()])
def test_is_enabled_never_raises(bad):
    assert mc.is_enabled({"ai_context_enabled": bad}) in (True, False)


@pytest.mark.parametrize("settings", [None, ["x"], "str", 5, object()])
def test_non_dict_settings_never_raises(settings):
    out = mc.get_market_context("AAPL", {"last_close": 1.0}, settings)
    assert out["available"] is False
    assert mc.is_enabled(settings) is False


@pytest.mark.parametrize("env_var,bad", [
    ("GEMINI_TIMEOUT_SECONDS", "abc"),
    ("GEMINI_TIMEOUT_SECONDS", ""),
    ("GEMINI_TIMEOUT_SECONDS", "-1"),
    ("GEMINI_TIMEOUT_SECONDS", "1e999"),
    ("GEMINI_MAX_CALLS_PER_MINUTE", "lots"),
    ("GEMINI_MAX_CALLS_PER_MINUTE", "-3"),
    ("GEMINI_MAX_CALLS_PER_DAY", "0"),
    ("GEMINI_MAX_CALLS_PER_DAY", "9" * 40),
])
def test_malformed_env_never_raises(monkeypatch, env_var, bad):
    monkeypatch.setenv(env_var, bad)
    cfg = mc._cfg({"ai_context_enabled": 1})
    assert cfg["timeout"] > 0
    assert cfg["max_per_minute"] >= 1
    assert cfg["max_per_day"] >= 1


def test_env_values_are_clamped_to_sane_bounds(monkeypatch):
    monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "100000")
    monkeypatch.setenv("GEMINI_MAX_CALLS_PER_MINUTE", "999999999")
    cfg = mc._cfg({"ai_context_enabled": 1})
    assert cfg["timeout"] <= mc._MAX_TIMEOUT_SECONDS
    assert cfg["max_per_minute"] <= mc._MAX_CALLS_PER_MINUTE


def test_safe_coercion_helpers_do_not_raise():
    assert mc._safe_int("12", 5) == 12
    assert mc._safe_int("abc", 5) == 5
    assert mc._safe_int(True, 5) == 5            # bool is not a count
    assert mc._safe_int(None, 5) == 5
    assert mc._safe_int(10 ** 9, 5, maximum=100) == 100
    assert mc._safe_int(-9, 5, minimum=1) == 1
    assert mc._safe_float("1.5", 2.0) == 1.5
    assert mc._safe_float("abc", 2.0) == 2.0
    assert mc._safe_float(float("nan"), 2.0) == 2.0
    assert mc._safe_float(float("inf"), 2.0) == 2.0
    assert mc._safe_float("1e999", 2.0) == 2.0
    assert mc._safe_flag("true") is True
    assert mc._safe_flag("false") is False
    assert mc._safe_flag("abc") is False          # fails closed
    assert mc._safe_flag(1) is True
    assert mc._safe_flag(True) is True


def test_malformed_flag_fails_closed_not_open():
    """A garbage flag must not silently ENABLE the advisory layer."""
    assert mc._cfg({"ai_context_enabled": "maybe"})["enabled"] is False
    assert mc._cfg({"ai_context_enabled": "true"})["enabled"] is True
    assert mc._cfg({"ai_context_enabled": "1"})["enabled"] is True
    assert mc._cfg({"ai_context_enabled": 0})["enabled"] is False


def test_disabled_still_means_no_network(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    calls = {"n": 0}
    real = mc._post_generate_content

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(mc, "_post_generate_content", _spy)

    out = mc.get_market_context("AAPL", {"last_close": 1.0},
                                {"ai_context_enabled": 0})
    assert out["reason"] == "disabled"
    assert calls["n"] == 0


# ===========================================================================
# Fix 3 - boolean settings round-trip
# ===========================================================================

@pytest.fixture
def fresh_settings(tmp_path, monkeypatch):
    """Reload decision_log + settings against an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "settings.db"))
    import decision_log
    importlib.reload(decision_log)
    import settings as settings_mod
    importlib.reload(settings_mod)
    return decision_log, settings_mod


@pytest.mark.parametrize("key,value,expected", [
    ("ai_context_enabled", True, 1),
    ("ai_context_enabled", False, 0),
    ("ai_context_enabled", "true", 1),
    ("ai_context_enabled", "false", 0),
    ("regime_enabled", True, 1),
    ("regime_enabled", False, 0),
    ("news_enabled", True, 1),
    ("news_enabled", False, 0),
])
def test_bool_settings_round_trip(fresh_settings, key, value, expected):
    decision_log, settings_mod = fresh_settings
    settings_mod.update_settings({key: value})
    # Stored canonically, never as "True"/"False".
    assert decision_log.get_setting(key) in ("true", "false")
    assert settings_mod.get_all_settings()[key] == expected


def test_json_bool_write_no_longer_crashes_get_all_settings(fresh_settings):
    """The exact defect from the review: POST true -> int('True') -> crash."""
    decision_log, settings_mod = fresh_settings
    settings_mod.update_settings({"regime_enabled": True})
    assert decision_log.get_setting("regime_enabled") == "true"
    got = settings_mod.get_all_settings()          # must not raise
    assert got["regime_enabled"] == 1


def test_allow_extended_hours_round_trip(fresh_settings):
    _, settings_mod = fresh_settings
    settings_mod.update_settings({"allow_extended_hours": True})
    assert settings_mod.get_all_settings()["allow_extended_hours"] is True
    settings_mod.update_settings({"allow_extended_hours": "false"})
    assert settings_mod.get_all_settings()["allow_extended_hours"] is False


@pytest.mark.parametrize("bad", ["abc", "[1]", "{}", "not-a-number", "1,5"])
def test_invalid_numeric_value_falls_back_to_default(fresh_settings, bad):
    """A bad write must not crash, and must not invent a trading value."""
    decision_log, settings_mod = fresh_settings
    settings_mod.get_all_settings()                # initialise defaults
    decision_log.set_setting("min_signal_score", bad)
    got = settings_mod.get_all_settings()          # must not raise
    assert got["min_signal_score"] == int(float(
        settings_mod.DEFAULTS["min_signal_score"]))


def test_invalid_float_value_falls_back_to_default(fresh_settings):
    decision_log, settings_mod = fresh_settings
    settings_mod.get_all_settings()
    decision_log.set_setting("stop_atr_multiplier", "garbage")
    got = settings_mod.get_all_settings()
    assert got["stop_atr_multiplier"] == float(
        settings_mod.DEFAULTS["stop_atr_multiplier"])


def test_invalid_symbols_json_does_not_crash(fresh_settings):
    decision_log, settings_mod = fresh_settings
    settings_mod.get_all_settings()
    decision_log.set_setting("symbols", "{not valid json")
    assert settings_mod.get_all_settings()["symbols"] == []


def test_valid_legacy_values_are_preserved(fresh_settings):
    """Correct existing behaviour must not regress."""
    _, settings_mod = fresh_settings
    settings_mod.get_all_settings()                # initialise at current version
    settings_mod.update_settings({"min_signal_score": 45})
    assert settings_mod.get_all_settings()["min_signal_score"] == 45
    settings_mod.update_settings({"max_open_positions": 7})
    assert settings_mod.get_all_settings()["max_open_positions"] == 7


def test_symbols_list_still_accepted(fresh_settings):
    _, settings_mod = fresh_settings
    settings_mod.update_settings({"symbols": ["AAPL", "MSFT"]})
    assert settings_mod.get_all_settings()["symbols"] == ["AAPL", "MSFT"]


def test_non_dict_payload_is_ignored(fresh_settings):
    _, settings_mod = fresh_settings
    settings_mod.get_all_settings()
    settings_mod.update_settings(["not", "a", "dict"])   # must not raise
    assert isinstance(settings_mod.get_all_settings(), dict)


def test_unknown_and_legacy_keys_are_still_rejected(fresh_settings):
    decision_log, settings_mod = fresh_settings
    settings_mod.get_all_settings()      # migration blanks legacy keys to ""
    legacy_before = decision_log.get_setting("hard_stop_loss_pct")
    settings_mod.update_settings({"hard_stop_loss_pct": 5, "nonsense_key": 1})
    # update_settings must not write legacy or unknown keys.
    assert decision_log.get_setting("hard_stop_loss_pct") == legacy_before
    assert decision_log.get_setting("nonsense_key") is None


def test_ai_context_enabled_defaults_off(fresh_settings):
    _, settings_mod = fresh_settings
    assert settings_mod.get_all_settings()["ai_context_enabled"] == 0
