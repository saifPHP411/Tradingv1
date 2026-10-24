"""Pre-campaign hardening tests: B2 (costs in the config hash), B3 (causal
timestamp enforcement), B4 (feature_source contract) and B8 (one cost model).

These tests exist because a preflight audit found four real gaps:

  B2  cost inputs were outside ``config_snapshot_hash``, so two runs with
      different costs could carry the same "frozen" hash.
  B3  timestamp ordering was recorded but not enforced, so a non-causal
      observation could be written and look valid.
  B4  the ``feature_source`` contract was undocumented and could be satisfied
      by re-hashing the engine's own indicator dict, which proves nothing.
  B8  the engine and the simplified replay disagreed on the SIGN of exit
      slippage, so a round trip cost 2 x bps more in one path than the other.

Nothing here changes a cost value, a threshold, a signal or ``C_MIN``.
"""
from __future__ import annotations

import json

import pytest

import execution_costs as ec
import market_context as mc
import shadow_mode as sm

GOOD = json.dumps({
    "trend_quality": "STRONG_UP", "volatility_regime": "NORMAL",
    "momentum_quality": "STEADY", "market_risk": "LOW",
    "confidence": 0.9, "recommendation": "NEUTRAL",
})

FEATURES = {
    "last_close": 150.0, "ema20": 148.0, "ema50": 145.0,
    "rsi14": 55.0, "atr14": 2.5, "adx": 30.0,
    "volume_ratio": 1.2, "bb_bandwidth": 0.04,
}

# The cost configuration the engine actually consumes.
COST_SETTINGS = {
    "fees_per_share": 0.005,
    "entry_slippage_bps": 5.0,
    "exit_slippage_bps": 5.0,
}


@pytest.fixture(autouse=True)
def _reset_budget():
    mc._reset_budget_for_tests()
    yield
    mc._reset_budget_for_tests()


@pytest.fixture()
def log(tmp_path):
    return sm.ShadowLog(tmp_path / "shadow.jsonl")


def _reply(text=GOOD):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def _runner(log, transport=None, **kw):
    return sm.ShadowRunner(
        log, run_id="hardening",
        transport=transport or (lambda u, b, t: _reply()), **kw)


def _observe(runner, **overrides):
    kwargs = dict(
        symbol="AAPL", market_data=dict(FEATURES),
        decision_ts="2026-11-03T15:59:00Z",
        feature_asof_ts="2026-11-03T15:59:00Z",
        execution_ts="2026-11-03T16:00:00Z",
        settings={"ai_context_enabled": 1},
    )
    kwargs.update(overrides)
    return runner.observe(**kwargs)


def _hash(cost_settings=None, **overrides):
    kwargs = dict(
        model=mc.DEFAULT_MODEL, c_min=sm.C_MIN,
        thinking_level=mc.DEFAULT_THINKING_LEVEL,
        max_output_tokens=mc.DEFAULT_MAX_OUTPUT_TOKENS,
        schema_version=sm.SCHEMA_VERSION,
        cost_inputs=cost_settings,
    )
    kwargs.update(overrides)
    return sm.config_snapshot_hash(**kwargs)


# ===========================================================================
# B2 — cost inputs must be inside config_snapshot_hash
# ===========================================================================

def test_b2_fees_per_share_changes_config_hash():
    base = _hash(COST_SETTINGS)
    other = dict(COST_SETTINGS, fees_per_share=0.01)
    assert base != _hash(other)


def test_b2_entry_slippage_changes_config_hash():
    base = _hash(COST_SETTINGS)
    other = dict(COST_SETTINGS, entry_slippage_bps=10.0)
    assert base != _hash(other)


def test_b2_exit_slippage_changes_config_hash():
    base = _hash(COST_SETTINGS)
    other = dict(COST_SETTINGS, exit_slippage_bps=10.0)
    assert base != _hash(other)


def test_b2_same_config_different_key_order_same_hash():
    a = {"fees_per_share": 0.005, "entry_slippage_bps": 5.0,
         "exit_slippage_bps": 5.0}
    b = {"exit_slippage_bps": 5.0, "fees_per_share": 0.005,
         "entry_slippage_bps": 5.0}
    assert _hash(a) == _hash(b)


def test_b2_missing_cost_inputs_still_hashes():
    """An old record with no costs must still produce a stable hash."""
    assert _hash(None) == _hash({})


def test_b2_short_borrow_mode_is_part_of_the_hash():
    """A campaign cannot claim a cost basis it did not use."""
    a = _hash(COST_SETTINGS)
    b = _hash(COST_SETTINGS, short_borrow_cost_mode="MODELED")
    assert a != b
    assert _hash(COST_SETTINGS, short_borrow_cost_mode="NOT_MODELED") == a


def test_b2_all_cost_keys_are_covered():
    """Every key in COST_SETTING_KEYS must actually move the hash."""
    base = _hash(COST_SETTINGS)
    for key in ec.COST_SETTING_KEYS:
        bumped = dict(COST_SETTINGS)
        bumped[key] = bumped[key] + 1.0
        assert _hash(bumped) != base, f"{key} is not covered by the hash"


def test_b2_cost_keys_match_what_the_engine_consumes():
    """The named cost keys must be exactly the engine's cost inputs."""
    import inspect
    from backtest import engine as engine_mod
    src = inspect.getsource(engine_mod)
    for key in ec.COST_SETTING_KEYS:
        assert f'settings["{key}"]' in src, f"engine does not use {key}"


def test_b2_record_hash_reflects_the_cost_config_actually_used(log):
    """The record's config hash must represent THAT observation's costs."""
    r1 = _runner(log)
    rec_a = _observe(r1, settings={"ai_context_enabled": 1, **COST_SETTINGS})
    rec_b = _observe(
        r1, decision_ts="2026-11-03T16:59:00Z",
        feature_asof_ts="2026-11-03T16:59:00Z",
        execution_ts="2026-11-03T17:00:00Z",
        settings={"ai_context_enabled": 1,
                  **dict(COST_SETTINGS, fees_per_share=0.02)})
    assert rec_a["config_snapshot_hash"] != rec_b["config_snapshot_hash"]
    assert rec_a["cost_inputs"]["fees_per_share"] == 0.005
    assert rec_b["cost_inputs"]["fees_per_share"] == 0.02
    # And the hash is reproducible from the recorded cost inputs alone.
    assert rec_a["config_snapshot_hash"] == _hash(rec_a["cost_inputs"])


def test_b2_record_reports_short_borrow_mode(log):
    rec = _observe(_runner(log), settings={"ai_context_enabled": 1})
    assert rec["short_borrow_cost_mode"] == "NOT_MODELED"


def test_b2_cost_inputs_are_canonicalised(log):
    """Extra settings keys must not leak into the hashed cost mapping."""
    rec = _observe(_runner(log), settings={
        "ai_context_enabled": 1, **COST_SETTINGS,
        "min_signal_score": 70, "take_profit_rr": 2.0})
    assert set(rec["cost_inputs"]) == set(ec.ALL_COST_SETTING_KEYS)


def test_b2_current_cost_values_are_unchanged():
    """This hardening step must not have moved any cost value."""
    import config
    assert config.FEES_PER_SHARE == 0.0
    assert config.ENTRY_SLIPPAGE_BPS == 5
    assert config.EXIT_SLIPPAGE_BPS == 5


# ===========================================================================
# B3 — causal timestamps are enforced, never repaired
# ===========================================================================

def test_b3_feature_asof_after_decision_is_rejected(log):
    with pytest.raises(sm.ShadowOrderingError):
        _observe(_runner(log), decision_ts="2026-11-03T15:59:00Z",
                 feature_asof_ts="2026-11-04T15:59:00Z")


def test_b3_decision_after_execution_is_rejected(log):
    with pytest.raises(sm.ShadowOrderingError):
        _observe(_runner(log), decision_ts="2026-11-03T16:30:00Z",
                 execution_ts="2026-11-03T16:00:00Z")


def test_b3_equal_feature_asof_and_decision_is_accepted(log):
    rec = _observe(_runner(log), decision_ts="2026-11-03T15:59:00Z",
                   feature_asof_ts="2026-11-03T15:59:00Z")
    assert rec["feature_asof_ts"] == rec["decision_ts"]


def test_b3_strictly_increasing_is_accepted(log):
    rec = _observe(_runner(log), feature_asof_ts="2026-11-03T15:58:00Z",
                   decision_ts="2026-11-03T15:59:00Z",
                   execution_ts="2026-11-03T16:00:00Z")
    assert (rec["feature_asof_ts"] <= rec["decision_ts"]
            <= rec["execution_ts"])


def test_b3_rejection_happens_before_any_call_or_write(log):
    """A non-causal call must not reach the transport and must not be logged."""
    calls = {"n": 0}

    def transport(u, b, t):
        calls["n"] += 1
        return _reply()

    with pytest.raises(sm.ShadowOrderingError):
        _observe(_runner(log, transport),
                 feature_asof_ts="2026-12-01T00:00:00Z",
                 decision_ts="2026-11-03T15:59:00Z")
    assert calls["n"] == 0
    assert sm.read_records(log.path) == []


def test_b3_timestamps_are_never_repaired(log):
    """The module must not clamp or substitute a violating timestamp."""
    bad_asof = "2026-11-04T15:59:00Z"
    with pytest.raises(sm.ShadowOrderingError) as exc:
        _observe(_runner(log), decision_ts="2026-11-03T15:59:00Z",
                 feature_asof_ts=bad_asof)
    assert sm.ORDERING_VIOLATION_PREFIX in str(exc.value)
    assert bad_asof in str(exc.value)   # the offending value is reported
    assert sm.read_records(log.path) == []


def test_b3_incomparable_timestamps_are_rejected(log):
    with pytest.raises(sm.ShadowOrderingError):
        _observe(_runner(log), decision_ts=123, feature_asof_ts="2026-01-01")


def test_b3_error_is_a_valueerror_subclass():
    """A caller that already catches ValueError keeps working."""
    assert issubclass(sm.ShadowOrderingError, ValueError)


def test_b3_none_bounds_are_skipped(log):
    """Not supplying a bound is not a violation."""
    rec = _observe(_runner(log), feature_asof_ts=None, execution_ts=None)
    assert rec["feature_asof_ts"] is None
    assert rec["execution_ts"] is None


def test_b3_valid_ordering_recorded_verbatim(log):
    rec = _observe(_runner(log), feature_asof_ts="2026-11-03T15:58:00Z",
                   decision_ts="2026-11-03T15:59:00Z",
                   execution_ts="2026-11-03T16:00:00Z")
    assert rec["feature_asof_ts"] == "2026-11-03T15:58:00Z"
    assert rec["decision_ts"] == "2026-11-03T15:59:00Z"
    assert rec["execution_ts"] == "2026-11-03T16:00:00Z"


# ===========================================================================
# B4 — feature_source contract: independent reconstruction
# ===========================================================================

def test_b4_no_source_is_recorded_as_unavailable(log):
    rec = _observe(_runner(log))
    assert rec["feature_rebuild"]["available"] is False
    assert rec["feature_rebuild"]["matches"] is None
    assert rec["feature_rebuild"]["source"] == "none"


def test_b4_matching_independent_source_reports_match(log):
    """A source that rebuilds the same vector must report matches=True."""
    source = lambda asof: dict(FEATURES)
    rec = _observe(_runner(log, feature_source=source))
    rb = rec["feature_rebuild"]
    assert rb["available"] is True
    assert rb["matches"] is True
    assert rb["source"] == "injected"
    assert rb["rebuilt_hash"] == rec["feature_hash"]


def test_b4_mismatching_source_reports_false_and_the_difference(log):
    """A leaky rebuild must be flagged AND diagnosed, never hidden."""
    leaky = dict(FEATURES, last_close=999.0)
    rec = _observe(_runner(log, feature_source=lambda asof: leaky))
    rb = rec["feature_rebuild"]
    assert rb["available"] is True
    assert rb["matches"] is False
    assert rb["differing_fields"] == ["last_close"]
    assert rb["sent_hash"] == rec["feature_hash"]
    assert rb["rebuilt_hash"] != rec["feature_hash"]


def test_b4_source_raising_is_recorded_not_hidden(log):
    def boom(asof):
        raise RuntimeError("source unavailable")

    rb = _observe(_runner(log, feature_source=boom))["feature_rebuild"]
    assert rb["available"] is False
    assert rb["matches"] is None
    assert rb["error"] == "RuntimeError"


def test_b4_non_mapping_source_is_recorded_not_hidden(log):
    rb = _observe(_runner(log,
                          feature_source=lambda asof: [1, 2, 3]
                          ))["feature_rebuild"]
    assert rb["available"] is False
    assert rb["error"].startswith("non_mapping")


def test_b4_source_receives_the_feature_asof_timestamp(log):
    seen = {}

    def source(asof):
        seen["asof"] = asof
        return dict(FEATURES)

    _observe(_runner(log, feature_source=source),
             feature_asof_ts="2026-11-03T15:58:00Z")
    assert seen["asof"] == "2026-11-03T15:58:00Z"


def test_b4_source_without_asof_timestamp_is_recorded(log):
    rb = _observe(_runner(log, feature_source=lambda a: dict(FEATURES)),
                  feature_asof_ts=None)["feature_rebuild"]
    assert rb["available"] is False
    assert rb["error"] == "missing_feature_asof_ts"


def test_b4_rebuild_uses_raw_ohlcv_only_not_the_engine_dict(log):
    """End-to-end: rebuild the eight features from an OHLCV prefix alone.

    This is the contract that matters. The source is given a dataframe cut at
    the decision bar and must reproduce the sent vector using nothing but that
    OHLCV. If it ever needed a later bar, the hashes would differ.
    """
    import numpy as np
    import pandas as pd
    from backtest.engine import compute_indicators_at

    rng = np.random.default_rng(3)
    n = 300
    close = 100.0 + rng.normal(0.3, 1.0, n).cumsum()
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)

    keys = ("last_close", "ema20", "ema50", "rsi14", "atr14", "adx",
            "volume_ratio", "bb_bandwidth")
    i = 250
    asof = df.index[i]
    sent = {k: compute_indicators_at(df, i)[k] for k in keys}

    def source(cutoff):
        # Causal by construction: only bars up to and including the cutoff.
        prefix = df.loc[:cutoff]
        ind = compute_indicators_at(prefix, len(prefix) - 1)
        return {k: ind[k] for k in keys}

    rec = _observe(_runner(log, feature_source=source),
                   market_data=sent, feature_asof_ts=asof,
                   decision_ts=asof, execution_ts=asof)
    rb = rec["feature_rebuild"]
    assert rb["available"] is True
    assert rb["matches"] is True, rb
    assert rb["rebuilt_hash"] == rec["feature_hash"]


def test_b4_rebuild_from_a_future_bar_is_caught(log):
    """If the source sneaks in a later bar, matches must be False."""
    import numpy as np
    import pandas as pd
    from backtest.engine import compute_indicators_at

    rng = np.random.default_rng(4)
    n = 300
    close = 100.0 + rng.normal(0.3, 1.0, n).cumsum()
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": close - 0.2, "High": close + 1.0, "Low": close - 1.0,
        "Close": close,
        "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)

    keys = ("last_close", "ema20", "ema50", "rsi14", "atr14", "adx",
            "volume_ratio", "bb_bandwidth")
    i = 250
    asof = df.index[i]
    sent = {k: compute_indicators_at(df, i)[k] for k in keys}

    def leaky_source(cutoff):
        # Deliberately uses 30 bars beyond the cutoff.
        future = df.iloc[:i + 30]
        ind = compute_indicators_at(future, len(future) - 1)
        return {k: ind[k] for k in keys}

    rb = _observe(_runner(log, feature_source=leaky_source),
                  market_data=sent, feature_asof_ts=asof,
                  decision_ts=asof, execution_ts=asof)["feature_rebuild"]
    assert rb["matches"] is False
    assert rb["differing_fields"]


def test_b4_no_network_or_live_source_is_wired():
    """This step must not have added a live market-data dependency."""
    import inspect
    src = inspect.getsource(sm)
    for banned in ("yfinance", "requests.get", "urllib.request",
                   "alpaca", "alpaca_trade_api"):
        assert banned not in src, f"unexpected live dependency: {banned}"


# ===========================================================================
# B8 — one cost model, proven to agree with the engine
# ===========================================================================

def test_b8_engine_entry_fill_matches_the_helper():
    from backtest.engine import normalize_slippage
    price = 123.45
    expected = price + normalize_slippage(price, 5.0, True)
    assert ec.entry_fill_price(price, True, COST_SETTINGS) == expected


def test_b8_engine_long_exit_fill_matches_the_helper():
    """The engine's long exit is the reference: it must PAY slippage."""
    from backtest.engine import normalize_slippage
    price = 123.45
    expected = price + normalize_slippage(price, 5.0, False)
    assert ec.exit_fill_price(price, False, COST_SETTINGS) == expected


def test_b8_long_exit_receives_less_than_the_raw_price():
    """The conservative direction: a closing sell must not gain money."""
    fill = ec.exit_fill_price(100.0, False, COST_SETTINGS)
    assert fill < 100.0


def test_b8_short_exit_pays_more_than_the_raw_price():
    """Closing a short is a BUY, so it pays up."""
    fill = ec.exit_fill_price(100.0, True, COST_SETTINGS)
    assert fill > 100.0


def test_b8_the_old_replay_sign_bug_is_gone():
    """Regression: ``- normalize_slippage(..., False)`` was optimistic.

    The simplified replay used to subtract a negative number, handing the
    trade 2 x bps of free money on exit. The helper must never do that.
    """
    from backtest.engine import normalize_slippage
    price = 100.0
    old_buggy = price - normalize_slippage(price, 5.0, False)   # 100.05
    correct = ec.exit_fill_price(price, False, COST_SETTINGS)   # 99.95
    assert correct != old_buggy
    assert correct == price + normalize_slippage(price, 5.0, False)


def test_b8_round_trip_fee_is_both_legs():
    qty = 100
    assert ec.round_trip_fee(qty, COST_SETTINGS) == pytest.approx(1.0)
    assert (ec.entry_fee(qty, COST_SETTINGS)
            + ec.exit_fee(qty, COST_SETTINGS)) == pytest.approx(1.0)


def test_b8_fees_are_zero_at_the_current_value():
    """Current config is FEES_PER_SHARE = 0.0; the helper must honour it."""
    assert ec.round_trip_fee(100, {"fees_per_share": 0.0,
                                   "entry_slippage_bps": 5.0,
                                   "exit_slippage_bps": 5.0}) == 0.0


def test_b8_short_and_long_pay_the_same_fee():
    """Same fee basis; only the slippage sign differs. No borrow cost."""
    assert (ec.entry_fee(50, COST_SETTINGS) == ec.exit_fee(50, COST_SETTINGS))
    assert ec.is_short_borrow_modeled() is False


def test_b8_missing_or_bad_cost_values_fall_back_safely():
    assert ec.resolve_cost_inputs(None) == ec.COST_DEFAULTS
    assert ec.resolve_cost_inputs({"fees_per_share": "nonsense"})[
        "fees_per_share"] == ec.COST_DEFAULTS["fees_per_share"]
    assert ec.resolve_cost_inputs({"fees_per_share": True})[
        "fees_per_share"] == ec.COST_DEFAULTS["fees_per_share"]


def test_b8_canonical_inputs_are_sorted_and_complete():
    canon = ec.canonical_cost_inputs({"exit_slippage_bps": 5.0})
    assert list(canon) == list(ec.COST_SETTING_KEYS)
    assert canon["fees_per_share"] == 0.0


def test_b8_replay_module_uses_the_canonical_helper():
    """The simplified replay must delegate, not re-derive, the fill price."""
    import inspect
    from evaluation import ab_ai_context as ab
    src = inspect.getsource(ab)
    assert "execution_costs" in src
    # The old optimistic expression must be gone.
    assert "- normalize_slippage(exit_px" not in src


def test_b8_replay_and_helper_agree_on_exit_fill():
    """Whatever the replay does, it must equal the canonical exit fill."""
    from evaluation import ab_ai_context as ab
    from risk_utils import normalize_slippage
    if not hasattr(ab, "_exit_fill"):
        pytest.skip("replay exposes no isolated exit-fill helper")
    price = 100.0
    assert ab._exit_fill(price, False, 5.0) == ec.exit_fill_price(
        price, False, COST_SETTINGS)
    assert ab._exit_fill(price, False, 5.0) == (
        price + normalize_slippage(price, 5.0, False))
