"""
Phase-1 audit: metrics, data handling, walk-forward robustness, and
no-lookahead regression tests.

Covers the remaining unfinished Phase 1 items:
  A. Metrics edge cases (zero trades, all wins, all losses, zero variance,
     zero/negative equity, complete loss, insufficient observations).
  B. Data handling (OHLC validity, duplicates, ordering, timezone,
     cache normalization, empty/corrupt frames).
  C. Walk-forward aggregation with invalid returns (NaN, +/-inf, < -100%,
     exactly -100%).
  D. No-lookahead: execution-time gates must not consume the current bar's
     close.
  E. Settings propagation: every key the engine reads is produced by
     get_all_settings().

Deterministic synthetic data only. No network, no credentials.

Run:
    python -m pytest tests/test_phase1_audit_metrics_data.py -v
"""
import math

import numpy as np
import pandas as pd
import pytest

from backtest.data import _normalize
from backtest.engine import simulate_portfolio, PortfolioState
from backtest.metrics import calculate_metrics
from backtest.walkforward import aggregate_walkforward, _naive_ts


def _settings(**overrides):
    s = {
        "min_confidence": 0.7,
        "max_risk_per_trade": 0.005,
        "max_open_positions": 3,
        "max_trades_per_day": 50,
        "max_daily_loss_pct": 0.02,
        "max_position_pct": 0.10,
        "max_portfolio_exposure_pct": 0.95,
        "max_consecutive_losses": 3,
        "cooldown_minutes": 120,
        "stop_atr_multiplier": 2.5,
        "take_profit_rr": 2.0,
        "trailing_activation_r": 1.5,
        "trailing_distance_r": 1.0,
        "min_stop_move_pct": 0.3,
        "min_signal_score": 50,
        "min_hold_minutes": 30,
        "min_loss_to_close_pct": 1.5,
        "llm_exit_min_confidence": 0.7,
        "exit_check_seconds": 60,
        "exit_llm_interval_seconds": 300,
        "allow_extended_hours": False,
        "cycle_interval_minutes": 15,
        "fees_per_share": 0.0,
        "entry_slippage_bps": 0,
        "exit_slippage_bps": 0,
        "trading_capital": 0,
        "regime_enabled": 0,
        "news_enabled": 0,
        "symbols": [],
        "settings_version": 3,
        "dry_run": True,
        "trading_mode": "BACKTEST",
        "enable_short": 0,
        "max_hold_bars": 60,
    }
    s.update(overrides)
    return s


def _curve(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return [{"time": t, "equity": float(v)} for t, v in zip(idx, values)]


def _trade(pnl, r=1.0):
    return {
        "net_pnl": float(pnl), "r_multiple": float(r), "bars_held": 1,
        "fees_entry": 0.0, "fees_exit": 0.0,
        "slippage_entry": 0.0, "slippage_exit": 0.0,
    }


# ==================== A. Metrics edge cases ====================

def test_metrics_zero_trades_returns_clean_numbers():
    m = calculate_metrics(_curve([10_000, 10_100, 10_050]), [], 10_000)
    assert m["num_trades"] == 0
    assert m["total_return_pct"] == pytest.approx(0.5, abs=0.01)
    assert m["equity_ruined"] is False


def test_metrics_all_winning_trades():
    m = calculate_metrics(_curve([10_000, 10_100, 10_200]),
                          [_trade(50), _trade(50)], 10_000)
    assert m["win_rate_pct"] == 100.0
    assert m["avg_loss"] == 0.0
    # No losing trades → profit factor is undefined, not inf or a crash.
    assert m["profit_factor"] is None


def test_metrics_all_losing_trades():
    m = calculate_metrics(_curve([10_000, 9_900, 9_800]),
                          [_trade(-50), _trade(-50)], 10_000)
    assert m["win_rate_pct"] == 0.0
    assert m["avg_win"] == 0.0
    assert m["profit_factor"] == 0.0


def test_metrics_zero_variance_flat_equity():
    m = calculate_metrics(_curve([10_000, 10_000, 10_000]), [], 10_000)
    assert m["sharpe"] == 0.0
    assert m["sortino"] == 0.0
    assert m["max_drawdown_pct"] == 0.0


def test_metrics_zero_equity_midcurve_emits_no_inf_or_warning():
    """
    A zero-equity point used to divide by zero (RuntimeWarning) and emit
    inf/NaN. It must now be handled: returns stay finite.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m = calculate_metrics(_curve([10_000, 5_000, 0, 5_000]), [], 10_000)
    for k, v in m.items():
        if isinstance(v, float):
            assert math.isfinite(v), f"{k} is not finite: {v}"
    assert m["equity_ruined"] is True


def test_metrics_negative_equity_is_clamped():
    """Negative equity is impossible for a cash-secured account; clamp to -100%."""
    m = calculate_metrics(_curve([10_000, 5_000, -1_000]), [], 10_000)
    assert m["total_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert m["max_drawdown_pct"] >= -100.0
    assert m["equity_ruined"] is True


def test_metrics_complete_loss_to_zero():
    m = calculate_metrics(_curve([10_000, 0]), [], 10_000)
    assert m["total_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert m["cagr_pct"] == pytest.approx(-100.0, abs=0.01)
    assert m["equity_ruined"] is True


def test_metrics_insufficient_observations():
    assert calculate_metrics([], [], 10_000) == {"error": "no data"}
    assert calculate_metrics(_curve([10_000]), [], 10_000) == {"error": "no data"}


def test_metrics_non_finite_points_are_dropped():
    curve = _curve([10_000, 10_100, 10_200])
    curve[1]["equity"] = float("nan")
    m = calculate_metrics(curve, [], 10_000)
    assert math.isfinite(m["total_return_pct"])


def test_metrics_zero_initial_equity_does_not_divide_by_zero():
    m = calculate_metrics(_curve([100, 110, 120]), [], 0)
    assert m["total_return_pct"] == 0.0
    assert math.isfinite(m["cagr_pct"])


def test_metrics_known_max_drawdown():
    m = calculate_metrics(_curve([10_000, 12_000, 9_000, 11_000]), [], 10_000)
    # peak 12,000 -> trough 9,000 = -25%
    assert m["max_drawdown_pct"] == pytest.approx(-25.0, abs=0.01)


# ==================== B. Data handling ====================

def _raw_df(rows, idx):
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


def test_normalize_sorts_and_dedupes_timestamps():
    idx = pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"])
    rows = [
        [10, 11, 9, 10, 100],
        [10, 11, 9, 10, 100],
        [10, 11, 9, 10, 100],
        [99, 99, 99, 99, 999],  # duplicate timestamp; keep first
    ]
    out = _normalize(_raw_df(rows, idx))
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    assert len(out) == 3
    # The first-seen row for 2024-01-02 is kept (Close 10, not 99).
    assert float(out.loc["2024-01-02", "Close"]) == 10.0


def test_normalize_converts_tz_aware_index_to_naive_utc():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="America/New_York")
    rows = [[10, 11, 9, 10, 100]] * 5
    out = _normalize(_raw_df(rows, idx))
    assert out.index.tz is None
    # Comparisons against naive boundaries must work.
    assert (out.index >= pd.Timestamp("2024-01-01")).all()


def test_normalize_drops_missing_and_invalid_ohlc():
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = [
        [10, 11, 9, 10, 100],       # valid
        [10, 11, 9, np.nan, 100],   # missing close
        [10, 9, 11, 10, 100],       # high < low
        [0, 1, 0, 0.5, 100],        # non-positive open/low
    ]
    out = _normalize(_raw_df(rows, idx))
    assert len(out) == 1


def test_normalize_rejects_missing_columns():
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    bad = pd.DataFrame({"Open": [1, 2], "Close": [1, 2]}, index=idx)
    with pytest.raises(ValueError):
        _normalize(bad)


def test_normalize_handles_empty_frame():
    idx = pd.DatetimeIndex([])
    out = _normalize(_raw_df([], idx))
    assert out.empty


def test_engine_accepts_tz_aware_index_without_typeerror():
    """
    Regression: yfinance returns a tz-aware index; the engine must not raise
    TypeError when comparing bars to window boundaries.
    """
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    closes = 100 + np.arange(n) * 0.5
    df = _raw_df([[c, c * 1.02, c * 0.98, c, 1_000_000] for c in closes], idx)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=_settings(), max_hold_bars=5)
    assert result["end_equity"] > 0


def test_engine_sorts_direct_input_before_building_indicator_windows(monkeypatch):
    """Unsorted caller data must not put later bars in an earlier prefix."""
    import backtest.engine as engine

    observed = []

    def inspect_prefix(df, i):
        prefix = df.iloc[:i + 1]
        observed.append((prefix.index.is_monotonic_increasing, prefix.index[-1]))
        return None  # This test isolates chronology, not strategy decisions.

    monkeypatch.setattr(engine, "compute_indicators_at", inspect_prefix)
    idx = pd.date_range("2024-01-01", periods=70, freq="D")
    closes = np.linspace(100.0, 110.0, len(idx))
    df = _raw_df([[c, c + 1, c - 1, c, 1_000] for c in closes], idx)
    unsorted = df.iloc[::-1].copy()

    engine.simulate_portfolio({"AAA": unsorted}, None, start_equity=10_000,
                              settings=_settings())

    assert observed, "expected indicator evaluation after warmup"
    assert observed == [(True, ts) for ts in idx[60:]]


def test_engine_normalizes_naive_window_boundary_against_tz_index():
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    closes = 100 + np.arange(n) * 0.5
    df = _raw_df([[c, c * 1.02, c * 0.98, c, 1_000_000] for c in closes], idx)
    start = pd.Timestamp("2024-04-01")
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(), trading_start_ts=start)
    for t in result["trades"]:
        assert _naive_ts(t["entry_time"]) >= start


# ==================== C. Walk-forward robustness ====================

def test_naive_ts_normalizes_both_forms():
    assert _naive_ts("2024-01-01") == pd.Timestamp("2024-01-01")
    aware = pd.Timestamp("2024-01-01", tz="America/New_York")
    assert _naive_ts(aware).tz is None


def test_aggregate_rejects_nan_window():
    agg = aggregate_walkforward([
        {"metrics": {"num_trades": 1, "total_return_pct": float("nan"), "sharpe": 0}},
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
    ])
    assert agg["num_valid_windows"] == 1
    assert math.isfinite(agg["cumulative_return_pct"])
    assert agg["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)
    assert agg["invalid_windows"]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_aggregate_rejects_infinite_window(bad):
    agg = aggregate_walkforward([
        {"metrics": {"num_trades": 1, "total_return_pct": bad, "sharpe": 0}},
    ])
    assert agg["num_valid_windows"] == 0
    assert math.isfinite(agg["cumulative_return_pct"])
    assert agg["cumulative_return_pct"] == 0.0


def test_aggregate_clamps_return_below_minus_100():
    agg = aggregate_walkforward([
        {"metrics": {"num_trades": 1, "total_return_pct": -150.0, "sharpe": 0}},
        {"metrics": {"num_trades": 1, "total_return_pct": 50.0, "sharpe": 0}},
    ])
    # -150% clamped to -100% → compounded (0.0)*(1.5) - 1 = -100%
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert agg["invalid_windows"][0]["reason"].startswith("return below -100%")


def test_aggregate_exactly_minus_100_is_valid():
    agg = aggregate_walkforward([
        {"metrics": {"num_trades": 1, "total_return_pct": -100.0, "sharpe": 0}},
    ])
    assert agg["num_valid_windows"] == 1
    assert "invalid_windows" not in agg
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)


def test_aggregate_compounds_and_does_not_sum():
    agg = aggregate_walkforward([
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
    ])
    assert agg["cumulative_return_pct"] == pytest.approx(21.0, abs=0.01)


def test_aggregate_empty_results():
    assert aggregate_walkforward([]) == {"error": "no windows"}


def test_filter_to_window_excludes_warmup_and_tolerates_tz():
    """
    Regression: _filter_to_window compared pd.Timestamp(p["time"]) (possibly
    tz-aware) against a naive boundary and raised TypeError. It must now keep
    only points/trades at or after the OOS start, regardless of tz-awareness.
    """
    from backtest.walkforward import _filter_to_window
    start = pd.Timestamp("2024-02-01")
    aware = pd.Timestamp("2024-03-01", tz="America/New_York")
    curve = [
        {"time": pd.Timestamp("2024-01-15"), "equity": 10_000},
        {"time": aware, "equity": 10_100},
    ]
    trades = [
        {"entry_time": pd.Timestamp("2024-01-20"), "net_pnl": 5},
        {"entry_time": pd.Timestamp("2024-02-10"), "net_pnl": 7},
    ]
    c, t = _filter_to_window(curve, trades, start)
    assert [p["equity"] for p in c] == [10_100]      # pre-window point dropped
    assert [x["net_pnl"] for x in t] == [7]          # pre-window trade dropped


def test_walk_forward_windows_do_not_overlap_or_leak(monkeypatch):
    """
    End-to-end walk-forward with synthetic data: every reported trade must
    fall inside its window, and no window may contain a trade from the warmup
    that precedes it.
    """
    import backtest.walkforward as wf

    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    closes = 100 + np.arange(n) * 0.4
    df = pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": [1_000_000] * n,
    }, index=idx)
    monkeypatch.setattr(wf, "load_ohlcv", lambda *a, **k: df)
    monkeypatch.setattr(wf, "get_spy_for_period", lambda *a, **k: None)

    results = wf.walk_forward("AAA", "2023-01-01", "2024-01-01",
                              train_months=6, test_months=3,
                              settings=_settings(max_hold_bars=5))
    assert results, "expected at least one window"

    prev_end = None
    for r in results:
        win_start, win_end = r["window"].split("_to_")
        ws = pd.Timestamp(win_start)
        we = pd.Timestamp(win_end)
        if prev_end is not None:
            assert ws >= prev_end, "walk-forward windows must not overlap"
        prev_end = we
        for t in r["result"]["trades"]:
            et = pd.Timestamp(t["entry_time"])
            assert ws <= et, "trade leaked from warmup into OOS window"


# ==================== D. No-lookahead execution gates ====================

def test_execution_gate_uses_prior_close_not_current_close():
    """
    The daily-loss gate must not see the current bar's close before it
    happens. Driven directly through PortfolioState: with a real open LONG
    position, equity at the gate price is what decides the gate, so a gate
    price that differs from the "current" price must produce a different
    verdict.
    """
    s = _settings(max_daily_loss_pct=0.02)
    state = PortfolioState(100_000.0, s)
    state.daily_start_equity = 100_000.0

    from backtest.engine import PortfolioPosition, LONG
    pos = PortfolioPosition("AAA", LONG, 100.0, 100.0, 95.0, 110.0, None, 0)
    state.open_position(pos, 0.0, 0.0)   # cash = 90,000

    # Prior close 100 → equity 100,000 → flat day → allowed.
    allowed, _ = state.can_trade("2024-01-02T00:00:00", {"AAA": 100.0})
    assert allowed is True

    # A crash to 70 → equity 97,000 → -3% → blocked. If the engine fed the
    # current close to this gate, the crash would retroactively veto an entry
    # that was decided before it happened.
    blocked, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 70.0})
    assert blocked is False and why == "daily_loss"


def test_engine_execution_gate_receives_prior_bar_close(monkeypatch):
    """
    Behavioral no-lookahead guard (not a source-string check).

    Wrap PortfolioState.can_trade and record the price map it receives. For
    the bar on which a queued entry executes, the FIRST call in that bar's
    iteration is the execution-time recheck, and it must be given the PRIOR
    bar's close — never the bar's own close.
    """
    from backtest.engine import PortfolioState

    recorded: list[tuple[str, dict]] = []
    original = PortfolioState.can_trade

    def spy(self, now_iso, prices):
        recorded.append((now_iso, dict(prices)))
        return original(self, now_iso, prices)

    monkeypatch.setattr(PortfolioState, "can_trade", spy)

    # Rise through index 65 so the entry bar's close differs from the prior
    # bar's close, then flatten (keeping stop/target out of reach).
    rise = list(100 + np.arange(66) * 0.5)
    flat = [rise[-1]] * 20
    closes = rise + flat
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": closes, "High": [c * 1.001 for c in closes],
        "Low": [c * 0.999 for c in closes], "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)

    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(), max_hold_bars=5)
    assert result["trades"], "expected a trade so an execution gate runs"

    entry_bar = pd.Timestamp(result["trades"][0]["entry_time"])
    assert entry_bar == idx[61]
    prior_close = float(df.loc[idx[61 - 1], "Close"])
    this_close = float(df.loc[entry_bar, "Close"])
    assert prior_close != this_close, "dataset must distinguish the two closes"

    calls = [p for ts, p in recorded if pd.Timestamp(ts) == entry_bar]
    assert calls, "no can_trade call recorded for the entry bar"
    # Execution runs before signal generation within a bar iteration.
    assert calls[0]["AAA"] == pytest.approx(prior_close), (
        "execution-time gate used the current bar close (lookahead)"
    )



# ==================== F. Execution timing: gaps, conflicts, trailing ====================

def _timing_settings(**over):
    s = _settings()
    s.update({
        "max_risk_per_trade": 0.02,
        "max_position_pct": 1.0,
        "max_portfolio_exposure_pct": 0.99,
        "max_daily_loss_pct": 0.5,
        "min_loss_to_close_pct": 99.0,
        "cooldown_minutes": 0,
        "min_signal_score": 30,
        "max_open_positions": 1,
    })
    s.update(over)
    return s


def _entry_df(extra_bars):
    """
    Rising series (so a LONG signal fires at index 60), then `extra_bars`
    caller-supplied candles. Returns (df, idx, entry_index=61).
    """
    rise = list(100 + np.arange(61) * 0.5)
    idx = pd.date_range("2024-01-01", periods=61 + len(extra_bars), freq="D")
    rows = []
    for c in rise:
        rows.append([c, c * 1.0005, c * 0.9995, c, 1_000_000])
    rows.extend(extra_bars)
    df = pd.DataFrame(rows, index=idx,
                      columns=["Open", "High", "Low", "Close", "Volume"])
    return df, idx


def _entry_df_rising(n_extra, step=0.5):
    """
    Rising series where the entry bar (index 61) GAPS UP from the signal bar's
    close, so the fill price proves next-bar execution. Keeps rising afterwards.
    """
    closes = [100 + i * step for i in range(61)]
    for _ in range(n_extra):
        closes.append(closes[-1] + step)
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    opens, highs, lows = [], [], []
    for i, c in enumerate(closes):
        o = closes[i - 1] + step * 0.5 if i > 0 else c
        opens.append(o)
        highs.append(max(o, c) + step * 0.1)
        lows.append(min(o, c) - step * 0.1)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows,
                       "Close": closes, "Volume": [1_000_000] * n}, index=idx)
    return df, idx


def _first_trade(df, settings):
    r = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                           settings=settings, max_hold_bars=1000)
    assert r["trades"], "expected an entry"
    return r["trades"][0]


def test_same_candle_stop_and_target_resolves_stop_first():
    """
    Documented rule: when one candle touches BOTH stop and target, the STOP
    wins (conservative). Exit must be at the stop price, reason "STOP".
    """
    # Two extra bars: entry fills on the first, conflict on the second.
    placeholder = [130.0, 130.0, 130.0, 130.0, 1_000_000]
    df, idx = _entry_df([placeholder, placeholder])
    settings = _timing_settings()

    base = _first_trade(df, settings)
    entry = float(base["entry_price"])
    stop = float(base["initial_stop"])
    target = float(base["take_profit"])
    assert stop < entry < target

    # Conflict candle: opens between stop and target, then touches both.
    conflict = [entry, target + 5.0, stop - 5.0, entry, 1_000_000]
    df2, _ = _entry_df([placeholder, conflict])
    trade = _first_trade(df2, settings)

    assert trade["exit_reason"] == "STOP"
    assert trade["exit_price"] == pytest.approx(stop, abs=1e-6)
    assert trade["exit_time"] == idx[62]


def test_gap_down_through_stop_fills_at_open():
    """A gap below the stop must exit at the Open (STOP_GAP), not the stop."""
    placeholder = [130.0, 130.0, 130.0, 130.0, 1_000_000]
    df, idx = _entry_df([placeholder, placeholder])
    settings = _timing_settings()
    base = _first_trade(df, settings)
    stop = float(base["initial_stop"])

    gap_open = stop - 3.0
    gap = [gap_open, gap_open + 0.5, gap_open - 0.5, gap_open, 1_000_000]
    df2, idx2 = _entry_df([placeholder, gap])
    trade = _first_trade(df2, settings)

    assert trade["exit_reason"] == "STOP_GAP"
    assert trade["exit_price"] == pytest.approx(gap_open, abs=1e-6)


def test_gap_up_through_target_fills_at_open():
    """A gap above the target must exit at the Open (TARGET_GAP)."""
    placeholder = [130.0, 130.0, 130.0, 130.0, 1_000_000]
    df, _ = _entry_df([placeholder, placeholder])
    settings = _timing_settings()
    base = _first_trade(df, settings)
    target = float(base["take_profit"])

    gap_open = target + 3.0
    gap = [gap_open, gap_open + 0.5, gap_open - 0.5, gap_open, 1_000_000]
    df2, _ = _entry_df([placeholder, gap])
    trade = _first_trade(df2, settings)

    assert trade["exit_reason"] == "TARGET_GAP"
    assert trade["exit_price"] == pytest.approx(gap_open, abs=1e-6)


def test_trailing_stop_ratchets_and_uses_only_available_info():
    """
    After a close reaches trailing_activation_r, the stop must ratchet up to
    (highest_close - trailing_distance_r * risk). A later dip to that trailed
    level must exit at the trailed stop, proving:
      - the stop moved up (not still at the initial stop), and
      - the new stop was derived from a close that had already happened.
    """
    placeholder = [130.0, 130.0, 130.0, 130.0, 1_000_000]
    df, idx = _entry_df([placeholder, placeholder])
    settings = _timing_settings(trailing_activation_r=1.5, trailing_distance_r=1.0)
    base = _first_trade(df, settings)

    entry = float(base["entry_price"])
    stop0 = float(base["initial_stop"])
    d = entry - stop0                       # risk per share, as the engine uses it
    assert d > 0

    # Bar 62: rally to +1.6R (above the 1.5R activation, below target at 2R).
    rally_close = entry + 1.6 * d
    rally = [entry, rally_close + 0.1 * d, entry + 0.1 * d, rally_close, 1_000_000]
    expected_trailed_stop = rally_close - d   # highest_close - 1.0R

    # Bar 63: dip to just below the trailed stop.
    dip = [expected_trailed_stop + 0.05 * d, expected_trailed_stop + 0.1 * d,
           expected_trailed_stop - 0.05 * d, expected_trailed_stop, 1_000_000]

    df2, idx2 = _entry_df([placeholder, rally, dip])
    trade = _first_trade(df2, settings)

    assert trade["exit_reason"] == "STOP"
    assert trade["exit_time"] == idx2[63]
    # Exit at the trailed stop, which is strictly above the initial stop.
    assert trade["final_stop"] > stop0
    assert trade["exit_price"] == pytest.approx(expected_trailed_stop, abs=1e-6)


def test_entry_fills_on_next_bar_open_not_signal_bar_close():
    """
    Signals are computed on bar T from its close, but the fill must happen on
    bar T+1's open. On a gapping series the next open is higher than the signal
    bar close, so the entry price proves next-bar execution (no same-bar fill).
    """
    df, idx = _entry_df_rising(n_extra=3)
    settings = _timing_settings()
    trade = _first_trade(df, settings)

    assert trade["entry_time"] == idx[61]
    entry_open = float(df.loc[idx[61], "Open"])
    signal_close = float(df.loc[idx[60], "Close"])
    assert entry_open != signal_close, "dataset must distinguish open from close"
    # Fill uses bar 61's Open, not bar 60's Close.
    assert trade["entry_price"] == pytest.approx(entry_open, abs=1e-6)
    assert trade["entry_price"] != pytest.approx(signal_close, abs=1e-6)


def test_engine_simultaneous_long_and_short_reconcile():
    """
    With SHORT enabled and two opposing trends, the engine holds longs and
    shorts at the same time. The sum of net P&L across all trades must equal
    the change in equity exactly: no double counting, and reserved short
    collateral is released on every close.
    """
    def mk(closes):
        n = len(closes)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "Open": closes, "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes], "Close": closes,
            "Volume": [1_000_000] * n,
        }, index=idx)

    up = mk(list(100 + np.arange(160) * 0.5))
    down = mk(list(300 - np.arange(160) * 0.5))
    s = _timing_settings(enable_short=1, max_hold_bars=10, max_open_positions=3)
    r = simulate_portfolio({"UP": up, "DN": down}, None,
                           start_equity=100_000, settings=s)

    sides = {t["side"] for t in r["trades"]}
    assert "BUY" in sides and "SELL" in sides, "expected both long and short trades"

    total_pnl = sum(t["net_pnl"] for t in r["trades"])
    assert total_pnl == pytest.approx(r["end_equity"] - 100_000, abs=0.01)
    # Every short released its collateral.
    assert r["final_reserved_collateral"] == pytest.approx(0.0, abs=0.01)


def test_engine_reserved_collateral_visible_in_equity_curve():
    """
    While a short is open, the equity curve must show reserved collateral and
    available_cash below raw cash; after it closes, both normalise.
    """
    def mk(closes):
        n = len(closes)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame({
            "Open": closes, "High": [c * 1.005 for c in closes],
            "Low": [c * 0.995 for c in closes], "Close": closes,
            "Volume": [1_000_000] * n,
        }, index=idx)

    down = mk(list(300 - np.arange(160) * 0.5))
    s = _timing_settings(enable_short=1, max_hold_bars=10, max_open_positions=1)
    r = simulate_portfolio({"DN": down}, None, start_equity=100_000, settings=s)

    short_open_points = [p for p in r["equity_curve"] if p["n_positions"] > 0]
    assert short_open_points, "expected an open short position at some point"
    for p in short_open_points:
        assert p["reserved_collateral"] > 0
        assert p["available_cash"] < p["cash"]
        assert p["available_cash"] == pytest.approx(
            p["cash"] - p["reserved_collateral"], abs=0.02)

    last = r["equity_curve"][-1]
    assert last["n_positions"] == 0
    assert last["reserved_collateral"] == pytest.approx(0.0, abs=0.01)


def test_get_all_settings_provides_every_engine_key():
    """
    Regression: the engine reads settings["entry_slippage_bps"] and friends.
    If get_all_settings() omits any of them, run_backtest.py raises KeyError.
    """
    from settings import get_all_settings
    s = get_all_settings()
    for key in ("fees_per_share", "entry_slippage_bps", "exit_slippage_bps",
                "max_hold_bars", "enable_short"):
        assert key in s, f"settings missing engine key {key!r}"
    assert isinstance(s["fees_per_share"], float)
    assert isinstance(s["entry_slippage_bps"], float)
    assert isinstance(s["exit_slippage_bps"], float)
    assert isinstance(s["max_hold_bars"], int)
    assert isinstance(s["enable_short"], int)


def test_cli_engine_settings_applies_overrides():
    from run_backtest import _engine_settings
    s = _engine_settings(max_hold_bars=7, enable_short=1)
    assert s["max_hold_bars"] == 7
    assert s["enable_short"] == 1
    # Untouched keys still come from persisted settings.
    s2 = _engine_settings()
    assert "fees_per_share" in s2


def test_cli_max_hold_bars_reaches_engine():
    """A CLI-style override must actually change exit timing."""
    rise = list(100 + np.arange(61) * 0.5)
    flat = [rise[-1]] * 20
    closes = rise + flat
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Open": closes, "High": [c * 1.001 for c in closes],
        "Low": [c * 0.999 for c in closes], "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)

    from run_backtest import _engine_settings
    s = _engine_settings(max_hold_bars=4)
    # Lower the signal threshold so a trade actually opens on this series;
    # the CLI override under test is max_hold_bars, not the score.
    s["min_signal_score"] = 30
    s["max_open_positions"] = 1
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=s)
    assert result["trades"]
    assert result["trades"][0]["exit_reason"] == "MAX_HOLD"
    assert result["trades"][0]["bars_held"] == 4
