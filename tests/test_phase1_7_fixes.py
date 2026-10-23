"""
Phase-1.7.1 targeted regression tests.

Small, focused fixes identified by the Phase 1.7 independent audit:

  1  aggregate_walkforward must not crash on malformed result entries or
     non-dict metrics, and such windows must contribute no trades/returns
     (consistent with the documented invalid-window policy).
  2  The engine's post-tick-rounding rejection branch must be exercised by a
     genuine engine run: a *positive* stop that tick-rounds to zero is
     rejected as ``invalid_stop_or_target_after_rounding``.
  3  The walk-forward driver's final window is open-ended and the last
     observation is included exactly once across all windows.
  4  run_backtest._print_monte_carlo handles the zero-trade/error payload.

Deterministic synthetic data only. No network, no credentials.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import simulate_portfolio, compute_indicators_at
from backtest.walkforward import aggregate_walkforward, walk_forward
from risk_utils import normalize_price_for_symbol


def _settings(**overrides):
    s = {
        "min_confidence": 0.7,
        "max_risk_per_trade": 0.02,
        "max_open_positions": 1,
        "max_trades_per_day": 50,
        "max_daily_loss_pct": 0.5,
        "max_position_pct": 1.0,
        "max_portfolio_exposure_pct": 0.99,
        "max_consecutive_losses": 999,
        "cooldown_minutes": 0,
        "stop_atr_multiplier": 2.5,
        "take_profit_rr": 2.0,
        "trailing_activation_r": 1.5,
        "trailing_distance_r": 1.0,
        "min_stop_move_pct": 0.3,
        "min_signal_score": 30,
        "min_hold_minutes": 0,
        "min_loss_to_close_pct": 99.0,
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
        "max_hold_bars": 5,
    }
    s.update(overrides)
    return s


def _frame(rows, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


# ============================================================================
# 1) aggregate_walkforward robustness to malformed entries
# ============================================================================

_VALID = {"window": "ok",
          "metrics": {"num_trades": 3, "total_return_pct": 10.0, "sharpe": 1.0}}


@pytest.mark.parametrize("malformed", [
    None,                       # entry itself is not a mapping
    "error",                    # entry is a bare string
    [1, 2, 3],                  # entry is a list
    {"window": "no-metrics"},   # metrics key absent
    {"window": "null", "metrics": None},   # metrics explicitly None
    {"window": "str", "metrics": "no data"},  # metrics is a bare string
    {"window": "list", "metrics": [1, 2]},    # metrics is a list
])
def test_malformed_entry_is_invalid_and_never_crashes(malformed):
    """A malformed entry must not raise and must contribute nothing."""
    agg = aggregate_walkforward([malformed, _VALID])
    assert agg["num_windows"] == 2
    assert agg["num_valid_windows"] == 1
    assert agg["num_invalid_windows"] == 1
    # The one valid window's figures are untouched by the malformed entry.
    assert agg["total_trades"] == 3
    assert agg["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)
    assert agg["avg_return_per_window"] == pytest.approx(10.0, abs=0.01)
    assert "invalid_windows" in agg
    # A malformed entry is labelled by its index; a mapping keeps its window.
    label = agg["invalid_windows"][0]["window"]
    assert label == (malformed.get("window", 0)
                     if isinstance(malformed, dict) else 0)


def test_all_malformed_entries_yield_empty_aggregate_not_crash():
    agg = aggregate_walkforward([None, "x", {"metrics": "y"}, {}])
    assert agg["num_valid_windows"] == 0
    assert agg["num_invalid_windows"] == 4
    assert agg["total_trades"] == 0
    assert agg["cumulative_return_pct"] == 0.0
    assert agg["avg_return_per_window"] == 0.0


def test_non_numeric_num_trades_does_not_crash_and_is_not_counted():
    """
    A window with a usable return but a junk num_trades must not raise a
    TypeError on the trade tally; a non-numeric count is treated as 0.
    """
    agg = aggregate_walkforward([
        {"window": "junk-trades",
         "metrics": {"num_trades": "many", "total_return_pct": 5.0}},
        _VALID,
    ])
    assert agg["total_trades"] == 3          # only the valid window's trades
    assert agg["cumulative_return_pct"] == pytest.approx(15.5, abs=0.01)


def test_non_numeric_num_trades_on_invalid_window_counts_zero():
    agg = aggregate_walkforward([
        {"window": "bad",
         "metrics": {"num_trades": "lots", "total_return_pct": float("nan")}},
        _VALID,
    ])
    assert agg["total_trades"] == 3
    assert agg["total_trades_invalid_windows"] == 0
    assert agg["num_invalid_windows"] == 1


def test_malformed_metrics_entry_records_reason():
    agg = aggregate_walkforward([{"window": "broken", "metrics": 42}])
    assert agg["num_invalid_windows"] == 1
    reason = agg["invalid_windows"][0]["reason"]
    assert "malformed" in reason


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_non_finite_num_trades_does_not_crash(bad):
    """
    ``int(float("inf"))`` raises ``OverflowError``, which is NOT a ``ValueError``.
    A non-finite trade count on an otherwise valid window must be treated as 0
    rather than aborting the whole aggregation.
    """
    agg = aggregate_walkforward([
        {"window": "inf-trades",
         "metrics": {"num_trades": bad, "total_return_pct": 5.0}},
        _VALID,
    ])
    # The junk count contributes nothing; the valid window is untouched.
    assert agg["total_trades"] == 3
    assert agg["cumulative_return_pct"] == pytest.approx(15.5, abs=0.01)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_non_finite_num_trades_on_invalid_window_counts_zero(bad):
    """A non-finite count on a non-finite-return window must not crash the
    invalid tally either (the same conversion happens on that path)."""
    agg = aggregate_walkforward([
        {"window": "bad",
         "metrics": {"num_trades": bad, "total_return_pct": float("nan")}},
        _VALID,
    ])
    assert agg["num_invalid_windows"] == 1
    assert agg["total_trades"] == 3
    assert agg["total_trades_invalid_windows"] == 0


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_non_finite_num_trades_on_clamped_window_counts_zero(bad):
    """The clamped (< -100%) branch also converts num_trades; it must not crash
    on a non-finite count."""
    agg = aggregate_walkforward([
        {"window": "clamped",
         "metrics": {"num_trades": bad, "total_return_pct": -150.0}},
        _VALID,
    ])
    assert agg["num_invalid_windows"] == 1
    assert agg["total_trades"] == 3
    assert agg["total_trades_invalid_windows"] == 0
    # The clamped window compounds at -100% (product forced to 0); the valid
    # window's +10% then leaves the cumulative at -100%.
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)


def test_unbounded_int_total_return_is_invalid_not_crash():
    """
    ``float(10**400)`` raises ``OverflowError``. An unbounded integer return is
    not a usable number, so it must be treated as a non-finite (invalid) window
    instead of crashing aggregation.
    """
    agg = aggregate_walkforward([
        {"window": "huge", "metrics": {"num_trades": 1,
                                       "total_return_pct": 10 ** 400}},
        _VALID,
    ])
    assert agg["num_invalid_windows"] == 1
    assert agg["num_valid_windows"] == 1
    # The invalid window contributes no trades and no return.
    assert agg["total_trades"] == 3
    assert agg["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)
    assert "non-finite" in agg["invalid_windows"][0]["reason"]


def test_unbounded_int_sharpe_does_not_crash():
    """A junk (unbounded-int) sharpe must fall back to 0.0, not raise."""
    agg = aggregate_walkforward([
        {"window": "huge-sharpe",
         "metrics": {"num_trades": 1, "total_return_pct": 5.0,
                     "sharpe": 10 ** 400}},
        _VALID,
    ])
    # sharpe 10**400 -> 0.0; the valid window's 1.0 -> mean 0.5
    assert agg["avg_sharpe"] == pytest.approx(0.5, abs=0.01)
    assert agg["cumulative_return_pct"] == pytest.approx(15.5, abs=0.01)


# ============================================================================
# 2) Engine-level post-rounding rejection branch
# ============================================================================

def _high_atr_uptrend(n_signal=61, step=2.0):
    """Zigzag uptrend with an inflated ATR, so a LONG signal fires at idx 60."""
    closes = []
    c = 100.0
    for i in range(n_signal):
        c += step + (step if i % 2 else -step)
        closes.append(c)
    return closes


def _long_gap_df(gap_open):
    """Uptrend that signals LONG at idx 60; the entry bar (idx 61) opens at
    ``gap_open``. The two trailing bars keep the position manageable."""
    closes = _high_atr_uptrend()
    rows = [[x, x + 3.0, x - 3.0, x, 1_000_000] for x in closes]
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    return _frame(rows)


def test_engine_rejects_stop_that_rounds_to_zero_after_tick_rounding():
    """
    A genuinely positive pre-rounding stop that collapses to zero under tick
    rounding must be rejected via ``invalid_stop_or_target_after_rounding``
    (NOT the earlier pre-rounding ``invalid_stop_or_target`` branch).

    The gap open is derived from the engine's own ATR so the pre-rounding stop
    is deterministically a small positive value below half a tick.
    """
    settings = _settings(enable_short=0)
    probe = _long_gap_df(gap_open=100.0)   # gap bars do not affect idx-60 ATR
    ind = compute_indicators_at(probe, 60)
    assert ind is not None
    stop_dist = ind["atr14"] * settings["stop_atr_multiplier"]
    assert stop_dist > 0

    # pre-round stop = gap_open - stop_dist; choose a strictly positive value
    # that is < 0.005 so it rounds to 0.00 at the 0.01 tick.
    gap_open = stop_dist + 0.001
    pre_round_stop = gap_open - stop_dist
    assert 0.0 < pre_round_stop < 0.005
    assert normalize_price_for_symbol("AAA", pre_round_stop) == 0.0

    result = simulate_portfolio({"AAA": _long_gap_df(gap_open)}, None,
                                start_equity=100_000, settings=settings,
                                max_hold_bars=5)

    assert result["trades"] == [], "no position may open on a zeroed stop"
    assert result["rejections"], "rejection must be observable"
    rej = result["rejections"][0]
    assert rej["symbol"] == "AAA"
    assert rej["reason"] == "invalid_stop_or_target_after_rounding"
    # The stored (post-rounding) stop is zero; the fill was positive.
    assert rej["stop_price"] == 0.0
    assert rej["fill_price"] > 0


def test_engine_rejects_target_that_rounds_to_zero_after_tick_rounding():
    """SHORT mirror: a positive target that rounds to zero is rejected."""
    settings = _settings(enable_short=1)

    closes = []
    c = 500.0
    for i in range(61):
        c -= 2.0 + (2.0 if i % 2 else -2.0)
        closes.append(c)
    rows = [[x, x + 3.0, x - 3.0, x, 1_000_000] for x in closes]

    def build(gap):
        r = list(rows)
        r.append([gap, gap * 1.05, gap * 0.95, gap, 1_000_000])
        r.append([gap, gap * 1.05, gap * 0.95, gap, 1_000_000])
        return _frame(r)

    ind = compute_indicators_at(build(500.0), 60)
    assert ind is not None
    target_dist = ind["atr14"] * settings["stop_atr_multiplier"] \
        * settings["take_profit_rr"]
    gap_open = target_dist + 0.001      # target = gap_open - target_dist > 0
    pre_round_target = gap_open - target_dist
    assert 0.0 < pre_round_target < 0.005
    assert normalize_price_for_symbol("AAA", pre_round_target) == 0.0

    result = simulate_portfolio({"AAA": build(gap_open)}, None,
                                start_equity=100_000, settings=settings,
                                max_hold_bars=5)
    assert result["trades"] == []
    assert result["rejections"][0]["reason"] == \
        "invalid_stop_or_target_after_rounding"


# ============================================================================
# 3) Driver-level: final window open-ended, last observation counted once
# ============================================================================

def test_walk_forward_final_window_open_ended_last_obs_once(monkeypatch):
    """
    End-to-end through the walk_forward driver: every non-final window must
    exclude its nominal end (owned by the next window), the final window must
    retain a data point at its nominal end (open-ended), and the final
    timestamp must appear in exactly one window.
    """
    import backtest.walkforward as wf

    idx = pd.date_range("2022-06-01", "2024-04-01", freq="D")
    closes = 100 + np.arange(len(idx)) * 0.3
    full = pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": [1_000_000] * len(idx),
    }, index=idx)
    last_data_ts = full.index[-1]

    def loader(symbol, start, end):
        s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
        out = full[(full.index >= s_ts) & (full.index <= e_ts)]
        return out if len(out) else None

    monkeypatch.setattr(wf, "load_ohlcv", loader)
    monkeypatch.setattr(wf, "get_spy_for_period", lambda *a, **k: None)

    results = walk_forward("AAA", "2023-01-01", "2024-04-01",
                           train_months=6, test_months=3,
                           initial_equity=10_000,
                           settings=_settings(max_hold_bars=5))
    assert len(results) >= 2, "need a non-final and a final window"

    seen: dict[pd.Timestamp, int] = {}
    for i, r in enumerate(results):
        start_s, end_s = r["window"].split("_to_")
        nominal_end = pd.Timestamp(end_s)
        curve = r["result"]["equity_curve"]
        assert curve, f"window {r['window']} produced an empty curve"
        times = [pd.Timestamp(p["time"]) for p in curve]
        for t in times:
            seen[t] = seen.get(t, 0) + 1

        if i < len(results) - 1:
            # Non-final window is half-open: it must NOT own its end boundary.
            assert max(times) < nominal_end, (
                f"non-final window {r['window']} owns the boundary point")
        else:
            # Final window is open-ended: it retains its nominal end.
            assert nominal_end in times, (
                "final window must keep the open-ended end data point")

    # The very last observation is present and owned by exactly one window.
    assert last_data_ts in seen, "final data point missing from all windows"
    assert seen[last_data_ts] == 1, (
        f"final data point counted {seen[last_data_ts]} times")
    assert all(c == 1 for c in seen.values()), (
        "no equity timestamp may be double counted across windows")


# ============================================================================
# 4) _print_monte_carlo with the zero-trade / error payload
# ============================================================================

def test_print_monte_carlo_zero_trade_error_payload(capsys):
    """
    The zero-trade payload is ``{"error": "no trades"}``. The printer must
    report it and return without touching the missing distribution keys.
    """
    import run_backtest

    rc = run_backtest._print_monte_carlo({"error": "no trades"})
    out = capsys.readouterr().out
    assert rc is None
    assert "=== Monte Carlo (1000 shuffles) ===" in out
    assert "skipped: no trades" in out
    # No partial metric lines must be emitted for an error payload.
    assert "Final equity median" not in out
    assert "Prob. of ruin" not in out


@pytest.mark.parametrize("payload", [
    {"error": "no trades"},
    {"error": "no PnLs"},
    {"error": "any other reason"},
])
def test_print_monte_carlo_never_raises_on_error_payload(payload, capsys):
    import run_backtest

    run_backtest._print_monte_carlo(payload)   # must not raise KeyError
    assert f"skipped: {payload['error']}" in capsys.readouterr().out
