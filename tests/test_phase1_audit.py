"""
Phase-1 audit regression tests.

Cover:
  - max_open_positions enforced at pending-execution time.
  - risk gates rechecked at execution time (not only at signal time).
  - warmup data does not leak OOS trades (trading_start_ts).
  - aggregate_walkforward compounds rather than sums.
  - simulate() honours max_hold_bars.

Run:
    python -m pytest tests/test_phase1_audit.py -v
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import simulate_portfolio, simulate
from backtest.walkforward import aggregate_walkforward


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
        "settings_version": 2,
        "dry_run": True,
        "trading_mode": "BACKTEST",
        "enable_short": 0,
    }
    s.update(overrides)
    return s


def _df(n=200, seed=0, start=100.0, slope=0.5):
    rng = np.random.default_rng(seed)
    closes = start + np.arange(n) * slope + rng.normal(0, 0.1, n)
    closes = np.maximum(closes, 1.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": closes, "High": closes * 1.02, "Low": closes * 0.98,
        "Close": closes, "Volume": [1_000_000] * n,
    }, index=idx)


def _max_positions(result):
    return max((pt["n_positions"] for pt in result["equity_curve"]), default=0)


# ---- max_open_positions at execution time ----

@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_open_positions_enforced_at_execution(limit):
    symbols = [f"S{i}" for i in range(5)]
    data = {sym: _df(seed=i) for i, sym in enumerate(symbols)}
    settings = _settings(max_open_positions=limit)
    result = simulate_portfolio(data, None, start_equity=100_000, settings=settings)
    assert _max_positions(result) <= limit


# ---- trading_start_ts suppresses warmup trades ----

def test_trading_start_ts_suppresses_early_trades():
    n = 250
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = 100 + np.arange(n) * 0.5
    df = pd.DataFrame({
        "Open": closes, "High": closes * 1.02, "Low": closes * 0.98,
        "Close": closes, "Volume": [1_000_000] * n,
    }, index=idx)

    # Start trading 100 days in: before that, no trades should open.
    start_ts = idx[100]
    result = simulate_portfolio(
        {"AAA": df}, None, start_equity=100_000,
        settings=_settings(), trading_start_ts=start_ts,
    )
    for t in result["trades"]:
        assert pd.Timestamp(t["entry_time"]) >= pd.Timestamp(start_ts)


# ---- aggregate_walkforward compounding ----

def test_aggregate_compounds_not_sums():
    results = [
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
    ]
    agg = aggregate_walkforward(results)
    # Compound: (1.10)*(1.10) - 1 = 21.0%
    assert agg["cumulative_return_pct"] == pytest.approx(21.0, abs=0.01)
    # Not arithmetic sum (20.0)


def test_aggregate_handles_negative_returns():
    results = [
        {"metrics": {"num_trades": 1, "total_return_pct": 10.0, "sharpe": 1.0}},
        {"metrics": {"num_trades": 1, "total_return_pct": -10.0, "sharpe": 1.0}},
    ]
    agg = aggregate_walkforward(results)
    # (1.10)*(0.90) - 1 = -1.0%
    assert agg["cumulative_return_pct"] == pytest.approx(-1.0, abs=0.01)


# ---- max_hold_bars respected deterministically ----

def _max_hold_df():
    """
    Rising trend for the first 61 bars (so a LONG signal fires at index 60),
    then a flat market with a +/-0.1% range.

    The stop distance is ATR * 2.5, which on this series is far larger than
    the flat range, so neither STOP nor TARGET can ever be touched after
    entry. The only possible exit is MAX_HOLD (or END_OF_DATA if the data
    ends first). This isolates the max_hold_bars rule.
    """
    rise = list(100 + np.arange(61) * 0.5)
    flat = [rise[-1]] * 20
    closes = rise + flat
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": closes, "High": [c * 1.001 for c in closes],
        "Low": [c * 0.999 for c in closes], "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx), idx


def test_max_hold_bars_exits_at_exactly_5_bars():
    """
    With max_hold_bars=5 and neither STOP nor TARGET reachable, the position
    MUST exit with reason MAX_HOLD after exactly 5 bars.

    END_OF_DATA is explicitly rejected as a substitute: the data continues
    for 15 bars past the exit, so an END_OF_DATA exit here would mean the
    rule was not applied.
    """
    df, idx = _max_hold_df()
    result = simulate_portfolio(
        {"AAA": df}, None, start_equity=100_000,
        settings=_settings(), max_hold_bars=5,
    )

    assert result["trades"], "expected at least one trade"
    first = result["trades"][0]

    assert first["exit_reason"] == "MAX_HOLD", (
        f"expected MAX_HOLD, got {first['exit_reason']}"
    )
    assert first["bars_held"] == 5, f"expected 5 bars, got {first['bars_held']}"

    # Deterministic entry/exit timing: signal on bar 60, entry on bar 61
    # (next open), exit on the 5th bar held = bar 65.
    assert first["entry_time"] == idx[61]
    assert first["exit_time"] == idx[65]

    # No trade may ever exit on END_OF_DATA while MAX_HOLD is binding.
    for t in result["trades"]:
        assert t["exit_reason"] != "END_OF_DATA"


def test_max_hold_bars_scales_with_setting():
    """bars_held must track max_hold_bars, proving the setting is honoured."""
    df, idx = _max_hold_df()
    for limit in (2, 3, 5, 8):
        result = simulate_portfolio(
            {"AAA": df}, None, start_equity=100_000,
            settings=_settings(), max_hold_bars=limit,
        )
        assert result["trades"], f"no trades for max_hold_bars={limit}"
        t = result["trades"][0]
        assert t["exit_reason"] == "MAX_HOLD"
        assert t["bars_held"] == limit
        # Entry bar counts as bars_held == 1, so the exit lands on
        # idx[61 + limit - 1].
        assert t["exit_time"] == idx[61 + limit - 1]


def test_no_binding_exit_can_reach_end_of_data():
    """
    A position with no binding exit condition (huge max_hold_bars, no
    stop/target touch) must legitimately exit with END_OF_DATA.
    """
    df, idx = _max_hold_df()
    result = simulate_portfolio(
        {"AAA": df}, None, start_equity=100_000,
        settings=_settings(), max_hold_bars=10_000,
    )
    assert result["trades"], "expected at least one trade"
    assert result["trades"][-1]["exit_reason"] == "END_OF_DATA"
    assert result["final_positions"] == 0


def test_max_hold_bars_defaults_to_settings_when_not_passed():
    """Omitting max_hold_bars must use settings['max_hold_bars']."""
    df, idx = _max_hold_df()
    result = simulate_portfolio(
        {"AAA": df}, None, start_equity=100_000,
        settings=_settings(max_hold_bars=3),
    )
    assert result["trades"]
    assert result["trades"][0]["exit_reason"] == "MAX_HOLD"
    assert result["trades"][0]["bars_held"] == 3


def test_max_hold_bars_rejects_non_positive():
    df, _ = _max_hold_df()
    with pytest.raises(ValueError):
        simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                           settings=_settings(), max_hold_bars=0)


# ---- engine reads the same settings dict the CLI produces ----

def test_simulate_with_real_settings_dict_does_not_raise():
    """
    Regression: settings.get_all_settings() used to omit fees/slippage keys,
    so run_backtest.py crashed with KeyError the moment a trade opened.
    """
    from settings import get_all_settings
    df, _ = _max_hold_df()
    s = get_all_settings()
    s["min_signal_score"] = 30
    s["max_open_positions"] = 1
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=s, max_hold_bars=5)
    assert result["trades"], "engine produced no trades with real settings"

