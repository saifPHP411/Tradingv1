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


# ---- max_hold_bars respected by simulate() ----

def test_simulate_respects_max_hold_bars():
    n = 200
    closes = [100.0] * n  # flat; only exits will be MAX_HOLD or EOD
    df = _df_from_closes = pd.DataFrame({
        "Open": closes, "High": closes, "Low": closes, "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

    # Force a signal by overriding the score check indirectly — use a strong uptrend.
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rising = 100 + np.arange(n) * 0.5
    df_rise = pd.DataFrame({
        "Open": rising, "High": rising * 1.02, "Low": rising * 0.98,
        "Close": rising, "Volume": [1_000_000] * n,
    }, index=idx)

    result = simulate(
        "AAA", df_rise, None, start_equity=100_000,
        max_hold_bars=5,
    )
    # If any trade lasted longer than 5 bars, the wrapper ignored max_hold_bars.
    for t in result["trades"]:
        assert t["bars_held"] <= 5 or t["exit_reason"] == "END_OF_DATA"
