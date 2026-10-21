"""
Regression tests for max_open_positions enforcement.

The pre-fix bug: max_open_positions was checked only in can_trade()
(signal-generation time), not at order-execution time. When multiple
symbols were queued in the same bar, all of them were opened in the
next bar, exceeding the limit.

These tests exercise the actual execution loop with multiple symbols
sharing the same bar, and verify the live position count never exceeds
the configured limit.

Run:
    python -m pytest tests/test_max_positions_fix.py -v
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import simulate_portfolio


def _settings(**overrides):
    s = {
        "min_confidence": 0.7,
        "max_risk_per_trade": 0.005,
        "max_open_positions": 2,
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


def _trending_df(n=200, seed=0, start_price=100.0):
    """Strong uptrend so every symbol generates LONG signals."""
    rng = np.random.default_rng(seed)
    closes = start_price + np.arange(n) * 0.5 + rng.normal(0, 0.1, n)
    closes = np.maximum(closes, 1.0)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": closes,
        "High": closes * 1.02,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)


def _max_positions_seen(result):
    return max((pt["n_positions"] for pt in result["equity_curve"]), default=0)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_open_positions_never_exceeded(limit):
    """
    Five symbols share the same bar and all generate strong LONG signals.
    The live position count in equity_curve must never exceed `limit`.
    """
    symbols = [f"S{i}" for i in range(5)]
    data = {sym: _trending_df(seed=i) for i, sym in enumerate(symbols)}
    settings = _settings(max_open_positions=limit)

    result = simulate_portfolio(data, None, start_equity=100_000,
                                settings=settings)

    seen = _max_positions_seen(result)
    assert seen <= limit, (
        f"n_positions reached {seen} with max_open_positions={limit}"
    )


def test_max_open_positions_one_blocks_second_entry():
    """
    With max_open_positions=1, at most one trade should be open at any
    time, and a second symbol's queued signal must be dropped, not
    executed later while the first is still open.
    """
    symbols = ["AAA", "BBB"]
    data = {
        "AAA": _trending_df(seed=1),
        "BBB": _trending_df(seed=2),
    }
    settings = _settings(max_open_positions=1)
    result = simulate_portfolio(data, None, start_equity=100_000,
                                settings=settings)

    assert _max_positions_seen(result) <= 1


def test_max_open_positions_does_not_break_long_accounting():
    """
    Sanity check: with a small limit and successful trades, the final
    equity must equal start_equity + sum(net_pnl), proving the guard
    did not corrupt cash/equity accounting.
    """
    symbols = [f"S{i}" for i in range(4)]
    data = {sym: _trending_df(seed=i) for i, sym in enumerate(symbols)}
    settings = _settings(max_open_positions=2)
    start = 100_000.0
    result = simulate_portfolio(data, None, start_equity=start,
                                settings=settings)

    total_pnl = sum(t["net_pnl"] for t in result["trades"])
    assert result["end_equity"] == pytest.approx(start + total_pnl, abs=1.0)
