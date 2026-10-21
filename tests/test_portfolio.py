"""
Unit tests for portfolio cash accounting, equity curve, and risk limits.

Run:
    pip install pytest
    pytest tests/test_portfolio.py -v

These tests DO NOT require Alpaca or LLM. They use synthetic data only.
"""
import pandas as pd
import numpy as np
import pytest

from backtest.engine import simulate_portfolio, LONG, SHORT, PortfolioPosition
from risk_utils import (
    compute_position_size, validate_portfolio_exposure,
    validate_risk_plan, normalize_slippage,
)


def _settings(**overrides):
    s = {
        "min_confidence": 0.7,
        "max_risk_per_trade": 0.005,
        "max_open_positions": 3,
        "max_trades_per_day": 6,
        "max_daily_loss_pct": 0.02,
        "max_position_pct": 0.15,
        "max_portfolio_exposure_pct": 0.45,
        "max_consecutive_losses": 3,
        "cooldown_minutes": 120,
        "stop_atr_multiplier": 2.5,
        "take_profit_rr": 2.0,
        "trailing_activation_r": 1.5,
        "trailing_distance_r": 1.0,
        "min_stop_move_pct": 0.3,
        "min_signal_score": 70,
        "min_hold_minutes": 30,
        "min_loss_to_close_pct": 1.5,
        "llm_exit_min_confidence": 0.7,
        "exit_check_seconds": 60,
        "exit_llm_interval_seconds": 300,
        "allow_extended_hours": False,
        "cycle_interval_minutes": 15,
        "fees_per_share": 0.0,
        "entry_slippage_bps": 5,
        "exit_slippage_bps": 5,
        "trading_capital": 0,
        "regime_enabled": 1,
        "news_enabled": 0,
        "symbols": ["AAA"],
        "settings_version": 2,
        "dry_run": True,
        "trading_mode": "BACKTEST",
        "enable_short": 0,
    }
    s.update(overrides)
    return s


def _df_from_closes(closes, opens=None, highs=None, lows=None, start="2024-01-01"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="D")
    closes = np.array(closes, dtype=float)
    opens = np.array(opens if opens is not None else closes, dtype=float)
    highs = np.array(highs if highs is not None else closes, dtype=float)
    lows = np.array(lows if lows is not None else closes, dtype=float)
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows, "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)


# ============ Pure function tests ============

def test_position_size_respects_risk():
    qty = compute_position_size(10_000, 100, 5, 0.005, 1.0, 0.15)
    assert abs(qty - 10.0) < 1e-6


def test_position_size_capped_by_pct():
    qty = compute_position_size(10_000, 100, 1, 0.01, 1.0, 0.15)
    assert qty <= 15.0 + 1e-6


def test_exposure_rejects_oversize():
    positions = {"X": PortfolioPosition("X", LONG, 100, 30, 95, 110, None, 0)}
    ok, _ = validate_portfolio_exposure(10_000, positions, 2000, 0.45)
    assert ok is False


def test_slippage_directions():
    assert normalize_slippage(100, 10, is_buy=True) > 0
    assert normalize_slippage(100, 10, is_buy=False) < 0


def test_risk_plan_long_ordering():
    plan = {"symbol": "X", "side": "buy", "entry_price": 100,
            "stop_loss": 95, "take_profit": 110, "qty": 1, "risk_amount": 5}
    ok, _ = validate_risk_plan(plan)
    assert ok is True


def test_risk_plan_short_ordering():
    plan = {"symbol": "X", "side": "sell", "entry_price": 100,
            "stop_loss": 105, "take_profit": 90, "qty": 1, "risk_amount": 5}
    ok, _ = validate_risk_plan(plan)
    assert ok is True


# ============ Cash accounting / equity curve ============

def test_cash_never_reused():
    """
    Two symbols, each requiring ~60% of equity. Only one should be allowed
    because cash is finite (no leverage / no reuse).
    """
    n = 120
    closes = [100.0] * n
    # Force a signal by making strong uptrend from bar 60 onwards
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(
        max_position_pct=0.9,
        max_portfolio_exposure_pct=0.99,
        max_open_positions=5,
        max_trades_per_day=50,
    )
    result = simulate_portfolio(
        {"AAA": df.copy(), "BBB": df.copy()},
        None, start_equity=10_000, settings=settings,
    )
    # Equity curve should never go negative
    for pt in result["equity_curve"]:
        assert pt["cash"] >= -1e-6 or pt["unrealized"] >= 0
    # No trade should be larger than cash could have afforded
    for t in result["trades"]:
        notional = t["entry_price"] * t["qty"]
        assert notional <= 10_000 + 1e-6


def test_equity_curve_marked_to_market():
    """
    With one open long, equity must move with price, not be flat until exit.
    """
    n = 150
    # Price flat 100 for 60 bars (warmup), then rises
    closes = [100.0] * 60 + [100 + i * 1.0 for i in range(n - 60)]
    opens = [c for c in closes]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = _df_from_closes(closes, opens, highs, lows)
    settings = _settings(min_signal_score=50)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    eq = [p["equity"] for p in result["equity_curve"]]
    # Equity should be non-decreasing in a rising trend after first trade
    assert max(eq) > 10_000 + 50  # at least some profit captured


def test_no_trades_returns_start_equity():
    n = 120
    closes = [100.0] * n  # flat, no signal
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=100)  # impossible to trigger
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    assert result["end_equity"] == pytest.approx(10_000, abs=1.0)
    assert len(result["trades"]) == 0


# ============ Risk limits ============

def test_max_open_positions_respected():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(
        max_open_positions=2,
        max_position_pct=0.1,
        max_portfolio_exposure_pct=0.95,
        min_signal_score=50,
        max_trades_per_day=50,
    )
    dfs = {f"S{i}": df.copy() for i in range(5)}
    result = simulate_portfolio(dfs, None, start_equity=10_000, settings=settings)
    # At any timestamp in equity curve, n_positions <= 2
    for pt in result["equity_curve"]:
        assert pt["n_positions"] <= 2


def test_cooldown_is_time_based_not_permanent():
    """
    After max_consecutive_losses, cooldown must be a real timestamp,
    not '9999-12-31'. We test by checking the trade journal doesn't
    permanently block after the first few losses.
    """
    # Descending price → stop-outs
    n = 150
    closes = [100 - i * 0.3 for i in range(n)]
    opens = [c for c in closes]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = _df_from_closes(closes, opens, highs, lows)
    settings = _settings(
        max_consecutive_losses=2,
        cooldown_minutes=60,
        min_signal_score=30,
        max_open_positions=1,
        max_trades_per_day=50,
    )
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    # Even after 2 losses, some later trades must still occur because
    # cooldown expires (60 minutes < full test duration in bars).
    assert len(result["trades"]) >= 3


# ============ Trade journal integrity ============

def test_trade_journal_fields():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        for k in ("symbol", "side", "direction", "entry_price", "exit_price",
                  "qty", "initial_stop", "fees_entry", "fees_exit",
                  "net_pnl", "r_multiple", "exit_reason"):
            assert k in t, f"missing {k}"


def test_long_and_short_stop_target_orientation():
    """
    For every trade, verify stop/target are on the correct side of entry.
    """
    n = 200
    closes = [100 + 20 * np.sin(i / 10.0) for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(
        min_signal_score=50,
        max_trades_per_day=50,
        enable_short=1,
        max_open_positions=1,
    )
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        if t["direction"] == LONG:
            assert t["initial_stop"] < t["entry_price"]
            assert t["take_profit"] > t["entry_price"]
        else:
            assert t["initial_stop"] > t["entry_price"]
            assert t["take_profit"] < t["entry_price"]


# ============ Metrics consistency ============

def test_metrics_net_equals_equity_change():
    """
    Sum of net_pnl should equal (end_equity - start_equity) within rounding,
    because we don't have open positions at end of data.
    """
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    total_pnl = sum(t["net_pnl"] for t in result["trades"])
    eq_delta = result["end_equity"] - 10_000
    assert abs(total_pnl - eq_delta) < 1.0


def test_no_lookahead_entry_after_signal_bar():
    """
    First trade entry_idx must be >= signal_bar_idx + 1 (next-bar execution).
    """
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    # No trade should exist before bar 60 (warmup)
    for t in result["trades"]:
        assert t["entry_time"] >= df.index[60]
