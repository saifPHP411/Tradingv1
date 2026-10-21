"""
Regression tests for portfolio accounting, equity curve, and risk limits.
Run:  pytest tests/test_portfolio.py -v
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import simulate_portfolio, LONG, SHORT, PortfolioPosition
from signals import compute_signal_score, compute_short_score
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


# ============ Cash / equity accounting ============

def test_no_trades_returns_start_equity():
    n = 120
    closes = [100.0] * n
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=100)  # impossible
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    assert result["end_equity"] == pytest.approx(10_000, abs=1.0)
    assert len(result["trades"]) == 0


def test_equity_curve_is_marked_to_market():
    """
    With a rising market, equity curve must move with the position,
    not be flat until exit.
    """
    n = 200
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    eq = [p["equity"] for p in result["equity_curve"]]
    assert max(eq) > 10_000 + 20


def test_cash_is_actually_deducted_on_entry():
    """
    After any open long, portfolio cash must be < start_equity
    (notional was paid), even before the trade closes.
    """
    n = 200
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    # Find first bar with an open position, and confirm cash dropped below 10k.
    for pt in result["equity_curve"]:
        if pt["n_positions"] > 0:
            assert pt["cash"] < 10_000 - 1e-6
            break


def test_no_double_counting_of_pnl():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    total_pnl = sum(t["net_pnl"] for t in result["trades"])
    eq_delta = result["end_equity"] - 10_000
    assert abs(total_pnl - eq_delta) < 1.0


def test_trade_journal_has_entry_and_exit_fees():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20,
                         fees_per_share=0.005)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        assert "fees_entry" in t and "fees_exit" in t
        assert "slippage_entry" in t and "slippage_exit" in t


# ============ Risk limits ============

def test_max_open_positions_respected():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(max_open_positions=2, max_position_pct=0.1,
                         max_portfolio_exposure_pct=0.95,
                         min_signal_score=50, max_trades_per_day=50)
    dfs = {f"S{i}": df.copy() for i in range(5)}
    result = simulate_portfolio(dfs, None, start_equity=10_000, settings=settings)
    for pt in result["equity_curve"]:
        assert pt["n_positions"] <= 2


# ============ Long / Short orientation ============

def test_long_stop_target_orientation():
    n = 200
    closes = [100 + 20 * np.sin(i / 10.0) for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=50,
                         max_open_positions=1)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        if t["direction"] == LONG:
            assert t["initial_stop"] < t["entry_price"]
            assert t["take_profit"] > t["entry_price"]


def test_short_stop_target_orientation():
    n = 200
    closes = [200 - i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=50,
                         max_open_positions=1, enable_short=1)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        if t["direction"] == SHORT:
            assert t["initial_stop"] > t["entry_price"]
            assert t["take_profit"] < t["entry_price"]


# ============ Cooldown ============

def test_cooldown_is_time_based_not_permanent():
    """
    On daily bars, a 60-minute cooldown expires at the next bar. So after
    max_consecutive_losses, the engine must resume trading on the next bar.
    """
    n = 200
    closes = [200 - i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(
        max_consecutive_losses=2,
        cooldown_minutes=60,
        min_signal_score=30,
        max_open_positions=1,
        max_trades_per_day=50,
        enable_short=1,
    )
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    # At least some trades must occur — cooldown is not a permanent lock.
    assert len(result["trades"]) >= 2


# ============ Signal parity ============

def test_backtest_uses_shared_signal_score():
    """
    Guard against future duplication: the engine must not define its own
    scoring. We verify by checking that the imported helper is the one
    from signals.py.
    """
    import inspect
    from backtest import engine as eng
    src = inspect.getsource(eng.simulate_portfolio)
    assert "compute_signal_score" in src
    assert "def compute_score(" not in src


# ============ No lookahead ============

def test_no_lookahead_entry_before_warmup():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for t in result["trades"]:
        assert t["entry_time"] >= df.index[60]
