"""
Focused regression tests for the three remaining fixes:
  A) can_open() now takes side and documents the cash-secured model.
  B) news is passed explicitly to compute_signal_score().
  C) equity_curve[-1] reconciles with end_equity after force-close.

Run:
    pytest tests/test_remaining_fixes.py -v

These tests are DETERMINISTIC, use synthetic OHLCV only, and require no
network, no Alpaca, no paid APIs.
"""
import numpy as np
import pandas as pd
import pytest

from backtest import engine as eng
from backtest.engine import (
    simulate_portfolio, PortfolioState, PortfolioPosition, LONG, SHORT,
)
from signals import compute_signal_score


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


# ---- A) Cash-secured gate is explicit and side-aware ----

def test_can_open_requires_full_collateral_for_both_sides():
    s = _settings()
    state = PortfolioState(1000.0, s)

    ok, _ = state.can_open(500.0, 1.0, LONG)
    assert ok is True

    ok, why = state.can_open(2000.0, 1.0, LONG)
    assert ok is False and "insufficient_cash" in why

    ok, _ = state.can_open(500.0, 1.0, SHORT)
    assert ok is True

    ok, why = state.can_open(2000.0, 1.0, SHORT)
    assert ok is False and "insufficient_cash" in why


# ---- B) News is passed explicitly (deterministic, no lookahead) ----

def test_news_is_explicitly_unavailable_in_backtest():
    # Constant must be present and documented.
    assert eng.BACKTEST_NEWS_AVAILABLE is False
    # The unavailable news dict must be well-formed.
    assert eng._NEWS_UNAVAILABLE["available"] is False
    assert eng._NEWS_UNAVAILABLE["score"] == 0.0

    # When news is passed as unavailable, the score must equal the
    # score computed without news.
    ind = {
        "ema20": 105, "ema50": 100, "last_close": 110, "sma50": 100,
        "adx": 30, "momentum_5d_pct": 1.0, "momentum_10d_pct": 1.0,
        "macd_hist": 1.0, "rsi14": 55, "bb_pct_b": 0.5, "volume_ratio": 1.0,
    }
    regime = {"regime": "BULL", "size_multiplier": 1.0}
    score_with_no_news = compute_signal_score(ind, regime, news=None)["score"]
    score_with_unavailable = compute_signal_score(
        ind, regime, news=eng._NEWS_UNAVAILABLE)["score"]
    assert score_with_no_news == score_with_unavailable

    # And it must NOT include the +10 boost a real positive news item gives.
    score_with_positive_news = compute_signal_score(
        ind, regime, news={"available": True, "score": 1.0})["score"]
    assert score_with_unavailable < score_with_positive_news


# ---- C) Equity curve reconciles with end_equity after force-close ----

def test_equity_curve_last_point_matches_end_equity():
    """
    Force-close at end of data must update both end_equity and the final
    snapshot — otherwise the equity curve is inconsistent.
    """
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)

    assert result["equity_curve"], "equity curve is empty"
    last = result["equity_curve"][-1]
    assert last["equity"] == pytest.approx(result["end_equity"], abs=0.01)
    assert last["n_positions"] == 0  # everything force-closed
    assert last["unrealized"] == pytest.approx(0.0, abs=0.01)


def test_equity_curve_reconciles_with_trade_pnls():
    """
    After force-close, end_equity - start_equity must equal sum(net_pnl)
    within rounding.
    """
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20,
                         fees_per_share=0.001)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    total_pnl = sum(t["net_pnl"] for t in result["trades"])
    eq_delta = result["end_equity"] - 10_000
    assert abs(total_pnl - eq_delta) < 1.0


# ---- D) Cash actually leaves the account on entry ----

def test_cash_is_deducted_on_entry():
    n = 200
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    # The first bar with any open position must show cash strictly below start.
    for pt in result["equity_curve"]:
        if pt["n_positions"] > 0:
            assert pt["cash"] < 10_000 - 1e-6
            break


# ---- E) Cooldown uses a real timestamp, not a magic string ----

def test_cooldown_is_not_9999():
    """
    Guard against the regression that used '9999-12-31' as a permanent lock.
    After max_consecutive_losses triggers, cooldown_until must be a
    timestamp close to the trigger bar (not a far-future date).
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

    # Rebuild a state to inspect behavior indirectly via trades count.
    # At least some trades must occur — cooldown expires, not permanent.
    assert len(result["trades"]) >= 2

    # Direct assertion on the raw string: no code path may ever set 9999-12-31.
    import inspect
    src = inspect.getsource(eng)
    assert "9999-12-31" not in src


# ---- F) No duplicated scoring ----

def test_engine_does_not_redefine_compute_score():
    import inspect
    src = inspect.getsource(eng)
    assert "def compute_score(" not in src
    assert "compute_signal_score" in src
