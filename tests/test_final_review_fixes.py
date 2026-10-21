"""
Regression tests for the final review fixes:

  1) Cooldown actually expires and allows trading to resume.
  2) SHORT accounting reserves collateral; proceeds are not spendable.
  3) Multiple short positions respect available cash.
  4) News is passed explicitly (deterministic, no lookahead).
  5) Final equity snapshot reconciles with end_equity after force-close.
  6) LONG accounting unchanged.

Run:
    pytest tests/test_final_review_fixes.py -v

Deterministic synthetic OHLCV only. No Alpaca, no network, no paid APIs.
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


# ---- 1) Cooldown transitions: allowed → blocked → allowed ----

def test_cooldown_transitions_allow_block_resume():
    """
    Exercise the actual PortfolioState.can_trade() logic across the three
    states: before the loss limit, inside cooldown, after cooldown.
    """
    s = _settings(max_consecutive_losses=2, cooldown_minutes=30,
                  max_trades_per_day=100, max_open_positions=10)
    state = PortfolioState(10_000.0, s)
    prices = {"AAA": 100.0}

    # State 1: before the loss limit — allowed.
    ok, why = state.can_trade("2024-01-01T10:00:00", prices)
    assert ok is True, why

    # Simulate two consecutive losses reaching the limit.
    state.consecutive_losses = 2
    state.cooldown_until = "2024-01-01T10:30:00"

    # State 2: inside cooldown — blocked with reason "cooldown".
    ok, why = state.can_trade("2024-01-01T10:10:00", prices)
    assert ok is False and why == "cooldown"

    # State 3: after cooldown — allowed again, and the loss counter is reset.
    ok, why = state.can_trade("2024-01-01T10:31:00", prices)
    assert ok is True, why
    assert state.cooldown_until is None
    assert state.consecutive_losses == 0


def test_cooldown_zero_minutes_does_not_lock_permanently():
    """
    cooldown_minutes=0 means "no cooldown": the loss counter must reset
    immediately and entries must remain allowed.
    """
    s = _settings(max_consecutive_losses=1, cooldown_minutes=0,
                  max_trades_per_day=100, max_open_positions=10)
    state = PortfolioState(10_000.0, s)
    prices = {"AAA": 100.0}

    state.consecutive_losses = 1  # threshold reached
    state.cooldown_until = None

    ok, why = state.can_trade("2024-01-01T10:00:00", prices)
    assert ok is True, why
    assert state.consecutive_losses == 0


def test_active_cooldown_is_not_extended():
    """
    A losing trade that closes while cooldown is still active must not
    extend the cooldown window.
    """
    s = _settings(max_consecutive_losses=1, cooldown_minutes=60)
    state = PortfolioState(10_000.0, s)
    state.cooldown_until = "2024-01-01T12:00:00"

    # Simulate a losing trade closing at 11:00 while cooldown is active.
    pos = PortfolioPosition("AAA", LONG, 100.0, 10.0, 95.0, 110.0, None, 0)
    state.positions["AAA"] = pos
    state.close_position("AAA", 95.0, 0.0, 0.0, "STOP",
                         "2024-01-01T11:00:00", {"AAA": 95.0})

    # Cooldown must remain the original 12:00, not extended.
    assert state.cooldown_until == "2024-01-01T12:00:00"


# ---- 2) SHORT accounting: reserved collateral ----

def test_short_entry_reserves_collateral():
    s = _settings(fees_per_share=0.0)
    state = PortfolioState(10_000.0, s)

    pos = PortfolioPosition("AAA", SHORT, 100.0, 50.0, 110.0, 90.0, None, 0)
    state.open_position(pos, 0.0, 0.0)

    # Proceeds credited: cash = 10,000 + 5,000 = 15,000
    assert state.cash == pytest.approx(15_000.0)
    # Same notional reserved as collateral.
    assert state.reserved_collateral == pytest.approx(5_000.0)
    # Available cash stays at pre-trade level.
    assert state.available_cash == pytest.approx(10_000.0)
    # Equity unchanged at entry (no fees, price == entry).
    assert state.equity({"AAA": 100.0}) == pytest.approx(10_000.0)


def test_short_proceeds_do_not_inflate_available_cash():
    """
    Without reserved collateral, a second short of 15k would pass because
    cash would show 19k. With reserved collateral, available stays at 10k,
    so the 15k short must be rejected.
    """
    s = _settings(fees_per_share=0.0)
    state = PortfolioState(10_000.0, s)

    ok, _ = state.can_open(9_000.0, 0.0, SHORT)
    assert ok is True

    pos = PortfolioPosition("AAA", SHORT, 100.0, 90.0, 110.0, 90.0, None, 0)
    state.open_position(pos, 0.0, 0.0)

    # raw cash is now 19,000 but available is still 10,000
    assert state.cash == pytest.approx(19_000.0)
    assert state.available_cash == pytest.approx(10_000.0)

    ok, why = state.can_open(15_000.0, 0.0, SHORT)
    assert ok is False
    assert "insufficient_cash" in why


def test_short_adverse_move_then_close_reconciles():
    s = _settings(fees_per_share=0.0, entry_slippage_bps=0, exit_slippage_bps=0)
    state = PortfolioState(10_000.0, s)

    pos = PortfolioPosition("AAA", SHORT, 100.0, 10.0, 110.0, 90.0, None, 0)
    state.open_position(pos, 0.0, 0.0)
    assert state.cash == pytest.approx(11_000.0)
    assert state.reserved_collateral == pytest.approx(1_000.0)
    assert state.equity({"AAA": 100.0}) == pytest.approx(10_000.0)

    # Adverse move: price up to 105 → unrealized -50.
    prices_up = {"AAA": 105.0}
    assert state.unrealized_pnl(prices_up) == pytest.approx(-50.0)
    assert state.equity(prices_up) == pytest.approx(9_950.0)

    # Close at 105 (stop).
    state.close_position("AAA", 105.0, 0.0, 0.0, "STOP",
                         "2024-01-02", prices_up)

    assert state.positions == {}
    assert state.reserved_collateral == pytest.approx(0.0)
    assert state.cash == pytest.approx(9_950.0)
    assert state.equity(prices_up) == pytest.approx(9_950.0)
    assert state.realized_pnl == pytest.approx(-50.0)


def test_short_favorable_move_then_close_reconciles():
    s = _settings(fees_per_share=0.0, entry_slippage_bps=0, exit_slippage_bps=0)
    state = PortfolioState(10_000.0, s)

    pos = PortfolioPosition("AAA", SHORT, 100.0, 10.0, 110.0, 90.0, None, 0)
    state.open_position(pos, 0.0, 0.0)

    prices_down = {"AAA": 90.0}
    assert state.unrealized_pnl(prices_down) == pytest.approx(100.0)
    assert state.equity(prices_down) == pytest.approx(10_100.0)

    state.close_position("AAA", 90.0, 0.0, 0.0, "TARGET",
                         "2024-01-02", prices_down)

    assert state.reserved_collateral == pytest.approx(0.0)
    assert state.cash == pytest.approx(10_100.0)
    assert state.equity(prices_down) == pytest.approx(10_100.0)
    assert state.realized_pnl == pytest.approx(100.0)


# ---- 3) Multiple short positions respect available cash ----

def test_multiple_short_positions_respect_available_cash():
    s = _settings(fees_per_share=0.0)
    state = PortfolioState(10_000.0, s)

    p1 = PortfolioPosition("AAA", SHORT, 100.0, 10.0, 110.0, 90.0, None, 0)
    p2 = PortfolioPosition("BBB", SHORT, 100.0, 10.0, 110.0, 90.0, None, 0)
    state.open_position(p1, 0.0, 0.0)
    state.open_position(p2, 0.0, 0.0)

    assert state.reserved_collateral == pytest.approx(2_000.0)
    assert state.available_cash == pytest.approx(10_000.0)

    # A third 9k short is fine (9k <= 10k available).
    ok, _ = state.can_open(9_000.0, 0.0, SHORT)
    assert ok is True

    # A 15k short is rejected.
    ok, why = state.can_open(15_000.0, 0.0, SHORT)
    assert ok is False and "insufficient_cash" in why


# ---- 4) News is passed explicitly ----

def test_news_is_explicitly_unavailable_in_backtest():
    assert eng.BACKTEST_NEWS_AVAILABLE is False
    assert eng._NEWS_UNAVAILABLE["available"] is False
    assert eng._NEWS_UNAVAILABLE["score"] == 0.0

    ind = {
        "ema20": 105, "ema50": 100, "last_close": 110, "sma50": 100,
        "adx": 30, "momentum_5d_pct": 1.0, "momentum_10d_pct": 1.0,
        "macd_hist": 1.0, "rsi14": 55, "bb_pct_b": 0.5, "volume_ratio": 1.0,
    }
    regime = {"regime": "BULL", "size_multiplier": 1.0}
    score_no_news = compute_signal_score(ind, regime, news=None)["score"]
    score_unavailable = compute_signal_score(
        ind, regime, news=eng._NEWS_UNAVAILABLE)["score"]
    assert score_no_news == score_unavailable

    # The explicit unavailable dict must not add the +10 boost that
    # real positive news would.
    score_positive = compute_signal_score(
        ind, regime, news={"available": True, "score": 1.0})["score"]
    assert score_unavailable < score_positive


# ---- 5) Final snapshot reconciles with end_equity ----

def test_final_snapshot_matches_end_equity():
    n = 150
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)

    assert result["equity_curve"], "equity curve is empty"
    last = result["equity_curve"][-1]
    assert last["equity"] == pytest.approx(result["end_equity"], abs=0.01)
    assert last["n_positions"] == 0
    assert last["unrealized"] == pytest.approx(0.0, abs=0.01)
    assert result["final_cash"] == pytest.approx(result["end_equity"], abs=0.01)


def test_equity_curve_reconciles_with_trade_pnls():
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


# ---- 6) LONG accounting unchanged ----

def test_long_entry_and_exit_reconcile():
    s = _settings(fees_per_share=0.0, entry_slippage_bps=0, exit_slippage_bps=0)
    state = PortfolioState(10_000.0, s)

    pos = PortfolioPosition("AAA", LONG, 100.0, 50.0, 95.0, 110.0, None, 0)
    state.open_position(pos, 0.0, 0.0)

    assert state.cash == pytest.approx(5_000.0)
    assert state.reserved_collateral == pytest.approx(0.0)
    assert state.equity({"AAA": 100.0}) == pytest.approx(10_000.0)

    prices_up = {"AAA": 110.0}
    assert state.equity(prices_up) == pytest.approx(10_500.0)
    assert state.unrealized_pnl(prices_up) == pytest.approx(500.0)

    state.close_position("AAA", 110.0, 0.0, 0.0, "TARGET",
                         "2024-01-02", prices_up)

    assert state.cash == pytest.approx(10_500.0)
    assert state.equity(prices_up) == pytest.approx(10_500.0)
    assert state.realized_pnl == pytest.approx(500.0)


def test_cash_is_deducted_on_long_entry():
    n = 200
    closes = [100 + i * 0.5 for i in range(n)]
    df = _df_from_closes(closes)
    settings = _settings(min_signal_score=50, max_trades_per_day=20)
    result = simulate_portfolio({"AAA": df}, None, start_equity=10_000,
                                settings=settings)
    for pt in result["equity_curve"]:
        if pt["n_positions"] > 0:
            assert pt["cash"] < 10_000 - 1e-6
            break
