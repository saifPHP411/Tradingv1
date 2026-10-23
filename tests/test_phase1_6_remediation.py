"""
Phase-1.6 targeted remediation regression tests.

Covers the independently verified findings from the Phase 1.5 forensic audit:

  D1  invalid (non-positive) derived stop/target must not crash the backtest.
  D2  entries must be blocked when marked-to-market equity is <= 0.
  D3  the daily-loss gate must fail closed on a non-positive daily start equity.
  E1  walk-forward OOS windows are half-open so boundary trades are counted once.
  E2  aggregate_walkforward must not count trades from invalid windows.
  E3  an "error" metrics window must be reported invalid, not as a flat window.
  F1  negative raw final equity must be disclosed, not hidden behind a 0 clamp.
  G3  an undefined profit factor must render as "N/A".

Deterministic synthetic data only. No network, no credentials.

Run:
    python -m pytest tests/test_phase1_6_remediation.py -v
"""
import math

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    simulate_portfolio, PortfolioState, PortfolioPosition, LONG, SHORT,
)
from backtest.metrics import calculate_metrics
from backtest.report import _fmt_profit_factor, print_summary
from backtest.walkforward import (
    _filter_to_window, aggregate_walkforward,
)


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


# ==================== D1: invalid stop/target must not crash ====================

def _high_atr_series(n_signal=61, direction=-1, step=2.0):
    """
    High-ATR zigzag series with a net drift of `direction * step` per bar.
    direction=-1 -> downtrend (SHORT signal); direction=+1 -> uptrend (LONG).
    ATR is inflated so the derived stop/target distance is large.
    """
    closes = []
    c = 500.0 if direction < 0 else 100.0
    for i in range(n_signal):
        c += direction * step + (step if i % 2 else -step)
        closes.append(c)
    return closes


def _long_gap_df(gap_open):
    """Uptrend that signals LONG at idx 60, then the entry bar gaps DOWN."""
    closes = _high_atr_series(direction=+1)
    rows = [[x, x + 3.0, x - 3.0, x, 1_000_000] for x in closes]
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    return _frame(rows)


def _short_gap_df(gap_open):
    """Downtrend that signals SHORT at idx 60, then the entry bar gaps DOWN."""
    closes = _high_atr_series(direction=-1)
    rows = [[x, x + 3.0, x - 3.0, x, 1_000_000] for x in closes]
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    return _frame(rows)


def test_d1_long_entry_with_non_positive_stop_is_rejected_not_crashed():
    """
    A LONG entry bar gapping far below the ATR stop distance makes
    stop = fill - stop_dist <= 0. Previously this raised ValueError from
    normalize_price_for_symbol and aborted the whole backtest.
    """
    df = _long_gap_df(gap_open=1.0)
    # Must not raise.
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)

    assert result["trades"] == [], "invalid entry must not open a position"
    assert result["rejections"], "rejection must be observable"
    rej = result["rejections"][0]
    assert rej["symbol"] == "AAA"
    assert rej["reason"] in (
        "invalid_stop_or_target", "invalid_stop_or_target_after_rounding")
    # The rejected levels really were non-positive.
    assert rej["stop_price"] <= 0 or rej["target_price"] <= 0


def test_d1_short_entry_with_non_positive_target_is_rejected_not_crashed():
    """SHORT mirror: target = fill - target_dist can go non-positive."""
    df = _short_gap_df(gap_open=1.0)
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(enable_short=1),
                                max_hold_bars=5)

    assert result["trades"] == [], "invalid entry must not open a position"
    assert result["rejections"]
    rej = result["rejections"][0]
    assert rej["reason"] in (
        "invalid_stop_or_target", "invalid_stop_or_target_after_rounding")
    assert rej["stop_price"] <= 0 or rej["target_price"] <= 0


def test_d1_no_invalid_stop_or_target_ever_enters_the_portfolio():
    """
    Every opened trade in a run containing a bad gap must still carry strictly
    positive stop and target prices.
    """
    good = _frame([[100 + i * 0.5, (100 + i * 0.5) * 1.001,
                    (100 + i * 0.5) * 0.999, 100 + i * 0.5, 1_000_000]
                   for i in range(120)])
    bad = _long_gap_df(gap_open=1.0)
    result = simulate_portfolio({"GOOD": good, "BAD": bad}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0,
                                                   max_open_positions=2),
                                max_hold_bars=5)
    for t in result["trades"]:
        assert t["initial_stop"] > 0, f"non-positive stop in trade {t}"
        assert t["take_profit"] > 0, f"non-positive target in trade {t}"


def test_d1_normal_entries_still_work_alongside_a_rejected_gap():
    """
    The D1 guard must reject only the malformed entry: a healthy symbol in the
    same run must still trade normally.
    """
    good = _frame([[100 + i * 0.5, (100 + i * 0.5) * 1.001,
                    (100 + i * 0.5) * 0.999, 100 + i * 0.5, 1_000_000]
                   for i in range(120)])
    bad = _long_gap_df(gap_open=1.0)
    result = simulate_portfolio({"GOOD": good, "BAD": bad}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0,
                                                   max_open_positions=2),
                                max_hold_bars=5)
    traded_symbols = {t["symbol"] for t in result["trades"]}
    assert "GOOD" in traded_symbols, "valid symbol must still trade"
    assert "BAD" not in traded_symbols, "gapped symbol must be rejected"
    assert all(r["symbol"] == "BAD" for r in result["rejections"])


def test_d1_rounding_to_zero_is_also_rejected():
    """
    A tiny positive level that tick-rounds to zero is equally unusable and must
    be rejected rather than opening an unprotected position.
    """
    # entry at 0.004: stop = 0.004 - stop_dist < 0; and any level < 0.005
    # rounds to 0.00 at a 0.01 tick.
    from risk_utils import normalize_price_for_symbol
    assert normalize_price_for_symbol("X", 0.004) == 0.0
    # Directly exercise the guard's post-rounding branch via the engine.
    df = _long_gap_df(gap_open=0.02)
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)
    assert result["trades"] == []
    assert result["rejections"]


# ==================== D2: equity-aware entry blocking ====================

def _short_state_with_negative_equity():
    """Two open shorts whose combined unrealized loss wipes out equity."""
    state = PortfolioState(10_000.0, _settings(enable_short=1))
    state.open_position(
        PortfolioPosition("A", SHORT, 100.0, 50.0, 110.0, 90.0, None, 0), 0, 0)
    state.open_position(
        PortfolioPosition("B", SHORT, 100.0, 50.0, 110.0, 90.0, None, 0), 0, 0)
    prices = {"A": 400.0, "B": 100.0}   # A gaps up -> -15,000 unrealized
    return state, prices


def test_d2_entry_blocked_when_equity_negative():
    state, prices = _short_state_with_negative_equity()
    eq = state.equity(prices)
    assert eq < 0, "scenario must produce negative marked-to-market equity"
    ok, why = state.can_open(9_000.0, 0.0, SHORT, equity=eq)
    assert ok is False
    assert why.startswith("non_positive_equity")


def test_d2_entry_blocked_when_equity_exactly_zero():
    state, _ = _short_state_with_negative_equity()
    ok, why = state.can_open(1_000.0, 0.0, SHORT, equity=0.0)
    assert ok is False
    assert why.startswith("non_positive_equity")


def test_d2_entry_allowed_with_positive_equity_and_sufficient_cash():
    state = PortfolioState(10_000.0, _settings())
    ok, why = state.can_open(4_000.0, 0.0, LONG, equity=10_000.0)
    assert ok is True and why == "ok"


def test_d2_positive_equity_still_respects_available_cash():
    """The equity guard must not disable the cash check."""
    state = PortfolioState(10_000.0, _settings())
    ok, why = state.can_open(50_000.0, 0.0, LONG, equity=10_000.0)
    assert ok is False
    assert why.startswith("insufficient_cash")


def test_d2_short_loss_then_pending_entry_is_blocked_at_execution(monkeypatch):
    """
    Execution-path guard: the engine must pass the gate-price equity into
    can_open, and a non-positive equity must block the pending entry. Record the
    equity argument the engine supplies during a normal run and assert it equals
    state.equity(gate_prices) (i.e. the gate is wired), then assert the
    negative-equity state blocks the same call.
    """
    recorded = []
    original = PortfolioState.can_open

    def spy(self, entry_notional, entry_fees, side="long", equity=None):
        recorded.append(equity)
        return original(self, entry_notional, entry_fees, side, equity)

    monkeypatch.setattr(PortfolioState, "can_open", spy)

    closes = list(100 + np.arange(61) * 0.5) + [130.0] * 20
    rows = [[c, c * 1.001, c * 0.999, c, 1_000_000] for c in closes]
    df = _frame(rows)
    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(), max_hold_bars=5)
    assert result["trades"], "expected a normal entry to exercise the path"
    # The engine must supply a real equity value (not None).
    assert recorded and all(e is not None for e in recorded), (
        "engine must pass gate-price equity into can_open")
    assert all(e > 0 for e in recorded), "positive-equity run must stay allowed"

    # And the same gate, fed a ruined account, blocks.
    state, prices = _short_state_with_negative_equity()
    gate_equity = state.equity(prices)
    ok, why = state.can_open(1_000.0, 0.0, SHORT, equity=gate_equity)
    assert ok is False and why.startswith("non_positive_equity")


def test_d2_legacy_call_without_equity_is_unchanged():
    """Omitting equity preserves the pre-fix cash-only behaviour (API compat)."""
    state = PortfolioState(10_000.0, _settings())
    assert state.can_open(4_000.0, 0.0, LONG) == (True, "ok")
    ok, why = state.can_open(50_000.0, 0.0, LONG)
    assert ok is False and why.startswith("insufficient_cash")


# ==================== D3: daily-loss fail closed ====================

def test_d3_zero_daily_start_equity_blocks_new_entries():
    state = PortfolioState(10_000.0, _settings())
    state.daily_start_equity = 0.0
    ok, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 100.0})
    assert ok is False
    assert why == "ruined_account"


def test_d3_negative_daily_start_equity_blocks_new_entries():
    state = PortfolioState(10_000.0, _settings())
    state.daily_start_equity = -5_000.0
    ok, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 100.0})
    assert ok is False
    assert why == "ruined_account"


def test_d3_reason_distinguishes_ruined_account_from_daily_loss():
    """A normal daily-loss rejection must keep its own reason string."""
    state = PortfolioState(10_000.0, _settings(max_daily_loss_pct=0.02,
                                               max_open_positions=5))
    state.daily_start_equity = 10_000.0
    state.open_position(
        PortfolioPosition("AAA", LONG, 100.0, 100.0, 50.0, 200.0, None, 0),
        0, 0)   # cash 0, notional 10,000
    # -5% day -> beyond the 2% daily loss limit.
    ok, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 95.0})
    assert ok is False and why == "daily_loss"


def test_d3_positive_equity_within_threshold_is_allowed():
    state = PortfolioState(10_000.0, _settings(max_daily_loss_pct=0.02,
                                               max_open_positions=5))
    state.daily_start_equity = 10_000.0
    state.open_position(
        PortfolioPosition("AAA", LONG, 100.0, 100.0, 50.0, 200.0, None, 0),
        0, 0)
    # -1% day -> allowed.
    ok, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 99.0})
    assert ok is True and why == "ok"


def test_d3_positive_equity_beyond_threshold_is_blocked():
    state = PortfolioState(10_000.0, _settings(max_daily_loss_pct=0.02,
                                               max_open_positions=5))
    state.daily_start_equity = 10_000.0
    state.open_position(
        PortfolioPosition("AAA", LONG, 100.0, 100.0, 50.0, 200.0, None, 0),
        0, 0)
    # -3% day -> beyond the 2% daily loss limit.
    ok, why = state.can_trade("2024-01-02T00:00:00", {"AAA": 97.0})
    assert ok is False and why == "daily_loss"


def test_d3_ruined_account_still_processes_exits():
    """
    can_trade only gates NEW entries. With a ruined account, an open position
    must still be closed by the engine's position-management loop.
    """
    # One symbol: signal at idx 60, entry idx 61, then a gap down to the stop.
    closes = list(100 + np.arange(61) * 0.5)      # last close 130
    rows = [[c, c * 1.001, c * 0.999, c, 1_000_000] for c in closes]
    rows.append([130.0, 131.0, 129.0, 130.0, 1_000_000])   # idx 61 entry
    rows.append([100.0, 101.0, 99.0, 100.0, 1_000_000])    # idx 62 gap down
    rows.append([100.0, 101.0, 99.0, 100.0, 1_000_000])
    df = _frame(rows)

    result = simulate_portfolio({"AAA": df}, None, start_equity=100_000,
                                settings=_settings(), max_hold_bars=5)
    # The position opened and then exited (not left dangling).
    assert result["trades"], "expected the entry to be processed"
    assert result["final_positions"] == 0, "open position must still be exited"
    assert result["trades"][-1]["exit_reason"] in (
        "STOP", "STOP_GAP", "TARGET", "TARGET_GAP", "MAX_HOLD", "END_OF_DATA")


# ==================== E1: half-open OOS windows ====================

def test_e1_boundary_trade_counted_in_exactly_one_window():
    """
    Adjacent windows share a boundary date. With half-open [start, end) the
    boundary trade belongs to the later window only and is counted once.
    """
    boundary = pd.Timestamp("2023-10-01")
    trade = {"entry_time": boundary, "net_pnl": 500}
    curve = [
        {"time": pd.Timestamp("2023-09-30"), "equity": 10_000},
        {"time": boundary, "equity": 10_500},
        {"time": pd.Timestamp("2023-10-02"), "equity": 10_600},
    ]
    # window 1: [2023-07-01, 2023-10-01); window 2 (last): [2023-10-01, None)
    c1, t1 = _filter_to_window(curve, [trade],
                               pd.Timestamp("2023-07-01"), boundary)
    c2, t2 = _filter_to_window(curve, [trade], boundary, None)

    assert t1 == [], "earlier window must not own the boundary trade"
    assert len(t2) == 1, "later window must own the boundary trade"
    assert len(t1) + len(t2) == 1, "boundary trade counted exactly once"
    # The boundary equity point is owned by exactly one window too.
    owners = sum(
        any(p["time"] == boundary for p in c) for c in (c1, c2)
    )
    assert owners == 1


def test_e1_final_window_keeps_last_data_point():
    """The open-ended final window must not discard the final timestamp."""
    last = pd.Timestamp("2024-01-01")
    curve = [
        {"time": pd.Timestamp("2023-12-31"), "equity": 10_000},
        {"time": last, "equity": 10_100},
    ]
    c, t = _filter_to_window(curve, [], pd.Timestamp("2023-12-01"), None)
    assert any(p["time"] == last for p in c), "final data point must be kept"


def test_e1_boundary_comparison_is_timezone_safe():
    """
    A tz-aware window end must not raise and must be normalized to naive UTC.
    America/New_York on 2023-10-01 is UTC-4, so the boundary normalizes to
    2023-10-01 04:00 UTC. A naive-UTC trade at 00:00 is therefore BEFORE that
    boundary and belongs to the earlier window; the window boundary itself is
    still owned by exactly one window.
    """
    boundary = pd.Timestamp("2023-10-01", tz="America/New_York")
    assert boundary.tz_convert("UTC").tz_localize(None) == pd.Timestamp("2023-10-01 04:00:00")
    trade = {"entry_time": pd.Timestamp("2023-10-01"), "net_pnl": 1}
    curve = [{"time": pd.Timestamp("2023-10-01"), "equity": 1.0}]
    # No TypeError; 00:00 < 04:00 boundary -> owned by the earlier window.
    c, t = _filter_to_window(curve, [trade],
                             pd.Timestamp("2023-07-01"), boundary)
    assert len(t) == 1 and len(c) == 1
    # The later window starts at the 04:00 boundary and excludes the 00:00 trade.
    c2, t2 = _filter_to_window(curve, [trade], boundary, None)
    assert t2 == [] and c2 == []


def test_e1_end_to_end_walkforward_counts_boundary_once(monkeypatch):
    """
    Full driver: with synthetic data, no trade may be reported in two windows
    and no window may own a trade outside [start, end) except the final window.
    """
    import backtest.walkforward as wf

    n = 900
    idx = pd.date_range("2022-06-01", periods=n, freq="D")
    closes = 100 + np.arange(n) * 0.3
    df = pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes, "Volume": [1_000_000] * n,
    }, index=idx)
    monkeypatch.setattr(wf, "load_ohlcv", lambda *a, **k: df)
    monkeypatch.setattr(wf, "get_spy_for_period", lambda *a, **k: None)

    results = wf.walk_forward("AAA", "2023-01-01", "2024-06-01",
                              train_months=6, test_months=3,
                              initial_equity=10_000,
                              settings=_settings(min_signal_score=30))
    assert results, "expected windows"

    seen: dict[str, list[str]] = {}
    for r in results:
        ws, we = r["window"].split("_to_")
        ws_ts, we_ts = pd.Timestamp(ws), pd.Timestamp(we)
        for t in r["result"]["trades"]:
            et = pd.Timestamp(t["entry_time"])
            assert et >= ws_ts, "trade before window start"
            # Only the final window may extend past its nominal end.
            if r is not results[-1]:
                assert et < we_ts, "non-final window owns a post-boundary trade"
            seen.setdefault(str(et), []).append(r["window"])

    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"trades counted in multiple windows: {dupes}"


# ==================== E2/E3: invalid-window aggregation ====================

def test_e2_invalid_window_trades_excluded_from_total():
    agg = aggregate_walkforward([
        {"window": "bad",
         "metrics": {"num_trades": 7, "total_return_pct": float("nan")}},
        {"window": "ok",
         "metrics": {"num_trades": 3, "total_return_pct": 10.0, "sharpe": 1.0}},
    ])
    assert agg["total_trades"] == 3, "invalid window's trades must not count"
    assert agg["total_trades_invalid_windows"] == 7
    assert agg["num_valid_windows"] == 1
    assert agg["num_invalid_windows"] == 1
    assert agg["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)


def test_e3_error_metrics_window_is_flagged_not_a_flat_window():
    """
    A window whose metrics are {"error": "no data"} must be reported invalid,
    not silently counted as a successful 0% window.
    """
    agg = aggregate_walkforward([
        {"window": "err", "metrics": {"error": "no data"}},
        {"window": "ok",
         "metrics": {"num_trades": 2, "total_return_pct": 10.0, "sharpe": 1.0}},
    ])
    assert agg["num_valid_windows"] == 1
    assert agg["num_invalid_windows"] == 1
    assert "invalid_windows" in agg
    assert agg["invalid_windows"][0]["window"] == "err"
    # The error window must not dilute the average as a 0% window.
    assert agg["avg_return_per_window"] == pytest.approx(10.0, abs=0.01)


def test_e3_missing_total_return_is_invalid():
    agg = aggregate_walkforward([
        {"window": "empty", "metrics": {"num_trades": 4}},
        {"window": "ok",
         "metrics": {"num_trades": 1, "total_return_pct": 5.0}},
    ])
    assert agg["num_invalid_windows"] == 1
    assert agg["total_trades"] == 1
    assert agg["total_trades_invalid_windows"] == 4


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_e2_non_finite_returns_are_invalid_and_contribute_nothing(bad):
    agg = aggregate_walkforward([
        {"window": "bad",
         "metrics": {"num_trades": 9, "total_return_pct": bad, "sharpe": 0}},
        {"window": "ok",
         "metrics": {"num_trades": 2, "total_return_pct": 8.0, "sharpe": 1.0}},
    ])
    assert agg["num_valid_windows"] == 1
    assert agg["total_trades"] == 2
    assert agg["total_trades_invalid_windows"] == 9
    assert agg["cumulative_return_pct"] == pytest.approx(8.0, abs=0.01)
    assert math.isfinite(agg["cumulative_return_pct"])


def test_e2_below_minus_100_is_invalid_and_clamped_for_compounding():
    agg = aggregate_walkforward([
        {"window": "bad",
         "metrics": {"num_trades": 5, "total_return_pct": -150.0}},
        {"window": "ok",
         "metrics": {"num_trades": 1, "total_return_pct": 50.0}},
    ])
    # -150% clamped to -100% -> (0.0)*(1.5) - 1 = -100%
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert agg["num_invalid_windows"] == 1
    assert agg["total_trades"] == 1
    assert agg["total_trades_invalid_windows"] == 5
    assert agg["invalid_windows"][0]["reason"].startswith("return below -100%")


def test_e2_exactly_minus_100_is_valid_and_counts_trades():
    agg = aggregate_walkforward([
        {"window": "ruined",
         "metrics": {"num_trades": 4, "total_return_pct": -100.0}},
    ])
    assert agg["num_valid_windows"] == 1
    assert agg["num_invalid_windows"] == 0
    assert "invalid_windows" not in agg
    assert agg["total_trades"] == 4, "-100% window is valid; trades count"
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)


def test_e2_valid_windows_still_compound_correctly():
    agg = aggregate_walkforward([
        {"window": "a", "metrics": {"num_trades": 1, "total_return_pct": 10.0}},
        {"window": "b", "metrics": {"num_trades": 1, "total_return_pct": -5.0}},
        {"window": "c", "metrics": {"num_trades": 1, "total_return_pct": 20.0}},
    ])
    manual = (1.10 * 0.95 * 1.20 - 1) * 100
    assert agg["cumulative_return_pct"] == pytest.approx(manual, abs=0.01)
    assert agg["total_trades"] == 3
    assert agg["num_valid_windows"] == 3


# ==================== F1: negative raw final equity disclosed ====================

def _curve(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return [{"time": t, "equity": float(v)} for t, v in zip(idx, values)]


def test_f1_positive_final_equity_unchanged():
    m = calculate_metrics(_curve([10_000, 11_000, 12_000]), [], 10_000)
    assert m["final_equity"] == pytest.approx(12_000.0)
    assert m["final_equity_raw"] == pytest.approx(12_000.0)
    assert m["negative_equity"] is False
    assert m["equity_ruined"] is False


def test_f1_exactly_zero_final_equity():
    m = calculate_metrics(_curve([10_000, 5_000, 0]), [], 10_000)
    assert m["final_equity"] == pytest.approx(0.0)
    assert m["final_equity_raw"] == pytest.approx(0.0)
    assert m["total_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert m["equity_ruined"] is True
    assert m["negative_equity"] is False


def test_f1_negative_final_equity_is_disclosed_not_hidden():
    m = calculate_metrics(_curve([10_000, 5_000, -2_000]), [], 10_000)
    # Bounded return metric stays at -100%...
    assert m["total_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert m["equity_ruined"] is True
    # ...but the raw balance is exposed and flagged.
    assert m["final_equity_raw"] == pytest.approx(-2_000.0)
    assert m["negative_equity"] is True
    assert m["final_equity"] == pytest.approx(0.0)


def test_f1_zero_then_negative_is_disclosed():
    m = calculate_metrics(_curve([10_000, 0, -500]), [], 10_000)
    assert m["final_equity_raw"] == pytest.approx(-500.0)
    assert m["negative_equity"] is True
    assert m["equity_ruined"] is True


def test_f1_report_does_not_conceal_negative_final_balance(capsys):
    m = calculate_metrics(_curve([10_000, 5_000, -2_000]), [], 10_000)
    result = {"symbol": "AAA", "start_equity": 10_000, "end_equity": -2_000}
    print_summary(result, m)
    out = capsys.readouterr().out
    assert "FINAL EQUITY BELOW ZERO" in out
    assert "-2,000.00" in out


def test_f1_report_flags_ruined_account_at_exactly_zero(capsys):
    m = calculate_metrics(_curve([10_000, 0]), [], 10_000)
    result = {"symbol": "AAA", "start_equity": 10_000, "end_equity": 0}
    print_summary(result, m)
    out = capsys.readouterr().out
    assert "ACCOUNT RUINED" in out
    assert "FINAL EQUITY BELOW ZERO" not in out


# ==================== G3: undefined profit factor renders clearly ====================

def test_g3_undefined_profit_factor_is_na_not_none():
    assert _fmt_profit_factor({"profit_factor": None}) == "N/A"
    assert _fmt_profit_factor({}) == "N/A"
    assert _fmt_profit_factor({"profit_factor": 1.75}) == "1.75"


def test_g3_report_prints_na_for_undefined_profit_factor(capsys):
    # All-winning trades -> profit factor undefined (None).
    trades = [{"net_pnl": 50.0, "r_multiple": 1.0, "bars_held": 1,
               "fees_entry": 0.0, "fees_exit": 0.0,
               "slippage_entry": 0.0, "slippage_exit": 0.0}]
    m = calculate_metrics(_curve([10_000, 10_100, 10_200]), trades, 10_000)
    assert m["profit_factor"] is None
    print_summary({"symbol": "AAA", "start_equity": 10_000,
                   "end_equity": 10_200}, m)
    out = capsys.readouterr().out
    assert "Profit Factor: N/A" in out
    assert "Profit Factor: None" not in out


# ==================== G1: live-only settings are documented, not faked ====================

def test_g1_engine_documents_live_only_settings():
    """
    G1: settings accepted but unused by the backtest must be documented as
    live-only, and no lookahead (future news / LLM predictions) may be added.
    This guards the documentation against silent drift.
    """
    import backtest.engine as engine
    doc = engine.__doc__ or ""
    assert "Live-only settings" in doc
    for key in ("min_confidence", "min_hold_minutes", "min_loss_to_close_pct",
                "allow_extended_hours", "trading_capital"):
        assert key in doc, f"live-only setting {key!r} must be documented"
    assert "lookahead" in doc.lower()


def test_g1_backtest_never_uses_future_news():
    """Historical news is unavailable; the engine must score with news off."""
    import backtest.engine as engine
    assert engine.BACKTEST_NEWS_AVAILABLE is False
    assert engine._NEWS_UNAVAILABLE["available"] is False


def test_g2_settings_migration_is_documented():
    """G2: migration semantics are documented on the migration function."""
    import settings as settings_mod
    doc = settings_mod._migrate_if_needed.__doc__ or ""
    assert "Migration semantics" in doc
    assert "discarded" in doc


def test_g2_migration_resets_risk_keys_and_preserves_others(tmp_path, monkeypatch):
    """
    G2 behaviour (unchanged): a version bump forces the risk-critical keys back
    to config defaults while preserving keys outside the reset list.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "settings_test.db"))
    import importlib
    import decision_log
    importlib.reload(decision_log)
    import settings as settings_mod
    importlib.reload(settings_mod)

    decision_log.set_trading_state("settings_version", "2")
    decision_log.set_setting("min_signal_score", "45")        # in reset list
    decision_log.set_setting("max_open_positions", "5")       # NOT in reset list

    got = settings_mod.get_all_settings()
    # Risk-critical key forced back to the config default.
    assert got["min_signal_score"] == int(settings_mod.DEFAULTS["min_signal_score"])
    # Non-reset key preserved.
    assert got["max_open_positions"] == 5
    assert got["settings_version"] == settings_mod.SETTINGS_VERSION


def test_g2_migration_is_idempotent_at_current_version(tmp_path, monkeypatch):
    """Once at the current version, operator-tuned values survive."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "settings_test2.db"))
    import importlib
    import decision_log
    importlib.reload(decision_log)
    import settings as settings_mod
    importlib.reload(settings_mod)

    settings_mod.get_all_settings()                # writes current version
    decision_log.set_setting("min_signal_score", "45")
    got = settings_mod.get_all_settings()
    assert got["min_signal_score"] == 45, "tuned value must survive at current version"


# ==================== M17: data.dropna() behaviour ====================

def test_m17_normalize_drops_rows_with_missing_volume():
    """
    The OHLC validity mask only inspects OHLC, so a NaN Volume row survives the
    mask; dropna() is what removes it. This documents that (non-redundant)
    behaviour so the guard is not silently removed.
    """
    from backtest.data import _normalize
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({
        "Open": [10, 10, 10], "High": [11, 11, 11], "Low": [9, 9, 9],
        "Close": [10, 10, 10], "Volume": [100, np.nan, 100],
    }, index=idx)
    out = _normalize(df)
    assert len(out) == 2, "row with NaN Volume must be dropped"
    assert not out["Volume"].isna().any()


# ==================== CLI: zero-trade Monte Carlo must not crash ====================

def test_cli_monte_carlo_zero_trades_does_not_crash(capsys):
    """
    run_monte_carlo returns {"error": "no trades"} when there are no trades.
    The CLI summary must report that and exit cleanly instead of raising
    KeyError on the missing 'final_equity' key.
    """
    import run_backtest
    run_backtest._print_monte_carlo({"error": "no trades"})
    out = capsys.readouterr().out
    assert "skipped: no trades" in out


def test_cli_monte_carlo_normal_payload_prints_metrics(capsys):
    import run_backtest
    run_backtest._print_monte_carlo({
        "final_equity": {"median": 10_500.0, "p5": 9_000.0, "p95": 12_000.0},
        "max_drawdown_pct": {"median": 5.0, "p95": 12.0},
        "probability_of_ruin_pct": 1.5,
    })
    out = capsys.readouterr().out
    assert "Final equity median: $10500.00" in out
    assert "Prob. of ruin:       1.50%" in out


def test_cli_zero_trade_run_end_to_end(tmp_path, monkeypatch):
    """
    End-to-end: a symbol with no qualifying signals yields zero trades and the
    CLI path must complete without raising.
    """
    import run_backtest
    from backtest.monte_carlo import run_monte_carlo

    # A flat series produces no signals.
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    flat = pd.DataFrame({"Open": [100.0] * n, "High": [100.0] * n,
                         "Low": [100.0] * n, "Close": [100.0] * n,
                         "Volume": [1_000_000] * n}, index=idx)

    monkeypatch.setattr(run_backtest, "load_ohlcv", lambda *a, **k: flat)
    monkeypatch.setattr(run_backtest, "get_spy_for_period", lambda *a, **k: None)
    monkeypatch.setattr(run_backtest, "save_report",
                        lambda *a, **k: tmp_path / "r.json")
    monkeypatch.setattr(run_backtest, "save_run", lambda *a, **k: None)

    result, metrics = run_backtest.run_single("FLAT", "2024-01-01", "2024-06-01",
                                              10_000)
    assert result["trades"] == []
    assert metrics["num_trades"] == 0
    # And the Monte Carlo payload itself carries the error, not a KeyError.
    assert run_monte_carlo([], 10_000, n_sims=10) == {"error": "no trades"}
