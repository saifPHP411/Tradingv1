"""
Walk-forward evaluation.

Naming note: this module performs ROLLING OUT-OF-SAMPLE EVALUATION.
It does NOT perform parameter optimization or in-sample training.
Train windows are advanced over time, but no fitting occurs. Do not
call this an "optimization" pipeline in reports or UI.

Warmup handling:
    Each OOS window loads data from `test_start - WARMUP_DAYS` to `test_end`.
    Indicators use the full range, but no signals are queued before
    `test_start` because `simulate_portfolio` is called with
    `trading_start_ts=test_start`. Reported metrics and trades are filtered
    to `>= test_start`, so warmup trades do not leak into OOS results.

Aggregate:
    cumulative_return_pct compounds per-window returns:
        cum = Π(1 + r_i / 100); cumulative = (cum - 1) * 100
    It is NOT the arithmetic sum.
"""
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import pandas as pd

from backtest.data import load_ohlcv, load_multi, get_spy_for_period
from backtest.engine import simulate, simulate_portfolio
from backtest.metrics import calculate_metrics
from settings import get_all_settings


WARMUP_DAYS = 200


def _naive_ts(x):
    """Normalize any timestamp-like value to a naive UTC pandas Timestamp.

    Data is normalized to naive UTC by backtest.data; walk-forward window
    boundaries are built from naive datetimes. Normalizing here keeps both
    sides comparable even if a caller passes a tz-aware boundary.
    """
    t = pd.Timestamp(x)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _filter_to_window(equity_curve: list, trades: list, test_start_ts):
    """Keep only the test-window portion of the result."""
    start = _naive_ts(test_start_ts)
    curve = [p for p in equity_curve if _naive_ts(p["time"]) >= start]
    trade_list = [
        t for t in trades
        if _naive_ts(t["entry_time"]) >= start
    ]
    return curve, trade_list


def walk_forward(symbol: str, start_date: str, end_date: str,
                 train_months: int = 12, test_months: int = 3,
                 initial_equity: float = 10000.0,
                 max_hold_bars: int | None = None,
                 settings: dict | None = None) -> list[dict]:
    """Single-symbol rolling OOS evaluation (legacy, kept for compatibility)."""
    results = []
    cur = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while True:
        test_start = cur + relativedelta(months=train_months)
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end:
            break
        print(f"\n Window: {test_start.date()} → {test_end.date()}")

        # Load warmup + test range so indicators have >= 60 bars at test_start.
        load_start = (test_start - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        df = load_ohlcv(symbol, load_start, test_end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 70:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue

        spy_df = get_spy_for_period(load_start, test_end.strftime("%Y-%m-%d"))
        test_start_ts = pd.Timestamp(test_start)

        full_result = simulate(
            symbol, df, spy_df, start_equity=initial_equity,
            trading_start_ts=test_start_ts,
            max_hold_bars=max_hold_bars, settings=settings,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"], test_start_ts
        )
        metrics = calculate_metrics(curve, trades, initial_equity)

        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": {"equity_curve": curve, "trades": trades},
            "metrics": metrics,
        })
        print(f" Return: {metrics.get('total_return_pct', 0):+.2f}% | "
              f"DD: {metrics.get('max_drawdown_pct', 0):.2f}% | "
              f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
              f"Trades: {metrics.get('num_trades', 0)}")
        cur += relativedelta(months=test_months)
    return results


def walk_forward_portfolio(symbols: list[str], start_date: str, end_date: str,
                           train_months: int = 12,
                           test_months: int = 3,
                           initial_equity: float = 10000.0,
                           max_hold_bars: int | None = None,
                           settings: dict | None = None) -> list[dict]:
    """
    Portfolio rolling OOS evaluation.
    Each window shares cash/equity/exposure via simulate_portfolio.
    Warmup bars are included, but no trades are opened before test_start.
    """
    results = []
    cur = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while True:
        test_start = cur + relativedelta(months=train_months)
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end:
            break
        print(f"\n Window: {test_start.date()} → {test_end.date()}")

        load_start = (test_start - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        data = load_multi(symbols, load_start, test_end.strftime("%Y-%m-%d"))
        if not data:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue

        spy = get_spy_for_period(load_start, test_end.strftime("%Y-%m-%d"))
        test_start_ts = pd.Timestamp(test_start)

        full_result = simulate_portfolio(
            data, spy, start_equity=initial_equity,
            trading_start_ts=test_start_ts,
            max_hold_bars=max_hold_bars, settings=settings,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"], test_start_ts
        )
        metrics = calculate_metrics(curve, trades, initial_equity)

        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": {"equity_curve": curve, "trades": trades},
            "metrics": metrics,
        })
        print(f" Return: {metrics.get('total_return_pct', 0):+.2f}% | "
              f"DD: {metrics.get('max_drawdown_pct', 0):.2f}% | "
              f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
              f"Trades: {metrics.get('num_trades', 0)}")
        cur += relativedelta(months=test_months)
    return results


def aggregate_walkforward(results: list[dict]) -> dict:
    """
    Compound per-window returns into a single cumulative figure.

    Per-window returns are validated before compounding:
      - NaN / +inf / -inf are rejected (a window whose equity curve was
        degenerate must not silently poison the cumulative result).
      - a return below -100% is impossible for a cash-secured account and is
        treated as a complete loss (-100%), never as a negative multiplier
        that would flip the sign of the cumulative product.

    Rejected windows are reported in `invalid_windows` rather than dropped
    silently.
    """
    if not results:
        return {"error": "no windows"}
    import numpy as np
    total_trades = 0
    all_returns: list[float] = []
    all_sharpes: list[float] = []
    invalid_windows: list[dict] = []
    compounded = 1.0
    for idx, r in enumerate(results):
        m = r.get("metrics", {}) or {}
        total_trades += m.get("num_trades", 0) or 0

        raw_ret = m.get("total_return_pct", 0)
        try:
            ret = float(raw_ret)
        except (TypeError, ValueError):
            ret = float("nan")

        if not math.isfinite(ret):
            invalid_windows.append({"window": r.get("window", idx),
                                    "reason": "non-finite total_return_pct",
                                    "value": raw_ret})
            continue

        if ret < -100.0:
            # A cash-secured account cannot lose more than 100%. Clamp rather
            # than let (1 + r) go negative and flip the product's sign.
            invalid_windows.append({"window": r.get("window", idx),
                                    "reason": "return below -100%; clamped",
                                    "value": ret})
            ret = -100.0

        all_returns.append(ret)
        sharpe = m.get("sharpe", 0)
        try:
            sharpe = float(sharpe)
        except (TypeError, ValueError):
            sharpe = 0.0
        all_sharpes.append(sharpe if math.isfinite(sharpe) else 0.0)
        compounded *= (1.0 + ret / 100.0)

    cumulative_return_pct = (compounded - 1.0) * 100.0

    out = {
        "num_windows": len(results),
        "num_valid_windows": len(all_returns),
        "total_trades": total_trades,
        "cumulative_return_pct": round(cumulative_return_pct, 2),
        "avg_return_per_window": (round(float(np.mean(all_returns)), 2)
                                  if all_returns else 0.0),
        "std_return_per_window": (round(float(np.std(all_returns)), 2)
                                  if all_returns else 0.0),
        "avg_sharpe": (round(float(np.mean(all_sharpes)), 2)
                       if all_sharpes else 0.0),
        "positive_windows": sum(1 for r in all_returns if r > 0),
        "negative_windows": sum(1 for r in all_returns if r < 0),
    }
    if invalid_windows:
        out["invalid_windows"] = invalid_windows
    return out
