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
import pandas as pd

from backtest.data import load_ohlcv, load_multi, get_spy_for_period
from backtest.engine import simulate, simulate_portfolio
from backtest.metrics import calculate_metrics


WARMUP_DAYS = 200


def _filter_to_window(equity_curve: list, trades: list, test_start_ts):
    """Keep only the test-window portion of the result."""
    curve = [p for p in equity_curve if pd.Timestamp(p["time"]) >= test_start_ts]
    trade_list = [
        t for t in trades
        if pd.Timestamp(t["entry_time"]) >= test_start_ts
    ]
    return curve, trade_list


def walk_forward(symbol: str, start_date: str, end_date: str,
                 train_months: int = 12, test_months: int = 3) -> list[dict]:
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
            symbol, df, spy_df, start_equity=10000.0,
            trading_start_ts=test_start_ts,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"], test_start_ts
        )
        metrics = calculate_metrics(curve, trades, 10000.0)

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
                           test_months: int = 3) -> list[dict]:
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
            data, spy, start_equity=10000.0,
            trading_start_ts=test_start_ts,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"], test_start_ts
        )
        metrics = calculate_metrics(curve, trades, 10000.0)

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
    if not results:
        return {"error": "no windows"}
    import numpy as np
    total_trades = 0
    all_returns: list[float] = []
    all_sharpes: list[float] = []
    compounded = 1.0
    for r in results:
        m = r["metrics"]
        total_trades += m.get("num_trades", 0)
        ret = float(m.get("total_return_pct", 0))
        all_returns.append(ret)
        all_sharpes.append(float(m.get("sharpe", 0)))
        compounded *= (1.0 + ret / 100.0)

    cumulative_return_pct = (compounded - 1.0) * 100.0

    return {
        "num_windows": len(results),
        "total_trades": total_trades,
        "cumulative_return_pct": round(cumulative_return_pct, 2),
        "avg_return_per_window": round(float(np.mean(all_returns)), 2),
        "std_return_per_window": round(float(np.std(all_returns)), 2),
        "avg_sharpe": round(float(np.mean(all_sharpes)), 2),
        "positive_windows": sum(1 for r in all_returns if r > 0),
        "negative_windows": sum(1 for r in all_returns if r < 0),
                   }
