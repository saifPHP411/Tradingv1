from datetime import datetime
from dateutil.relativedelta import relativedelta
from backtest.data import load_ohlcv, load_multi, get_spy_for_period
from backtest.engine import simulate, simulate_portfolio
from backtest.metrics import calculate_metrics


def walk_forward(symbol: str, start_date: str, end_date: str,
                 train_months: int = 12, test_months: int = 3) -> list[dict]:
    """Legacy single-symbol rolling OOS. Kept for compatibility."""
    results = []
    cur = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while True:
        test_start = cur + relativedelta(months=train_months)
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end:
            break
        print(f"\n Window: {test_start.date()} → {test_end.date()}")
        df = load_ohlcv(symbol, test_start.strftime("%Y-%m-%d"),
                        test_end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 70:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue
        spy_df = get_spy_for_period(test_start.strftime("%Y-%m-%d"),
                                    test_end.strftime("%Y-%m-%d"))
        result = simulate(symbol, df, spy_df, start_equity=10000.0)
        metrics = calculate_metrics(result["equity_curve"], result["trades"], 10000.0)
        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": result,
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
    Rolling OOS evaluation across a symbol set.
    Uses simulate_portfolio so all symbols share one portfolio per window.
    NO parameter optimization — this is pure OOS evaluation.
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
        data = load_multi(symbols, test_start.strftime("%Y-%m-%d"),
                          test_end.strftime("%Y-%m-%d"))
        if not data:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue
        spy = get_spy_for_period(test_start.strftime("%Y-%m-%d"),
                                 test_end.strftime("%Y-%m-%d"))
        result = simulate_portfolio(data, spy, start_equity=10000.0)
        metrics = calculate_metrics(result["equity_curve"], result["trades"], 10000.0)
        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": result,
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
    total_return = 0.0
    all_returns = []
    all_sharpes = []
    for r in results:
        m = r["metrics"]
        total_trades += m.get("num_trades", 0)
        total_return += m.get("total_return_pct", 0)
        all_returns.append(m.get("total_return_pct", 0))
        all_sharpes.append(m.get("sharpe", 0))
    return {
        "num_windows": len(results),
        "total_trades": total_trades,
        "cumulative_return_pct": round(total_return, 2),
        "avg_return_per_window": round(float(np.mean(all_returns)), 2),
        "std_return_per_window": round(float(np.std(all_returns)), 2),
        "avg_sharpe": round(float(np.mean(all_sharpes)), 2),
        "positive_windows": sum(1 for r in all_returns if r > 0),
        "negative_windows": sum(1 for r in all_returns if r < 0),
    }
