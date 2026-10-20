import numpy as np
import pandas as pd


def calculate_metrics(equity_curve: list, trades: list, initial_equity: float) -> dict:
    """Compute comprehensive performance metrics."""
    if not equity_curve or len(equity_curve) < 2:
        return {"error": "no data"}

    eq = pd.DataFrame(equity_curve)
    eq["time"] = pd.to_datetime(eq["time"])
    eq = eq.set_index("time").sort_index()
    values = eq["equity"].values

    # Returns
    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]

    total_return_pct = (values[-1] / initial_equity - 1) * 100

    # Annualized (assuming daily bars, 252 trading days)
    n_days = len(values)
    years = max(n_days / 252, 1 / 252)
    cagr = ((values[-1] / initial_equity) ** (1 / years) - 1) * 100 if initial_equity > 0 else 0

    # Drawdown
    running_max = np.maximum.accumulate(values)
    drawdowns = (values - running_max) / running_max
    max_dd = drawdowns.min() * 100

    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(252)
    else:
        sortino = 0.0

    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    # Trade stats
    if not trades:
        return {
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "calmar": round(calmar, 2),
            "num_trades": 0,
        }

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    rs = [t["r_multiple"] for t in trades]

    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    largest_win = max(wins) if wins else 0
    largest_loss = min(losses) if losses else 0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = np.mean(pnls) if pnls else 0
    avg_r = np.mean(rs) if rs else 0

    # Consecutive
    max_cons_wins = 0
    max_cons_losses = 0
    cur_w = 0
    cur_l = 0
    for p in pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
            max_cons_wins = max(max_cons_wins, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_cons_losses = max(max_cons_losses, cur_l)

    total_fees = sum(t.get("fees_entry", 0) + t.get("fees_exit", 0) for t in trades)
    total_slip = sum(t.get("slippage_entry", 0) + t.get("slippage_exit", 0) for t in trades)
    avg_hold = np.mean([t.get("bars_held", 0) for t in trades]) if trades else 0

    # Monthly returns
    try:
        monthly = eq["equity"].resample("ME").last().pct_change() * 100
        monthly = monthly.dropna()
    except Exception:
        monthly = pd.Series(dtype=float)

    return {
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "largest_win": round(float(largest_win), 2),
        "largest_loss": round(float(largest_loss), 2),
        "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else None,
        "expectancy": round(float(expectancy), 2),
        "avg_r_multiple": round(float(avg_r), 3),
        "max_consecutive_wins": max_cons_wins,
        "max_consecutive_losses": max_cons_losses,
        "avg_hold_bars": round(float(avg_hold), 1),
        "total_fees": round(float(total_fees), 2),
        "total_slippage": round(float(total_slip), 2),
        "final_equity": round(float(values[-1]), 2),
      }
