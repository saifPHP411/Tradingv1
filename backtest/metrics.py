import numpy as np
import pandas as pd


def calculate_metrics(equity_curve: list, trades: list, initial_equity: float,
                      periods_per_year: int = 252) -> dict:
    """
    periods_per_year: 252 for daily bars, 252*390 for 1-minute intraday,
                      52 for weekly, 12 for monthly.
    """
    if not equity_curve or len(equity_curve) < 2:
        return {"error": "no data"}
    eq = pd.DataFrame(equity_curve)
    eq["time"] = pd.to_datetime(eq["time"])
    eq = eq.set_index("time").sort_index()
    values = eq["equity"].values.astype(float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"error": "no data"}

    equity_ruined = bool(np.any(values <= 0.0))

    prev = values[:-1]
    safe_prev = np.where(prev > 0, prev, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(values) / safe_prev
    returns = returns[np.isfinite(returns)]

    # A cash-secured account cannot lose more than 100% of its capital, so
    # floor the terminal equity at zero when computing return-based metrics.
    final_equity = max(float(values[-1]), 0.0)
    total_return_pct = ((final_equity / initial_equity - 1) * 100
                        if initial_equity > 0 else 0.0)

    n_periods = len(values)
    years = max(n_periods / periods_per_year, 1 / periods_per_year)
    if initial_equity > 0 and final_equity > 0:
        cagr = ((final_equity / initial_equity) ** (1 / years) - 1) * 100
    elif initial_equity > 0:
        cagr = -100.0  # complete loss
    else:
        cagr = 0.0

    running_max = np.maximum.accumulate(values)
    denom = np.where(running_max > 0, running_max, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = (values - running_max) / denom
    drawdowns = drawdowns[np.isfinite(drawdowns)]
    max_dd = float(drawdowns.min()) * 100 if len(drawdowns) else 0.0
    max_dd = max(max_dd, -100.0)

    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0

    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    if not trades:
        return {
            "total_return_pct": round(float(total_return_pct), 2),
            "cagr_pct": round(float(cagr), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "sharpe": round(float(sharpe), 2),
            "sortino": round(float(sortino), 2),
            "calmar": round(float(calmar), 2),
            "num_trades": 0,
            "equity_ruined": equity_ruined,
            "final_equity": round(final_equity, 2),
        }

    pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    rs = [t.get("r_multiple", 0.0) for t in trades]

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
    median_r = float(np.median(rs)) if rs else 0

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

    return {
        "total_return_pct": round(float(total_return_pct), 2),
        "cagr_pct": round(float(cagr), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "calmar": round(float(calmar), 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "largest_win": round(float(largest_win), 2),
        "largest_loss": round(float(largest_loss), 2),
        "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else None,
        "expectancy": round(float(expectancy), 2),
        "avg_r_multiple": round(float(avg_r), 3),
        "median_r_multiple": round(median_r, 3),
        "max_consecutive_wins": max_cons_wins,
        "max_consecutive_losses": max_cons_losses,
        "avg_hold_bars": round(float(avg_hold), 1),
        "total_fees": round(float(total_fees), 2),
        "total_slippage": round(float(total_slip), 2),
        "final_equity": round(final_equity, 2),
        "equity_ruined": equity_ruined,
    }
