"""
Monte Carlo analysis on a set of trades.
Shuffles trade order and recomputes equity metrics to test robustness.
"""
import random
import numpy as np


def run_monte_carlo(trades: list, initial_equity: float = 10000.0,
                    n_sims: int = 1000, ruin_threshold_pct: float = 0.5) -> dict:
    """
    Args:
        trades: list of trade dicts each containing 'net_pnl'
        initial_equity: starting capital
        n_sims: number of shuffles
        ruin_threshold_pct: ruin = equity drops below (1 - pct) of initial

    Returns:
        dict with distribution statistics.
    """
    if not trades:
        return {"error": "no trades"}

    pnls = [float(t.get("net_pnl", 0)) for t in trades]
    if not pnls:
        return {"error": "no PnLs"}

    final_equities = []
    max_drawdowns = []
    ruins = 0
    ruin_level = initial_equity * (1 - ruin_threshold_pct)

    for _ in range(n_sims):
        shuffled = pnls.copy()
        random.shuffle(shuffled)

        equity = initial_equity
        peak = equity
        max_dd = 0.0
        ruined = False

        for pnl in shuffled:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
            if equity < ruin_level:
                ruined = True

        final_equities.append(equity)
        max_drawdowns.append(max_dd)
        if ruined:
            ruins += 1

    fe = np.array(final_equities)
    mdd = np.array(max_drawdowns) * 100  # %

    return {
        "n_sims": n_sims,
        "n_trades": len(pnls),
        "initial_equity": initial_equity,
        "final_equity": {
            "mean": round(float(np.mean(fe)), 2),
            "median": round(float(np.median(fe)), 2),
            "p5": round(float(np.percentile(fe, 5)), 2),
            "p25": round(float(np.percentile(fe, 25)), 2),
            "p75": round(float(np.percentile(fe, 75)), 2),
            "p95": round(float(np.percentile(fe, 95)), 2),
            "min": round(float(np.min(fe)), 2),
            "max": round(float(np.max(fe)), 2),
        },
        "max_drawdown_pct": {
            "mean": round(float(np.mean(mdd)), 2),
            "median": round(float(np.median(mdd)), 2),
            "p5": round(float(np.percentile(mdd, 5)), 2),
            "p95": round(float(np.percentile(mdd, 95)), 2),
        },
        "probability_of_ruin_pct": round(100.0 * ruins / n_sims, 2),
                      }
