import json
from pathlib import Path
from datetime import datetime, timezone

REPORT_DIR = Path("backtest_reports")
REPORT_DIR.mkdir(exist_ok=True)


def _fmt_profit_factor(metrics: dict) -> str:
    """Human-readable profit factor; 'N/A' when undefined (None or missing)."""
    pf = metrics.get("profit_factor")
    return "N/A" if pf is None else f"{pf}"


def _rejections_of(result: dict) -> list:
    """Rejection events the engine recorded for this run.

    Only the invalid stop/target classes the engine already records are
    included. This is NOT a count of every reason a candidate failed to trade.
    """
    return list(result.get("rejections") or [])


def save_report(result: dict, metrics: dict, name: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"{name}_{ts}.json"
    rejections = _rejections_of(result)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "symbol": result.get("symbol"),
        "start_equity": result.get("start_equity"),
        "end_equity": result.get("end_equity"),
        "metrics": metrics,
        "trades": result.get("trades", []),
        "equity_curve": result.get("equity_curve", []),
        # Rejection EVENTS (not symbols affected); records kept verbatim.
        "rejections": rejections,
        "num_rejections": len(rejections),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _print_rejections(result: dict):
    """Additive warning for recorded invalid stop/target rejections.

    Printed only when at least one rejection event exists, so a clean run stays
    quiet. The count is rejection EVENTS, not symbols, and covers only the
    invalid stop/target classes the engine records.
    """
    n = len(_rejections_of(result))
    if n:
        print(f"⚠️ Rejected entries: {n} "
              f"(invalid stop/target level; see report 'rejections')")


def print_summary(result: dict, metrics: dict):
    print(f"\n{'='*60}")
    print(f" Backtest: {result.get('symbol', '?')}")
    print(f"{'='*60}")
    print(f"Start Equity: ${result.get('start_equity', 0):,.2f}")
    print(f"End Equity: ${result.get('end_equity', 0):,.2f}")
    raw_final = metrics.get("final_equity_raw")
    if raw_final is not None and raw_final < 0:
        print(f"⚠️ FINAL EQUITY BELOW ZERO: ${raw_final:,.2f} "
              f"(account balance is negative; return capped at -100%)")
    elif metrics.get("equity_ruined"):
        print("⚠️ ACCOUNT RUINED: equity reached zero "
              "(reported final equity floored at $0.00)")
    print(f"Total Return: {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"CAGR: {metrics.get('cagr_pct', 0):+.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe: {metrics.get('sharpe', 0):.2f}")
    print(f"Sortino: {metrics.get('sortino', 0):.2f}")
    print(f"Calmar: {metrics.get('calmar', 0):.2f}")
    print(f"{'─'*60}")
    print(f"Trades: {metrics.get('num_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"Profit Factor: {_fmt_profit_factor(metrics)}")
    print(f"Expectancy: ${metrics.get('expectancy', 0):.2f}")
    print(f"Avg R: {metrics.get('avg_r_multiple', 0):.2f}R")
    print(f"Max Consec Losses: {metrics.get('max_consecutive_losses', 0)}")
    print(f"Avg Hold: {metrics.get('avg_hold_bars', 0):.1f} bars")
    print(f"Fees: ${metrics.get('total_fees', 0):.2f}")
    print(f"Slippage: ${metrics.get('total_slippage', 0):.2f}")
    _print_rejections(result)
    print(f"{'='*60}\n")


def print_portfolio_summary(symbols: list[str], result: dict, metrics: dict):
    print(f"\n{'='*60}")
    print(f" PORTFOLIO BACKTEST")
    print(f" Symbols: {', '.join(symbols)}")
    print(f"{'='*60}")
    print(f"Start Equity: ${result.get('start_equity', 0):,.2f}")
    print(f"End Equity: ${result.get('end_equity', 0):,.2f}")
    raw_final = metrics.get("final_equity_raw")
    if raw_final is not None and raw_final < 0:
        print(f"⚠️ FINAL EQUITY BELOW ZERO: ${raw_final:,.2f} "
              f"(account balance is negative; return capped at -100%)")
    elif metrics.get("equity_ruined"):
        print("⚠️ ACCOUNT RUINED: equity reached zero "
              "(reported final equity floored at $0.00)")
    print(f"Total Return: {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe: {metrics.get('sharpe', 0):.2f}")
    print(f"Sortino: {metrics.get('sortino', 0):.2f}")
    print(f"Profit Factor: {_fmt_profit_factor(metrics)}")
    print(f"Win Rate: {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"Avg R: {metrics.get('avg_r_multiple', 0):.2f}R")
    print(f"Trades: {metrics.get('num_trades', 0)}")
    _print_rejections(result)
    print(f"{'='*60}\n")

    # Per-symbol breakdown
    if not result.get("trades"):
        print("(no trades)\n")
        return
    by_symbol: dict[str, list[dict]] = {}
    for t in result["trades"]:
        by_symbol.setdefault(t["symbol"], []).append(t)
    print(f"{'Symbol':<8} {'Trades':>7} {'Win%':>7} {'Net$':>10} {'AvgR':>7}")
    print(f"{'-'*45}")
    for sym in sorted(by_symbol):
        trades = by_symbol[sym]
        wins = [t for t in trades if t["net_pnl"] > 0]
        net = sum(t["net_pnl"] for t in trades)
        avg_r = sum(t.get("r_multiple", 0) for t in trades) / len(trades)
        print(f"{sym:<8} {len(trades):>7} "
              f"{100 * len(wins) / len(trades):>6.1f}% "
              f"{net:>10.2f} {avg_r:>7.2f}")
    print(f"{'='*60}\n")
