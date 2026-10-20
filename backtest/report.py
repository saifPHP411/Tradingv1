import json
from pathlib import Path
from datetime import datetime

REPORT_DIR = Path("backtest_reports")
REPORT_DIR.mkdir(exist_ok=True)


def save_report(result: dict, metrics: dict, name: str) -> Path:
    """Saves equity curve + trades + metrics to disk."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"{name}_{ts}.json"

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "symbol": result.get("symbol"),
        "start_equity": result.get("start_equity"),
        "end_equity": result.get("end_equity"),
        "metrics": metrics,
        "trades": result.get("trades", []),
        "equity_curve": result.get("equity_curve", []),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def print_summary(result: dict, metrics: dict):
    print(f"\n{'='*60}")
    print(f"📊 Backtest: {result.get('symbol', '?')}")
    print(f"{'='*60}")
    print(f"Start Equity:       ${result.get('start_equity', 0):,.2f}")
    print(f"End Equity:         ${result.get('end_equity', 0):,.2f}")
    print(f"Total Return:       {metrics.get('total_return_pct', 0):+.2f}%")
    print(f"CAGR:               {metrics.get('cagr_pct', 0):+.2f}%")
    print(f"Max Drawdown:       {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe:             {metrics.get('sharpe', 0):.2f}")
    print(f"Sortino:            {metrics.get('sortino', 0):.2f}")
    print(f"Calmar:             {metrics.get('calmar', 0):.2f}")
    print(f"{'─'*60}")
    print(f"Trades:             {metrics.get('num_trades', 0)}")
    print(f"Win Rate:           {metrics.get('win_rate_pct', 0):.1f}%")
    print(f"Profit Factor:      {metrics.get('profit_factor', 'N/A')}")
    print(f"Expectancy:         ${metrics.get('expectancy', 0):.2f}")
    print(f"Avg R:              {metrics.get('avg_r_multiple', 0):.2f}R")
    print(f"Max Consec Losses:  {metrics.get('max_consecutive_losses', 0)}")
    print(f"Avg Hold:           {metrics.get('avg_hold_bars', 0):.1f} bars")
    print(f"Fees:               ${metrics.get('total_fees', 0):.2f}")
    print(f"Slippage:           ${metrics.get('total_slippage', 0):.2f}")
    print(f"{'='*60}\n")
