import argparse
import sys
from datetime import datetime, timedelta

from backtest.data import load_ohlcv, get_spy_for_period
from backtest.engine import simulate
from backtest.metrics import calculate_metrics
from backtest.report import save_report, print_summary
from backtest.walkforward import walk_forward, aggregate_walkforward
from backtest.storage import save_run


def run_single(symbol: str, start: str, end: str, initial: float):
    print(f"🔄 Loading {symbol} from {start} to {end}...")
    df = load_ohlcv(symbol, start, end)
    if df is None or len(df) < 70:
        print(f"❌ insufficient data for {symbol}")
        return None

    spy_df = get_spy_for_period(start, end)
    print(f"✓ {len(df)} bars loaded")

    result = simulate(symbol, df, spy_df, start_equity=initial)
    metrics = calculate_metrics(result["equity_curve"], result["trades"], initial)

    print_summary(result, metrics)
    path = save_report(result, metrics, f"single_{symbol}")
    print(f"📁 Report saved: {path}")

    save_run(symbol, start, end, {}, metrics)
    return result, metrics


def run_multi(symbols: list[str], start: str, end: str, initial: float):
    """Run backtest on each symbol separately (portfolio simulation = Phase 5)."""
    all_metrics = []
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {sym}")
        print(f"{'='*60}")
        out = run_single(sym, start, end, initial)
        if out:
            all_metrics.append((sym, out[1]))

    print(f"\n\n{'='*60}")
    print("📊 SUMMARY ACROSS SYMBOLS")
    print(f"{'='*60}")
    print(f"{'Symbol':<8} {'Return':>10} {'DD':>10} {'Sharpe':>8} {'Trades':>8} {'Win%':>8}")
    print(f"{'-'*60}")
    for sym, m in all_metrics:
        print(f"{sym:<8} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('max_drawdown_pct', 0):>9.2f}% "
              f"{m.get('sharpe', 0):>8.2f} "
              f"{m.get('num_trades', 0):>8} "
              f"{m.get('win_rate_pct', 0):>7.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Backtest the consensus trader")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)

    args = parser.parse_args()

    if args.walkforward:
        results = walk_forward(args.symbol, args.start, args.end,
                                args.train_months, args.test_months)
        agg = aggregate_walkforward(results)
        print("\n" + "="*60)
        print("📊 WALK-FORWARD AGGREGATE")
        print("="*60)
        for k, v in agg.items():
            print(f"  {k}: {v}")
        return

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        run_multi(syms, args.start, args.end, args.initial)
    else:
        run_single(args.symbol, args.start, args.end, args.initial)


if __name__ == "__main__":
    main()
