import argparse
from datetime import datetime, timedelta

from backtest.data import load_ohlcv, load_multi, get_spy_for_period
from backtest.engine import simulate, simulate_portfolio
from backtest.metrics import calculate_metrics
from backtest.report import (
    save_report, print_summary, print_portfolio_summary,
)
from backtest.walkforward import walk_forward, aggregate_walkforward
from backtest.storage import save_run
from backtest.monte_carlo import run_monte_carlo


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

    mc = run_monte_carlo(result["trades"], initial, n_sims=1000)
    print("\n=== Monte Carlo (1000 shuffles) ===")
    print(f"  Final equity median: ${mc['final_equity']['median']:.2f}")
    print(f"  Final equity P5:     ${mc['final_equity']['p5']:.2f}")
    print(f"  Final equity P95:    ${mc['final_equity']['p95']:.2f}")
    print(f"  Max DD median:       {mc['max_drawdown_pct']['median']:.2f}%")
    print(f"  Max DD P95:          {mc['max_drawdown_pct']['p95']:.2f}%")
    print(f"  Prob. of ruin:       {mc['probability_of_ruin_pct']:.2f}%")

    return result, metrics


def run_multi(symbols: list[str], start: str, end: str, initial: float):
    """Legacy per-symbol mode (kept for compatibility)."""
    all_metrics = []
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {sym}")
        print(f"{'='*60}")
        out = run_single(sym, start, end, initial)
        if out:
            all_metrics.append((sym, out[1]))
    print(f"\n\n{'='*60}")
    print(" SUMMARY ACROSS SYMBOLS (each treated independently)")
    print(f"{'='*60}")
    print(f"{'Symbol':<8} {'Return':>10} {'DD':>10} {'Sharpe':>8} {'Trades':>8} {'Win%':>8}")
    print(f"{'-'*60}")
    for sym, m in all_metrics:
        print(f"{sym:<8} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('max_drawdown_pct', 0):>9.2f}% "
              f"{m.get('sharpe', 0):>8.2f} "
              f"{m.get('num_trades', 0):>8} "
              f"{m.get('win_rate_pct', 0):>7.1f}%")


def run_portfolio(symbols: list[str], start: str, end: str, initial: float):
    """TRUE portfolio backtest (shared cash, equity, exposure)."""
    print(f"🔄 Loading {len(symbols)} symbols from {start} to {end}...")
    data = load_multi(symbols, start, end)
    if not data:
        print("❌ no data")
        return
    spy = get_spy_for_period(start, end)
    print(f"✓ {len(data)} symbols loaded")

    result = simulate_portfolio(data, spy, start_equity=initial)
    metrics = calculate_metrics(
        result["equity_curve"], result["trades"], initial,
    )
    print_portfolio_summary(list(data.keys()), result, metrics)

    path = save_report(result, metrics, "portfolio")
    print(f"📁 Report saved: {path}")

    mc = run_monte_carlo(result["trades"], initial, n_sims=1000)
    print("\n=== Monte Carlo (1000 shuffles) ===")
    print(f"  Final equity median: ${mc['final_equity']['median']:.2f}")
    print(f"  Final equity P5:     ${mc['final_equity']['p5']:.2f}")
    print(f"  Final equity P95:    ${mc['final_equity']['p95']:.2f}")
    print(f"  Max DD median:       {mc['max_drawdown_pct']['median']:.2f}%")
    print(f"  Max DD P95:          {mc['max_drawdown_pct']['p95']:.2f}%")
    print(f"  Prob. of ruin:       {mc['probability_of_ruin_pct']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Backtest the consensus trader")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--initial", type=float, default=10000.0)
    parser.add_argument("--portfolio", action="store_true",
                        help="Run true portfolio backtest (shared equity)")
    parser.add_argument("--walkforward", action="store_true")
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    args = parser.parse_args()

    if args.walkforward:
        results = walk_forward(args.symbol, args.start, args.end,
                               args.train_months, args.test_months)
        agg = aggregate_walkforward(results)
        print("\n" + "=" * 60)
        print("📊 WALK-FORWARD AGGREGATE")
        print("=" * 60)
        for k, v in agg.items():
            print(f"  {k}: {v}")
        return

    if args.portfolio:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not syms:
            print("❌ --portfolio requires --symbols")
            return
        run_portfolio(syms, args.start, args.end, args.initial)
        return

    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        run_multi(syms, args.start, args.end, args.initial)
    else:
        run_single(args.symbol, args.start, args.end, args.initial)


if __name__ == "__main__":
    main()
