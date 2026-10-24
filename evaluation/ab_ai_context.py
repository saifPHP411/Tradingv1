"""
Experimental A/B evaluation: baseline strategy vs an AI-conditioned variant.

PURPOSE
    Measure, on identical data / periods / costs, whether gating entries on the
    advisory AI context would have changed outcomes — and by how much.

IMPORTANT LIMITATIONS (read before trusting any number)
    The strategy's own execution engine deliberately refuses to consume LLM
    output in historical backtests, because the model is trained on data that
    postdates the simulated period (see the header of ``backtest/engine.py``).
    A genuine, leak-free historical replay of an LLM overlay is therefore not
    possible with this codebase.

    To keep the experiment honest and leak-free, this driver does NOT call any
    live API for history. Instead it builds a *deterministic stand-in* for the
    AI layer using only information available at the decision bar (a simple
    trend rule over trailing indicators). That stand-in is:

      - strictly causal (no future data), so it cannot leak;
      - reproducible (fixed seed, no network);
      - transparent (the rule is in ``ai_gate_decision``).

    The result is a *mechanism* test: "if an advisor with this behaviour
    existed, how would the gate change the numbers?" It is NOT evidence that
    the real Gemini model improves performance. Any claim that the real model
    helps requires a separate, forward-only paper evaluation.

WHAT IS HELD CONSTANT
    Same symbols, same date range, same costs (fees/slippage), same seed, same
    exit rules. The ONLY difference between arms is the entry gate.

USAGE
    python -m evaluation.ab_ai_context --symbols AAPL,MSFT --start ... --end ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.data import load_multi, get_spy_for_period  # noqa: E402
from backtest.engine import compute_indicators_at  # noqa: E402
from backtest.metrics import calculate_metrics  # noqa: E402
from settings import get_all_settings  # noqa: E402

# Imported for documentation/contract only: the driver never calls the live API.
import market_context  # noqa: E402,F401  (documents the contract only)


# --------------------------------------------------------------------------
# Deterministic, causal stand-in for the AI gate
# --------------------------------------------------------------------------

def ai_gate_decision(ind: dict) -> str:
    """A leak-free stand-in for an AI market-context call.

    Uses ONLY trailing indicators already computed for the decision bar:
      - bullish trend alignment (close > ema20 > ema50) and adx > 25 -> allow
      - otherwise -> block (the AI "sees" an unfavourable regime)

    Deliberately simple and fully reproducible. A behavioural stand-in, not a
    model call.
    """
    if ind is None:
        return "block"
    aligned = (ind["last_close"] > ind["ema20"] > ind["ema50"])
    trending = ind["adx"] > 25
    return "allow" if (aligned and trending) else "block"


# --------------------------------------------------------------------------
# A/B replay over identical bars
# --------------------------------------------------------------------------

def run_arm(symbol: str, df: pd.DataFrame, spy_df: pd.DataFrame | None,
            initial: float, gate: str) -> dict:
    """Replay entries on identical bars, applying the gate.

    gate: "none" (baseline) or "ai" (skip entries the stand-in blocks).

    Position management is a fixed, simple, identical rule for both arms so the
    ONLY difference is which entries are taken:
      - enter at next bar open after a signal bar
      - exit at the first of: stop (stop_atr_multiplier x ATR),
        target (take_profit_rr R), or max_hold bars
    Costs use the settings' fees_per_share and slippage bps.
    """
    from signals import compute_signal_score
    from risk_utils import normalize_slippage

    settings = get_all_settings()
    fees = float(settings.get("fees_per_share", 0.0))
    slip_in = float(settings.get("entry_slippage_bps", 0.0))
    slip_out = float(settings.get("exit_slippage_bps", 0.0))
    max_hold = int(settings.get("max_hold_bars", 60))
    min_score = int(settings.get("min_signal_score", 30))
    stop_mult = float(settings.get("stop_atr_multiplier", 2.5))
    tp_rr = float(settings.get("take_profit_rr", 2.0))
    risk_pct = float(settings.get("max_risk_per_trade", 0.02))

    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    opens = df["Open"].values.astype(float)
    times = list(df.index)

    equity = initial
    curve = []
    trades = []
    i = 60
    blocked = 0
    taken = 0

    while i < len(df) - 1:
        ind = compute_indicators_at(df, i)
        if ind is None:
            i += 1
            continue

        score = compute_signal_score(
            ind, None, news={"available": False, "score": 0.0})["score"]
        if score < min_score:
            i += 1
            continue

        if gate == "ai" and ai_gate_decision(ind) == "block":
            blocked += 1
            i += 1
            continue

        entry_idx = i + 1
        entry_px = opens[entry_idx]
        atr = ind["atr14"]
        stop_dist = atr * stop_mult
        if stop_dist <= 0 or stop_dist > entry_px * 0.20:
            i += 1
            continue
        target_dist = stop_dist * tp_rr

        qty = int((equity * risk_pct) // stop_dist) if stop_dist > 0 else 0
        if qty <= 0 or qty * entry_px < 1.0:
            i += 1
            continue

        fill_in = entry_px + normalize_slippage(entry_px, slip_in, is_buy=True)
        stop_px = fill_in - stop_dist
        target_px = fill_in + target_dist

        exit_px = None
        exit_reason = None
        j = entry_idx
        last = min(entry_idx + max_hold, len(df) - 1)
        while j <= last:
            if lows[j] <= stop_px:
                exit_px = stop_px
                exit_reason = "stop"
                break
            if highs[j] >= target_px:
                exit_px = target_px
                exit_reason = "target"
                break
            if j == last:
                exit_px = closes[j]
                exit_reason = "max_hold"
                break
            j += 1

        fill_out = exit_px - normalize_slippage(exit_px, slip_out, is_buy=False)
        gross = (fill_out - fill_in) * qty
        cost = fees * qty * 2
        net = gross - cost

        equity += net
        trades.append({
            "symbol": symbol,
            "entry_time": str(times[entry_idx]),
            "entry_price": float(fill_in),
            "exit_time": str(times[j]),
            "exit_price": float(fill_out),
            "qty": qty,
            "net_pnl": float(net),
            "exit_reason": exit_reason,
        })
        taken += 1
        curve.append({"time": str(times[entry_idx]), "equity": equity,
                      "n_positions": 1})
        i = j + 1

    if not curve:
        curve.append({"time": str(times[0]), "equity": equity, "n_positions": 0})
    curve.append({"time": str(times[-1]), "equity": equity, "n_positions": 0})

    metrics = calculate_metrics(curve, trades, initial)
    metrics["blocked_entries"] = blocked
    metrics["taken_entries"] = taken
    return metrics


def run_experiment(symbols: list[str], start: str, end: str,
                   initial: float = 10000.0) -> dict:
    """Run baseline and AI-gated arms on identical data. No network for AI."""
    data = load_multi(symbols, start, end)
    spy = get_spy_for_period(start, end)
    out = {"start": start, "end": end, "symbols": list(data.keys()), "arms": {}}
    for gate in ("none", "ai"):
        per_symbol = {}
        for sym, df in data.items():
            per_symbol[sym] = run_arm(sym, df, spy, initial, gate)
        out["arms"][gate] = {"per_symbol": per_symbol,
                             "aggregate": _aggregate(per_symbol)}
    return out


def _aggregate(per_symbol: dict) -> dict:
    rets = [m.get("total_return_pct", 0.0) for m in per_symbol.values()]
    trades = [m.get("num_trades", 0) for m in per_symbol.values()]
    dds = [m.get("max_drawdown_pct", 0.0) for m in per_symbol.values()]
    wins = [m.get("win_rate_pct", 0.0)
            for m in per_symbol.values() if m.get("num_trades")]
    blocked = [m.get("blocked_entries", 0) for m in per_symbol.values()]
    n = len(rets)
    if n == 0:
        return {
            "num_symbols": 0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "total_trades": 0,
            "mean_win_rate_pct": 0.0,
            "worst_drawdown_pct": 0.0,
            "total_blocked_entries": 0,
        }
    return {
        "num_symbols": len(per_symbol),
        "mean_return_pct": round(sum(rets) / n, 4),
        "median_return_pct": round(sorted(rets)[n // 2], 4),
        "total_trades": sum(trades),
        "mean_win_rate_pct": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "worst_drawdown_pct": round(min(dds), 4) if dds else 0.0,
        "total_blocked_entries": sum(blocked),
    }


def compare(result: dict) -> dict:
    """Delta between the AI-gated arm and the baseline arm."""
    a = result["arms"]["none"]["aggregate"]
    b = result["arms"]["ai"]["aggregate"]
    return {
        "delta_mean_return_pct": round(
            b["mean_return_pct"] - a["mean_return_pct"], 4),
        "delta_total_trades": b["total_trades"] - a["total_trades"],
        "delta_mean_win_rate_pct": round(
            b["mean_win_rate_pct"] - a["mean_win_rate_pct"], 2),
        "delta_worst_drawdown_pct": round(
            b["worst_drawdown_pct"] - a["worst_drawdown_pct"], 4),
    }


def main():
    p = argparse.ArgumentParser(description="A/B: baseline vs AI-gated entries")
    p.add_argument("--symbols", default="AAPL,MSFT")
    p.add_argument("--start", default="2023-01-02")
    p.add_argument("--end", default="2024-08-30")
    p.add_argument("--initial", type=float, default=10000.0)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    res = run_experiment(syms, args.start, args.end, args.initial)
    res["comparison"] = compare(res)

    print("=" * 78)
    print("A/B EXPERIMENT (deterministic causal stand-in, NOT a live AI call)")
    print(f"period {args.start}..{args.end}  symbols={res['symbols']}")
    print("=" * 78)
    for arm in ("none", "ai"):
        a = res["arms"][arm]["aggregate"]
        label = "baseline" if arm == "none" else "ai-gated"
        print(f"{label:>9}: mean_ret={a['mean_return_pct']:+7.2f}%  "
              f"trades={a['total_trades']:3d}  win={a['mean_win_rate_pct']:5.1f}%  "
              f"worstDD={a['worst_drawdown_pct']:6.2f}%  "
              f"blocked={a['total_blocked_entries']}")
    c = res["comparison"]
    print("-" * 78)
    print(f"delta: mean_ret={c['delta_mean_return_pct']:+.2f}pp  "
          f"trades={c['delta_total_trades']:+d}  "
          f"win={c['delta_mean_win_rate_pct']:+.1f}pp  "
          f"worstDD={c['delta_worst_drawdown_pct']:+.2f}pp")
    print("\nNOTE: the stand-in is causal but is NOT the Gemini model. This "
          "measures the gate mechanism, not real AI alpha.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {args.json_out}")
    return res


if __name__ == "__main__":
    main()
