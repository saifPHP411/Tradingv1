"""
Experimental A/B evaluation: baseline strategy vs an AI-conditioned variant.

TWO MODES — read this before trusting any number
    This driver can run the comparison in two different ways, and they do NOT
    have the same evidential weight.

    1. ``run_engine_experiment`` (PREFERRED, mode="engine")
       Runs BOTH arms through the real strategy engine
       (``backtest.engine.simulate_portfolio``), unchanged. The only difference
       between the arms is the entry gate, injected at run time by wrapping
       ``compute_signal_score`` — the engine source is not modified. This keeps
       cooldown, regime gating, position sizing, exit rules, capital
       management and costs identical by construction, so the comparison is
       fair. Its weakness: the engine's own entry rule ALREADY requires
       ``close > ema20 > ema50 and adx > 25`` (``strong_uptrend``), which is
       exactly the condition the stand-in gate tests. A gate built on that
       condition is therefore REDUNDANT and cannot change results; it is a
       plumbing/leak test, not evidence of alpha.

    2. ``run_experiment`` / ``run_arm`` (LEGACY, mode="simplified")
       A simplified re-implementation of entry and exit. It does NOT model the
       engine's cooldown, consecutive-loss guard, regime gate (BEAR blocks
       longs), multi-symbol capital sharing or the ``strong_uptrend`` filter.
       It is a rough approximation and its numbers MUST NOT be described as
       "the strategy". Treat its output as exploratory only.

WHY NOT A LIVE LLM REPLAY
    The engine deliberately refuses to consume LLM output in historical
    backtests, because the model is trained on data that postdates the
    simulated period. A leak-free historical replay of a real LLM overlay is
    therefore not possible with this codebase; it needs a forward-only paper
    evaluation. Both modes here use a deterministic, causal stand-in, so they
    measure the GATE MECHANISM, not Gemini's alpha.

WHAT IS HELD CONSTANT
    Same symbols, same date range, same costs (fees/slippage), same seed, same
    exit rules. In engine mode this is guaranteed by the engine itself.

USAGE
    python -m evaluation.ab_ai_context --mode engine --symbols AAPL,MSFT \\
        --start 2023-01-02 --end 2024-08-30
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

import backtest.engine as engine_mod  # noqa: E402


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


# The engine's own entry rule (backtest/engine.py) already requires
# ``close > ema20 > ema50 and adx > 25``. The stand-in above tests the same
# condition, so it can never change which entries the engine takes. This is
# recorded here so a reader is not misled into reading the engine-mode delta
# as evidence of alpha.
STAND_IN_IS_REDUNDANT_WITH_ENGINE = True


def _exit_fill(price: float, is_buy_exit: bool, bps: float) -> float:
    """Exit fill for the simplified replay, delegating to execution_costs.

    Exposed so a contract test can compare this replay's exit fill against the
    engine's without running a whole backtest. It must never be optimistic.
    """
    from execution_costs import exit_fill_price
    return exit_fill_price(price, is_buy_exit, {
        "fees_per_share": 0.0,
        "entry_slippage_bps": 0.0,
        "exit_slippage_bps": bps,
    })


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
    from execution_costs import (
        entry_fill_price as _entry_fill_impl,
        exit_fill_price as _exit_fill_impl,
        round_trip_fee as _round_trip_fee,
    )

    settings = get_all_settings()
    fees = float(settings.get("fees_per_share", 0.0))
    slip_in = float(settings.get("entry_slippage_bps", 0.0))
    slip_out = float(settings.get("exit_slippage_bps", 0.0))
    # The canonical cost inputs. This replay must charge the SAME round-trip
    # cost as backtest/engine.py, so the fill arithmetic is delegated to
    # execution_costs rather than re-derived here.
    cost_inputs = {"fees_per_share": fees,
                   "entry_slippage_bps": slip_in,
                   "exit_slippage_bps": slip_out}
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

        fill_in = _entry_fill_impl(entry_px, True, cost_inputs)
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

        fill_out = _exit_fill_impl(exit_px, False, cost_inputs)
        gross = (fill_out - fill_in) * qty
        cost = _round_trip_fee(qty, cost_inputs)
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


def run_engine_arm(symbols_data: dict, spy_df, initial: float,
                   gate: str) -> dict:
    """Run ONE arm through the REAL engine, gating entries when gate == "ai".

    The engine source is not modified. The gate is applied by temporarily
    wrapping ``backtest.engine.compute_signal_score`` so that a blocked entry
    scores below any threshold; the wrapper is always restored.

    Because the engine decides everything else (cooldown, regime gate, sizing,
    exits, capital, costs), both arms are identical except for the gate.

    Returns the engine's own result dict plus a ``blocked_entries`` count.
    """
    if gate not in ("none", "ai"):
        raise ValueError(f"gate must be 'none' or 'ai', got {gate!r}")

    original = engine_mod.compute_signal_score
    blocked = {"n": 0}

    if gate == "ai":
        def _gated(ind, regime, **kwargs):
            result = original(ind, regime, **kwargs)
            if ai_gate_decision(ind) == "block":
                blocked["n"] += 1
                result = dict(result)
                # Below any plausible threshold: the engine's own
                # ``score >= min_signal_score`` check then rejects the entry.
                result["score"] = -1.0
            return result
        engine_mod.compute_signal_score = _gated

    try:
        result = engine_mod.simulate_portfolio(
            symbols_data, spy_df, start_equity=initial,
            settings=get_all_settings())
    finally:
        engine_mod.compute_signal_score = original

    result = dict(result)
    result["blocked_entries"] = blocked["n"]
    return result


def _engine_arm_metrics(result: dict) -> dict:
    """Reduce an engine result to the metrics ``compare`` needs."""
    curve = result.get("equity_curve") or []
    trades = result.get("trades") or []
    initial = float(result.get("start_equity", 10000.0))
    metrics = calculate_metrics(curve, trades, initial)
    metrics["blocked_entries"] = result.get("blocked_entries", 0)
    metrics["end_equity"] = result.get("end_equity")
    return metrics


def run_engine_experiment(symbols: list[str], start: str, end: str,
                          initial: float = 10000.0) -> dict:
    """Fair A/B through the real engine. No network, no LLM call.

    Both arms share one ``simulate_portfolio`` run each on identical data,
    settings and costs; the only difference is the entry gate.
    """
    data = load_multi(symbols, start, end)
    spy = get_spy_for_period(start, end)
    out = {"mode": "engine", "start": start, "end": end,
           "symbols": list(data.keys()), "arms": {}}
    for gate in ("none", "ai"):
        result = run_engine_arm(data, spy, initial, gate)
        metrics = _engine_arm_metrics(result)
        out["arms"][gate] = {
            "metrics": metrics,
            "num_trades": len(result.get("trades") or []),
            "end_equity": result.get("end_equity"),
            "blocked_entries": result.get("blocked_entries", 0),
        }
    return out


def compare_engine(result: dict) -> dict:
    """Delta between the AI-gated and baseline arms of an engine run."""
    a = result["arms"]["none"]
    b = result["arms"]["ai"]
    return {
        "delta_total_return_pct": round(
            b["metrics"].get("total_return_pct", 0.0)
            - a["metrics"].get("total_return_pct", 0.0), 4),
        "delta_num_trades": b["num_trades"] - a["num_trades"],
        "delta_max_drawdown_pct": round(
            b["metrics"].get("max_drawdown_pct", 0.0)
            - a["metrics"].get("max_drawdown_pct", 0.0), 4),
        "ai_blocked_entries": b["blocked_entries"],
        "baseline_trades": a["num_trades"],
        "ai_trades": b["num_trades"],
    }


def run_experiment(symbols: list[str], start: str, end: str,
                   initial: float = 10000.0) -> dict:
    """LEGACY simplified arms (approximate; see module docstring)."""
    data = load_multi(symbols, start, end)
    spy = get_spy_for_period(start, end)
    out = {"mode": "simplified", "start": start, "end": end,
           "symbols": list(data.keys()), "arms": {}}
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
    p.add_argument("--mode", choices=("engine", "simplified"), default="engine",
                   help="engine = real strategy engine (fair, preferred); "
                        "simplified = legacy approximate replay")
    p.add_argument("--symbols", default="AAPL,MSFT")
    p.add_argument("--start", default="2023-01-02")
    p.add_argument("--end", default="2024-08-30")
    p.add_argument("--initial", type=float, default=10000.0)
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print("=" * 78)
    print("A/B EXPERIMENT (deterministic causal stand-in, NOT a live AI call)")
    print(f"period {args.start}..{args.end}  symbols={syms}  mode={args.mode}")
    print("=" * 78)

    if args.mode == "engine":
        res = run_engine_experiment(syms, args.start, args.end, args.initial)
        res["comparison"] = compare_engine(res)
        for arm in ("none", "ai"):
            a = res["arms"][arm]
            label = "baseline" if arm == "none" else "ai-gated"
            print(f"{label:>9}: ret={a['metrics'].get('total_return_pct', 0.0):+7.2f}%  "
                  f"trades={a['num_trades']:3d}  end_eq={a['end_equity']}  "
                  f"blocked={a['blocked_entries']}")
        c = res["comparison"]
        print("-" * 78)
        print(f"delta: ret={c['delta_total_return_pct']:+.2f}pp  "
              f"trades={c['delta_num_trades']:+d}  "
              f"maxDD={c['delta_max_drawdown_pct']:+.2f}pp")
        print("\nNOTE: this ran the REAL engine for both arms (cooldown, regime,"
              "\nsizing, exits and costs are identical by construction). The "
              "stand-in gate\nhowever duplicates the engine's own strong_uptrend "
              "filter, so a zero delta\nhere is EXPECTED and is a plumbing check, "
              "not evidence of alpha.")
    else:
        res = run_experiment(syms, args.start, args.end, args.initial)
        res["comparison"] = compare(res)
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
        print("\nWARNING: 'simplified' mode is an APPROXIMATE replay. It does "
              "not model\ncooldown, regime gating, position sizing or the "
              "engine's strong_uptrend\nfilter. Treat these numbers as "
              "exploratory only — not as the strategy.")

    print("\nNOTE: the stand-in is causal but is NOT the Gemini model. This "
          "measures the gate mechanism, not real AI alpha.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(res, indent=2))
        print(f"\nwritten: {args.json_out}")
    return res


if __name__ == "__main__":
    main()
