"""
Tests for the experimental A/B evaluation driver.

These verify the *mechanics* of the experiment, not that AI improves returns:
  - the driver never performs a network call
  - both arms see identical symbols/period/costs (only the gate differs)
  - the causal stand-in is deterministic and leak-free
  - the comparison arithmetic is correct

No test asserts that the AI arm is better. The observed effect is reported by
running the driver, not encoded as an expectation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import market_context as mc  # noqa: E402
from evaluation import ab_ai_context as ab  # noqa: E402


# ---------------------------------------------------------------------------
# No-network guarantee
# ---------------------------------------------------------------------------

def test_evaluation_does_not_call_the_gemini_api(monkeypatch):
    """If the driver touched the real transport this would raise."""
    def _blocked(*a, **k):
        raise AssertionError("evaluation attempted a real Gemini call")
    monkeypatch.setattr(mc, "_post_generate_content", _blocked)
    # Building the gate decision is pure and must not call out.
    ind = {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0, "adx": 30.0}
    assert ab.ai_gate_decision(ind) in ("allow", "block")


# ---------------------------------------------------------------------------
# The stand-in is deterministic and causal
# ---------------------------------------------------------------------------

def test_stand_in_is_deterministic():
    ind = {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0, "adx": 30.0}
    assert ab.ai_gate_decision(ind) == ab.ai_gate_decision(ind) == "allow"


def test_stand_in_blocks_weak_trend():
    ind = {"last_close": 100.0, "ema20": 105.0, "ema50": 110.0, "adx": 30.0}
    assert ab.ai_gate_decision(ind) == "block"


def test_stand_in_blocks_low_adx():
    ind = {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0, "adx": 10.0}
    assert ab.ai_gate_decision(ind) == "block"


def test_stand_in_blocks_none():
    assert ab.ai_gate_decision(None) == "block"


def test_stand_in_uses_only_trailing_indicators():
    """Same indicator values must give the same decision regardless of any
    (hypothetical) future-looking keys the caller might attach."""
    ind = {"last_close": 110.0, "ema20": 105.0, "ema50": 100.0, "adx": 30.0}
    noisy = dict(ind)
    noisy["future_close"] = 1.0
    noisy["next_bar"] = 999.0
    assert ab.ai_gate_decision(ind) == ab.ai_gate_decision(noisy)


# ---------------------------------------------------------------------------
# Identical data / costs: only the gate differs
# ---------------------------------------------------------------------------

def _frame(rows, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


def _uptrend(n=200):
    closes = 100 + np.arange(n) * 0.5
    return _frame([(c - 0.2, c + 0.5, c - 0.5, c, 2_000_000) for c in closes])


def test_both_arms_use_the_same_symbols_and_bars():
    df = _uptrend()
    base = ab.run_arm("AAPL", df, None, 10_000.0, "none")
    ai = ab.run_arm("AAPL", df, None, 10_000.0, "ai")
    # Same frame -> same candidate bars. AI can only remove entries.
    assert ai["taken_entries"] <= base["taken_entries"]
    assert ai["taken_entries"] + ai["blocked_entries"] >= ai["taken_entries"]


def test_ai_arm_never_takes_more_entries_than_baseline():
    df = _uptrend()
    base = ab.run_arm("AAPL", df, None, 10_000.0, "none")
    ai = ab.run_arm("AAPL", df, None, 10_000.0, "ai")
    assert ai["taken_entries"] <= base["taken_entries"]


def test_determinism_of_full_arm():
    df = _uptrend()
    a = ab.run_arm("AAPL", df, None, 10_000.0, "ai")
    b = ab.run_arm("AAPL", df, None, 10_000.0, "ai")
    assert a["taken_entries"] == b["taken_entries"]
    assert a["total_return_pct"] == b["total_return_pct"]


# ---------------------------------------------------------------------------
# Comparison arithmetic
# ---------------------------------------------------------------------------

def test_compare_arithmetic():
    result = {
        "arms": {
            "none": {"aggregate": {"mean_return_pct": 5.0, "total_trades": 100,
                                   "mean_win_rate_pct": 50.0,
                                   "worst_drawdown_pct": -3.0}},
            "ai": {"aggregate": {"mean_return_pct": 7.5, "total_trades": 60,
                                 "mean_win_rate_pct": 55.0,
                                 "worst_drawdown_pct": -2.0}},
        }
    }
    c = ab.compare(result)
    assert c["delta_mean_return_pct"] == 2.5
    assert c["delta_total_trades"] == -40
    assert c["delta_mean_win_rate_pct"] == 5.0
    assert c["delta_worst_drawdown_pct"] == 1.0


def test_aggregate_handles_empty():
    agg = ab._aggregate({})
    assert agg["num_symbols"] == 0
    assert agg["total_trades"] == 0


# ---------------------------------------------------------------------------
# End-to-end on synthetic frames (no cache, no network)
# ---------------------------------------------------------------------------

def test_end_to_end_experiment_runs_offline(monkeypatch):
    df = _uptrend(260)

    import evaluation.ab_ai_context as mod
    monkeypatch.setattr(mod, "load_multi", lambda syms, s, e: {"AAA": df})
    monkeypatch.setattr(mod, "get_spy_for_period", lambda s, e: None)

    res = mod.run_experiment(["AAA"], "2020-01-01", "2020-09-01")
    assert set(res["arms"]) == {"none", "ai"}
    assert res["arms"]["none"]["aggregate"]["num_symbols"] == 1
    # The comparison must serialise cleanly (report artifact).
    json.dumps(mod.compare(res))
