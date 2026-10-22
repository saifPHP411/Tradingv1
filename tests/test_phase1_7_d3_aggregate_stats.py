"""
Phase-1.7 D3 / NEW-1 regression tests — invalid-window aggregate statistics.

``aggregate_walkforward`` documents that an invalid window contributes NOTHING
to the aggregate: not to the compounded return, not to the trade count, not to
the averages. The ``total_return_pct < -100%`` branch clamped the return for
compounding (correct and intentional) but also appended the clamped value to
the *valid-window* statistical population, so a clamped invalid window
contaminated ``avg_return_per_window`` / ``std_return_per_window`` /
``avg_sharpe`` / ``positive_windows`` / ``negative_windows`` while every other
invalid window did not.

These tests pin the corrected contract: an invalid window never enters the
valid-window statistics, regardless of *which* kind of invalidity it carries,
while the compounding clamp is preserved.

Deterministic. No network, no credentials.
"""
import math

import pytest

from backtest.walkforward import aggregate_walkforward


def _win(label, ret, trades=0, sharpe=0.0):
    return {"window": label,
            "metrics": {"total_return_pct": ret, "num_trades": trades,
                        "sharpe": sharpe}}


def _stat_population(agg):
    """The fields that describe the VALID-window population."""
    return {k: agg[k] for k in ("avg_return_per_window", "std_return_per_window",
                                "avg_sharpe", "positive_windows",
                                "negative_windows")}


def test_clamped_invalid_window_excluded_from_valid_statistics():
    """valid +10% alongside a clamped -150% invalid window."""
    agg = aggregate_walkforward([_win("A", 10.0, 3, 1.0),
                                 _win("B", -150.0, 5, 2.0)])

    assert agg["num_valid_windows"] == 1
    assert agg["num_invalid_windows"] == 1
    # Averages reflect ONLY the valid window.
    assert agg["avg_return_per_window"] == pytest.approx(10.0, abs=0.01)
    assert agg["std_return_per_window"] == pytest.approx(0.0, abs=0.01)
    assert agg["avg_sharpe"] == pytest.approx(1.0, abs=0.01)
    # The clamped window is not a "negative window".
    assert agg["positive_windows"] == 1
    assert agg["negative_windows"] == 0
    # Compounding clamp preserved: (0.0)*(1.10) - 1 = -100%.
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)
    # Trade accounting unchanged.
    assert agg["total_trades"] == 3
    assert agg["total_trades_invalid_windows"] == 5
    assert agg["invalid_windows"][0]["reason"].startswith("return below -100%")


def test_clamped_and_missing_return_invalids_have_equivalent_statistics():
    """The two invalid kinds must not differ merely because one was clamped."""
    clamped = aggregate_walkforward([_win("A", 10.0, 3, 1.0),
                                     _win("B", -150.0, 5, 2.0)])
    missing = aggregate_walkforward([
        _win("A", 10.0, 3, 1.0),
        {"window": "B", "metrics": {"num_trades": 5, "sharpe": 2.0}},
    ])

    assert _stat_population(clamped) == _stat_population(missing)
    assert clamped["num_valid_windows"] == missing["num_valid_windows"] == 1
    assert clamped["num_invalid_windows"] == missing["num_invalid_windows"] == 1
    # Compounding still differs by design: the clamped window compounds at
    # -100% while the missing-return window contributes nothing.
    assert clamped["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert missing["cumulative_return_pct"] == pytest.approx(10.0, abs=0.01)


def test_clamped_window_sharpe_does_not_dilute_avg_sharpe():
    """A clamped window's Sharpe must not enter the valid-window population."""
    agg = aggregate_walkforward([_win("A", 5.0, 1, 0.5),
                                 _win("B", -200.0, 1, 9.9),
                                 _win("C", -3.0, 1, 0.1)])
    assert agg["avg_sharpe"] == pytest.approx((0.5 + 0.1) / 2, abs=0.01)
    assert agg["avg_return_per_window"] == pytest.approx((5.0 - 3.0) / 2, abs=0.01)
    assert agg["positive_windows"] == 1
    assert agg["negative_windows"] == 1


def test_multiple_valid_windows_average_over_valid_only():
    agg = aggregate_walkforward([_win("A", 10.0, 1),
                                 _win("B", -150.0, 1),
                                 _win("C", 20.0, 1),
                                 _win("D", -30.0, 1)])
    assert agg["num_valid_windows"] == 3
    assert agg["num_invalid_windows"] == 1
    assert agg["avg_return_per_window"] == pytest.approx((10 + 20 - 30) / 3,
                                                         abs=0.01)
    assert agg["positive_windows"] == 2
    assert agg["negative_windows"] == 1
    assert agg["total_trades"] == 3
    assert agg["total_trades_invalid_windows"] == 1


def test_clamped_window_only_is_safe():
    """A single clamped window: no valid population, compounding still safe."""
    agg = aggregate_walkforward([_win("B", -150.0, 4)])
    assert agg["num_valid_windows"] == 0
    assert agg["num_invalid_windows"] == 1
    assert agg["avg_return_per_window"] == 0.0
    assert agg["std_return_per_window"] == 0.0
    assert agg["avg_sharpe"] == 0.0
    assert agg["positive_windows"] == 0
    assert agg["negative_windows"] == 0
    assert agg["cumulative_return_pct"] == pytest.approx(-100.0, abs=0.01)
    assert math.isfinite(agg["cumulative_return_pct"])


def test_valid_only_aggregation_is_unchanged():
    """Regression guard: valid-only inputs must behave exactly as before."""
    agg = aggregate_walkforward([_win("a", 10.0, 1, 1.0),
                                 _win("b", -5.0, 1, 0.5),
                                 _win("c", 20.0, 1, 2.0)])
    manual = (1.10 * 0.95 * 1.20 - 1) * 100
    assert agg["cumulative_return_pct"] == pytest.approx(manual, abs=0.01)
    assert agg["num_valid_windows"] == 3
    assert agg["num_invalid_windows"] == 0
    assert "invalid_windows" not in agg
    assert agg["avg_return_per_window"] == pytest.approx((10 - 5 + 20) / 3,
                                                         abs=0.01)
    assert agg["positive_windows"] == 2
    assert agg["negative_windows"] == 1
    assert agg["total_trades"] == 3
