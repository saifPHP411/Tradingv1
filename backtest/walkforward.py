"""
Walk-forward evaluation.

Naming note: this module performs ROLLING OUT-OF-SAMPLE EVALUATION.
It does NOT perform parameter optimization or in-sample training.
Train windows are advanced over time, but no fitting occurs. Do not
call this an "optimization" pipeline in reports or UI.

Warmup handling:
    Each OOS window loads data from `test_start - WARMUP_DAYS` to `test_end`.
    Indicators use the full range, but no signals are queued before
    `test_start` because `simulate_portfolio` is called with
    `trading_start_ts=test_start`. Reported metrics and trades are filtered
    to `>= test_start`, so warmup trades do not leak into OOS results.

Aggregate:
    cumulative_return_pct compounds per-window returns:
        cum = Π(1 + r_i / 100); cumulative = (cum - 1) * 100
    It is NOT the arithmetic sum.
"""
from collections.abc import Collection, Mapping
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import pandas as pd

from backtest.data import load_ohlcv, load_multi, get_spy_for_period
from backtest.engine import simulate, simulate_portfolio
from backtest.metrics import calculate_metrics
from settings import get_all_settings


WARMUP_DAYS = 200


def _naive_ts(x):
    """Normalize any timestamp-like value to a naive UTC pandas Timestamp.

    Data is normalized to naive UTC by backtest.data; walk-forward window
    boundaries are built from naive datetimes. Normalizing here keeps both
    sides comparable even if a caller passes a tz-aware boundary.
    """
    t = pd.Timestamp(x)
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _filter_to_window(equity_curve: list, trades: list, test_start_ts,
                      test_end_ts=None):
    """Keep only the test-window portion of the result.

    Interval convention: OOS windows are **half-open** ``[start, end)`` so a
    timestamp on the shared boundary between two adjacent windows belongs to
    exactly one window and is never double counted. The walk-forward driver
    passes ``test_end_ts=None`` for the final window, which means "no upper
    bound": there is no subsequent window to own that boundary, so the final
    data point must not be discarded.

    Comparisons are timezone-safe (both sides normalized to naive UTC).
    """
    start = _naive_ts(test_start_ts)
    end = _naive_ts(test_end_ts) if test_end_ts is not None else None

    def _in_window(t) -> bool:
        t = _naive_ts(t)
        if t < start:
            return False
        if end is not None and t >= end:
            return False
        return True

    curve = [p for p in equity_curve if _in_window(p["time"])]
    trade_list = [t for t in trades if _in_window(t["entry_time"])]
    return curve, trade_list


def _filter_rejections_to_window(rejections, test_start_ts, test_end_ts=None):
    """Keep only the rejection events recorded inside the OOS window.

    Same half-open ``[start, end)`` convention as ``_filter_to_window`` (the
    final window is open-ended). Rejections carry a ``time`` field. A malformed
    entry without a usable ``time`` is dropped rather than crashing the driver;
    rejections never affect metrics, trade counts, or returns.
    """
    if not rejections:
        return []
    start = _naive_ts(test_start_ts)
    end = _naive_ts(test_end_ts) if test_end_ts is not None else None
    kept = []
    for r in rejections:
        if not isinstance(r, Mapping):
            continue
        t = r.get("time")
        if t is None:
            continue
        t = _naive_ts(t)
        if t < start:
            continue
        if end is not None and t >= end:
            continue
        kept.append(r)
    return kept


def walk_forward(symbol: str, start_date: str, end_date: str,
                 train_months: int = 12, test_months: int = 3,
                 initial_equity: float = 10000.0,
                 max_hold_bars: int | None = None,
                 settings: dict | None = None) -> list[dict]:
    """Single-symbol rolling OOS evaluation (legacy, kept for compatibility)."""
    results = []
    cur = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while True:
        test_start = cur + relativedelta(months=train_months)
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end:
            break
        print(f"\n Window: {test_start.date()} → {test_end.date()}")

        # Load warmup + test range so indicators have >= 60 bars at test_start.
        load_start = (test_start - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        df = load_ohlcv(symbol, load_start, test_end.strftime("%Y-%m-%d"))
        if df is None or len(df) < 70:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue

        spy_df = get_spy_for_period(load_start, test_end.strftime("%Y-%m-%d"))
        test_start_ts = pd.Timestamp(test_start)

        # Adjacent OOS windows share a boundary date. Use half-open intervals
        # [test_start, test_end) so the shared boundary belongs to the NEXT
        # window only. The final window has no successor, so it is open-ended
        # and keeps the last data point.
        is_last_window = (test_end + relativedelta(months=test_months)) > end
        test_end_ts = None if is_last_window else pd.Timestamp(test_end)

        full_result = simulate(
            symbol, df, spy_df, start_equity=initial_equity,
            trading_start_ts=test_start_ts,
            max_hold_bars=max_hold_bars, settings=settings,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"],
            test_start_ts, test_end_ts,
        )
        rejections = _filter_rejections_to_window(
            full_result.get("rejections"), test_start_ts, test_end_ts,
        )
        metrics = calculate_metrics(curve, trades, initial_equity)

        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": {"equity_curve": curve, "trades": trades,
                       "rejections": rejections,
                       "num_rejections": len(rejections)},
            "metrics": metrics,
        })
        print(f" Return: {metrics.get('total_return_pct', 0):+.2f}% | "
              f"DD: {metrics.get('max_drawdown_pct', 0):.2f}% | "
              f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
              f"Trades: {metrics.get('num_trades', 0)}")
        cur += relativedelta(months=test_months)
    return results


def walk_forward_portfolio(symbols: list[str], start_date: str, end_date: str,
                           train_months: int = 12,
                           test_months: int = 3,
                           initial_equity: float = 10000.0,
                           max_hold_bars: int | None = None,
                           settings: dict | None = None) -> list[dict]:
    """
    Portfolio rolling OOS evaluation.
    Each window shares cash/equity/exposure via simulate_portfolio.
    Warmup bars are included, but no trades are opened before test_start.
    """
    results = []
    cur = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    while True:
        test_start = cur + relativedelta(months=train_months)
        test_end = test_start + relativedelta(months=test_months)
        if test_end > end:
            break
        print(f"\n Window: {test_start.date()} → {test_end.date()}")

        load_start = (test_start - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
        data = load_multi(symbols, load_start, test_end.strftime("%Y-%m-%d"))
        if not data:
            print(" ⚠️ insufficient data")
            cur += relativedelta(months=test_months)
            continue

        spy = get_spy_for_period(load_start, test_end.strftime("%Y-%m-%d"))
        test_start_ts = pd.Timestamp(test_start)

        # Half-open [test_start, test_end) windows; the final window is
        # open-ended so the last data point is retained (see _filter_to_window).
        is_last_window = (test_end + relativedelta(months=test_months)) > end
        test_end_ts = None if is_last_window else pd.Timestamp(test_end)

        full_result = simulate_portfolio(
            data, spy, start_equity=initial_equity,
            trading_start_ts=test_start_ts,
            max_hold_bars=max_hold_bars, settings=settings,
        )
        curve, trades = _filter_to_window(
            full_result["equity_curve"], full_result["trades"],
            test_start_ts, test_end_ts,
        )
        rejections = _filter_rejections_to_window(
            full_result.get("rejections"), test_start_ts, test_end_ts,
        )
        metrics = calculate_metrics(curve, trades, initial_equity)

        results.append({
            "window": f"{test_start.date()}_to_{test_end.date()}",
            "result": {"equity_curve": curve, "trades": trades,
                       "rejections": rejections,
                       "num_rejections": len(rejections)},
            "metrics": metrics,
        })
        print(f" Return: {metrics.get('total_return_pct', 0):+.2f}% | "
              f"DD: {metrics.get('max_drawdown_pct', 0):.2f}% | "
              f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
              f"Trades: {metrics.get('num_trades', 0)}")
        cur += relativedelta(months=test_months)
    return results


def aggregate_walkforward(results: list[dict]) -> dict:
    """
    Compound per-window returns into a single cumulative figure.

    Validity rule (single, explicit): a window is **valid** only if its
    ``metrics`` is a mapping that contains a numeric, finite
    ``total_return_pct``. Everything else is an **invalid** window and
    contributes NOTHING to the aggregate — neither to the compounded return,
    nor to the trade count, nor to the averages.

    Invalid cases:
      - a result entry that is not a mapping (e.g. ``None``), or whose
        ``metrics`` is missing / not a mapping (e.g. ``{"error": "no data"}``
        has no ``total_return_pct`` key). Malformed entries must not crash
        aggregation.
      - ``total_return_pct`` absent, non-numeric, NaN, +inf or -inf.
      - ``total_return_pct < -100%``: impossible for a cash-secured account.
        It is recorded as invalid. Its clamped value (-100%) is used **only**
        for the compounded-return product, so ``(1 + r)`` can never go negative
        and flip the product's sign; it does not enter the valid-window
        statistics (average, std, sharpe, positive/negative counts).
      - ``total_return_pct == -100%`` is valid and means a ruined account.

    A non-numeric ``num_trades`` on an invalid window is treated as 0 rather
    than crashing the ``total_trades_invalid_windows`` tally. A non-finite
    (``inf``/``-inf``) or otherwise unconvertible count is treated the same way:
    ``int(float("inf"))`` raises ``OverflowError``, so it is caught alongside
    ``TypeError``/``ValueError``. An unbounded integer ``total_return_pct`` (whose
    ``float()`` also raises ``OverflowError``) is treated as non-finite, i.e. an
    invalid window. Invalid windows are reported in ``invalid_windows`` (never
    dropped silently), and
    ``num_invalid_windows`` / ``total_trades_invalid_windows`` are surfaced so
    the exclusion is visible rather than implied.

    The invalid-window policy above applies to the *entries* of ``results``.
    The container itself is the caller's contract: ``results`` must be a sized,
    iterable collection of window entries (a list, as every caller passes). A
    falsy value (``[]``, ``None``, ``{}``, ``()``) yields the documented
    ``{"error": "no windows"}``; a truthy value that is not a collection — a
    non-iterable such as ``5``, or a one-shot iterator such as a generator that
    cannot be measured by ``len`` — is a caller error and is rejected with a
    clear ``ValueError`` instead of the incidental ``TypeError`` that
    ``enumerate`` / ``len`` used to raise.
    """
    if not results:
        return {"error": "no windows"}
    if not isinstance(results, Collection):
        raise ValueError(
            "aggregate_walkforward expects a sized iterable collection of "
            f"window result dicts, got {type(results).__name__}")
    import numpy as np

    def _window_label(r, idx):
        return r.get("window", idx) if isinstance(r, Mapping) else idx

    def _trades_of(m) -> int:
        """Trades attributed to a window; a junk count is treated as 0.

        ``int(float("inf"))`` raises ``OverflowError`` (not ``ValueError``), and
        an unbounded Python int raises ``OverflowError`` too, so both must be
        caught here or a malformed count still crashes aggregation.
        """
        if not isinstance(m, Mapping):
            return 0
        try:
            return int(m.get("num_trades", 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    total_trades = 0
    total_trades_invalid = 0
    all_returns: list[float] = []
    all_sharpes: list[float] = []
    invalid_windows: list[dict] = []
    compounded = 1.0
    for idx, r in enumerate(results):
        m = r.get("metrics", {}) if isinstance(r, Mapping) else None
        if not isinstance(m, Mapping):
            invalid_windows.append({
                "window": _window_label(r, idx),
                "reason": "malformed result entry (missing or non-mapping metrics)",
                "value": m,
            })
            continue
        n_trades = _trades_of(m)

        # A window is valid only if it carries a numeric finite return.
        raw_ret = m.get("total_return_pct", None)
        try:
            ret = float(raw_ret)
        except (TypeError, ValueError, OverflowError):
            # OverflowError: an unbounded Python int return is not a usable
            # number; treat it as a non-finite (invalid) window like NaN/inf.
            ret = float("nan")

        if not math.isfinite(ret):
            invalid_windows.append({"window": _window_label(r, idx),
                                    "reason": "missing or non-finite total_return_pct",
                                    "value": raw_ret})
            total_trades_invalid += n_trades
            continue

        if ret < -100.0:
            # A cash-secured account cannot lose more than 100%. Clamp for
            # compounding rather than let (1 + r) go negative and flip the
            # product's sign. The window is reported invalid.
            invalid_windows.append({"window": _window_label(r, idx),
                                    "reason": "return below -100%; clamped",
                                    "value": ret})
            ret = -100.0
            total_trades_invalid += n_trades
            stat_eligible = False
        else:
            # Valid window: its trades count toward the aggregate.
            total_trades += n_trades
            stat_eligible = True

        # Statistics describe VALID windows only. An invalid clamped window
        # still participates in compounding below, but must not enter the
        # valid-window population (avg/std/sharpe, positive/negative counts).
        if stat_eligible:
            all_returns.append(ret)
            sharpe = m.get("sharpe", 0)
            try:
                sharpe = float(sharpe)
            except (TypeError, ValueError, OverflowError):
                sharpe = 0.0
            all_sharpes.append(sharpe if math.isfinite(sharpe) else 0.0)
        compounded *= (1.0 + ret / 100.0)

    cumulative_return_pct = (compounded - 1.0) * 100.0

    out = {
        "num_windows": len(results),
        "num_valid_windows": len(results) - len(invalid_windows),
        "num_invalid_windows": len(invalid_windows),
        "total_trades": total_trades,
        "total_trades_invalid_windows": total_trades_invalid,
        "cumulative_return_pct": round(cumulative_return_pct, 2),
        "avg_return_per_window": (round(float(np.mean(all_returns)), 2)
                                  if all_returns else 0.0),
        "std_return_per_window": (round(float(np.std(all_returns)), 2)
                                  if all_returns else 0.0),
        "avg_sharpe": (round(float(np.mean(all_sharpes)), 2)
                       if all_sharpes else 0.0),
        "positive_windows": sum(1 for r in all_returns if r > 0),
        "negative_windows": sum(1 for r in all_returns if r < 0),
    }
    if invalid_windows:
        out["invalid_windows"] = invalid_windows
    return out
