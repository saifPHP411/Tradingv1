"""
Phase-1.7 D2 regression tests — backtest rejection observability.

The engine already records invalid stop/target rejection EVENTS in
``simulate_portfolio`` (``invalid_stop_or_target`` /
``invalid_stop_or_target_after_rounding``). D2 makes those existing events
visible in the report JSON, the console summary, the run DB row, and each
walk-forward window. It is an observability-only change: no signal, threshold,
sizing, accounting, or execution behaviour is altered.

Deterministic synthetic data only. No network, no credentials, no broker.
"""
import json
import sqlite3

import pandas as pd

from backtest.engine import simulate_portfolio, compute_indicators_at
from backtest.report import save_report, print_summary, print_portfolio_summary
from backtest.walkforward import (
    walk_forward, _filter_rejections_to_window,
)
from risk_utils import normalize_price_for_symbol


def _settings(**overrides):
    s = {
        "min_confidence": 0.7,
        "max_risk_per_trade": 0.02,
        "max_open_positions": 1,
        "max_trades_per_day": 50,
        "max_daily_loss_pct": 0.5,
        "max_position_pct": 1.0,
        "max_portfolio_exposure_pct": 0.99,
        "max_consecutive_losses": 999,
        "cooldown_minutes": 0,
        "stop_atr_multiplier": 2.5,
        "take_profit_rr": 2.0,
        "trailing_activation_r": 1.5,
        "trailing_distance_r": 1.0,
        "min_stop_move_pct": 0.3,
        "min_signal_score": 30,
        "min_hold_minutes": 0,
        "min_loss_to_close_pct": 99.0,
        "llm_exit_min_confidence": 0.7,
        "exit_check_seconds": 60,
        "exit_llm_interval_seconds": 300,
        "allow_extended_hours": False,
        "cycle_interval_minutes": 15,
        "fees_per_share": 0.0,
        "entry_slippage_bps": 0,
        "exit_slippage_bps": 0,
        "trading_capital": 0,
        "regime_enabled": 0,
        "news_enabled": 0,
        "symbols": [],
        "settings_version": 3,
        "dry_run": True,
        "trading_mode": "BACKTEST",
        "enable_short": 0,
        "max_hold_bars": 5,
    }
    s.update(overrides)
    return s


def _frame(rows, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


def _high_atr_uptrend(n_signal=61, step=2.0):
    """Zigzag uptrend with inflated ATR, so a LONG signal fires at idx 60."""
    closes = []
    c = 100.0
    for i in range(n_signal):
        c += step + (step if i % 2 else -step)
        closes.append(c)
    return closes


def _long_gap_df(gap_open, trailing=0, start="2024-01-01"):
    """Uptrend signalling LONG at idx 60; the entry bar (idx 61) opens at
    ``gap_open``. Optional ``trailing`` extra bars at the same tiny price."""
    rows = [[x, x + 3.0, x - 3.0, x, 1_000_000] for x in _high_atr_uptrend()]
    rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open, 1_000_000])
    for _ in range(trailing):
        rows.append([gap_open, gap_open * 1.05, gap_open * 0.95, gap_open,
                     1_000_000])
    return _frame(rows, start=start)


def _clean_df(n=120):
    """Healthy trending frame that trades normally and rejects nothing."""
    return _frame([[100 + i * 0.5, (100 + i * 0.5) * 1.001,
                    (100 + i * 0.5) * 0.999, 100 + i * 0.5, 1_000_000]
                   for i in range(n)])


def _post_rounding_gap_open():
    """Gap open whose pre-rounding stop is a tiny *positive* value that
    tick-rounds to zero, hitting the post-rounding branch."""
    settings = _settings(enable_short=0)
    ind = compute_indicators_at(_long_gap_df(gap_open=100.0), 60)
    stop_dist = ind["atr14"] * settings["stop_atr_multiplier"]
    return stop_dist + 0.001, stop_dist


# ==================== Engine still records the events (baseline) ====================

def test_engine_records_pre_rounding_rejection_event():
    result = simulate_portfolio({"AAA": _long_gap_df(gap_open=1.0)}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)
    assert result["trades"] == []
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["reason"] == "invalid_stop_or_target"


def test_engine_records_post_rounding_rejection_event():
    gap_open, stop_dist = _post_rounding_gap_open()
    pre_round_stop = gap_open - stop_dist
    assert 0.0 < pre_round_stop < 0.005
    assert normalize_price_for_symbol("AAA", pre_round_stop) == 0.0

    result = simulate_portfolio({"AAA": _long_gap_df(gap_open)}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)
    assert result["trades"] == []
    assert len(result["rejections"]) == 1
    assert result["rejections"][0]["reason"] == \
        "invalid_stop_or_target_after_rounding"


def test_both_rejection_branches_remain_distinguishable():
    """The two reasons must stay distinct, not collapsed into one string."""
    pre = simulate_portfolio({"AAA": _long_gap_df(gap_open=1.0)}, None,
                             start_equity=100_000,
                             settings=_settings(enable_short=0), max_hold_bars=5)
    gap_open, _ = _post_rounding_gap_open()
    post = simulate_portfolio(
        {"AAA": _long_gap_df(gap_open)}, None,
        start_equity=100_000, settings=_settings(enable_short=0), max_hold_bars=5)
    r_pre = {r["reason"] for r in pre["rejections"]}
    r_post = {r["reason"] for r in post["rejections"]}
    assert r_pre == {"invalid_stop_or_target"}
    assert r_post == {"invalid_stop_or_target_after_rounding"}
    assert r_pre.isdisjoint(r_post)


def test_no_double_counting_one_bad_candidate_is_one_event():
    """One deliberately invalid candidate must yield exactly one event, not one
    per validation stage (the branches are mutually exclusive)."""
    result = simulate_portfolio({"AAA": _long_gap_df(gap_open=1.0)}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)
    assert len(result["rejections"]) == 1
    keys = [(r["symbol"], str(r["time"])) for r in result["rejections"]]
    assert len(keys) == len(set(keys))


def test_rejected_entry_is_not_an_executed_trade():
    """A rejected (symbol, time) must never appear as an executed entry."""
    good = _clean_df()
    bad = _long_gap_df(gap_open=1.0)
    result = simulate_portfolio({"GOOD": good, "BAD": bad}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0,
                                                   max_open_positions=2),
                                max_hold_bars=5)
    assert result["rejections"]
    assert all(r["symbol"] == "BAD" for r in result["rejections"])
    executed = {(t["symbol"], str(t["entry_time"])) for t in result["trades"]}
    rejected = {(r["symbol"], str(r["time"])) for r in result["rejections"]}
    assert executed.isdisjoint(rejected)
    assert "GOOD" in {t["symbol"] for t in result["trades"]}


# ==================== A) Report JSON ====================

def test_save_report_persists_rejections_and_count(tmp_path, monkeypatch):
    import backtest.report as report_mod
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)

    result = simulate_portfolio({"AAA": _long_gap_df(gap_open=1.0)}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0),
                                max_hold_bars=5)
    result["symbol"] = "AAA"
    path = save_report(result, {"num_trades": 0}, "rej")
    payload = json.loads(path.read_text())

    assert payload["num_rejections"] == 1
    # Record structure preserved verbatim (JSON stringifies the Timestamp).
    rec = payload["rejections"][0]
    assert set(rec) == {"symbol", "time", "reason", "fill_price",
                        "stop_price", "target_price"}
    assert rec["reason"] == "invalid_stop_or_target"
    assert rec["symbol"] == result["rejections"][0]["symbol"]
    assert rec["stop_price"] == result["rejections"][0]["stop_price"]
    assert rec["time"] == str(result["rejections"][0]["time"])


def test_save_report_clean_run_is_empty(tmp_path, monkeypatch):
    import backtest.report as report_mod
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)

    result = simulate_portfolio({"AAA": _clean_df()}, None, start_equity=10_000,
                                settings=_settings(enable_short=0), max_hold_bars=5)
    result["symbol"] = "AAA"
    payload = json.loads(save_report(result, {"num_trades": 0}, "clean").read_text())
    assert payload["rejections"] == []
    assert payload["num_rejections"] == 0


def test_save_report_keeps_existing_fields(tmp_path, monkeypatch):
    """Backward compatibility: the original payload keys are unchanged."""
    import backtest.report as report_mod
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)

    result = {"symbol": "AAA", "start_equity": 1.0, "end_equity": 2.0,
              "trades": [{"x": 1}], "equity_curve": [{"time": "t"}]}
    payload = json.loads(save_report(result, {"num_trades": 1}, "bc").read_text())
    for key in ("generated_at", "symbol", "start_equity", "end_equity",
                "metrics", "trades", "equity_curve"):
        assert key in payload
    assert payload["trades"] == [{"x": 1}]
    assert payload["equity_curve"] == [{"time": "t"}]
    assert payload["num_rejections"] == 0


def test_save_report_tolerates_missing_rejections_key(tmp_path, monkeypatch):
    """A legacy result dict without 'rejections' must not raise."""
    import backtest.report as report_mod
    monkeypatch.setattr(report_mod, "REPORT_DIR", tmp_path)
    payload = json.loads(save_report({"symbol": "X"}, {}, "legacy").read_text())
    assert payload["rejections"] == []
    assert payload["num_rejections"] == 0


# ==================== B) Console summary ====================

def test_print_summary_shows_rejection_warning_when_nonzero(capsys):
    result = {"symbol": "AAA", "start_equity": 1.0, "end_equity": 1.0,
              "rejections": [{"symbol": "AAA", "time": "t",
                              "reason": "invalid_stop_or_target"}]}
    print_summary(result, {"num_trades": 0})
    out = capsys.readouterr().out
    assert "Rejected entries: 1" in out
    assert "invalid stop/target" in out


def test_print_summary_is_quiet_when_zero(capsys):
    result = {"symbol": "AAA", "start_equity": 1.0, "end_equity": 1.0,
              "rejections": []}
    print_summary(result, {"num_trades": 0})
    out = capsys.readouterr().out
    assert "Rejected entries" not in out


def test_print_summary_quiet_when_key_absent(capsys):
    print_summary({"symbol": "AAA", "start_equity": 1.0, "end_equity": 1.0},
                  {"num_trades": 0})
    assert "Rejected entries" not in capsys.readouterr().out


def test_print_portfolio_summary_shows_and_hides(capsys):
    with_rej = {"start_equity": 1.0, "end_equity": 1.0, "trades": [],
                "rejections": [{"symbol": "AAA", "time": "t",
                                "reason": "invalid_stop_or_target"}]}
    print_portfolio_summary(["AAA"], with_rej, {"num_trades": 0})
    assert "Rejected entries: 1" in capsys.readouterr().out

    clean = {"start_equity": 1.0, "end_equity": 1.0, "trades": [],
             "rejections": []}
    print_portfolio_summary(["AAA"], clean, {"num_trades": 0})
    assert "Rejected entries" not in capsys.readouterr().out


# ==================== C) Database (no schema change) ====================

def _patch_db(monkeypatch, tmp_path):
    import backtest.storage as storage
    monkeypatch.setattr(storage, "DB", tmp_path / "runs.db")
    return storage


def test_save_run_adds_num_rejections_without_schema_change(tmp_path, monkeypatch):
    storage = _patch_db(monkeypatch, tmp_path)
    storage.save_run("AAA", "2024-01-01", "2024-06-01",
                     {"foo": 1, "bar": "x"}, {"num_trades": 0},
                     num_rejections=3)

    # Schema is untouched: still the original 9 columns.
    cols = [c[1] for c in sqlite3.connect(storage.DB)
            .execute("PRAGMA table_info(runs)")]
    assert cols == ["id", "timestamp", "symbol", "run_type", "start_date",
                    "end_date", "params", "metrics", "num_trades"]

    row = storage.list_runs(1)[0]
    assert row["params"]["num_rejections"] == 3
    # Existing params preserved.
    assert row["params"]["foo"] == 1
    assert row["params"]["bar"] == "x"


def test_save_run_does_not_mutate_caller_params(tmp_path, monkeypatch):
    storage = _patch_db(monkeypatch, tmp_path)
    original = {"foo": 1}
    storage.save_run("AAA", "a", "b", original, {"num_trades": 0},
                     num_rejections=2)
    assert original == {"foo": 1}, "caller's params dict must not be mutated"


def test_save_run_clean_run_records_zero(tmp_path, monkeypatch):
    storage = _patch_db(monkeypatch, tmp_path)
    storage.save_run("AAA", "a", "b", {}, {"num_trades": 0})
    assert storage.list_runs(1)[0]["params"]["num_rejections"] == 0


def test_save_run_rejected_run_records_count(tmp_path, monkeypatch):
    storage = _patch_db(monkeypatch, tmp_path)
    result = simulate_portfolio({"AAA": _long_gap_df(gap_open=1.0)}, None,
                                start_equity=100_000,
                                settings=_settings(enable_short=0), max_hold_bars=5)
    storage.save_run("AAA", "a", "b", {}, {"num_trades": 0},
                     num_rejections=len(result["rejections"]))
    assert storage.list_runs(1)[0]["params"]["num_rejections"] == 1


# ==================== D) CLI run paths ====================

def test_single_run_path_preserves_rejection_count(tmp_path, monkeypatch):
    import run_backtest
    # >= 70 bars so run_single does not bail on insufficient data.
    gap_df = _long_gap_df(gap_open=1.0, trailing=12)
    assert len(gap_df) >= 70

    captured = {}
    monkeypatch.setattr(run_backtest, "load_ohlcv", lambda *a, **k: gap_df)
    monkeypatch.setattr(run_backtest, "get_spy_for_period", lambda *a, **k: None)
    monkeypatch.setattr(run_backtest, "save_report",
                        lambda result, metrics, name: captured.update(result=result)
                        or tmp_path / "r.json")
    monkeypatch.setattr(run_backtest, "save_run", lambda *a, **k: None)

    result, metrics = run_backtest.run_single(
        "AAA", "2024-01-01", "2024-06-01", 100_000,
        settings=_settings(enable_short=0), max_hold_bars=5)

    assert captured["result"] is result
    assert len(result["rejections"]) >= 1
    assert result["trades"] == []
    assert metrics["num_trades"] == 0


def test_portfolio_run_path_preserves_rejection_count(tmp_path, monkeypatch):
    import run_backtest
    data = {"BAD": _long_gap_df(gap_open=1.0), "GOOD": _clean_df()}
    captured = {}
    monkeypatch.setattr(run_backtest, "load_multi", lambda *a, **k: data)
    monkeypatch.setattr(run_backtest, "get_spy_for_period", lambda *a, **k: None)
    monkeypatch.setattr(run_backtest, "save_report",
                        lambda result, metrics, name: captured.update(result=result)
                        or tmp_path / "p.json")

    run_backtest.run_portfolio(["BAD", "GOOD"], "2024-01-01", "2024-06-01",
                               100_000, settings=_settings(enable_short=0,
                                                           max_open_positions=2),
                               max_hold_bars=5)
    assert len(captured["result"]["rejections"]) >= 1
    assert all(r["symbol"] == "BAD" for r in captured["result"]["rejections"])


# ==================== E) Walk-forward ====================

def test_filter_rejections_to_window_half_open_and_malformed():
    start = pd.Timestamp("2024-02-01")
    end = pd.Timestamp("2024-03-01")
    rejs = [
        {"symbol": "A", "time": pd.Timestamp("2024-01-31"), "reason": "r"},  # before
        {"symbol": "A", "time": pd.Timestamp("2024-02-01"), "reason": "r"},  # in
        {"symbol": "A", "time": pd.Timestamp("2024-03-01"), "reason": "r"},  # end excl
        {"symbol": "A", "time": None, "reason": "r"},                        # malformed
        "not-a-mapping",                                                     # malformed
    ]
    kept = _filter_rejections_to_window(rejs, start, end)
    assert [r["time"] for r in kept] == [pd.Timestamp("2024-02-01")]


def test_filter_rejections_final_window_is_open_ended():
    start = pd.Timestamp("2024-02-01")
    rejs = [{"symbol": "A", "time": pd.Timestamp("2099-01-01"), "reason": "r"}]
    assert len(_filter_rejections_to_window(rejs, start, None)) == 1


def test_filter_rejections_handles_none_and_empty():
    assert _filter_rejections_to_window(None, pd.Timestamp("2024-01-01"), None) == []
    assert _filter_rejections_to_window([], pd.Timestamp("2024-01-01"), None) == []


def test_walk_forward_window_retains_rejections(monkeypatch):
    """Driver-level: every window exposes rejection info and the single gap
    event is counted exactly once across all windows."""
    import backtest.walkforward as wf

    gap_df = _long_gap_df(gap_open=1.0, trailing=12, start="2023-06-01")
    monkeypatch.setattr(wf, "load_ohlcv", lambda *a, **k: gap_df)
    monkeypatch.setattr(wf, "get_spy_for_period", lambda *a, **k: None)

    results = wf.walk_forward("AAA", "2023-01-02", "2024-08-30", 6, 3,
                              initial_equity=100_000.0,
                              settings=_settings(enable_short=0))
    assert results, "walk-forward must produce windows"
    for w in results:
        assert isinstance(w["result"]["rejections"], list)
        assert isinstance(w["result"]["num_rejections"], int)
        assert w["result"]["num_rejections"] == len(w["result"]["rejections"])
        # existing keys preserved
        assert "equity_curve" in w["result"] and "trades" in w["result"]

    total = sum(w["result"]["num_rejections"] for w in results)
    assert total == 1, "the single gap event must be counted exactly once"


def test_walk_forward_rejections_do_not_become_trades(monkeypatch):
    import backtest.walkforward as wf
    gap_df = _long_gap_df(gap_open=1.0, trailing=12, start="2023-06-01")
    monkeypatch.setattr(wf, "load_ohlcv", lambda *a, **k: gap_df)
    monkeypatch.setattr(wf, "get_spy_for_period", lambda *a, **k: None)

    results = wf.walk_forward("AAA", "2023-01-02", "2024-08-30", 6, 3,
                              initial_equity=100_000.0,
                              settings=_settings(enable_short=0))
    n_rej = sum(w["result"]["num_rejections"] for w in results)
    n_trades = sum(len(w["result"]["trades"]) for w in results)
    assert n_rej == 1
    assert n_trades == 0, "rejected entries must not appear as trades"
