"""
Phase 2B regression tests — Part 3 (CLI smoke tests).

Runs the real ``run_backtest`` CLI paths against a synthetic, on-disk parquet
cache so ``load_ohlcv`` exercises its cache-read branch. No network, no broker,
no live credentials. Report/storage artifacts are redirected to tmp_path.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _synthetic_ohlcv(n=420, start="2023-01-02", seed=7):
    """Trending-with-noise OHLCV guaranteed to pass _normalize()."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n)
    ret = rng.normal(0.0006, 0.010, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    hi = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.01, n))
    lo = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.01, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": vol},
        index=idx,
    )


@pytest.fixture
def cache_env(tmp_path, monkeypatch):
    """Populate an isolated parquet cache and redirect all outputs."""
    cache = tmp_path / "cache"
    cache.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()

    start, end = "2023-01-02", "2024-08-30"
    for sym in ("AAPL", "MSFT"):
        df = _synthetic_ohlcv(seed=hash(sym) % 1000)
        df.to_parquet(cache / f"{sym}_{start}_{end}_1d.parquet")

    # SPY warmup range used by get_spy_for_period.
    spy_start = "2021-11-28"
    _synthetic_ohlcv(n=700, start="2021-11-01", seed=3).to_parquet(
        cache / f"SPY_{spy_start}_{end}_1d.parquet")

    env = dict(os.environ)
    env["BACKTEST_CACHE_DIR"] = str(cache)
    env["BACKTEST_DB_PATH"] = str(tmp_path / "bt.db")
    env["DB_PATH"] = str(tmp_path / "decisions.db")
    env["PYTHONPATH"] = os.getcwd()
    return {"env": env, "cache": cache, "reports": reports,
            "start": start, "end": end, "tmp": tmp_path}


def _run_cli(cache_env, args, monkeypatch):
    import backtest.report as report_mod

    monkeypatch.setattr(report_mod, "REPORT_DIR", cache_env["reports"])
    from run_backtest import main
    argv = ["run_backtest.py"] + args
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return list(cache_env["reports"].glob("*.json"))


def test_cli_single_backtest_cache_only(cache_env, monkeypatch, capsys):
    reports = _run_cli(cache_env, [
        "--symbol", "AAPL",
        "--start", cache_env["start"], "--end", cache_env["end"],
        "--initial", "10000"], monkeypatch)
    out = capsys.readouterr().out
    assert "Monte Carlo" in out
    assert "insufficient data" not in out
    assert reports, "a report must be written"


def test_cli_walkforward_cache_only(cache_env, monkeypatch, capsys):
    _run_cli(cache_env, [
        "--symbol", "AAPL", "--walkforward",
        "--start", cache_env["start"], "--end", cache_env["end"],
        "--train-months", "6", "--test-months", "3",
        "--initial", "10000"], monkeypatch)
    out = capsys.readouterr().out
    assert "WALK-FORWARD AGGREGATE" in out


def test_cli_portfolio_cache_only(cache_env, monkeypatch, capsys):
    _run_cli(cache_env, [
        "--portfolio", "--symbols", "AAPL,MSFT",
        "--start", cache_env["start"], "--end", cache_env["end"],
        "--initial", "10000"], monkeypatch)
    out = capsys.readouterr().out
    assert "Monte Carlo" in out


def test_cli_no_network(monkeypatch):
    """The CLI must never reach yfinance when the cache has no entry."""
    import backtest.data as data

    def boom(*a, **k):
        raise AssertionError("CLI attempted a network download")

    monkeypatch.setattr(data.yf, "Ticker", boom)
    monkeypatch.setattr(data, "CACHE_DIR", Path("/nonexistent-cache-dir"))
    assert data.load_ohlcv("ZZZZ", "2023-01-02", "2024-01-01") is None
