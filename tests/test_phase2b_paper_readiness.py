"""
Phase 2B regression tests — Part 1 (P0 blockers, mode selection, data
contract, whole-share sizing and the $50 budget).

Deterministic synthetic data and fake clients only. No network, no Alpaca
contact, no order submission.
"""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

from risk_utils import whole_share_qty, can_protect_with_stop


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ohlcv(n=140, start=100.0, drift=0.003, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.004, n)
    closes = start * np.exp(np.cumsum(steps))
    highs = closes * 1.01
    lows = closes * 0.99
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vol = rng.integers(800_000, 1_200_000, n).astype(float)
    idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="B")
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows,
         "Close": closes, "Volume": vol}, index=idx)


def _resolve_mode(monkeypatch, env):
    for k in ("TRADING_MODE", "ENABLE_LIVE_TRADING", "LIVE_TRADING_CONFIRMATION"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    for m in ("config", "settings"):
        sys.modules.pop(m, None)
    import config
    importlib.reload(config)
    import settings
    importlib.reload(settings)
    return config, settings


# ===========================================================================
# 1) monitor.py and app.py must import/compile (P0-1)
# ===========================================================================
def test_monitor_and_app_import_and_compile():
    import py_compile
    py_compile.compile("monitor.py", doraise=True)
    py_compile.compile("app.py", doraise=True)

    import monitor  # noqa: F401
    import app  # noqa: F401

    # Importing must NOT start a background trading loop.
    assert monitor.monitor.is_running() is False
    assert app.controller.is_running() is False


def test_monitor_start_reconciles_before_trading(monkeypatch):
    import monitor as mon

    calls = {"sync": 0}

    def fake_sync():
        calls["sync"] += 1
        return [{"symbol": "AAPL"}]

    class FakeThread:
        def __init__(self, target=None, daemon=None, name=None, **k):
            self._t = target

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(mon, "_sync_positions", fake_sync)
    monkeypatch.setattr(mon.threading, "Thread", FakeThread)

    pm = mon.PositionMonitor()
    pm.start()
    assert calls["sync"] == 1, "startup must reconcile broker state"


def test_monitor_start_reconciliation_failure_blocks_entries(monkeypatch):
    import monitor as mon
    import decision_log

    monkeypatch.setenv("DB_PATH", os.environ["DB_PATH"])
    decision_log.set_protection_unverified(False)

    def boom():
        raise RuntimeError("broker unreachable")

    class FakeThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr(mon, "_sync_positions", boom)
    monkeypatch.setattr(mon.threading, "Thread", FakeThread)

    pm = mon.PositionMonitor()
    pm.start()
    assert decision_log.get_protection_unverified() is True
    decision_log.set_protection_unverified(False)


# ===========================================================================
# 2) PAPER / BACKTEST / LIVE mode selection (P0-2)
# ===========================================================================
def test_mode_selection_table(monkeypatch):
    # BACKTEST: never submits
    c, s = _resolve_mode(monkeypatch, {"TRADING_MODE": "backtest"})
    assert c.get_trading_mode() == "BACKTEST"
    assert s.is_dry_run() is True
    assert s.broker_submission_allowed() is False

    # PAPER: paper endpoint, may submit
    c, s = _resolve_mode(monkeypatch, {"TRADING_MODE": "paper"})
    assert c.get_trading_mode() == "PAPER"
    assert c.ALPACA_PAPER is True
    assert s.is_dry_run() is False
    assert s.is_paper() is True
    assert s.broker_submission_allowed() is True

    # LIVE without safeguards → fail safe to PAPER
    c, s = _resolve_mode(monkeypatch, {"TRADING_MODE": "live"})
    assert c.get_trading_mode() == "PAPER"

    # Unknown / typo → PAPER, never LIVE
    c, s = _resolve_mode(monkeypatch, {"TRADING_MODE": "lvie"})
    assert c.get_trading_mode() == "PAPER"
    assert c.ALPACA_PAPER is True

    # LIVE only with all three safeguards
    c, s = _resolve_mode(monkeypatch, {
        "TRADING_MODE": "live", "ENABLE_LIVE_TRADING": "true",
        "LIVE_TRADING_CONFIRMATION": "I_UNDERSTAND_LIVE_RISK"})
    assert c.get_trading_mode() == "LIVE"
    assert c.ALPACA_PAPER is False
    assert s.is_live() is True


def test_backtest_cannot_submit_orders(monkeypatch):
    import broker

    monkeypatch.setattr(broker, "is_dry_run", lambda: True)
    monkeypatch.setattr(broker, "ALPACA_API_KEY", "k")
    monkeypatch.setattr(broker, "ALPACA_SECRET_KEY", "s")

    def explode():
        raise AssertionError("BACKTEST must not create a broker client")

    monkeypatch.setattr(broker, "_client", explode)

    res = broker.execute_order({"symbol": "AAPL", "side": "buy",
                                "qty": 5, "stop_loss": 90.0})
    assert res["status"] == "dry_run"


def test_paper_mode_can_submit_with_fake_client(monkeypatch):
    import broker
    from conftest import FakeClient, filled_order, stop_order

    client = FakeClient(orders=[filled_order("E1", 5, 100.0)],
                        open_orders=[stop_order("S1", "AAPL", 5, "sell", 90.0)])
    monkeypatch.setattr(broker, "_client", lambda: client)
    monkeypatch.setattr(broker, "check_safety_gates", lambda: (True, "ok"))
    monkeypatch.setattr(broker, "is_dry_run", lambda: False)
    monkeypatch.setattr(broker, "broker_submission_allowed", lambda: True)
    monkeypatch.setattr(broker, "ALPACA_API_KEY", "k")
    monkeypatch.setattr(broker, "ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(broker, "is_kill_switch_active", lambda: (False, ""))

    res = broker.execute_order({"symbol": "AAPL", "side": "buy", "qty": 5,
                                "stop_loss": 90.0, "estimated_cost": 500.0})
    assert res["status"] == "submitted"
    assert client.submitted, "PAPER mode must be able to submit"
    req = client.submitted[0]
    assert getattr(req, "client_order_id", "").startswith("tv1-AAPL-")


# ===========================================================================
# 3) live market-data / indicator contract (P0-3)
# ===========================================================================
def test_data_layer_provides_full_indicator_contract():
    from signals import REQUIRED_INDICATOR_KEYS, compute_indicators_from_ohlcv

    # Call the pure contract builder directly (no network).
    ind = compute_indicators_from_ohlcv(_ohlcv())
    assert ind is not None
    for k in REQUIRED_INDICATOR_KEYS:
        assert k in ind, f"missing {k}"
        assert np.isfinite(float(ind[k])), f"non-finite {k}"


def test_fetch_market_data_uses_shared_indicators(monkeypatch):
    import data_layer

    monkeypatch.setattr(data_layer, "fetch_ohlcv", lambda sym, period="6mo": _ohlcv())
    monkeypatch.setattr(data_layer, "fetch_finnhub_sentiment", lambda s: {})
    monkeypatch.setattr(data_layer, "fetch_alpha_vantage_news", lambda s: {})

    data = data_layer.fetch_market_data("AAPL")
    assert "error" not in data
    assert "atr14" in data and "ema20" in data and "adx" in data
    assert data["symbol"] == "AAPL"


def test_fetch_market_data_fails_closed_on_bad_data(monkeypatch):
    import data_layer

    # too few bars → invalid
    monkeypatch.setattr(data_layer, "fetch_ohlcv", lambda sym, period="6mo": _ohlcv(n=10))
    assert "error" in data_layer.fetch_market_data("AAPL")

    # NaN prices → invalid
    bad = _ohlcv()
    bad.iloc[5, bad.columns.get_loc("Close")] = np.nan
    monkeypatch.setattr(data_layer, "fetch_ohlcv", lambda sym, period="6mo": bad)
    assert "error" in data_layer.fetch_market_data("AAPL")


def test_analyst_receives_indicators_and_does_not_hold_on_valid_data(monkeypatch):
    import agents
    from signals import compute_indicators_from_ohlcv

    ind = compute_indicators_from_ohlcv(_ohlcv())
    state = {"symbol": "AAPL", "market_data": ind, "memory_context": ""}
    out = agents.analyst_agent(state)
    # Valid indicators must NOT produce the "no valid indicators" HOLD.
    assert out["analyst_report"].get("reason") != "no valid indicators"


def test_analyst_diagnoses_score_gate_before_llm(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
    })
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a: {
        "score": 56, "trend": 20, "momentum": 13, "rsi": 15,
        "bb": 0, "volume": 8, "news": 0,
    })
    monkeypatch.setattr(agents, "llm", type("NoCallLLM", (), {
        "invoke": lambda *a, **k: pytest.fail("LLM must not be called")
    })())

    out = agents.analyst_agent({"symbol": "AAPL", "market_data": {"atr14": 1}})
    report = out["analyst_report"]
    assert report["signal"] == "HOLD"
    assert report["diagnostic"] == {
        "stage": "signal_score", "llm_called": False, "threshold": 70,
        "signal_score": 56, "market_regime": "UNKNOWN",
        "score_components": {"trend": 20, "momentum": 13, "rsi": 15,
                             "bb": 0, "volume": 8, "news": 0},
    }
    assert "LLM not called" in out["logs"][0]


def test_analyst_diagnoses_bear_gate_without_calling_llm(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 1, "news_enabled": 0, "min_signal_score": 70,
    })
    monkeypatch.setattr(agents, "get_market_regime", lambda: {"regime": "BEAR"})
    monkeypatch.setattr(agents, "get_news_sentiment", lambda _s: None)
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a: {
        "score": 80, "trend": 25, "momentum": 20, "rsi": 15,
        "bb": 8, "volume": 12, "news": 0,
    })
    monkeypatch.setattr(agents, "llm", type("NoCallLLM", (), {
        "invoke": lambda *a, **k: pytest.fail("LLM must not be called")
    })())

    report = agents.analyst_agent({"symbol": "AAPL", "market_data": {"atr14": 1}})["analyst_report"]
    assert report["reason"] == "BEAR market — no new longs"
    assert report["diagnostic"]["stage"] == "regime"
    assert report["diagnostic"]["llm_called"] is False
    assert report["diagnostic"]["signal_score"] == 80
    assert report["diagnostic"]["market_regime"] == "BEAR"


def test_analyst_reads_valid_llm_json_and_records_success(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
    })
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a: {
        "score": 82, "trend": 25, "momentum": 20, "rsi": 15,
        "bb": 8, "volume": 12, "news": 0,
    })

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            return type("Response", (), {"content":
                '{"signal":"BUY","confidence":0.91,"reason":"trend"}'})()

    fake = FakeLLM()
    monkeypatch.setattr(agents, "llm", fake)
    report = agents.analyst_agent({
        "symbol": "AAPL", "market_data": {
            "atr14": 1, "last_close": 100, "ema20": 99, "ema50": 98,
            "rsi14": 55, "adx": 30, "macd_hist": 1,
            "bb_pct_b": 0.5, "bb_bandwidth": 5, "stoch_k": 50,
            "stoch_d": 45, "volume_ratio": 1.1,
            "momentum_5d_pct": 1, "momentum_10d_pct": 2,
        },
    })["analyst_report"]

    assert fake.calls == 1
    assert report["signal"] == "BUY"
    assert report["confidence"] == pytest.approx(0.91)
    assert report["diagnostic"]["llm_status"] == "success"
    assert report["diagnostic"]["llm_attempts"] == 1
    assert report["diagnostic"]["signal_score"] == 82
    assert report["diagnostic"]["market_regime"] == "UNKNOWN"


def test_analyst_retries_invalid_json_then_parses_valid_response(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
    })
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a: {
        "score": 82, "trend": 25, "momentum": 20, "rsi": 15,
        "bb": 8, "volume": 12, "news": 0,
    })

    class RetryLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            content = (
                "not valid JSON" if self.calls == 1 else
                '{"signal":"SELL","confidence":0.82,"reason":"test response"}'
            )
            return type("Response", (), {"content": content})()

    fake = RetryLLM()
    monkeypatch.setattr(agents, "llm", fake)
    report = agents.analyst_agent({
        "symbol": "AAPL", "market_data": {
            "atr14": 1, "last_close": 100, "ema20": 99, "ema50": 98,
            "rsi14": 55, "adx": 30, "macd_hist": 1,
            "bb_pct_b": 0.5, "bb_bandwidth": 5, "stoch_k": 50,
            "stoch_d": 45, "volume_ratio": 1.1,
            "momentum_5d_pct": 1, "momentum_10d_pct": 2,
        },
    })["analyst_report"]

    assert fake.calls == 2
    assert report["signal"] == "SELL"
    assert report["confidence"] == pytest.approx(0.82)
    assert report["diagnostic"]["llm_called"] is True
    assert report["diagnostic"]["llm_attempts"] == 2
    assert report["diagnostic"]["llm_status"] == "success"


def test_hold_is_not_misreported_as_a_risk_veto(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "min_confidence": 0.65, "min_signal_score": 70,
    })
    report = agents.risk_manager_agent({
        "symbol": "AAPL", "market_data": {},
        "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                           "signal_score": 0, "reason": "below threshold"},
    })

    assert report["risk_report"] == {
        "approved": False, "veto": False, "reason": "analyst HOLD",
    }


def test_decision_log_persists_safe_analyst_diagnostics(tmp_path, monkeypatch):
    import json
    import decision_log

    monkeypatch.setattr(decision_log, "DB", tmp_path / "decisions.db")
    decision_log.init_db()
    decision_log.log_decision({
        "symbol": "AAPL",
        "analyst_report": {
            "signal": "HOLD", "confidence": 0.0,
            "reason": "signal score below threshold (56 < 70)",
            "diagnostic": {
                "stage": "signal_score", "llm_called": False,
                "signal_score": 56, "threshold": 70,
                "market_regime": "BULL",
                "score_components": {"trend": 20, "momentum": 13},
            },
        },
        "risk_report": {"approved": False, "veto": False,
                        "reason": "analyst HOLD"},
        "final_decision": "HOLD", "consensus_reached": False,
        "market_data": {},
    }, None)

    row = decision_log.recent_decisions(1)[0]
    diagnostic = json.loads(row["analyst_diagnostic"])
    assert diagnostic["signal_score"] == 56
    assert diagnostic["score_components"]["trend"] == 20
    assert diagnostic["market_regime"] == "BULL"
    assert diagnostic["llm_called"] is False
    assert row["risk_reason"] == "analyst HOLD"
    assert row["final_decision"] == "HOLD"


def test_analyst_retries_invalid_llm_output_and_sanitizes_errors(monkeypatch):
    import agents

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
    })
    monkeypatch.setattr(agents, "compute_signal_score", lambda *a: {
        "score": 82, "trend": 25, "momentum": 20, "rsi": 15,
        "bb": 8, "volume": 12, "news": 0,
    })

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            if self.calls == 1:
                return type("Response", (), {"content": "not json"})()
            raise RuntimeError("Authorization: Bearer secret-value")

    fake = FakeLLM()
    monkeypatch.setattr(agents, "llm", fake)
    report = agents.analyst_agent({
        "symbol": "AAPL", "market_data": {
            "atr14": 1, "last_close": 100, "ema20": 99, "ema50": 98,
            "rsi14": 55, "adx": 30, "macd_hist": 1,
            "bb_pct_b": 0.5, "bb_bandwidth": 5, "stoch_k": 50,
            "stoch_d": 45, "volume_ratio": 1.1,
            "momentum_5d_pct": 1, "momentum_10d_pct": 2,
        },
    })["analyst_report"]

    assert fake.calls == 2
    assert report["signal"] == "HOLD"
    assert report["reason"] == "LLM request failed"
    assert "secret-value" not in str(report)
    assert report["diagnostic"]["llm_status"] == "error"
    assert report["diagnostic"]["llm_error_type"] == "RuntimeError"
    assert report["diagnostic"]["llm_attempts"] == 2


def test_validate_indicators_rejects_missing_and_nonfinite():
    from signals import validate_indicators

    ok, _ = validate_indicators({"last_close": 1.0})
    assert ok is False
    good = {k: 1.0 for k in [
        "last_close", "atr14", "rsi14", "sma20", "sma50", "ema20", "ema50",
        "macd_hist", "bb_pct_b", "bb_bandwidth", "adx", "stoch_k", "stoch_d",
        "volume_ratio", "momentum_5d_pct", "momentum_10d_pct"]}
    assert validate_indicators(good)[0] is True
    good["adx"] = float("nan")
    assert validate_indicators(good)[0] is False
    good["adx"] = float("inf")
    assert validate_indicators(good)[0] is False


# ===========================================================================
# 4) whole-share sizing policy
# ===========================================================================
def test_whole_share_qty_flooring_and_invalid():
    assert whole_share_qty(0.02) == 0
    assert whole_share_qty(0.99) == 0
    assert whole_share_qty(1.0) == 1
    assert whole_share_qty(8.9) == 8
    assert whole_share_qty(0) == 0
    assert whole_share_qty(-3) == 0
    assert whole_share_qty(float("nan")) == 0
    assert whole_share_qty(None) == 0


def test_can_protect_with_stop_rules():
    assert can_protect_with_stop(0, 100, 90, "buy")[0] is False
    assert can_protect_with_stop(1, 100, 90, "buy")[0] is True
    # long stop must be below entry
    assert can_protect_with_stop(1, 100, 100, "buy")[0] is False
    assert can_protect_with_stop(1, 100, 101, "buy")[0] is False
    # short stop must be above entry
    assert can_protect_with_stop(1, 100, 101, "sell")[0] is True
    assert can_protect_with_stop(1, 100, 99, "sell")[0] is False
    assert can_protect_with_stop(1, 100, 0, "buy")[0] is False


# ===========================================================================
# 5) $50 trading_capital budget, end to end through risk_manager_agent
# ===========================================================================
def _agent_settings(**over):
    s = {
        "min_confidence": 0.7, "min_signal_score": 30,
        "max_risk_per_trade": 0.02, "max_position_pct": 1.0,
        "max_portfolio_exposure_pct": 0.99, "take_profit_rr": 2.0,
        "stop_atr_multiplier": 2.5, "trading_capital": 0.0,
        "enable_short": 0,
    }
    s.update(over)
    return s


def _run_risk(monkeypatch, capital, equity=100_000.0, buying_power=200_000.0,
              price=100.0, atr=5.0, committed=0.0):
    import agents

    monkeypatch.setattr(agents, "get_all_settings",
                        lambda: _agent_settings(trading_capital=capital))
    monkeypatch.setattr(agents, "get_account_snapshot",
                        lambda: {"available": True, "equity": equity,
                                 "buying_power": buying_power})
    monkeypatch.setattr(agents, "get_committed_exposure_value", lambda: committed)
    monkeypatch.setattr(agents, "check_portfolio_exposure", lambda s, c: (True, "ok"))

    state = {
        "symbol": "AAPL",
        "market_data": {"atr14": atr, "last_close": price, "rsi14": 50.0},
        "analyst_report": {"signal": "BUY", "confidence": 0.9,
                           "signal_score": 100, "regime": "BULL"},
    }
    return agents.risk_manager_agent(state)["risk_report"]


def test_fifty_dollar_budget_cannot_fund_a_protected_share(monkeypatch):
    # risk_amount = 50 * 0.02 = $1 ; stop_distance = 5 * 2.5 = 12.5
    # raw_qty = 0.08 shares → floors to 0 → trade must be SKIPPED, not inflated.
    rep = _run_risk(monkeypatch, capital=50.0)
    assert rep["approved"] is False
    assert rep["veto"] is True
    assert "cannot protect" in rep["reason"]


def test_large_broker_equity_cannot_bypass_fifty_dollar_budget(monkeypatch):
    rep = _run_risk(monkeypatch, capital=50.0,
                    equity=10_000_000.0, buying_power=50_000_000.0)
    assert rep["approved"] is False
    assert "cannot protect" in rep["reason"]


def test_budget_sufficient_for_one_share_is_approved(monkeypatch):
    # capital 5000 → risk $100 → raw_qty 8 → whole 8 shares, cost $800
    rep = _run_risk(monkeypatch, capital=5000.0)
    assert rep["approved"] is True
    assert rep["qty"] == 8
    assert float(rep["qty"]).is_integer()


def test_zero_capital_uses_account_equity(monkeypatch):
    # capital 0 → sizing uses real equity (100k) → whole shares, approved
    rep = _run_risk(monkeypatch, capital=0.0)
    assert rep["approved"] is True
    assert rep["qty"] >= 1


def test_budget_blocks_when_exposure_unknown(monkeypatch):
    import agents
    monkeypatch.setattr(agents, "get_all_settings",
                        lambda: _agent_settings(trading_capital=5000.0))
    monkeypatch.setattr(agents, "get_account_snapshot",
                        lambda: {"available": True, "equity": 100_000.0,
                                 "buying_power": 200_000.0})
    monkeypatch.setattr(agents, "get_committed_exposure_value", lambda: None)
    state = {
        "symbol": "AAPL",
        "market_data": {"atr14": 5.0, "last_close": 100.0, "rsi14": 50.0},
        "analyst_report": {"signal": "BUY", "confidence": 0.9,
                           "signal_score": 100, "regime": "BULL"},
    }
    rep = agents.risk_manager_agent(state)["risk_report"]
    assert rep["approved"] is False
    assert "exposure unknown" in rep["reason"]


def test_risk_engine_exposure_caps_use_budget(monkeypatch):
    import risk_engine

    class Acct:
        equity = "100000"

    class FakeClient:
        def get_account(self):
            return Acct()

        def get_all_positions(self):
            return []

    monkeypatch.setattr(risk_engine, "_client", lambda: FakeClient())
    monkeypatch.setattr(risk_engine, "get_all_settings",
                        lambda: {"trading_capital": 50.0,
                                 "max_portfolio_exposure_pct": 0.45,
                                 "max_position_pct": 0.15})

    # 50 * 0.15 = $7.50 cap; an $8 position must be rejected even though the
    # account has $100k (which would allow $15k).
    ok, reason = risk_engine.check_portfolio_exposure("AAPL", 8.0)
    assert ok is False
    assert "budget" in reason


def test_daily_loss_gate_measured_against_budget(monkeypatch):
    """The daily-loss gate is anchored to the $50 budget, not the $100k account.

    Discriminator: on a fresh day the gate records the day-start equity. It must
    record min(budget, account equity) = 50, not the raw account equity. Without
    the fix it stores 100000, so the daily-loss baseline is 2000x too large and
    the gate can never trip on the process's real $50 risk.

    NOTE (documented limitation, not silently changed): because both the
    baseline and the current value are clamped to the budget, this gate only
    trips when the *account* equity falls to/below the budget level. It is
    effectively inert while the paper account equity far exceeds the budget.
    Detecting a real loss of the process's own $50 requires tracking this
    process's realized P&L, which is a larger change than malformed-input
    rejection and is intentionally out of scope here.
    """
    import safety
    import decision_log

    monkeypatch.setattr(safety, "get_all_settings",
                        lambda: {"trading_capital": 50.0,
                                 "max_daily_loss_pct": 0.05,
                                 "max_open_positions": 5,
                                 "max_trades_per_day": 100,
                                 "max_consecutive_losses": 5})
    monkeypatch.setattr(safety, "is_market_open", lambda: True)
    monkeypatch.setattr(safety, "get_cooldown_until", lambda: None)
    monkeypatch.setattr(safety, "get_consecutive_losses", lambda: 0)
    monkeypatch.setattr(decision_log, "is_kill_switch_active", lambda: (False, ""))
    monkeypatch.setattr(decision_log, "get_protection_unverified", lambda: False)
    monkeypatch.setattr(safety, "get_account_snapshot", lambda: {
        "available": True, "equity": 100_000.0, "open_positions": 0,
        "trades_today": 0, "buying_power": 100_000.0})

    # Fresh day: the baseline must be the budget (50), never the raw 100000.
    recorded = {}
    monkeypatch.setattr(safety, "get_day_start_equity", lambda: None)
    monkeypatch.setattr(safety, "set_day_start_equity",
                        lambda v: recorded.setdefault("v", v))
    ok, reason = safety.check_safety_gates()
    assert ok is True, reason
    assert recorded["v"] == 50.0, (
        f"day-start equity anchored to {recorded['v']}, expected budget 50")

    # A real drop within the budget trips the limit.
    monkeypatch.setattr(safety, "get_day_start_equity", lambda: 50.0)
    monkeypatch.setattr(safety, "get_account_snapshot", lambda: {
        "available": True, "equity": 47.0, "open_positions": 0,
        "trades_today": 0, "buying_power": 47.0})
    ok, reason = safety.check_safety_gates()
    assert ok is False
    assert "daily loss" in reason
