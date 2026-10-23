"""
Phase-1.8 RC-4 regression tests — cycle state / checkpoint isolation.

Root cause of the contradictory analyst logs (RC-4): ``TradingState.logs`` used
``Annotated[list[str], operator.add]`` while the graph is compiled with a
*persistent* ``SqliteSaver`` checkpointer and invoked under a stable per-symbol
``thread_id``. The reducer therefore concatenated every earlier cycle's
transcript onto the current one, so a cycle replayed a previous cycle's
"بيانات غير صالحة" (invalid data) HOLD next to its own current reason.

These tests drive the REAL compiled graph with the REAL checkpointer; only
external I/O (LLM, memory store, market regime, news, broker snapshot) is
replaced with deterministic fakes. They assert:

  * a cycle's logs contain only its own evidence (no cross-cycle replay),
  * repeated cycles stay independent,
  * symbol A's transcript never leaks into symbol B,
  * a non-executing decision never inherits a previous cycle's order_details,
  * a node runs exactly once per cycle (no duplicate invocation / retry blow-up).

No network, no credentials, no broker order.
"""
import importlib
import sys

import pytest

pytest.importorskip("langgraph", reason="requires the real langgraph")

from state import CYCLE_BOUNDARY, _merge_cycle_logs

INVALID_MSG = "بيانات غير صالحة"

VALID_DATA = {
    "symbol": "ASML", "last_close": 100.0, "atr14": 1.0, "rsi14": 50.0,
    "ema20": 99.0, "ema50": 98.0, "adx": 30.0, "macd_hist": 0.1,
    "bb_pct_b": 0.5, "bb_bandwidth": 2.0, "stoch_k": 50.0, "stoch_d": 50.0,
    "volume_ratio": 1.2, "momentum_5d_pct": 1.0, "momentum_10d_pct": 2.0,
    "sma20": 99.0, "sma50": 98.0,
}


# ============================================================================
# Unit: the reducer itself
# ============================================================================

def test_boundary_discards_accumulated_transcript():
    prev = ["cycle-1 A", "cycle-1 B"]
    new = [CYCLE_BOUNDARY, "cycle-2 A"]
    assert _merge_cycle_logs(prev, new) == ["cycle-2 A"]


def test_non_boundary_right_appends():
    assert _merge_cycle_logs(["a"], ["b", "c"]) == ["a", "b", "c"]


def test_boundary_only_honoured_in_first_position():
    """A log line equal to the marker must survive as data when not leading."""
    out = _merge_cycle_logs(["keep"], ["data", CYCLE_BOUNDARY])
    assert out == ["keep", "data", CYCLE_BOUNDARY]


def test_reducer_tolerates_none_sides():
    assert _merge_cycle_logs(None, [CYCLE_BOUNDARY, "x"]) == ["x"]
    assert _merge_cycle_logs(["x"], None) == ["x"]
    assert _merge_cycle_logs(None, None) == []


def test_reducer_does_not_mutate_inputs():
    left, right = ["a"], [CYCLE_BOUNDARY, "b"]
    _merge_cycle_logs(left, right)
    assert left == ["a"] and right == [CYCLE_BOUNDARY, "b"]


# ============================================================================
# Graph-level harness
# ============================================================================

@pytest.fixture
def rc4(monkeypatch, tmp_path):
    """Real graph + real SqliteSaver, deterministic external fakes."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "decisions.db"))
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "graph.db"))
    monkeypatch.setenv("MEMORY_DB_PATH", str(tmp_path / "memory.db"))

    import decision_log
    importlib.reload(decision_log)

    import agents
    import signals
    import memory_layer

    calls = {"analyst": 0, "risk": 0, "pm": 0}

    monkeypatch.setattr(agents, "get_all_settings", lambda: {
        "regime_enabled": 0, "news_enabled": 0, "min_signal_score": 70,
        "min_confidence": 0.70, "max_risk_per_trade": 0.005,
        "stop_atr_multiplier": 2.5, "take_profit_rr": 2.0,
        "max_position_pct": 0.15, "trading_capital": 10000,
    })
    monkeypatch.setattr(agents, "get_market_regime",
                        lambda: {"regime": "SIDEWAYS"})
    monkeypatch.setattr(agents, "get_news_sentiment",
                        lambda s: {"available": False})
    monkeypatch.setattr(agents, "get_account_snapshot", lambda: {
        "available": True, "equity": 10000.0, "buying_power": 10000.0,
        "open_positions": 0, "trades_today": 0})
    monkeypatch.setattr(agents, "check_portfolio_exposure",
                        lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(agents, "get_committed_exposure_value", lambda: 0.0)

    score = {"value": 68}          # default: below threshold -> HOLD
    monkeypatch.setattr(
        agents, "compute_signal_score",
        lambda *a, **k: {"score": score["value"], "trend": 25, "momentum": 20,
                         "rsi": 15, "bb": 8, "volume": 8, "news": 0})

    class _LLM:
        def invoke(self, *a, **k):
            return type("R", (), {
                "content": '{"signal":"BUY","confidence":0.9,"reason":"buy"}'})()

    monkeypatch.setattr(agents, "llm", _LLM())

    # Count node executions: wrap the agent callables *before* graph binds them.
    for name, key in (("analyst_agent", "analyst"),
                      ("risk_manager_agent", "risk"),
                      ("portfolio_manager_agent", "pm")):
        real = getattr(agents, name)

        def make(real=real, key=key):
            def wrapper(state):
                calls[key] += 1
                return real(state)
            return wrapper

        monkeypatch.setattr(agents, name, make())

    # Reload graph so it binds the wrapped agents and a fresh checkpoint DB.
    if "graph" in sys.modules:
        del sys.modules["graph"]
    import graph
    monkeypatch.setattr(graph, "get_similar_experiences",
                        lambda s, k=3: "none")

    # Guard against a stubbed langgraph making these tests vacuous.
    if type(graph.trading_graph).__module__.startswith("conftest"):
        pytest.skip("langgraph is stubbed; graph-level test would be vacuous")

    def run(symbol, data, omit_order_details=False):
        st = {"symbol": symbol, "market_data": data, "memory_context": "",
              "analyst_report": None, "risk_report": None,
              "final_decision": None, "broker_result": None,
              "consensus_reached": False, "logs": []}
        if not omit_order_details:
            st["order_details"] = None
        return graph.trading_graph.invoke(
            st, config={"configurable": {"thread_id": f"live_{symbol}"}})

    return {"run": run, "score": score, "calls": calls, "graph": graph}


# ============================================================================
# 1) A cycle's logs contain only its own evidence
# ============================================================================

def test_cycle_logs_do_not_replay_previous_cycle(rc4):
    r1 = rc4["run"]("ASML", {})              # invalid data -> HOLD
    assert any(INVALID_MSG in l for l in r1["logs"])

    r2 = rc4["run"]("ASML", VALID_DATA)      # valid, low score -> HOLD
    joined = " ".join(r2["logs"])
    assert INVALID_MSG not in joined, "cycle 2 replayed cycle 1's evidence"
    assert any("below threshold" in l for l in r2["logs"])
    assert len(r2["logs"]) == 4, r2["logs"]


def test_repeated_cycles_stay_independent(rc4):
    for i in range(6):
        data = {} if i % 2 == 0 else VALID_DATA
        r = rc4["run"]("ASML", data)
        assert len(r["logs"]) == 4, f"cycle {i}: {r['logs']}"
        assert not any(CYCLE_BOUNDARY in l for l in r["logs"])
        if i % 2 == 0:
            assert any(INVALID_MSG in l for l in r["logs"])
        else:
            assert not any(INVALID_MSG in l for l in r["logs"])


def test_symbol_a_does_not_contaminate_symbol_b(rc4):
    rc4["run"]("CRM", {})                    # CRM invalid
    r_b = rc4["run"]("NVDA", VALID_DATA)     # NVDA valid
    assert not any(INVALID_MSG in l for l in r_b["logs"])
    assert all("NVDA" in l or "الذاكرة" in l for l in r_b["logs"]), r_b["logs"]

    # ...and the reverse direction.
    r_a = rc4["run"]("CRM", VALID_DATA)
    assert not any("NVDA" in l for l in r_a["logs"])
    assert len(r_a["logs"]) == 4


# ============================================================================
# 2) Decision-level isolation (not just logs)
# ============================================================================

def test_new_decision_does_not_inherit_stale_order_details(rc4):
    rc4["score"]["value"] = 90                # reach the LLM -> EXECUTE_BUY
    r1 = rc4["run"]("ASML", VALID_DATA)
    assert r1["final_decision"] == "EXECUTE_BUY"
    assert r1["order_details"] is not None

    rc4["score"]["value"] = 10                # HOLD, and caller omits the key
    r2 = rc4["run"]("ASML", VALID_DATA, omit_order_details=True)
    assert r2["final_decision"] == "HOLD"
    assert r2["order_details"] is None, "stale order leaked into a HOLD"


def test_each_node_runs_once_per_cycle(rc4):
    rc4["run"]("ASML", VALID_DATA)
    assert rc4["calls"] == {"analyst": 1, "risk": 1, "pm": 1}
    rc4["run"]("ASML", VALID_DATA)
    assert rc4["calls"] == {"analyst": 2, "risk": 2, "pm": 2}


def test_analyst_retry_does_not_duplicate_state(rc4):
    """A failing first LLM attempt retries *inside* the analyst node; the graph
    must not re-run the node, and the transcript must stay a single cycle."""
    import agents

    attempts = {"n": 0}

    class FlakyLLM:
        def invoke(self, *a, **k):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient")
            return type("R", (), {
                "content": '{"signal":"HOLD","confidence":0.0,"reason":"x"}'})()

    agents.llm = FlakyLLM()
    rc4["score"]["value"] = 90

    r = rc4["run"]("ASML", VALID_DATA)
    assert attempts["n"] == 2, "retry did not happen inside the node"
    assert rc4["calls"]["analyst"] == 1, "graph re-ran the analyst node"
    assert len(r["logs"]) == 4, r["logs"]
