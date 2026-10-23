import os
import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from state import TradingState, CYCLE_BOUNDARY
from agents import analyst_agent, risk_manager_agent, portfolio_manager_agent
from memory_layer import get_similar_experiences

# تجهيز مسار قاعدة البيانات داخل /data (Railway Volume)
_db_path = os.getenv("GRAPH_DB_PATH", "graph_checkpoints.db")
Path(_db_path).parent.mkdir(parents=True, exist_ok=True)

# اتصال sqlite3 عادي (وليس from_conn_string)
_conn = sqlite3.connect(_db_path, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
checkpointer.setup()

# NOTE (known limitation, measured): the checkpointer persists every node write
# per thread and is never pruned, so a long-lived per-symbol thread grows by a
# fixed number of rows each cycle (measured: 6 checkpoint rows / 24 write rows
# per cycle) regardless of the ``logs`` reducer. That is disk growth, not a
# correctness problem: each cycle's decision state is rebuilt from the caller's
# full input, and durable audit history lives in decision_log, not here. Pruning
# the thread per cycle (``checkpointer.delete_thread``) would bound it but
# changes checkpoint-restore semantics, so it is deliberately left as a separate,
# explicitly-authorized change rather than folded into this fix.


def enrich_with_memory(state: TradingState) -> dict:
    context = get_similar_experiences(state)
    # Entry node of every cycle: the leading CYCLE_BOUNDARY resets the
    # append-only ``logs`` transcript so this cycle never replays a previous
    # cycle's evidence (see state._merge_cycle_logs).
    return {"memory_context": context,
            "logs": [CYCLE_BOUNDARY, "الذاكرة: تم استرجاع السياق"]}

builder = StateGraph(TradingState)
builder.add_node("memory_enrich", enrich_with_memory)
builder.add_node("analyst", analyst_agent)
builder.add_node("risk_manager", risk_manager_agent)
builder.add_node("portfolio_manager", portfolio_manager_agent)

builder.add_edge(START, "memory_enrich")
builder.add_edge("memory_enrich", "analyst")
builder.add_edge("analyst", "risk_manager")
builder.add_edge("risk_manager", "portfolio_manager")
builder.add_edge("portfolio_manager", END)

trading_graph = builder.compile(checkpointer=checkpointer)
