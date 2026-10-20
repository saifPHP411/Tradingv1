from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import TradingState
from agents import analyst_agent, risk_manager_agent, portfolio_manager_agent
from memory_layer import get_similar_experiences

import os
checkpointer = SqliteSaver.from_conn_string(os.getenv("GRAPH_DB_PATH", "graph_checkpoints.db"))

def enrich_with_memory(state: TradingState) -> dict:
    context = get_similar_experiences(state)
    return {"memory_context": context, "logs": [f"الذاكرة: تم استرجاع السياق"]}

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
