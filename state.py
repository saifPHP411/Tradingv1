from typing import TypedDict, Literal, Optional, Annotated
import operator

class TradingState(TypedDict):
    symbol: str
    market_data: dict
    memory_context: str          # سياق من الذاكرة (مواقف مشابهة)
    analyst_report: Optional[dict]
    risk_report: Optional[dict]
    final_decision: Optional[Literal["EXECUTE_BUY", "EXECUTE_SELL", "HOLD", "VETO"]]
    order_details: Optional[dict]
    broker_result: Optional[dict]
    consensus_reached: bool
    logs: Annotated[list[str], operator.add]