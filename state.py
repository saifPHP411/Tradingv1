from typing import TypedDict, Literal, Optional, Annotated

# First log emitted by the entry node of every cycle. ``_merge_cycle_logs``
# treats a leading CYCLE_BOUNDARY as the start of a fresh transcript, so a
# cycle's logs never contain a previous cycle's evidence.
CYCLE_BOUNDARY = "__CYCLE_START__"


def _merge_cycle_logs(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append logs within a cycle, but reset the transcript at a cycle boundary.

    ``logs`` is a per-cycle diagnostic transcript. The graph is compiled with a
    *persistent* ``SqliteSaver`` checkpointer and invoked under a stable
    per-symbol ``thread_id`` (``f"live_{symbol}"``), so a plain ``operator.add``
    reducer concatenates every previous cycle's logs onto the current one. That
    replays stale evidence -- an earlier cycle's "بيانات غير صالحة" (invalid
    data) HOLD appears alongside the current cycle's "signal score below
    threshold" HOLD -- and grows the transcript without bound.

    ``graph.enrich_with_memory`` emits :data:`CYCLE_BOUNDARY` as the first log of
    each cycle; that leading marker discards the accumulated transcript so a
    decision can only ever report its own cycle's evidence. The boundary is only
    honoured in first position, so a log line that happens to equal the marker
    is still preserved as data.
    """
    left = list(left or [])
    right = list(right or [])
    if right and right[0] == CYCLE_BOUNDARY:
        return right[1:]
    return left + right


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
    logs: Annotated[list[str], _merge_cycle_logs]
