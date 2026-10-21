import time
from apscheduler.schedulers.blocking import BlockingScheduler

from data_layer import fetch_market_data
from graph import trading_graph
from broker import execute_order
from learner import learn_from_closed_trades
from safety import check_safety_gates, get_account_snapshot
from settings import get_all_settings, get_symbols
from decision_log import log_decision, log_equity
from state import TradingState


def run_cycle(symbol: str) -> dict:
    """One analysis + decision + (optional) execution cycle for a symbol."""
    print(f"\n🔄 دورة {symbol}")
    data = fetch_market_data(symbol)
    if "error" in data:
        print("❌ فشل البيانات:", data["error"])
        return {"error": data["error"]}

    state: TradingState = {
        "symbol": symbol,
        "market_data": data,
        "memory_context": "",
        "analyst_report": None,
        "risk_report": None,
        "final_decision": None,
        "order_details": None,
        "broker_result": None,
        "consensus_reached": False,
        "logs": [],
    }

    result = trading_graph.invoke(
        state,
        config={"configurable": {"thread_id": f"live_{symbol}"}},
    )

    for log in result.get("logs", []):
        print("  •", log)
    print(f"⚖️ القرار: {result.get('final_decision')} "
          f"| إجماع: {result.get('consensus_reached')}")

    broker_res = None
    if result.get("final_decision") and result["final_decision"].startswith("EXECUTE"):
        broker_res = execute_order(result["order_details"])
        print("📤 نتيجة التنفيذ:", broker_res)

    log_decision(result, broker_res)

    # FAIL-CLOSED: only log equity when broker actually reports it.
    snap = get_account_snapshot()
    if snap.get("available"):
        log_equity(snap["equity"], snap["open_positions"])
    else:
        print(f"⚠️ main: account snapshot unavailable ({snap.get('reason')}) — "
              f"equity not logged")

    return result


def main():
    print("🚀 بدء النظام (CLI)")
    settings = get_all_settings()
    symbols = get_symbols()
    interval_min = max(1, settings["cycle_interval_minutes"])

    for sym in symbols:
        try:
            run_cycle(sym)
        except Exception as e:
            print(f"⚠️ فشل دورة {sym}: {e}")
        time.sleep(2)

    try:
        learn_from_closed_trades()
    except Exception as e:
        print(f"⚠️ learner failed: {e}")

    scheduler = BlockingScheduler()
    scheduler.add_job(
        lambda: [run_cycle(s) for s in get_symbols()],
        "interval",
        minutes=interval_min,
        id="cycle",
    )
    scheduler.add_job(
        learn_from_closed_trades,
        "interval",
        hours=6,
        id="learn",
    )
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("⏹️ إيقاف.")


if __name__ == "__main__":
    main()
