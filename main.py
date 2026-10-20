import time
from apscheduler.schedulers.blocking import BlockingScheduler
from data_layer import fetch_market_data
from graph import trading_graph
from broker import execute_order
from learner import learn_from_closed_trades
from safety import check_safety_gates, get_account_snapshot
from decision_log import log_decision, log_equity
from state import TradingState
from config import SYMBOLS

def run_cycle(symbol: str) -> dict:
    print(f"\n🔄 دورة {symbol}")
    data = fetch_market_data(symbol)
    if "error" in data:
        print("❌ فشل البيانات:", data["error"])
        return {"error": data["error"]}

    state: TradingState = {
        "symbol": symbol, "market_data": data, "memory_context": "",
        "analyst_report": None, "risk_report": None, "final_decision": None,
        "order_details": None, "broker_result": None, "consensus_reached": False, "logs": [],
    }
    result = trading_graph.invoke(state, config={"configurable": {"thread_id": f"live_{symbol}"}})

    for log in result["logs"]:
        print("  •", log)
    print(f"⚖️ القرار: {result['final_decision']} | إجماع: {result['consensus_reached']}")

    broker_res = None
    if result["final_decision"] and result["final_decision"].startswith("EXECUTE"):
        broker_res = execute_order(result["order_details"])
        print("📤 نتيجة التنفيذ:", broker_res)

    log_decision(result, broker_res)
    snap = get_account_snapshot()
    log_equity(snap["equity"], snap["open_positions"])
    return result

def main():
    print("🚀 بدء النظام (CLI)")
    for sym in SYMBOLS:
        run_cycle(sym)
        time.sleep(2)
    learn_from_closed_trades()
    scheduler = BlockingScheduler()
    scheduler.add_job(lambda: [run_cycle(s) for s in SYMBOLS], "interval", minutes=30)
    scheduler.add_job(learn_from_closed_trades, "interval", hours=6)
    try:
        scheduler.start()
    except KeyboardInterrupt:
        print("⏹️ إيقاف.")

if __name__ == "__main__":
    main()