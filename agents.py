from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, MIN_CONFIDENCE, MAX_RISK_PER_TRADE
from state import TradingState
import json, re

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

def analyst_agent(state: TradingState) -> dict:
    data = state["market_data"]
    memory = state.get("memory_context", "")
    prompt = f"""أنت محلل مالي خبير. حلل {data['symbol']} بناءً على:
    - السعر: {data.get('last_close')}, SMA20: {data.get('sma20')}, SMA50: {data.get('sma50')}
    - RSI: {data.get('rsi14', 0):.1f}, تغير 5 أيام: {data.get('price_change_5d_pct', 0):.2f}%
    - مشاعر أخبار Alpha Vantage: {data.get('alpha_sentiment_avg', 'غير متوفر')}

    {memory}

    أجب بـ JSON فقط: {{"signal":"BUY/SELL/HOLD","confidence":0-1,"reason":"..."}}
    """
    response = llm.invoke(prompt)
    report = _parse_json(response.content)
    report["agent"] = "analyst"
    log = f"المحلل: {report.get('signal')} ثقة={report.get('confidence', 0):.2f} — {report.get('reason', '')}"
    return {"analyst_report": report, "logs": [log]}

def risk_manager_agent(state: TradingState) -> dict:
    analyst = state["analyst_report"]
    data = state["market_data"]

    if analyst.get("confidence", 0) < MIN_CONFIDENCE:
        return {"risk_report": {"approved": False, "veto": True, "reason": f"ثقة أقل من {MIN_CONFIDENCE}"}, "logs": ["مدير المخاطر: فيتو (ثقة منخفضة)"]}

    rsi = data.get("rsi14", 50)
    sma20, sma50 = data.get("sma20", 0), data.get("sma50", 0)
    last = data.get("last_close", 0)

    if analyst.get("signal") == "BUY" and (rsi > 75 or (sma20 < sma50 and last < sma50)):
        return {"risk_report": {"approved": False, "veto": True, "reason": "مخاطر شراء مرتفعة (RSI/اتجاه)"}, "logs": ["مدير المخاطر: فيتو (مخاطر شراء)"]}
    if analyst.get("signal") == "SELL" and rsi < 25:
        return {"risk_report": {"approved": False, "veto": True, "reason": "مخاطر بيع مرتفعة (RSI)"}, "logs": ["مدير المخاطر: فيتو (مخاطر بيع)"]}

    risk_amount = 10000 * MAX_RISK_PER_TRADE
    stop_distance = last * 0.02
    pos_size = max(1, int(risk_amount / stop_distance)) if stop_distance > 0 else 0
    report = {"approved": True, "veto": False, "position_size": pos_size, "stop_loss": round(last * 0.98, 2), "take_profit": round(last * 1.04, 2)}
    return {"risk_report": report, "logs": [f"مدير المخاطر: موافقة (حجم={pos_size})"]}

def portfolio_manager_agent(state: TradingState) -> dict:
    risk = state["risk_report"]
    if risk.get("veto") or not risk.get("approved"):
        return {"final_decision": "VETO" if risk.get("veto") else "HOLD", "consensus_reached": False, "logs": ["مدير المحفظة: لا تنفيذ (فيتو/رفض)"]}

    signal = state["analyst_report"].get("signal", "HOLD")
    decision = {"BUY": "EXECUTE_BUY", "SELL": "EXECUTE_SELL"}.get(signal, "HOLD")
    order = None
    if decision.startswith("EXECUTE"):
        order = {"symbol": state["symbol"], "side": "buy" if decision == "EXECUTE_BUY" else "sell", "qty": risk.get("position_size", 1), "stop_loss": risk.get("stop_loss"), "take_profit": risk.get("take_profit")}
    return {"final_decision": decision, "order_details": order, "consensus_reached": decision != "HOLD", "logs": [f"مدير المحفظة: {decision} (إجماع مكتمل)"]}

def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group()) if match else json.loads(text)
    except Exception:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "فشل التحليل"}