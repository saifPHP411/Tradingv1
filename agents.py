from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY
from settings import get_all_settings
from state import TradingState
import json, re

_llm_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    headers=_llm_headers,
    format="json",
)


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

    report = {"signal": "HOLD", "confidence": 0.0, "reason": "فشل"}
    for attempt in range(2):
        try:
            suffix = "\n\nأعد الإجابة بصيغة JSON صالحة فقط بدون أي نص إضافي." if attempt else ""
            response = llm.invoke(prompt + suffix)
            report = _parse_json(response.content)
            if report.get("confidence", 0) > 0 and report.get("signal") in ("BUY", "SELL", "HOLD"):
                break
        except Exception as e:
            report = {"signal": "HOLD", "confidence": 0.0, "reason": f"خطأ في الاتصال: {e}"}

    report["agent"] = "analyst"
    log = f"المحلل: {report.get('signal')} ثقة={report.get('confidence', 0):.2f} — {report.get('reason', '')[:120]}"
    return {"analyst_report": report, "logs": [log]}


def risk_manager_agent(state: TradingState) -> dict:
    analyst = state["analyst_report"]
    data = state["market_data"]
    symbol = state["symbol"]
    settings = get_all_settings()
    min_conf = settings["min_confidence"]
    max_risk = settings["max_risk_per_trade"]
    signal = analyst.get("signal", "HOLD")

    if analyst.get("confidence", 0) < min_conf:
        return {"risk_report": {"approved": False, "veto": True,
            "reason": f"ثقة ({analyst.get('confidence', 0):.2f}) < حد ({min_conf:.2f})"},
            "logs": [f"مدير المخاطر: فيتو — ثقة أقل من {min_conf:.2f}"]}

    if signal == "HOLD":
        return {"risk_report": {"approved": False, "veto": False, "reason": "إشارة HOLD"},
                "logs": ["مدير المخاطر: HOLD — لا صفقة"]}

    rsi = data.get("rsi14", 50)
    sma20 = data.get("sma20", 0)
    sma50 = data.get("sma50", 0)
    last = data.get("last_close", 0)

    if signal == "BUY" and (rsi > 75 or (sma20 < sma50 and last < sma50)):
        return {"risk_report": {"approved": False, "veto": True, "reason": "مخاطر شراء مرتفعة (RSI/اتجاه)"},
                "logs": ["مدير المخاطر: فيتو (RSI/اتجاه)"]}

    from safety import get_account_snapshot
    snap = get_account_snapshot()
    real_equity = snap.get("equity", 10000) or 10000
    real_buying_power = snap.get("buying_power", real_equity) or real_equity

    # تطبيق رأس المال المخصص
    trading_capital = settings.get("trading_capital", 0) or 0
    if trading_capital > 0:
        equity = min(real_equity, trading_capital)
        buying_power = min(real_buying_power, trading_capital)
        capital_note = f" (رأس مال ${trading_capital:.0f})"
    else:
        equity = real_equity
        buying_power = real_buying_power
        capital_note = ""

    # جلب المركز الحالي
    current_qty = 0.0
    try:
        from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            from alpaca.trading.client import TradingClient
            client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
            for p in client.get_all_positions():
                if p.symbol == symbol:
                    current_qty = float(p.qty)
                    break
    except Exception as e:
        print(f"⚠️ تعذر جلب المراكز: {e}")

    # === SELL ===
    if signal == "SELL":
        if current_qty <= 0:
            return {"risk_report": {"approved": False, "veto": False,
                "reason": f"لا مركز في {symbol} لإغلاقه"},
                "logs": [f"مدير المخاطر: تجاهل SELL — لا مركز لإغلاقه"]}
        order_qty = round(current_qty, 4)  # كسري
        return {"risk_report": {"approved": True, "veto": False,
            "position_size": order_qty,
            "stop_loss": round(last * 1.02, 2),
            "take_profit": round(last * 0.96, 2),
            "entry_price": round(last, 2), "side": "sell",
            "estimated_cost": round(order_qty * last, 2)},
            "logs": [f"مدير المخاطر: موافقة إغلاق {symbol} ({order_qty} سهم){capital_note}"]}

    # === BUY ===
    if signal == "BUY":
        stop_loss = round(last * 0.98, 2)
        take_profit = round(last * 1.04, 2)
        risk_amount = equity * max_risk
        stop_distance = abs(last - stop_loss)

        # الحجم بناءً على المخاطرة
        pos_size_by_risk = risk_amount / stop_distance if stop_distance > 0 else 0

        # سقف: 25% من رأس المال لكل صفقة
        max_pos_value = equity * 0.25
        pos_size_by_cap = max_pos_value / last if last > 0 else 0

        # الحجم النهائي = الأصغر بينهما، بدقة 4 منازل عشرية (كسرية)
        pos_size = min(pos_size_by_risk, pos_size_by_cap)
        pos_size = round(pos_size, 4)

        # ضمان حد أدنى 0.001 سهم وقيمة صفقة $1 على الأقل
        estimated_cost = pos_size * last
        if estimated_cost < 1.0:
            # نحاول رفع الحجم إلى الحد الأدنى $1
            pos_size = round(1.0 / last, 4) if last > 0 else 0
            estimated_cost = pos_size * last

        if estimated_cost > buying_power:
            return {"risk_report": {"approved": False, "veto": True,
                "reason": f"قوة شرائية غير كافية (تحتاج ~${estimated_cost:.2f}, متاح ${buying_power:.2f}){capital_note}"},
                "logs": [f"مدير المخاطر: فيتو — ${buying_power:.2f} < ${estimated_cost:.2f}{capital_note}"]}

        if pos_size <= 0:
            return {"risk_report": {"approved": False, "veto": True, "reason": "حجم صفر"},
                    "logs": ["مدير المخاطر: فيتو (حجم صفر)"]}

        return {"risk_report": {"approved": True, "veto": False,
            "position_size": pos_size,
            "stop_loss": stop_loss, "take_profit": take_profit,
            "entry_price": round(last, 2), "side": "buy",
            "estimated_cost": round(estimated_cost, 2)},
            "logs": [f"مدير المخاطر: موافقة شراء ({pos_size} سهم, ~${estimated_cost:.2f}){capital_note}"]}

    return {"risk_report": {"approved": False, "veto": True, "reason": "إشارة غير معروفة"},
            "logs": ["مدير المخاطر: فيتو (إشارة غير معروفة)"]}


def portfolio_manager_agent(state: TradingState) -> dict:
    risk = state["risk_report"]
    if risk.get("veto") or not risk.get("approved"):
        decision = "VETO" if risk.get("veto") else "HOLD"
        return {
            "final_decision": decision,
            "consensus_reached": False,
            "logs": [f"مدير المحفظة: لا تنفيذ ({decision})"]
        }

    signal = state["analyst_report"].get("signal", "HOLD")
    decision = {"BUY": "EXECUTE_BUY", "SELL": "EXECUTE_SELL"}.get(signal, "HOLD")

    order = None
    if decision.startswith("EXECUTE"):
        order = {
            "symbol": state["symbol"],
            "side": "buy" if decision == "EXECUTE_BUY" else "sell",
            "qty": risk.get("position_size", 1),
            "stop_loss": risk.get("stop_loss"),
            "take_profit": risk.get("take_profit"),
            "entry_price": risk.get("entry_price"),
            "estimated_cost": risk.get("estimated_cost"),
        }

    return {
        "final_decision": decision,
        "order_details": order,
        "consensus_reached": decision != "HOLD",
        "logs": [f"مدير المحفظة: {decision}"]
    }


def _parse_json(text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        parsed = json.loads(match.group()) if match else json.loads(text)
        # تطبيع الإشارة
        if isinstance(parsed.get("signal"), str):
            parsed["signal"] = parsed["signal"].upper().strip()
        # تطبيع الثقة
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0))
        except (ValueError, TypeError):
            parsed["confidence"] = 0.0
        return parsed
    except Exception:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "فشل التحليل"}
