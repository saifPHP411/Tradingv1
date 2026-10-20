from langchain_ollama import ChatOllama
from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY,
    MIN_CONFIDENCE, MIN_SIGNAL_SCORE,
)
from settings import get_all_settings
from state import TradingState
from signals import compute_all_indicators, compute_signal_score
from safety import get_account_snapshot
import json, re

_llm_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.1,
    headers=_llm_headers,
    format="json",
)


# =============== ANALYST ===============
def analyst_agent(state: TradingState) -> dict:
    symbol = state["symbol"]
    data = state.get("market_data") or {}

    if not data or "atr14" not in data:
        return {
            "analyst_report": {"signal": "HOLD", "confidence": 0.0, "reason": "no valid indicators"},
            "logs": [f"المحلل[{symbol}]: HOLD — بيانات غير صالحة"],
        }

    sig = compute_signal_score(data)
    if sig["score"] < MIN_SIGNAL_SCORE:
        return {
            "analyst_report": {
                "signal": "HOLD", "confidence": 0.0,
                "reason": f"signal_score={sig['score']} < {MIN_SIGNAL_SCORE}",
                "signal_score": sig["score"],
            },
            "logs": [f"المحلل[{symbol}]: HOLD — score {sig['score']} < {MIN_SIGNAL_SCORE}"],
        }

    memory = state.get("memory_context", "")
    prompt = f"""أنت محلل مالي. حلل {symbol}:
- السعر: {data['last_close']}
- EMA20: {data['ema20']} | EMA50: {data['ema50']}
- RSI14: {data['rsi14']} | ATR14: {data['atr14']}
- Volume ratio: {data['volume_ratio']}
- Momentum 5d: {data['momentum_5d_pct']}% | 10d: {data['momentum_10d_pct']}%
- Signal score: {sig['score']}/100

{memory}

أجب JSON فقط: {{"signal":"BUY"|"SELL"|"HOLD","confidence":0-1,"reason":"سطر واحد"}}"""

    report = {"signal": "HOLD", "confidence": 0.0, "reason": "LLM invalid"}
    for attempt in range(2):
        try:
            suffix = "\n\nJSON فقط بدون أي نص إضافي." if attempt else ""
            resp = llm.invoke(prompt + suffix)
            report = _parse_llm_json(resp.content)
            if report.get("signal") in ("BUY", "SELL", "HOLD") and report.get("confidence", 0) > 0:
                break
        except Exception as e:
            report = {"signal": "HOLD", "confidence": 0.0, "reason": f"LLM error: {e}"}

    report["signal_score"] = sig["score"]
    report["agent"] = "analyst"
    log = f"المحلل[{symbol}]: {report.get('signal')} conf={report.get('confidence', 0):.2f} score={sig['score']} — {report.get('reason', '')[:100]}"
    return {"analyst_report": report, "logs": [log]}


# =============== RISK MANAGER ===============
def risk_manager_agent(state: TradingState) -> dict:
    analyst = state["analyst_report"]
    data = state["market_data"]
    symbol = state["symbol"]
    settings = get_all_settings()

    signal = analyst.get("signal", "HOLD")
    confidence = float(analyst.get("confidence", 0))
    signal_score = analyst.get("signal_score", 0)

    # Rule 1: confidence
    if confidence < MIN_CONFIDENCE:
        return _veto(f"confidence {confidence:.2f} < {MIN_CONFIDENCE}")

    # Rule 2: signal score
    if signal_score < MIN_SIGNAL_SCORE:
        return _veto(f"signal_score {signal_score} < {MIN_SIGNAL_SCORE}")

    # Rule 3: HOLD
    if signal == "HOLD":
        return {"risk_report": {"approved": False, "veto": False, "reason": "HOLD"},
                "logs": [f"مدير المخاطر[{symbol}]: HOLD"]}

    # Rule 4: basic sanity on indicators
    atr = data.get("atr14", 0)
    last = data.get("last_close", 0)
    if atr <= 0 or last <= 0:
        return _veto("ATR or price invalid")

    # Rule 5: RSI extremes
    rsi = data.get("rsi14", 50)
    if signal == "BUY" and rsi > 78:
        return _veto(f"RSI {rsi:.1f} overbought")
    if signal == "SELL" and rsi < 22:
        return _veto(f"RSI {rsi:.1f} oversold")

    # ===== POSITION SIZING =====
    snap = get_account_snapshot()
    equity = snap.get("equity", 0) or 0
    buying_power = snap.get("buying_power", 0) or 0

    # Respect user-defined trading capital cap
    trading_capital = settings.get("trading_capital", 0) or 0
    if trading_capital > 0:
        equity = min(equity, trading_capital)
        buying_power = min(buying_power, trading_capital)

    if equity <= 0:
        return _veto("equity <= 0")

    risk_amount = equity * settings["max_risk_per_trade"]

    # ATR-based stop
    stop_distance = atr * settings["stop_atr_multiplier"]
    if stop_distance <= 0 or stop_distance > last * 0.15:
        return _veto(f"stop distance unreasonable ({stop_distance:.2f})")

    # R-multiple target
    take_profit_distance = stop_distance * settings["take_profit_rr"]

    if signal == "BUY":
        stop_price = last - stop_distance
        take_profit = last + take_profit_distance
    else:  # SELL -> only if we hold a position; caller will filter
        stop_price = last + stop_distance
        take_profit = last - take_profit_distance

    # Core sizing formula
    pos_size = risk_amount / stop_distance
    if pos_size <= 0:
        return _veto("position size <= 0")

    # Cap 1: position notional <= max_position_pct of equity
    max_notional = equity * settings["max_position_pct"]
    if pos_size * last > max_notional:
        pos_size = max_notional / last

    # Cap 2: buying power
    estimated_cost = pos_size * last
    if estimated_cost > buying_power:
        return _veto(f"buying_power insufficient ({estimated_cost:.2f} > {buying_power:.2f})")

    # Minimum meaningful position
    if estimated_cost < 1.0:
        return _veto("position too small (<$1)")

    # Round to 4 decimals (fractional)
    pos_size = round(pos_size, 4)

    # Sanity: expected risk must not exceed limit
    actual_risk = pos_size * stop_distance
    if actual_risk > risk_amount * 1.05:
        return _veto(f"actual risk {actual_risk:.2f} > limit {risk_amount:.2f}")

    report = {
        "approved": True, "veto": False,
        "position_size": pos_size,
        "stop_loss": round(stop_price, 2),
        "take_profit": round(take_profit, 2),
        "entry_price": round(last, 2),
        "atr": round(atr, 4),
        "stop_distance": round(stop_distance, 4),
        "side": "buy" if signal == "BUY" else "sell",
        "estimated_cost": round(estimated_cost, 2),
        "risk_amount": round(actual_risk, 2),
    }
    log = (f"مدير المخاطر[{symbol}]: موافقة | qty={pos_size} "
           f"cost=${estimated_cost:.2f} risk=${actual_risk:.2f} "
           f"stop={stop_price:.2f} tp={take_profit:.2f}")
    return {"risk_report": report, "logs": [log]}


def _veto(reason: str) -> dict:
    return {"risk_report": {"approved": False, "veto": True, "reason": reason},
            "logs": [f"مدير المخاطر: فيتو — {reason}"]}


# =============== PORTFOLIO MANAGER ===============
def portfolio_manager_agent(state: TradingState) -> dict:
    risk = state["risk_report"]
    if risk.get("veto") or not risk.get("approved"):
        decision = "VETO" if risk.get("veto") else "HOLD"
        return {"final_decision": decision, "consensus_reached": False,
                "logs": [f"مدير المحفظة: {decision}"]}

    signal = state["analyst_report"].get("signal", "HOLD")
    decision = {"BUY": "EXECUTE_BUY", "SELL": "EXECUTE_SELL"}.get(signal, "HOLD")

    order = None
    if decision.startswith("EXECUTE"):
        order = {
            "symbol": state["symbol"],
            "side": "buy" if decision == "EXECUTE_BUY" else "sell",
            "qty": risk["position_size"],
            "stop_loss": risk["stop_loss"],
            "take_profit": risk["take_profit"],
            "entry_price": risk["entry_price"],
            "estimated_cost": risk["estimated_cost"],
            "atr": risk["atr"],
        }

    return {"final_decision": decision, "order_details": order,
            "consensus_reached": decision != "HOLD",
            "logs": [f"مدير المحفظة: {decision}"]}


def _parse_llm_json(text) -> dict:
    try:
        if not isinstance(text, str):
            text = str(text)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group()) if m else json.loads(text)
        sig = str(data.get("signal", "HOLD")).upper().strip()
        if sig not in ("BUY", "SELL", "HOLD"):
            sig = "HOLD"
        try:
            conf = float(data.get("confidence", 0))
            conf = max(0.0, min(1.0, conf))
        except Exception:
            conf = 0.0
        reason = str(data.get("reason", ""))[:200]
        return {"signal": sig, "confidence": conf, "reason": reason}
    except Exception:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "invalid JSON"}
