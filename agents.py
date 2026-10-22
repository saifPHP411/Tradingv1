from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_API_KEY
from settings import get_all_settings
from state import TradingState
from signals import compute_signal_score
from safety import get_account_snapshot
from risk_engine import check_portfolio_exposure, get_committed_exposure_value
from regime import get_market_regime
from news import get_news_sentiment
from risk_utils import (
    validate_risk_plan, normalize_price_for_symbol,
    whole_share_qty, can_protect_with_stop,
)
import json, re

_llm_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
llm = ChatOllama(
    model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
    temperature=0.1, headers=_llm_headers, format="json",
)


def analyst_agent(state: TradingState) -> dict:
    symbol = state["symbol"]
    data = state.get("market_data") or {}

    if not data or "atr14" not in data:
        return {
            "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                               "reason": "no valid indicators", "signal_score": 0,
                               "diagnostic": {"stage": "input", "llm_called": False}},
            "logs": [f"المحلل[{symbol}]: HOLD — بيانات غير صالحة (LLM not called)"],
        }

    settings = get_all_settings()
    regime = get_market_regime() if settings["regime_enabled"] else None
    news = get_news_sentiment(symbol) if settings["news_enabled"] else None
    sig = compute_signal_score(data, regime, news)
    score_parts = {key: sig.get(key, 0) for key in
                   ("trend", "momentum", "rsi", "bb", "volume", "news")}
    regime_name = regime["regime"] if regime else "UNKNOWN"

    if sig["score"] < settings["min_signal_score"]:
        reason = f"signal score below threshold ({sig['score']} < {settings['min_signal_score']})"
        return {
            "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                               "reason": reason, "signal_score": sig["score"],
                               "regime": regime_name,
                               "diagnostic": {"stage": "signal_score", "llm_called": False,
                                              "threshold": settings["min_signal_score"],
                                              "score_components": score_parts}},
            "logs": [f"المحلل[{symbol}]: HOLD — {reason}; components={score_parts}; "
                     f"regime={regime_name}; LLM not called"],
        }

    if regime_name == "BEAR":
        reason = "BEAR market — no new longs"
        return {
            "analyst_report": {"signal": "HOLD", "confidence": 0.0,
                               "reason": reason, "signal_score": sig["score"],
                               "regime": regime_name,
                               "diagnostic": {"stage": "regime", "llm_called": False,
                                              "threshold": settings["min_signal_score"],
                                              "score_components": score_parts}},
            "logs": [f"المحلل[{symbol}]: HOLD — {reason}; score={sig['score']}; "
                     f"LLM not called"],
        }

    memory = state.get("memory_context", "")
    news_line = "غير متوفر"
    if news and news.get("available"):
        news_line = f"sentiment={news['score']:+.2f}, count={news['news_count']}"

    prompt = f"""أنت محلل مالي. حلل {symbol}:

--- Technical ---
- Price: {data['last_close']}
- EMA20: {data['ema20']} | EMA50: {data['ema50']}
- RSI14: {data['rsi14']} | ADX: {data['adx']}
- MACD hist: {data['macd_hist']}
- BB %B: {data['bb_pct_b']} (bandwidth {data['bb_bandwidth']}%)
- Stochastic K/D: {data['stoch_k']}/{data['stoch_d']}
- Vol ratio: {data['volume_ratio']}
- Momentum 5d: {data['momentum_5d_pct']}% | 10d: {data['momentum_10d_pct']}%

--- Market Regime ---
- Regime: {regime_name}
- SPY price: {regime['spy_price'] if regime else 0} (SMA50: {regime['spy_sma50'] if regime else 0})
- Regime ATR%: {regime['atr_pct'] if regime else 0}

--- News ---
- {news_line}

--- Signal Score ---
{sig['score']}/100 (trend={sig['trend']}, momentum={sig['momentum']}, rsi={sig['rsi']}, bb={sig['bb']}, volume={sig['volume']}, news={sig['news']})

{memory}

أجب JSON فقط: {{"signal":"BUY"|"SELL"|"HOLD","confidence":0-1,"reason":"سطر واحد"}}"""

    report = {"signal": "HOLD", "confidence": 0.0, "reason": "LLM invalid"}
    llm_attempts = 0
    llm_status = "invalid_response"
    llm_error_type = None
    for attempt in range(2):
        llm_attempts += 1
        try:
            suffix = "\n\nJSON فقط." if attempt else ""
            resp = llm.invoke(prompt + suffix)
            report = _parse_llm_json(resp.content)
            if report.get("signal") in ("BUY", "SELL", "HOLD") and report.get("confidence", 0) > 0:
                llm_status = "success"
                break
        except Exception as e:
            # Do not copy exception text into reports/logs: HTTP clients may
            # include URLs, headers, or other credential-bearing details.
            llm_status = "error"
            llm_error_type = type(e).__name__
            report = {"signal": "HOLD", "confidence": 0.0,
                      "reason": "LLM request failed"}

    report["signal_score"] = sig["score"]
    report["regime"] = regime_name
    report["agent"] = "analyst"
    report["diagnostic"] = {"stage": "llm", "llm_called": True,
                            "threshold": settings["min_signal_score"],
                            "score_components": score_parts,
                            "llm_attempts": llm_attempts,
                            "llm_status": llm_status}
    if llm_error_type:
        report["diagnostic"]["llm_error_type"] = llm_error_type
    log = (f"المحلل[{symbol}]: {report.get('signal')} "
           f"conf={report.get('confidence', 0):.2f} "
           f"score={sig['score']} regime={regime_name}")
    return {"analyst_report": report, "logs": [log]}


def risk_manager_agent(state: TradingState) -> dict:
    analyst = state["analyst_report"]
    data = state["market_data"]
    symbol = state["symbol"]
    settings = get_all_settings()

    signal = analyst.get("signal", "HOLD")
    confidence = float(analyst.get("confidence", 0))
    signal_score = analyst.get("signal_score", 0)
    regime_name = analyst.get("regime", "UNKNOWN")

    if confidence < settings["min_confidence"]:
        return _veto(symbol, f"confidence {confidence:.2f} < {settings['min_confidence']}")
    if signal_score < settings["min_signal_score"]:
        return _veto(symbol, f"score {signal_score} < {settings['min_signal_score']}")
    if signal == "HOLD":
        return {"risk_report": {"approved": False, "veto": False, "reason": "HOLD"},
                "logs": [f"مدير المخاطر[{symbol}]: HOLD"]}

    atr = data.get("atr14", 0)
    last = data.get("last_close", 0)
    if atr <= 0 or last <= 0:
        return _veto(symbol, "ATR or price invalid")

    rsi = data.get("rsi14", 50)
    if signal == "BUY" and rsi > 78:
        return _veto(symbol, f"RSI {rsi:.1f} overbought")
    if signal == "SELL" and rsi < 22:
        return _veto(symbol, f"RSI {rsi:.1f} oversold")

    snap = get_account_snapshot()
    if not snap.get("available"):
        return _veto(symbol, f"account data unavailable ({snap.get('reason')})")

    acct_equity = snap.get("equity", 0) or 0
    acct_buying_power = snap.get("buying_power", 0) or 0

    # ---- Risk budget ----
    # trading_capital > 0 is a HARD test budget. It caps sizing, per-trade risk
    # and the total committed exposure this process will take, regardless of the
    # (possibly very large) virtual account equity/buying power. It is a cap on
    # capital at risk, NOT a synthetic balance substituted for real broker data:
    # the real account numbers remain an additional, never-relaxed limit
    # (equity = min(budget, account equity), buying_power likewise).
    #
    # Semantics (deterministic, no P&L drift):
    #   budget            = trading_capital  (a hard exposure cap, constant)
    #   committed         = market value of currently open positions
    #   remaining buying  = max(budget - committed, 0)
    #   per-trade risk    = min(budget, account equity) * max_risk_per_trade * mult
    # Realized/unrealized P&L never RAISES the budget. Aggregate drawdown is
    # caught by the daily-loss gate, which is also measured against this budget.
    trading_capital = settings.get("trading_capital", 0) or 0
    if trading_capital > 0:
        committed = get_committed_exposure_value()
        if committed is None:
            # Cannot prove remaining budget → fail closed.
            return _veto(symbol, "open exposure unknown — budget cannot be enforced")
        remaining = max(trading_capital - committed, 0.0)
        equity = min(trading_capital, acct_equity) if acct_equity > 0 else trading_capital
        buying_power = min(remaining, acct_buying_power) \
            if acct_buying_power > 0 else remaining
    else:
        equity = acct_equity
        buying_power = acct_buying_power

    if equity <= 0:
        return _veto(symbol, "equity <= 0")

    risk_multiplier = 1.0
    if regime_name == "HIGH_VOLATILITY":
        risk_multiplier = 0.5
    elif regime_name == "SIDEWAYS":
        risk_multiplier = 0.8
    elif regime_name == "BEAR":
        return _veto(symbol, "BEAR market — no longs")
    elif regime_name == "UNKNOWN":
        risk_multiplier = 0.5

    risk_amount = equity * settings["max_risk_per_trade"] * risk_multiplier
    stop_distance = atr * settings["stop_atr_multiplier"]
    if stop_distance <= 0 or stop_distance > last * 0.20:
        return _veto(symbol, f"stop distance unreasonable ({stop_distance:.2f})")

    target_distance = stop_distance * settings["take_profit_rr"]

    if signal == "BUY":
        stop_price = last - stop_distance
        take_profit = last + target_distance
    else:
        stop_price = last + stop_distance
        take_profit = last - target_distance

    raw_qty = risk_amount / stop_distance
    if raw_qty <= 0:
        return _veto(symbol, "position size <= 0")

    max_notional = equity * settings["max_position_pct"]
    if raw_qty * last > max_notional:
        raw_qty = max_notional / last

    # Whole-share policy: a protective stop requires whole shares at Alpaca.
    # Floor to whole shares; if the budget cannot fund even one whole share
    # (with a valid stop) we SKIP the trade rather than fractional-size it or
    # raise the risk budget.
    pos_size = whole_share_qty(raw_qty)
    ok_prot, why_prot = can_protect_with_stop(
        pos_size, last, stop_price, "buy" if signal == "BUY" else "sell")
    if not ok_prot:
        return _veto(symbol, f"cannot protect: {why_prot} "
                             f"(budget {trading_capital or equity:.2f}, "
                             f"risk/size {raw_qty:.4f})")

    estimated_cost = pos_size * last
    if estimated_cost > buying_power:
        return _veto(symbol, f"buying_power ({estimated_cost:.2f} > {buying_power:.2f})")
    if estimated_cost < 1.0:
        return _veto(symbol, "position too small (<$1)")

    if signal == "BUY":
        exp_ok, exp_reason = check_portfolio_exposure(symbol, estimated_cost)
        if not exp_ok:
            return _veto(symbol, f"exposure: {exp_reason}")

    actual_risk = pos_size * stop_distance
    if actual_risk > risk_amount * 1.05:
        return _veto(symbol, f"actual risk {actual_risk:.2f} > limit {risk_amount:.2f}")

    plan = {
        "symbol": symbol,
        "side": "buy" if signal == "BUY" else "sell",
        "entry_price": normalize_price_for_symbol(symbol, last),
        "stop_loss": normalize_price_for_symbol(symbol, stop_price),
        "take_profit": normalize_price_for_symbol(symbol, take_profit),
        "qty": pos_size,
        "risk_amount": round(actual_risk, 2),
        "atr": round(atr, 4),
        "stop_distance": round(stop_distance, 4),
        "estimated_cost": round(estimated_cost, 2),
        "regime": regime_name,
        "risk_multiplier": risk_multiplier,
    }
    ok, why = validate_risk_plan(plan)
    if not ok:
        return _veto(symbol, f"invalid plan: {why}")

    plan["approved"] = True
    plan["veto"] = False
    log = (f"مدير المخاطر[{symbol}]: موافقة | qty={pos_size} "
           f"cost=${estimated_cost:.2f} risk=${actual_risk:.2f} "
           f"stop={plan['stop_loss']} tp={plan['take_profit']} "
           f"regime={regime_name}")
    return {"risk_report": plan, "logs": [log]}


def _veto(symbol: str, reason: str) -> dict:
    return {"risk_report": {"approved": False, "veto": True, "reason": reason},
            "logs": [f"مدير المخاطر[{symbol}]: فيتو — {reason}"]}


def portfolio_manager_agent(state: TradingState) -> dict:
    risk = state["risk_report"]
    symbol = state.get("symbol", "?")
    if risk.get("veto") or not risk.get("approved"):
        decision = "VETO" if risk.get("veto") else "HOLD"
        return {"final_decision": decision, "consensus_reached": False,
                "logs": [f"مدير المحفظة[{symbol}]: {decision}"]}

    signal = state["analyst_report"].get("signal", "HOLD")
    decision = {"BUY": "EXECUTE_BUY", "SELL": "EXECUTE_SELL"}.get(signal, "HOLD")

    order = None
    if decision.startswith("EXECUTE"):
        order = {
            "symbol": symbol,
            "side": "buy" if decision == "EXECUTE_BUY" else "sell",
            "qty": risk["qty"],
            "stop_loss": risk["stop_loss"],
            "take_profit": risk["take_profit"],
            "entry_price": risk["entry_price"],
            "estimated_cost": risk["estimated_cost"],
            "atr": risk["atr"],
        }
    return {"final_decision": decision, "order_details": order,
            "consensus_reached": decision != "HOLD",
            "logs": [f"مدير المحفظة[{symbol}]: {decision}"]}


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
            conf = max(0.0, min(1.0, float(data.get("confidence", 0))))
        except Exception:
            conf = 0.0
        return {"signal": sig, "confidence": conf,
                "reason": str(data.get("reason", ""))[:200]}
    except Exception:
        return {"signal": "HOLD", "confidence": 0.0, "reason": "invalid JSON"}
