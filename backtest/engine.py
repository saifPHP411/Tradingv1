import numpy as np
import pandas as pd
from datetime import datetime

from signals import (
    compute_rsi_wilder, compute_atr, compute_sma, compute_ema,
    compute_macd, compute_bollinger, compute_adx, compute_stochastic,
)
from config import (
    STOP_ATR_MULTIPLIER, TAKE_PROFIT_RR,
    MAX_RISK_PER_TRADE, MAX_POSITION_PCT,
    MIN_SIGNAL_SCORE, MIN_CONFIDENCE,
    FEES_PER_SHARE, ENTRY_SLIPPAGE_BPS, EXIT_SLIPPAGE_BPS,
)


# ============ Indicator extraction for backtest ============
def compute_indicators_at(df: pd.DataFrame, i: int) -> dict | None:
    """Compute indicators using ONLY data up to index i (no look-ahead)."""
    if i < 60:
        return None

    window = df.iloc[: i + 1]
    closes = window["Close"].values.astype(float)
    highs = window["High"].values.astype(float)
    lows = window["Low"].values.astype(float)
    volumes = window["Volume"].values.astype(float)

    if len(closes) < 60:
        return None

    rsi = compute_rsi_wilder(closes)
    atr = compute_atr(highs, lows, closes)
    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, 50)
    ema20 = compute_ema(closes, 20)
    ema50 = compute_ema(closes, 50)
    macd_line, macd_sig, macd_hist = compute_macd(closes)
    pct_b, bb_bw = compute_bollinger(closes)
    adx = compute_adx(highs, lows, closes)
    k, d = compute_stochastic(highs, lows, closes)

    if any(v is None for v in [rsi, atr, sma20, sma50, ema20, ema50, adx]):
        return None

    vol_mean = float(np.mean(volumes[-20:]))
    vol_ratio = float(volumes[-1] / vol_mean) if vol_mean > 0 else 1.0
    mom5 = float((closes[-1] / closes[-6] - 1) * 100) if len(closes) >= 6 else 0.0
    mom10 = float((closes[-1] / closes[-11] - 1) * 100) if len(closes) >= 11 else 0.0

    return {
        "last_close": float(closes[-1]),
        "atr14": float(atr),
        "rsi14": float(rsi),
        "sma20": float(sma20), "sma50": float(sma50),
        "ema20": float(ema20), "ema50": float(ema50),
        "macd_hist": float(macd_hist) if macd_hist is not None else 0.0,
        "bb_pct_b": float(pct_b) if pct_b is not None else 0.5,
        "adx": float(adx),
        "stoch_k": float(k) if k is not None else 50.0,
        "volume_ratio": float(vol_ratio),
        "momentum_5d_pct": float(mom5),
        "momentum_10d_pct": float(mom10),
    }


def compute_regime_at(spy_df: pd.DataFrame, ts) -> dict:
    """Regime using SPY data up to timestamp ts (no look-ahead)."""
    if spy_df is None:
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}

    try:
        window = spy_df.loc[:ts]
    except Exception:
        window = spy_df.iloc[:0]

    if len(window) < 200:
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}

    closes = window["Close"].values.astype(float)
    highs = window["High"].values.astype(float)
    lows = window["Low"].values.astype(float)

    price = float(closes[-1])
    sma50 = compute_sma(closes, 50)
    sma200 = compute_sma(closes, 200)
    atr = compute_atr(highs, lows, closes)
    rsi = compute_rsi_wilder(closes)

    if any(v is None for v in [sma50, sma200, atr, rsi]):
        return {"regime": "UNKNOWN", "size_multiplier": 0.5}

    atr_pct = (atr / price) * 100 if price > 0 else 0

    if atr_pct >= 3.0:
        return {"regime": "HIGH_VOLATILITY", "size_multiplier": 0.4}
    if atr_pct >= 2.2 and rsi < 40:
        return {"regime": "HIGH_VOLATILITY", "size_multiplier": 0.5}
    if price > sma50 and sma50 > sma200 and rsi > 50:
        return {"regime": "BULL", "size_multiplier": 1.0}
    if price < sma50 and sma50 < sma200:
        return {"regime": "BEAR", "size_multiplier": 0.0}
    return {"regime": "SIDEWAYS", "size_multiplier": 0.7}


def compute_score(indicators: dict, regime: dict) -> int:
    """Composite signal score 0-100, matching live signals.py logic."""
    trend = 0
    if indicators["ema20"] > indicators["ema50"]:
        trend += 12
    if indicators["last_close"] > indicators["ema20"]:
        trend += 8
    if indicators["last_close"] > indicators["sma50"]:
        trend += 5
    if indicators["adx"] > 25:
        trend += 5

    momentum = 0
    if indicators["momentum_5d_pct"] > 0:
        momentum += 8
    if indicators["momentum_10d_pct"] > 0:
        momentum += 5
    if indicators["macd_hist"] > 0:
        momentum += 7

    rsi = indicators["rsi14"]
    if 40 <= rsi <= 70:
        rsi_score = 15
    elif 30 <= rsi < 40:
        rsi_score = 10
    elif 70 < rsi <= 78:
        rsi_score = 8
    else:
        rsi_score = 0

    bb = 0
    if 0.2 <= indicators["bb_pct_b"] <= 0.8:
        bb += 8
    if indicators["bb_pct_b"] > 0.9:
        bb -= 5
    if indicators["bb_pct_b"] < 0.1:
        bb += 3
    bb = max(0, bb)

    if indicators["volume_ratio"] > 1.2:
        vol = 12
    elif indicators["volume_ratio"] > 0.8:
        vol = 8
    else:
        vol = 3

    raw = trend + momentum + rsi_score + bb + vol
    raw = max(0, min(100, raw))

    r = regime.get("regime", "UNKNOWN")
    if r == "BEAR":
        raw = int(raw * 0.5)
    elif r == "HIGH_VOLATILITY":
        raw = int(raw * 0.75)
    elif r == "SIDEWAYS":
        raw = int(raw * 0.9)

    return raw


# ============ Position dataclass ============
class Position:
    def __init__(self, symbol, entry_price, qty, stop, target, entry_time, entry_idx):
        self.symbol = symbol
        self.entry_price = entry_price
        self.qty = qty
        self.stop = stop
        self.initial_stop = stop
        self.target = target
        self.entry_time = entry_time
        self.entry_idx = entry_idx
        self.highest_close = entry_price
        self.trailing_active = False
        self.bars_held = 0

    def risk_per_share(self):
        return abs(self.entry_price - self.initial_stop)


# ============ Main simulation ============
def simulate(
    symbol: str,
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None,
    start_equity: float = 10000.0,
    trailing_activation_pct: float = 2.0,
    trailing_distance_pct: float = 1.5,
    max_hold_bars: int = 60,
) -> dict:
    """
    Candle-by-candle simulation with realistic fills, fees, slippage.
    
    Entry: at NEXT bar's open + slippage (realistic)
    Exit: intrabar check (SL first if both hit — conservative)
    Trailing: activates at +X%, moves stop by Y% from highest close
    """
    equity = start_equity
    equity_curve = []
    trades = []
    position: Position | None = None
    pending_entry = None  # {stop_dist, qty, target_dist, signal_bar_idx}

    n_bars = len(df)
    for i in range(60, n_bars):
        row = df.iloc[i]
        ts = df.index[i]
        open_p = float(row["Open"])
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])

        # ---- 1) Execute pending entry at this bar's open ----
        if pending_entry is not None:
            slip = pending_entry["entry_ref"] * (ENTRY_SLIPPAGE_BPS / 10000.0)
            fill_price = open_p + slip  # entry always long, so slip up

            stop_price = fill_price - pending_entry["stop_dist"]
            target_price = fill_price + pending_entry["target_dist"]
            qty = pending_entry["qty"]

            cost = fill_price * qty
            fees = qty * FEES_PER_SHARE
            equity -= fees

            if cost <= equity + 0.01:
                position = Position(
                    symbol=symbol,
                    entry_price=fill_price,
                    qty=qty,
                    stop=stop_price,
                    target=target_price,
                    entry_time=ts,
                    entry_idx=i,
                )
                trade_record = {
                    "symbol": symbol,
                    "side": "BUY",
                    "entry_time": ts,
                    "entry_price": fill_price,
                    "qty": qty,
                    "initial_stop": stop_price,
                    "target": target_price,
                    "signal_bar_idx": pending_entry["signal_bar_idx"],
                    "regime": pending_entry.get("regime", ""),
                    "score": pending_entry.get("score", 0),
                    "fees_entry": fees,
                    "slippage_entry": slip * qty,
                }
            pending_entry = None

        # ---- 2) Manage open position ----
        if position is not None:
            position.bars_held += 1
            exit_reason = None
            exit_price = None

            # Check SL first (conservative — worse assumption)
            if low <= position.stop:
                exit_price = position.stop
                exit_reason = "STOP"
                if open_p < position.stop:
                    # gapped down past stop
                    exit_price = open_p
                    exit_reason = "STOP_GAP"

            # Then TP (only if SL not hit this bar)
            elif high >= position.target:
                exit_price = position.target
                exit_reason = "TARGET"

            # Max hold time
            elif position.bars_held >= max_hold_bars:
                exit_price = close
                exit_reason = "MAX_HOLD"

            if exit_price is not None:
                # Apply exit slippage
                slip_out = exit_price * (EXIT_SLIPPAGE_BPS / 10000.0)
                final_exit = exit_price - slip_out  # sold below
                fees_out = position.qty * FEES_PER_SHARE

                gross_pnl = (final_exit - position.entry_price) * position.qty
                net_pnl = gross_pnl - fees_out
                equity += net_pnl

                trade_record["exit_time"] = ts
                trade_record["exit_price"] = final_exit
                trade_record["exit_reason"] = exit_reason
                trade_record["gross_pnl"] = round(gross_pnl, 4)
                trade_record["fees_exit"] = round(fees_out, 4)
                trade_record["slippage_exit"] = round(slip_out * position.qty, 4)
                trade_record["net_pnl"] = round(net_pnl, 4)
                trade_record["r_multiple"] = round(
                    (final_exit - position.entry_price) / position.risk_per_share(), 3
                ) if position.risk_per_share() > 0 else 0
                trade_record["bars_held"] = position.bars_held
                trades.append(trade_record)
                position = None

            else:
                # Update trailing stop (only if in profit)
                if close > position.highest_close:
                    position.highest_close = close

                unrealized_pct = (close - position.entry_price) / position.entry_price * 100

                if unrealized_pct >= trailing_activation_pct:
                    new_stop = position.highest_close * (1 - trailing_distance_pct / 100)
                    if new_stop > position.stop:
                        position.stop = new_stop

        # ---- 3) Look for new signals (if no position, no pending) ----
        if position is None and pending_entry is None:
            ind = compute_indicators_at(df, i)
            if ind is None:
                equity_curve.append({"time": ts, "equity": equity})
                continue

            regime = compute_regime_at(spy_df, ts)
            score = compute_score(ind, regime)

            if score >= MIN_SIGNAL_SCORE and regime["regime"] != "BEAR":
                atr = ind["atr14"]
                stop_dist = atr * STOP_ATR_MULTIPLIER
                target_dist = stop_dist * TAKE_PROFIT_RR

                if stop_dist > 0 and stop_dist < close * 0.15:
                    risk_mult = regime.get("size_multiplier", 1.0)
                    risk_amount = equity * MAX_RISK_PER_TRADE * risk_mult
                    qty = risk_amount / stop_dist

                    # Cap by position %
                    max_notional = equity * MAX_POSITION_PCT
                    if qty * close > max_notional:
                        qty = max_notional / close

                    if qty > 0 and qty * close >= 1.0:
                        pending_entry = {
                            "qty": qty,
                            "stop_dist": stop_dist,
                            "target_dist": target_dist,
                            "entry_ref": close,  # use close for sizing; fill = next open
                            "signal_bar_idx": i,
                            "regime": regime["regime"],
                            "score": score,
                        }

        equity_curve.append({"time": ts, "equity": round(equity, 2)})

    # Force-close open position at end
    if position is not None:
        final_close = float(df.iloc[-1]["Close"])
        slip_out = final_close * (EXIT_SLIPPAGE_BPS / 10000.0)
        final_exit = final_close - slip_out
        fees_out = position.qty * FEES_PER_SHARE
        gross_pnl = (final_exit - position.entry_price) * position.qty
        net_pnl = gross_pnl - fees_out
        equity += net_pnl
        trade_record["exit_time"] = df.index[-1]
        trade_record["exit_price"] = final_exit
        trade_record["exit_reason"] = "END_OF_DATA"
        trade_record["gross_pnl"] = round(gross_pnl, 4)
        trade_record["fees_exit"] = round(fees_out, 4)
        trade_record["net_pnl"] = round(net_pnl, 4)
        trade_record["r_multiple"] = round(
            (final_exit - position.entry_price) / position.risk_per_share(), 3
        ) if position.risk_per_share() > 0 else 0
        trade_record["bars_held"] = position.bars_held
        trades.append(trade_record)

    return {
        "symbol": symbol,
        "start_equity": start_equity,
        "end_equity": round(equity, 2),
        "trades": trades,
        "equity_curve": equity_curve,
  }
