from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER
from settings import get_all_settings

# توزيع القطاعات
SECTOR_MAP = {
    "AAPL": "tech", "NVDA": "semis", "AMD": "semis", "ASML": "semis",
    "INTC": "semis", "CRM": "software", "ADBE": "software", "NOW": "software",
    "PANW": "software", "TSLA": "auto", "RIVN": "auto",
    "FSLR": "energy", "ENPH": "energy", "JNJ": "health", "MRK": "health",
    "ISRG": "health", "LLY": "health", "PFE": "health",
    "AMZN": "retail", "FDX": "logistics", "UPS": "logistics",
    "MSFT": "tech",
}

# مجموعات الارتباط — أسهم تتحرك معاً
CORRELATION_GROUPS = {
    "tech_mega":  {"AAPL", "MSFT", "AMZN", "CRM", "ADBE", "NOW", "PANW"},
    "semis":      {"NVDA", "AMD", "ASML", "INTC"},
    "ev":         {"TSLA", "RIVN"},
    "clean":      {"FSLR", "ENPH"},
    "healthcare": {"JNJ", "MRK", "ISRG", "LLY", "PFE"},
    "logistics":  {"FDX", "UPS"},
}


def _group_of(symbol: str) -> str:
    for name, syms in CORRELATION_GROUPS.items():
        if symbol in syms:
            return name
    return "other"


def _client():
    from alpaca.trading.client import TradingClient
    return TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)


def _budget_equity(acct_equity: float) -> float:
    """Equity the exposure caps should be measured against.

    When ``trading_capital`` is set it is a hard budget: exposure caps are
    measured against min(budget, real account equity) so a large virtual
    account balance cannot widen the test budget. Falls back to account equity.
    """
    try:
        tc = get_all_settings().get("trading_capital", 0) or 0
    except Exception:
        tc = 0
    if tc > 0:
        return min(tc, acct_equity) if acct_equity > 0 else tc
    return acct_equity


def get_committed_exposure_value() -> float | None:
    """Market value currently committed to open positions.

    Returns ``None`` when it cannot be determined (fail closed) — callers must
    treat that as "budget cannot be proven" and block new entries.
    """
    try:
        positions = _client().get_all_positions()
        return sum(abs(float(p.market_value)) for p in positions)
    except Exception as e:
        print(f"⚠️ risk_engine: committed exposure fetch failed: {e}")
        return None


def get_current_exposure() -> dict:
    """Returns full portfolio exposure breakdown."""
    try:
        acct = _client().get_account()
        equity = float(acct.equity)
        positions = _client().get_all_positions()

        total = 0.0
        by_symbol = {}
        by_group = {}
        for p in positions:
            mv = abs(float(p.market_value))
            total += mv
            by_symbol[p.symbol] = mv
            g = _group_of(p.symbol)
            by_group[g] = by_group.get(g, 0) + mv

        return {
            "equity": equity,
            "total_exposure": total,
            "exposure_pct": (total / equity * 100) if equity > 0 else 0,
            "by_symbol": by_symbol,
            "by_group": by_group,
            "open_count": len(positions),
        }
    except Exception as e:
        print(f"⚠️ risk_engine: exposure fetch failed: {e}")
        return {"equity": 0, "total_exposure": 0, "exposure_pct": 0,
                "by_symbol": {}, "by_group": {}, "open_count": 0}


def check_portfolio_exposure(symbol: str, new_cost: float) -> tuple[bool, str]:
    """
    Returns (allowed, reason).
    Checks per-position cap, total exposure, and correlation concentration.
    Exposure is measured against the effective budget (trading_capital when
    set), never a larger virtual account balance.
    """
    settings = get_all_settings()
    max_total = settings["max_portfolio_exposure_pct"]
    max_position = settings["max_position_pct"]

    exp = get_current_exposure()
    if exp["equity"] <= 0:
        return False, "equity unavailable"
    equity = _budget_equity(exp["equity"])

    # 1) سقف المركز الواحد
    if new_cost > equity * max_position:
        return False, (f"position {new_cost:.2f} > "
                       f"{max_position*100:.0f}% of budget {equity:.2f}")

    # 2) سقف Exposure الكلي
    if exp["total_exposure"] + new_cost > equity * max_total:
        pct = (exp["total_exposure"] + new_cost) / equity * 100
        return False, f"total exposure {pct:.1f}% > {max_total*100:.0f}%"

    # 3) سقف تركيز المجموعة (Correlation)
    group = _group_of(symbol)
    group_exp = exp["by_group"].get(group, 0)
    group_cap = equity * max_total * 0.7  # 70% من الحد الكلي
    if group_exp + new_cost > group_cap:
        pct = (group_exp + new_cost) / equity * 100
        return False, (f"concentration in '{group}' would reach {pct:.1f}% "
                       f"> {group_cap/equity*100:.1f}%")

    return True, "ok"


def get_exposure_summary() -> dict:
    """Human-readable for logs."""
    exp = get_current_exposure()
    return {
        "exposure_pct": round(exp["exposure_pct"], 2),
        "open_positions": exp["open_count"],
        "by_group_pct": {
            g: round(v / exp["equity"] * 100, 2) if exp["equity"] > 0 else 0
            for g, v in exp["by_group"].items()
        },
      }
