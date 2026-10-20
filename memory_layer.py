from langgraph.store.sqlite import SqliteStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from langchain_ollama import ChatOllama
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
import json

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

# متجر ذاكرة دائم في ملف SQLite (يبقى بعد إغلاق البرنامج)
import os
store = SqliteStore.from_conn_string(os.getenv("MEMORY_DB_PATH", "trading_memory.db"))

def build_memory_tools():
    """أدوات تخزين واسترجاع الذاكرة (LangMem)."""
    manage = create_manage_memory_tool(namespace=("trading", "lessons"))
    search = create_search_memory_tool(namespace=("trading", "lessons"))
    return manage, search

manage_memory, search_memory = build_memory_tools()

def get_similar_experiences(state: dict, k: int = 3) -> str:
    """يبحث عن مواقف مشابهة في الذاكرة ويُرجع سياقًا نصيًا."""
    query = f"symbol={state['symbol']} rsi={state['market_data'].get('rsi14', 0):.1f} trend_sma20_vs_sma50={state['market_data'].get('sma20', 0) - state['market_data'].get('sma50', 0):.2f}"
    try:
        results = search_memory.invoke({"query": query, "limit": k})
        if not results:
            return "لا توجد تجارب مشابهة سابقة."
        lines = []
        for r in results:
            lines.append(f"- تجربة سابقة: {r.get('content', '')}")
        return "تجارب مشابهة من الذاكرة:\n" + "\n".join(lines)
    except Exception as e:
        return f"تعذر استرجاع الذاكرة: {e}"

def reflect_and_store(trade_record: dict):
    """بعد إغلاق الصفقة، نستخدم LLM لاستخلاص درس وتخزينه."""
    prompt = f"""أنت محلل أداء. هذه صفقة سابقة:
    - الرمز: {trade_record['symbol']}
    - القرار: {trade_record['decision']}
    - سبب المحلل: {trade_record.get('analyst_reason', '')}
    - نتيجة الصفقة (P&L%): {trade_record.get('pnl_pct', 'غير معروف')}
    - الظروف: RSI={trade_record.get('rsi', '')}, تغير 5 أيام={trade_record.get('change_5d', '')}%

    اكتب درسًا واحدًا مختصرًا (سطر واحد) يمكن استخدامه في المستقبل. ابدأ بـ "الدرس:".
    """
    response = llm.invoke(prompt)
    lesson = response.content.strip()
    try:
        manage_memory.invoke({"content": lesson})
    except Exception:
        pass
    return lesson
