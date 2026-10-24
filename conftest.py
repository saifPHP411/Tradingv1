"""Offline test support for the whole test suite.

The application imports Alpaca, LangChain/LangGraph, LangMem and FastAPI, which
are not installed in the test image and must never be contacted from tests.
This conftest installs minimal *stub* modules — only when the real package is
absent — so ``monitor.py``, ``app.py``, ``broker.py``, ``agents.py`` and
``data_layer.py`` can be imported and exercised with fake clients.

No network access is performed, and no real order can be submitted: the stubs
raise if a test accidentally reaches a real broker call without patching it.

It also redirects the SQLite artifacts (decisions.db, graph/memory stores) to a
temporary directory so the repository tree stays clean.
"""
import importlib.util
import os
import sys
import tempfile
import types

# ---- Keep generated SQLite artifacts out of the repository ----
_TMP = tempfile.mkdtemp(prefix="tradingv1-tests-")
os.environ.setdefault("DB_PATH", os.path.join(_TMP, "decisions.db"))
os.environ.setdefault("GRAPH_DB_PATH", os.path.join(_TMP, "graph_checkpoints.db"))
os.environ.setdefault("MEMORY_DB_PATH", os.path.join(_TMP, "trading_memory.db"))


def _installed(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _module(name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
    return mod


def _ensure_pkg(name: str):
    if not _installed(name):
        _module(name)


def _make_enum(**members):
    class _E:
        def __init__(self, n):
            self._n = n

        def __str__(self):
            return self._n

        __repr__ = __str__

    obj = types.SimpleNamespace()
    for k, v in members.items():
        setattr(obj, k, _E(v))
    return obj


def _install_alpaca():
    if _installed("alpaca"):
        return
    _ensure_pkg("alpaca")
    trading = _module("alpaca.trading")
    _ensure_pkg("alpaca.trading")

    class TradingClient:
        def __init__(self, key=None, secret=None, paper=True, **kw):
            self.key, self.secret, self.paper = key, secret, paper

        def _fail(self, *_a, **_k):
            raise AssertionError(
                "test reached a real Alpaca call — patch _client instead")

        get_account = get_all_positions = get_orders = get_order_by_id = _fail
        submit_order = cancel_order_by_id = close_position = _fail

    client_mod = _module("alpaca.trading.client")
    client_mod.TradingClient = TradingClient
    sys.modules["alpaca.trading.client"] = client_mod

    req = _module("alpaca.trading.requests")

    class _Req:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    req.MarketOrderRequest = req.StopOrderRequest = _Req
    req.GetOrdersRequest = _Req
    sys.modules["alpaca.trading.requests"] = req

    enums = _module("alpaca.trading.enums")
    enums.OrderSide = _make_enum(BUY="buy", SELL="sell")
    enums.TimeInForce = _make_enum(DAY="day", GTC="gtc")
    enums.QueryOrderStatus = _make_enum(OPEN="open", ALL="all", CLOSED="closed")
    enums.OrderClass = _make_enum(SIMPLE="simple")
    sys.modules["alpaca.trading.enums"] = enums


def _install_langchain_ollama():
    if _installed("langchain_ollama"):
        return
    mod = _module("langchain_ollama")

    class ChatOllama:
        def __init__(self, *a, **k):
            pass

        def invoke(self, *a, **k):
            raise AssertionError("test reached a real LLM call")

    mod.ChatOllama = ChatOllama


def _install_langgraph():
    if _installed("langgraph"):
        return
    _ensure_pkg("langgraph")

    graph = _module("langgraph.graph")
    graph.START = object()
    graph.END = object()

    class StateGraph:
        def __init__(self, *a, **k):
            pass

        def add_node(self, *a, **k):
            pass

        def add_edge(self, *a, **k):
            pass

        def compile(self, *a, **k):
            return self

        def invoke(self, *a, **k):
            return {}

    graph.StateGraph = StateGraph
    sys.modules["langgraph.graph"] = graph

    _ensure_pkg("langgraph.checkpoint")
    ckpt = _module("langgraph.checkpoint.sqlite")

    class SqliteSaver:
        def __init__(self, conn=None, *a, **k):
            pass

        def setup(self):
            pass

    ckpt.SqliteSaver = SqliteSaver
    sys.modules["langgraph.checkpoint.sqlite"] = ckpt

    _ensure_pkg("langgraph.store")
    store = _module("langgraph.store.sqlite")

    class SqliteStore:
        def __init__(self, conn=None, *a, **k):
            pass

        def setup(self):
            pass

    store.SqliteStore = SqliteStore
    sys.modules["langgraph.store.sqlite"] = store


def _install_langmem():
    if _installed("langmem"):
        return
    mod = _module("langmem")
    mod.create_manage_memory_tool = lambda *a, **k: object()
    mod.create_search_memory_tool = lambda *a, **k: object()


def _install_finnhub():
    if _installed("finnhub"):
        return
    mod = _module("finnhub")

    class Client:
        def __init__(self, *a, **k):
            pass

        def company_news(self, *a, **k):
            raise AssertionError("test reached a real finnhub call")

    mod.Client = Client


def _guard_gemini_network():
    """Make the advisory AI layer fail closed in tests.

    The real transport is replaced so that no test, anywhere in the suite, can
    reach the Gemini API. Tests that need a reply inject their own transport.
    """
    try:
        import market_context as _mc
    except Exception:
        return

    def _blocked(*a, **k):
        raise AssertionError("test attempted a real Gemini network call")

    _mc._post_generate_content = _blocked


def _install_fastapi():
    if _installed("fastapi"):
        return
    _ensure_pkg("fastapi")
    mod = _module("fastapi")

    class FastAPI:
        def __init__(self, *a, **k):
            self.routes = []

        def get(self, *a, **k):
            def deco(fn):
                return fn
            return deco

        post = get

    class Request:
        pass

    mod.FastAPI = FastAPI
    mod.Request = Request

    responses = _module("fastapi.responses")

    class HTMLResponse(str):
        pass

    responses.HTMLResponse = HTMLResponse
    sys.modules["fastapi.responses"] = responses

    templating = _module("fastapi.templating")

    class Jinja2Templates:
        def __init__(self, *a, **k):
            pass

        def TemplateResponse(self, *a, **k):
            return ""

    templating.Jinja2Templates = Jinja2Templates
    sys.modules["fastapi.templating"] = templating


def _install_apscheduler():
    if _installed("apscheduler"):
        return
    _ensure_pkg("apscheduler")
    _ensure_pkg("apscheduler.schedulers")
    mod = _module("apscheduler.schedulers.blocking")

    class BlockingScheduler:
        def __init__(self, *a, **k):
            pass

        def add_job(self, *a, **k):
            pass

        def start(self):
            raise AssertionError("test reached a real scheduler start")

    mod.BlockingScheduler = BlockingScheduler
    sys.modules["apscheduler.schedulers.blocking"] = mod


_install_alpaca()
_install_langchain_ollama()
_install_langgraph()
_install_langmem()
_install_finnhub()
_guard_gemini_network()
_install_fastapi()
_install_apscheduler()


# ---------------------------------------------------------------------------
# Shared fake Alpaca client for offline order-path tests.
# ---------------------------------------------------------------------------
class FakeOrder:
    def __init__(self, oid, symbol, qty, side, order_type="market",
                 status="new", filled_qty=0.0, filled_avg_price=0.0,
                 stop_price=None):
        self.id = oid
        self.symbol = symbol
        self.qty = qty
        self.side = side
        self.order_type = order_type
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = filled_avg_price
        self.stop_price = stop_price
        self.order_class = "simple"
        self.filled_at = None


class FakeAccount:
    def __init__(self, buying_power=1_000_000.0, equity=1_000_000.0):
        self.buying_power = str(buying_power)
        self.equity = str(equity)
        self.last_equity = str(equity)


def filled_order(oid, qty, price, symbol="AAPL", side="buy"):
    return FakeOrder(oid, symbol, qty, side, "market", "filled",
                     filled_qty=qty, filled_avg_price=price)


def stop_order(oid, symbol, qty, side, stop_price, status="new"):
    return FakeOrder(oid, symbol, qty, side, "stop", status,
                     stop_price=stop_price)


class FakeClient:
    """In-memory stand-in for Alpaca's TradingClient. No network."""

    def __init__(self, orders=None, open_orders=None, positions=None,
                 account=None, order_lookup_raises=False,
                 submit_raises=False, stop_raises=False,
                 auto_fill_entry=True, fill_price=100.0):
        self._orders = {o.id: o for o in (orders or [])}
        self._open = list(open_orders or [])
        self._positions = list(positions or [])
        self._account = account or FakeAccount()
        self.order_lookup_raises = order_lookup_raises
        self.submit_raises = submit_raises
        self.stop_raises = stop_raises
        self.auto_fill_entry = auto_fill_entry
        self.fill_price = fill_price
        self.submitted = []
        self.cancelled = []
        self.closed = []
        self._seq = 1000

    def _next_id(self, prefix):
        self._seq += 1
        return f"{prefix}{self._seq}"

    def get_account(self):
        return self._account

    def get_all_positions(self):
        return list(self._positions)

    def get_order_by_id(self, oid):
        if self.order_lookup_raises:
            raise RuntimeError("api down")
        o = self._orders.get(oid)
        if o is None:
            raise KeyError(oid)
        return o

    def get_orders(self, req=None):
        return list(self._open)

    def submit_order(self, req):
        self.submitted.append(req)
        if getattr(req, "stop_price", None) is not None:
            if self.stop_raises:
                raise RuntimeError("stop rejected by broker")
            oid = self._next_id("S")
            o = FakeOrder(oid, req.symbol, req.qty, req.side, "stop", "new",
                          stop_price=req.stop_price)
            self._orders[oid] = o
            self._open.append(o)
            return o
        if self.submit_raises:
            raise RuntimeError("entry rejected by broker")
        oid = self._next_id("E")
        if self.auto_fill_entry:
            o = FakeOrder(oid, req.symbol, req.qty, req.side, "market",
                          "filled", filled_qty=req.qty,
                          filled_avg_price=self.fill_price)
        else:
            o = FakeOrder(oid, req.symbol, req.qty, req.side, "market",
                          "accepted")
        self._orders[oid] = o
        return o

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)
        o = self._orders.get(oid)
        if o is not None and "filled" not in o.status:
            o.status = "canceled"

    def close_position(self, symbol):
        self.closed.append(symbol)
        self._positions = [p for p in self._positions if p.symbol != symbol]
