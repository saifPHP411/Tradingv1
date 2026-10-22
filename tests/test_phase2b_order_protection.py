"""
Phase 2B regression tests — Part 2 (order protection, fill reconciliation,
stop verification, kill switch).

All broker interactions use the in-memory ``FakeClient`` from conftest.
No network, no real Alpaca contact, no order submission.
"""
import pytest

import broker
from types import SimpleNamespace

from conftest import FakeClient, FakeOrder, filled_order, stop_order

def _wire(monkeypatch, client, gates=(True, "ok")):
    monkeypatch.setattr(broker, "_client", lambda: client)
    monkeypatch.setattr(broker, "check_safety_gates", lambda: gates)
    monkeypatch.setattr(broker, "is_dry_run", lambda: False)
    monkeypatch.setattr(broker, "broker_submission_allowed", lambda: True)
    monkeypatch.setattr(broker, "ALPACA_API_KEY", "k")
    monkeypatch.setattr(broker, "ALPACA_SECRET_KEY", "s")
    monkeypatch.setattr(broker, "is_kill_switch_active", lambda: (False, ""))


def _entry_order(**over):
    o = {"symbol": "AAPL", "side": "buy", "qty": 5, "stop_loss": 90.0,
         "take_profit": 120.0, "estimated_cost": 500.0}
    o.update(over)
    return o


def _patch_reconcile(monkeypatch, **state):
    monkeypatch.setattr(broker, "_reconcile_order",
                        lambda oid, max_wait_sec=30: state)


# ===========================================================================
# 5) fill-timeout reconciliation
# ===========================================================================
def test_late_full_fill_is_not_treated_as_no_fill(monkeypatch):
    # Reconcile times out as "unresolved", but the final re-read shows a full
    # fill. The timeout must NOT be read as "did not fill".
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="unresolved", status="new")

    orig_submit = client.submit_order

    def submit_then_fill(req):
        o = orig_submit(req)
        if getattr(req, "stop_price", None) is None:
            o.status = "filled"
            o.filled_qty = req.qty
            o.filled_avg_price = 100.0
        return o

    monkeypatch.setattr(client, "submit_order", submit_then_fill)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "submitted"
    assert res["qty"] == 5
    assert client.cancelled, "cancel must be attempted on an unresolved order"


def test_partial_fill_is_protected(monkeypatch):
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="unresolved", status="partially_filled")

    orig_submit = client.submit_order

    def submit_then_partial(req):
        o = orig_submit(req)
        if getattr(req, "stop_price", None) is None:
            o.status = "partially_filled"
            o.filled_qty = 3.0
            o.filled_avg_price = 100.0
        return o

    monkeypatch.setattr(client, "submit_order", submit_then_partial)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "submitted"
    assert res["qty"] == 3


def test_unresolved_state_fails_closed_and_is_visible(monkeypatch):
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="unresolved", status="accepted")

    def cancel_noop(oid):
        client.cancelled.append(oid)

    monkeypatch.setattr(client, "cancel_order_by_id", cancel_noop)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "unresolved"
    assert "no retry" in res["reason"]


def test_api_uncertainty_does_not_submit_duplicate(monkeypatch):
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="unresolved", status="unknown")

    def cancel_noop(oid):
        client.cancelled.append(oid)

    monkeypatch.setattr(client, "cancel_order_by_id", cancel_noop)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "unresolved"
    entry_submits = [r for r in client.submitted
                     if getattr(r, "stop_price", None) is None]
    assert len(entry_submits) == 1, "no duplicate entry may be submitted"


def test_dead_order_without_fill_is_error(monkeypatch):
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="dead", status="rejected")
    res = broker.execute_order(_entry_order())
    assert res["status"] == "error"
    assert "rejected" in res["reason"]


def test_cancel_fill_race_protects_what_filled(monkeypatch):
    # Reconcile says unresolved, but by the time we re-read after cancel the
    # order is fully filled → protect the position, do not report an error.
    client = FakeClient(auto_fill_entry=False)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="unresolved", status="accepted")

    def cancel_then_fill(oid):
        client.cancelled.append(oid)
        o = client._orders[oid]
        o.status = "filled"
        o.filled_qty = o.qty
        o.filled_avg_price = 100.0

    monkeypatch.setattr(client, "cancel_order_by_id", cancel_then_fill)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "submitted"
    assert res["qty"] == 5


def test_reconcile_order_detects_late_fill_at_timeout(monkeypatch):
    # Drive _reconcile_order itself: the order reads "new" while polling and
    # only fills at the final read after the deadline. It must return "filled".
    client = FakeClient(auto_fill_entry=False)
    order = FakeOrder("E1", "AAPL", 5, "buy", "market", "new")
    client._orders["E1"] = order
    monkeypatch.setattr(broker, "_client", lambda: client)

    clock = {"t": 0.0}
    reads = {"n": 0}

    def fake_time():
        return clock["t"]

    def fake_sleep(_s):
        clock["t"] += 100  # jump past the deadline

    real_get = client.get_order_by_id

    def counting_get(oid):
        reads["n"] += 1
        if reads["n"] >= 2:
            order.status = "filled"
            order.filled_qty = 5.0
            order.filled_avg_price = 100.0
        return real_get(oid)

    monkeypatch.setattr(client, "get_order_by_id", counting_get)
    monkeypatch.setattr(broker.time, "time", fake_time)
    monkeypatch.setattr(broker.time, "sleep", fake_sleep)

    rec = broker._reconcile_order("E1", max_wait_sec=30)
    assert rec["state"] == "filled"
    assert rec["filled_qty"] == 5.0


def test_reconcile_order_returns_dead_for_terminal_status(monkeypatch):
    client = FakeClient(orders=[FakeOrder("E1", "AAPL", 5, "buy",
                                          "market", "rejected")])
    monkeypatch.setattr(broker, "_client", lambda: client)
    rec = broker._reconcile_order("E1", max_wait_sec=30)
    assert rec["state"] == "dead"


def test_reconcile_order_unknown_on_api_error(monkeypatch):
    client = FakeClient(order_lookup_raises=True)
    monkeypatch.setattr(broker, "_client", lambda: client)
    rec = broker._reconcile_order("E1", max_wait_sec=30)
    assert rec["state"] == "unresolved"


# ===========================================================================
# 6) stop protection verification (side / qty / status / price)
# ===========================================================================
def test_verify_stop_requires_side_qty_and_price(monkeypatch):
    good = FakeClient(open_orders=[stop_order("S1", "AAPL", 5, "sell", 90.0)])
    monkeypatch.setattr(broker, "_client", lambda: good)
    assert broker._verify_stop_exists("AAPL", 5, "long", 90.0) is True
    assert broker._verify_stop_exists("AAPL", 5, "short", 90.0) is False
    assert broker._verify_stop_exists("AAPL", 6, "long", 90.0) is False
    assert broker._verify_stop_exists("AAPL", 5, "long", 88.0) is False


def test_missing_stop_triggers_emergency_close(monkeypatch):
    client = FakeClient(auto_fill_entry=True)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="filled", filled_qty=5.0,
                     filled_avg_price=100.0, status="filled")
    monkeypatch.setattr(broker, "_submit_stop",
                        lambda *a, **k: {"stop_order_id": "S1", "stop_price": 90.0})
    monkeypatch.setattr(broker, "_verify_stop_exists", lambda *a, **k: False)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "error"
    assert "STOP VERIFICATION FAILED" in res["reason"]


def test_stop_submission_failure_emergency_closes(monkeypatch):
    client = FakeClient(auto_fill_entry=True)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="filled", filled_qty=5.0,
                     filled_avg_price=100.0, status="filled")
    monkeypatch.setattr(broker, "_submit_stop", lambda *a, **k: None)

    res = broker.execute_order(_entry_order())
    assert res["status"] == "error"
    assert "STOP CREATION FAILED" in res["reason"]


def test_invalid_stop_price_emergency_closes(monkeypatch):
    client = FakeClient(auto_fill_entry=True)
    _wire(monkeypatch, client)
    _patch_reconcile(monkeypatch, state="filled", filled_qty=5.0,
                     filled_avg_price=100.0, status="filled")
    res = broker.execute_order(_entry_order(stop_loss=105.0))
    assert res["status"] == "error"
    assert "INVALID STOP" in res["reason"]


def test_entry_refuses_non_whole_share_qty(monkeypatch):
    client = FakeClient()
    _wire(monkeypatch, client)
    res = broker.execute_order(_entry_order(qty=0.5))
    assert res["status"] == "blocked"
    assert "protective stop" in res["reason"]
    assert client.submitted == []


# ===========================================================================
# 7) kill switch
# ===========================================================================
def test_kill_switch_blocks_entries(monkeypatch):
    client = FakeClient()
    _wire(monkeypatch, client)
    monkeypatch.setattr(broker, "is_kill_switch_active", lambda: (True, "manual"))
    res = broker.execute_order(_entry_order())
    assert res["status"] == "blocked"
    assert "kill switch" in res["reason"]
    assert client.submitted == []


def test_kill_switch_report_is_read_only(monkeypatch):
    class _Pos:
        def __init__(self, symbol):
            self.symbol = symbol
            self.qty = "5"
            self.side = type("S", (), {"value": "long"})()
            self.unrealized_plpc = "0.01"

    client = FakeClient(positions=[_Pos("AAPL")],
                        open_orders=[stop_order("S1", "AAPL", 5, "sell", 90.0)])
    monkeypatch.setattr(broker, "_client", lambda: client)
    rep = broker.kill_switch_report()
    assert rep["positions"][0]["symbol"] == "AAPL"
    assert rep["open_orders"][0]["id"] == "S1"
    assert client.cancelled == []
    assert client.closed == []


# ===========================================================================
# 8) protection flag blocks new entries
# ===========================================================================
def test_protection_unverified_blocks_safety_gate():
    import safety
    import decision_log

    decision_log.set_protection_unverified(True, "test")
    try:
        ok, reason = safety.check_safety_gates()
        assert ok is False
        assert "protection_unverified" in reason
    finally:
        decision_log.set_protection_unverified(False)


def test_sync_positions_fetch_failure_fails_closed(monkeypatch):
    # If broker state cannot be read, protection is unverifiable for every
    # position → the flag must be set so no new entry is opened.
    import monitor
    import decision_log

    class Boom:
        def get_all_positions(self):
            raise RuntimeError("alpaca down")

        def get_orders(self, *a, **k):
            raise RuntimeError("alpaca down")

    monkeypatch.setattr(monitor, "_alpaca", lambda: Boom())
    decision_log.set_protection_unverified(False)
    try:
        out = monitor._sync_positions()
        assert out == []
        assert decision_log.get_protection_unverified() is True
    finally:
        decision_log.set_protection_unverified(False)


def test_sync_positions_keeps_unverified_when_emergency_close_fails(monkeypatch):
    import monitor
    import decision_log

    class Pos:
        symbol = "AAPL"
        qty = "5"
        side = type("S", (), {"value": "long"})()
        avg_entry_price = "100"
        current_price = "100"
        unrealized_plpc = "0"

    class Client:
        def get_all_positions(self):
            return [Pos()]
        def get_orders(self, *a, **k):
            return []

    monkeypatch.setattr(monitor, "_alpaca", lambda: Client())
    monkeypatch.setattr(monitor, "get_position_state", lambda s: {"atr": 1})
    deleted = []
    monkeypatch.setattr(monitor, "delete_position_state", lambda sym: deleted.append(sym))
    import broker
    import safety
    monkeypatch.setattr(safety, "is_market_open", lambda: True)
    monkeypatch.setattr(broker, "_emergency_close_and_verify", lambda sym: False)
    monkeypatch.setattr(broker, "_submit_stop", lambda **kwargs: None)

    import signals
    monkeypatch.setattr(signals, "compute_all_indicators", lambda s: None)
    decision_log.set_protection_unverified(False)
    try:
        out = monitor._sync_positions()
        assert decision_log.get_protection_unverified() is True
        assert out == []
        assert deleted == [], 'position state must not be deleted when emergency close is unverified'
    finally:
        decision_log.set_protection_unverified(False)

def test_sync_positions_does_not_clear_flag_when_unresolved(monkeypatch):
    # A symbol whose protection cannot be reconstructed must keep the flag set
    # (the end-of-sync clear must not run while anything is unresolved).
    import monitor
    import decision_log

    class Pos:
        symbol = "AAPL"
        qty = "5"
        side = type("S", (), {"value": "long"})()
        avg_entry_price = "100"
        current_price = "100"
        unrealized_plpc = "0"

    class Client:
        def get_all_positions(self):
            return [Pos()]

        def get_orders(self, *a, **k):
            return []

    monkeypatch.setattr(monitor, "_alpaca", lambda: Client())
    # No indicators available → ATR missing → protection not reconstructable.
    import signals
    monkeypatch.setattr(signals, "compute_all_indicators", lambda s: None)
    monkeypatch.setattr(monitor, "get_position_state", lambda s: None)

    decision_log.set_protection_unverified(False)
    try:
        monitor._sync_positions()
        assert decision_log.get_protection_unverified() is True
    finally:
        decision_log.set_protection_unverified(False)

def test_move_broker_stop_handles_enum_order_type(monkeypatch):
    old_stop = FakeOrder(
        "OLD",
        "AAPL",
        5,
        SimpleNamespace(value="sell"),
        SimpleNamespace(value="stop"),
        "new",
        stop_price=90.0,
    )

    client = FakeClient(open_orders=[old_stop])
    _wire(monkeypatch, client)

    monkeypatch.setattr(
        broker,
        "_submit_stop",
        lambda *a, **k: {
            "stop_order_id": "NEW",
            "stop_price": 95.0,
        },
    )

    result = broker.move_broker_stop("AAPL", 95.0, 5, "long")

    assert result["stop_order_id"] == "NEW"
    assert client.cancelled == ["OLD"]


def test_move_broker_stop_does_not_cancel_when_new_stop_fails(monkeypatch):
    old_stop = stop_order("OLD", "AAPL", 5, "sell", 90.0)

    client = FakeClient(open_orders=[old_stop])
    _wire(monkeypatch, client)

    monkeypatch.setattr(broker, "_submit_stop", lambda *a, **k: None)

    result = broker.move_broker_stop("AAPL", 95.0, 5, "long")

    assert result is None
    assert client.cancelled == []
