"""Tests for BusClient: construction, publish, ack, nack, deregister, close."""

from __future__ import annotations

from khonliang_bus import BusClient, Message


def test_message_from_wire():
    msg = Message.from_wire({
        "id": "42",
        "topic": "test.event",
        "payload": {"key": "val"},
        "timestamp": "2026-04-10T00:00:00Z",
    })
    assert msg.id == "42"
    assert msg.topic == "test.event"
    assert msg.payload == {"key": "val"}


def test_client_construction():
    """Client constructs without error even when bus isn't running (register=False)."""
    client = BusClient(
        base_url="http://localhost:1",
        subscriber_id="test",
        register=False,
    )
    assert client.subscriber_id == "test"
    assert client.base_url == "http://localhost:1"
    client.close()


def test_client_requires_base_url():
    import pytest
    with pytest.raises(ValueError, match="base_url"):
        BusClient(base_url="", subscriber_id="test")


def test_client_requires_subscriber_id():
    import pytest
    with pytest.raises(ValueError, match="subscriber_id"):
        BusClient(base_url="http://x", subscriber_id="")


def test_client_context_manager():
    """Context manager calls close() on exit."""
    with BusClient("http://localhost:1", "test", register=False) as client:
        assert client.subscriber_id == "test"


def test_client_has_nack_method():
    client = BusClient("http://localhost:1", "test", register=False)
    assert callable(client.nack)
    client.close()


def test_client_has_deregister_method():
    client = BusClient("http://localhost:1", "test", register=False)
    assert callable(client.deregister)
    client.close()


def test_client_has_services_method():
    client = BusClient("http://localhost:1", "test", register=False)
    assert callable(client.services)
    client.close()
