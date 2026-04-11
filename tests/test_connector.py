"""Tests for the WebSocket connector."""

from __future__ import annotations

import pytest

from khonliang_bus.connector import BusConnector


def test_ws_url_http():
    c = BusConnector("http://localhost:8787", "test")
    assert c._ws_url("/v1/agent") == "ws://localhost:8787/v1/agent"


def test_ws_url_https():
    c = BusConnector("https://bus.example.com", "test")
    assert c._ws_url("/v1/agent") == "wss://bus.example.com/v1/agent"


def test_ws_url_strips_trailing_slash():
    c = BusConnector("http://localhost:8787/", "test")
    assert c._ws_url("/v1/agent") == "ws://localhost:8787/v1/agent"


def test_ws_url_with_port():
    c = BusConnector("https://bus.example.com:9443", "test")
    assert c._ws_url("/v1/agent") == "wss://bus.example.com:9443/v1/agent"


def test_initial_state():
    c = BusConnector("http://localhost:8787", "test-agent")
    assert c.agent_id == "test-agent"
    assert c.connected is False
    assert c.registered is False


@pytest.mark.asyncio
async def test_connect_fails_with_clear_error():
    c = BusConnector("http://localhost:1", "test-agent")
    with pytest.raises(RuntimeError, match="failed to connect"):
        await c.connect_and_register(
            agent_type="test",
            version="0.1.0",
            pid=1234,
            skills=[],
        )


@pytest.mark.asyncio
async def test_connect_error_includes_agent_id():
    c = BusConnector("http://localhost:1", "my-agent-42")
    with pytest.raises(RuntimeError, match="my-agent-42"):
        await c.connect_and_register("test", "0.1.0", 1, [])


@pytest.mark.asyncio
async def test_connect_error_includes_bus_url():
    c = BusConnector("http://localhost:1", "test")
    with pytest.raises(RuntimeError, match="localhost:1"):
        await c.connect_and_register("test", "0.1.0", 1, [])


@pytest.mark.asyncio
async def test_send_is_noop_when_not_connected():
    """send() must not raise when there is no active WebSocket."""
    c = BusConnector("http://localhost:8787", "test")
    # _ws is None — should silently drop without raising
    await c.send({"type": "heartbeat"})


@pytest.mark.asyncio
async def test_handle_bus_message_ping():
    """A 'ping' message should result in a 'pong' send."""
    sent = []

    async def _fake_send(msg):
        sent.append(msg)

    c = BusConnector("http://localhost:8787", "test")
    c.send = _fake_send

    await c._handle_bus_message({"type": "ping"})
    assert sent == [{"type": "pong"}]


@pytest.mark.asyncio
async def test_handle_bus_message_request_dispatched():
    """A 'request' message is forwarded to on_request and a 'response' sent back."""
    sent = []

    async def _on_request(msg):
        return {"answer": 42}

    async def _fake_send(msg):
        sent.append(msg)

    c = BusConnector("http://localhost:8787", "test", on_request=_on_request)
    c.send = _fake_send

    await c._handle_bus_message({
        "type": "request",
        "correlation_id": "abc",
        "operation": "solve",
        "args": {},
    })

    assert len(sent) == 1
    assert sent[0]["type"] == "response"
    assert sent[0]["correlation_id"] == "abc"
    assert sent[0]["result"] == {"answer": 42}


@pytest.mark.asyncio
async def test_handle_bus_message_request_error_returned():
    """When on_request returns an error dict, an 'error' frame is sent."""
    sent = []

    async def _on_request(msg):
        return {"error": "not found", "retryable": False}

    async def _fake_send(msg):
        sent.append(msg)

    c = BusConnector("http://localhost:8787", "test", on_request=_on_request)
    c.send = _fake_send

    await c._handle_bus_message({
        "type": "request",
        "correlation_id": "xyz",
        "operation": "find",
        "args": {},
    })

    assert sent[0]["type"] == "error"
    assert sent[0]["error"] == "not found"
    assert sent[0]["retryable"] is False
