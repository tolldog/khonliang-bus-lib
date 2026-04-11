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


def test_is_open_v14_state():
    """_is_open uses .state for websockets v14+ (no .open attribute)."""
    from unittest.mock import MagicMock
    from websockets.protocol import State

    c = BusConnector("http://localhost:8787", "test")
    mock_ws = MagicMock()
    mock_ws.state = State.OPEN
    del mock_ws.open  # v14+ doesn't have .open
    c._ws = mock_ws
    assert c._is_open() is True

    mock_ws.state = State.CLOSED
    assert c._is_open() is False


def test_is_open_legacy_fallback():
    """_is_open falls back to .open for legacy websockets."""
    from unittest.mock import MagicMock

    c = BusConnector("http://localhost:8787", "test")
    mock_ws = MagicMock(spec=[])  # empty spec — no .state
    mock_ws.open = True
    c._ws = mock_ws
    assert c._is_open() is True


def test_is_open_none():
    c = BusConnector("http://localhost:8787", "test")
    assert c._is_open() is False


@pytest.mark.asyncio
async def test_send_warns_when_not_connected():
    """send() logs a warning and drops (does not raise) when no WebSocket is active."""
    c = BusConnector("http://localhost:8787", "test")
    # _ws is None — should log warning but NOT raise
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
async def test_handle_bus_message_request_error_dict_is_legit_payload():
    """Handler returning a dict with 'error' key is a legitimate response, not a transport error."""
    sent = []

    async def _on_request(msg):
        return {"error": "not found", "detail": "no matching paper"}

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

    # Should be sent as a response, not an error — the handler chose to return this
    assert sent[0]["type"] == "response"
    assert sent[0]["result"]["error"] == "not found"


@pytest.mark.asyncio
async def test_handle_bus_message_request_exception_sends_error():
    """Handler raising an exception IS a transport error → error frame."""
    sent = []

    async def _on_request(msg):
        raise RuntimeError("handler crashed")

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
    assert "handler crashed" in sent[0]["error"]
