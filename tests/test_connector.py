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
