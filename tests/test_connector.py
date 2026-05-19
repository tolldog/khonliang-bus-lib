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
async def test_connect_and_register_omits_launch_fields_when_not_provided():
    """Backward compat: callers that don't pass launch_spec/launch_info keep the
    prior register-payload shape — older buses won't see unexpected keys.

    fr_khonliang-bus-lib_2cfc0de6 / fr_khonliang-bus-lib_cccaa6a9 — the
    contract extension is additive.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    with pytest.raises(RuntimeError):  # bus unreachable; payload still built
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1234, skills=[],
        )
    payload = c._registration_payload
    assert "launch_spec" not in payload
    assert "launch_info" not in payload
    # Existing fields stay intact.
    assert payload["type"] == "register"
    assert payload["id"] == "test-agent"
    assert payload["pid"] == 1234


@pytest.mark.asyncio
async def test_connect_and_register_includes_launch_spec_when_provided():
    """Fixture shape mirrors what ``capture_launch_spec()`` actually emits:
    field name is ``args`` (not ``argv``), interpreter is stripped.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    spec = {
        "executable": "/opt/x/.venv/bin/python",
        "args": ["-m", "x.agent", "--config", "/etc/x/config.yaml"],
        "cwd": "/opt/x",
        "config": "/etc/x/config.yaml",
    }
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test",
            version="0.1.0",
            pid=1234,
            skills=[],
            launch_spec=spec,
        )
    assert c._registration_payload["launch_spec"] == spec
    assert "launch_info" not in c._registration_payload


@pytest.mark.asyncio
async def test_connect_and_register_includes_launch_info_when_provided():
    """Fixture matches the real shape from ``capture_launch_info()``: no
    ``pid`` (top-level payload field) and no spec-side keys.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    info = {
        "started_at": 1234567.0,
        "commit_sha": "deadbeef" * 5,
        "branch": "main",
        "dirty": False,
    }
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test",
            version="0.1.0",
            pid=1234,
            skills=[],
            launch_info=info,
        )
    assert c._registration_payload["launch_info"] == info
    assert "launch_spec" not in c._registration_payload
    # Top-level pid is the authoritative source; launch_info doesn't carry one.
    assert c._registration_payload["pid"] == 1234


@pytest.mark.asyncio
async def test_connect_and_register_includes_both_launch_fields():
    """The common case: agent sends both at registration. Shapes mirror
    the real ``capture_launch_*`` outputs — ``args`` not ``argv``, no
    ``pid`` in ``launch_info``.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    spec = {"executable": "/x", "args": ["-m", "x"], "cwd": "/", "config": None}
    info = {"started_at": 0.0, "commit_sha": None, "branch": None, "dirty": None}
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
            launch_spec=spec, launch_info=info,
        )
    assert c._registration_payload["launch_spec"] == spec
    assert c._registration_payload["launch_info"] == info
    assert c._registration_payload["pid"] == 1


@pytest.mark.asyncio
async def test_connect_and_register_omits_welcome_when_not_provided():
    """fr_khonliang-bus_f96722dd: welcome is additive. Calls that don't
    pass it keep the prior payload shape so older buses see no
    unexpected keys.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
        )
    assert "welcome" not in c._registration_payload


@pytest.mark.asyncio
async def test_connect_and_register_includes_welcome_when_provided():
    """Welcome dict is forwarded verbatim — the connector is a passthrough.

    The shape is whatever ``handle_welcome`` returns; the bus is
    responsible for persistence + serving. This test only verifies the
    wire layer (fr_khonliang-bus_f96722dd contract extension on
    bus-lib).
    """
    c = BusConnector("http://localhost:1", "test-agent")
    welcome = {
        "agent_id": "test-agent",
        "agent_type": "test",
        "version": "0.1.0",
        "skill_count": 3,
        "role": "test fixture",
        "mission": "exercise the welcome path",
    }
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
            welcome=welcome,
        )
    assert c._registration_payload["welcome"] == welcome


@pytest.mark.asyncio
async def test_connect_and_register_carries_all_three_extension_fields():
    """The forward-shape: launch_spec + launch_info + welcome all
    coexist in one register handshake. All three are independent
    additive extensions; they don't interact.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    spec = {"executable": "/x", "args": ["-m", "x"], "cwd": "/", "config": None}
    info = {"started_at": 0.0, "commit_sha": None, "branch": None, "dirty": None}
    welcome = {"agent_id": "test-agent", "role": "r"}
    with pytest.raises(RuntimeError):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
            launch_spec=spec, launch_info=info, welcome=welcome,
        )
    payload = c._registration_payload
    assert payload["launch_spec"] == spec
    assert payload["launch_info"] == info
    assert payload["welcome"] == welcome


# ---------------------------------------------------------------------------
# Strict JSON encoding contract (Copilot PR #25 R6 #2, #3): serialization
# failures must honor each method's documented exception surface, not
# leak the underlying ValueError/TypeError.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_and_register_raises_runtime_error_on_nan_payload():
    """``connect_and_register`` docstring promises ``RuntimeError`` on
    registration failure. A payload containing NaN (rejected by
    ``allow_nan=False``) must surface as RuntimeError, not the raw
    ``ValueError`` ``json.dumps`` raises.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    bad_welcome = {"agent_id": "t", "score": float("nan")}
    with pytest.raises(RuntimeError, match="not JSON-encodable"):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
            welcome=bad_welcome,
        )


@pytest.mark.asyncio
async def test_connect_and_register_raises_runtime_error_on_non_serializable_payload():
    """Non-JSON-serializable values (e.g. ``object()``, ``set``,
    functions) trigger ``TypeError`` inside ``json.dumps``. Same
    contract — surface as RuntimeError.
    """
    c = BusConnector("http://localhost:1", "test-agent")
    bad_welcome = {"agent_id": "t", "broken": object()}
    with pytest.raises(RuntimeError, match="not JSON-encodable"):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
            welcome=bad_welcome,
        )


@pytest.mark.asyncio
async def test_connect_and_register_raises_runtime_error_on_overflow_error(monkeypatch):
    """Copilot PR #25 R7: ``json.dumps`` raises a wide range of exceptions
    (``OverflowError``, ``RecursionError``, custom-encoder errors), not just
    ``(TypeError, ValueError)``. The pre-flight serialization guard must
    wrap ALL of them as ``RuntimeError`` to honor the documented exception
    contract.

    Verified by monkeypatching the connector's bound ``json.dumps`` to
    raise ``OverflowError`` — independent of any payload shape.
    """
    import khonliang_bus.connector as connector_mod

    monkeypatch.setattr(
        connector_mod.json,
        "dumps",
        lambda *a, **kw: (_ for _ in ()).throw(OverflowError("synthetic")),
    )

    c = BusConnector("http://localhost:1", "test-agent")
    with pytest.raises(RuntimeError, match="not JSON-encodable"):
        await c.connect_and_register(
            agent_type="test", version="0.1.0", pid=1, skills=[],
        )


@pytest.mark.asyncio
async def test_send_drops_message_on_overflow_error(caplog, monkeypatch):
    """The matching broad-catch test for ``send()``: ``OverflowError``
    (and other non-(TypeError,ValueError) exceptions) must be caught
    and logged-as-dropped, never propagate to the caller.
    """
    import logging
    from unittest.mock import AsyncMock, MagicMock
    from websockets.protocol import State

    import khonliang_bus.connector as connector_mod

    monkeypatch.setattr(
        connector_mod.json,
        "dumps",
        lambda *a, **kw: (_ for _ in ()).throw(OverflowError("synthetic")),
    )

    c = BusConnector("http://localhost:1", "test-agent")
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.state = State.OPEN
    c._ws = fake_ws

    with caplog.at_level(logging.WARNING):
        await c.send({"type": "publish"})  # must not raise

    fake_ws.send.assert_not_awaited()
    assert any(
        "not JSON-encodable" in r.message and "publish" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_send_drops_unserializable_message_without_raising(caplog):
    """``send()`` docstring contract: warns-and-drops on problems out of
    the caller's control. A NaN/Inf or non-serializable value falls into
    that category — log warning + drop, don't propagate ValueError.
    """
    import logging
    from unittest.mock import AsyncMock, MagicMock
    from websockets.protocol import State

    c = BusConnector("http://localhost:1", "test-agent")
    # Stub _ws so _is_open() returns True without a real socket.
    fake_ws = MagicMock()
    fake_ws.send = AsyncMock()
    fake_ws.state = State.OPEN  # _is_open does ``ws.state == State.OPEN``
    c._ws = fake_ws

    with caplog.at_level(logging.WARNING):
        # NaN trips allow_nan=False. The whole call must not raise.
        await c.send({"type": "response", "value": float("nan")})

    # Nothing went on the wire.
    fake_ws.send.assert_not_awaited()
    # Warning logged with type info.
    assert any(
        "not JSON-encodable" in r.message and "response" in r.message
        for r in caplog.records
    )


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
