"""WebSocket connector for bus agents.

Handles the persistent WebSocket connection between an agent and the
bus. Multiplexes registration, heartbeat, request handling, publish,
and gap reporting over a single connection.

This replaces the HTTP callback model from v0.1. Agents are pure
WebSocket clients — no port binding, no FastAPI, no uvicorn.

Used internally by BaseAgent. Agent authors don't interact with
this module directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import websockets

logger = logging.getLogger(__name__)


class BusConnector:
    """Persistent WebSocket connection to the bus.

    Handles the multiplexed protocol:

    Agent → Bus:
      register, heartbeat, response, error, publish, gap, deregister

    Bus → Agent:
      registered, request, ping
    """

    def __init__(
        self,
        bus_url: str,
        agent_id: str,
        on_request: Callable[[dict], Any] | None = None,
        heartbeat_interval: float = 30.0,
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        self.bus_url = bus_url.rstrip("/")
        self.agent_id = agent_id
        self._on_request = on_request
        self._heartbeat_interval = heartbeat_interval
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._registered = False
        self._registration_payload: dict | None = None
        self._running = False

    @property
    def connected(self) -> bool:
        return self._ws is not None and self._ws.open

    @property
    def registered(self) -> bool:
        return self._registered

    async def connect_and_register(
        self,
        agent_type: str,
        version: str,
        pid: int,
        skills: list[dict],
        collaborations: list[dict] | None = None,
    ) -> None:
        """Connect to the bus and register.

        Raises RuntimeError if the bus is unreachable or registration fails.
        """
        self._registration_payload = {
            "type": "register",
            "id": self.agent_id,
            "agent_type": agent_type,
            "version": version,
            "pid": pid,
            "skills": skills,
            "collaborations": collaborations or [],
        }

        ws_url = self._ws_url("/v1/agent")
        try:
            self._ws = await websockets.connect(ws_url)
        except Exception as e:
            raise RuntimeError(
                f"Agent {self.agent_id} failed to connect to bus at "
                f"{ws_url}: {e}. The bus must be running before agents can start."
            ) from e

        # Send registration
        await self._ws.send(json.dumps(self._registration_payload))
        resp = json.loads(await self._ws.recv())
        if resp.get("type") != "registered":
            await self._ws.close()
            raise RuntimeError(
                f"Agent {self.agent_id} registration rejected by bus: {resp}"
            )

        self._registered = True
        logger.info("Agent %s registered with bus via WebSocket", self.agent_id)

    async def run(self) -> None:
        """Main loop: receive messages from bus, dispatch requests.

        Runs until the connection is closed or :meth:`disconnect` is called.
        Auto-reconnects on connection loss.
        """
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        initial_delay = self._reconnect_delay
        delay = initial_delay

        while self._running:
            try:
                async for raw in self._ws:
                    msg = json.loads(raw)
                    await self._handle_bus_message(msg)
                    delay = initial_delay  # reset on successful message

                # Clean close — the async for ended normally. If we're
                # still supposed to be running, treat it like a connection
                # loss and reconnect with backoff.
                if not self._running:
                    break
                logger.warning(
                    "Agent %s WebSocket closed cleanly, reconnecting in %.1fs",
                    self.agent_id, delay,
                )
                self._registered = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)
                try:
                    ws_url = self._ws_url("/v1/agent")
                    self._ws = await websockets.connect(ws_url)
                    if self._registration_payload:
                        await self._ws.send(json.dumps(self._registration_payload))
                        resp = json.loads(await self._ws.recv())
                        if resp.get("type") == "registered":
                            self._registered = True
                            delay = initial_delay
                            logger.info("Agent %s reconnected", self.agent_id)
                except Exception as e:
                    logger.warning("Reconnect after clean close failed: %s", e)

            except websockets.ConnectionClosed:
                if not self._running:
                    break
                logger.warning(
                    "Agent %s lost connection to bus, reconnecting in %.1fs",
                    self.agent_id, delay,
                )
                self._registered = False
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._max_reconnect_delay)

                # Reconnect and re-register
                try:
                    ws_url = self._ws_url("/v1/agent")
                    self._ws = await websockets.connect(ws_url)
                    if self._registration_payload:
                        await self._ws.send(json.dumps(self._registration_payload))
                        resp = json.loads(await self._ws.recv())
                        if resp.get("type") == "registered":
                            self._registered = True
                            delay = initial_delay  # reset after successful reconnect
                            logger.info("Agent %s reconnected and re-registered", self.agent_id)
                except Exception as e:
                    logger.warning("Reconnect failed: %s", e)

            except Exception as e:
                if not self._running:
                    break
                logger.error("Agent %s unexpected error: %s", self.agent_id, e)
                await asyncio.sleep(delay)

    async def _handle_bus_message(self, msg: dict) -> None:
        """Dispatch a message received from the bus."""
        msg_type = msg.get("type", "")

        if msg_type == "request" and self._on_request:
            # Dispatch to the agent's handler.
            # Handlers raise HandlerError for transport-level errors;
            # any dict they return (even one with an "error" key) is a
            # legitimate payload and gets sent as a response.
            try:
                result = await self._on_request(msg)
                await self.send({
                    "type": "response",
                    "correlation_id": msg.get("correlation_id", ""),
                    "result": result,
                })
            except Exception as e:
                await self.send({
                    "type": "error",
                    "correlation_id": msg.get("correlation_id", ""),
                    "error": str(e),
                    "retryable": True,
                })

        elif msg_type == "ping":
            await self.send({"type": "pong"})

    async def send(self, msg: dict) -> None:
        """Send a message to the bus. Raises ConnectionError if not connected."""
        if self._ws and self._ws.open:
            await self._ws.send(json.dumps(msg))
        else:
            raise ConnectionError(
                f"Agent {self.agent_id}: cannot send (not connected). "
                f"Message type={msg.get('type')}"
            )

    async def publish(self, topic: str, payload: Any) -> None:
        """Publish an event through the bus."""
        await self.send({
            "type": "publish",
            "topic": topic,
            "payload": payload,
        })

    async def report_gap(self, operation: str, reason: str, context: dict | None = None) -> None:
        """Report a capability gap to the bus."""
        await self.send({
            "type": "gap",
            "operation": operation,
            "reason": reason,
            "context": context or {},
        })

    async def disconnect(self) -> None:
        """Clean disconnect: deregister and close."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._ws and self._ws.open:
            try:
                await self.send({"type": "deregister"})
                await self._ws.close()
            except Exception:
                pass
        self._registered = False
        logger.info("Agent %s disconnected from bus", self.agent_id)

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                if self._ws and self._ws.open:
                    await self.send({"type": "heartbeat"})
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Heartbeat send failed: %s", e)

    def _ws_url(self, path: str) -> str:
        """Convert http(s) bus URL to ws(s) URL."""
        parsed = urlparse(self.bus_url + path)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return urlunparse(parsed._replace(scheme=scheme))
