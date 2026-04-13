"""BaseAgent — the base class every agent inherits from.

Connects to the bus via WebSocket. No port binding, no HTTP server.
Agent authors subclass BaseAgent, define skills + handlers, and run.

Usage::

    class ResearcherAgent(BaseAgent):
        agent_type = "researcher"
        module_name = "researcher.agent"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.pipeline = create_pipeline(self.config_path)

        def register_skills(self):
            return [
                Skill("find_papers", "Search arxiv", {"query": {"type": "string"}}),
            ]

        @handler("find_papers")
        async def find_papers(self, args):
            return await self.pipeline.search(args["query"])

    if __name__ == "__main__":
        agent = ResearcherAgent.from_cli()
        asyncio.run(agent.start())
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from khonliang_bus.connector import BusConnector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill descriptor
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A skill this agent can handle."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    since: str = ""


@dataclass
class Collaboration:
    """A multi-agent flow this agent declares."""

    name: str
    description: str = ""
    requires: dict[str, str] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Handler decorator
# ---------------------------------------------------------------------------

_HANDLER_ATTR = "_bus_handler_name"


def handler(operation: str) -> Callable:
    """Decorator marking a method as a skill handler.

    Usage::

        @handler("find_papers")
        async def find_papers(self, args):
            return {"papers": [...]}
    """

    def decorator(fn: Callable) -> Callable:
        setattr(fn, _HANDLER_ATTR, operation)
        return fn

    return decorator


# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Base class for bus agents. Connects via WebSocket — no HTTP server needed.

    Subclass and override :meth:`register_skills` and ``@handler`` methods.
    """

    agent_type: str = "base"
    module_name: str = "agent"
    version: str = "0.0.0"

    def __init__(
        self,
        agent_id: str,
        bus_url: str,
        config_path: str = "",
    ):
        self.agent_id = agent_id
        self.bus_url = bus_url.rstrip("/")
        self.config_path = config_path
        self._http = httpx.AsyncClient(timeout=30.0)
        self._connector: BusConnector | None = None
        self._handlers: dict[str, Callable] = {}
        self._started_at: float = time.time()
        self._collect_handlers()

    def _collect_handlers(self) -> None:
        """Discover @handler-decorated methods.

        Walks the MRO from most-base to most-derived so that a subclass
        handler for the same operation wins, even when its method name
        differs from the base's (e.g. ``handle_health_check`` overridden
        by a new ``custom_health`` method on the subclass).
        """
        seen: set[str] = set()
        for klass in reversed(type(self).__mro__):
            for attr_name, attr_value in vars(klass).items():
                if not callable(attr_value) or not hasattr(attr_value, _HANDLER_ATTR):
                    continue
                op = getattr(attr_value, _HANDLER_ATTR)
                bound = getattr(self, attr_name)
                self._handlers[op] = bound
                seen.add(op)

    # -- built-in skills --

    BUILT_IN_SKILLS: list[Skill] = [
        Skill(
            name="health_check",
            description="Agent liveness + identity probe. Always available.",
            parameters={},
        ),
    ]

    def _all_skills(self) -> list[Skill]:
        """Compose subclass skills with built-ins.

        Subclass names take precedence — a subclass can replace the
        built-in schema/description (e.g. to return a richer health
        payload) without losing the skill advertisement.
        """
        subclass_skills = self.register_skills()
        subclass_names = {s.name for s in subclass_skills}
        extras = [s for s in self.BUILT_IN_SKILLS if s.name not in subclass_names]
        return subclass_skills + extras

    @handler("health_check")
    async def handle_health_check(self, args: dict) -> dict:
        """Default health payload: identity, version, pid, uptime, bus link.

        Subclasses can override (regular Python method override) to add
        domain-specific checks (stores reachable, model pool ready, etc.)
        — they should call ``super().handle_health_check(args)`` and merge
        to keep the baseline fields.
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "pid": os.getpid(),
            "uptime_seconds": round(time.time() - self._started_at, 3),
            "bus_url": self.bus_url,
            "connected": self._connector is not None,
        }

    # -- override these --

    def register_skills(self) -> list[Skill]:
        """Return the skills this agent provides. Override in subclass.

        `health_check` is always advertised via :attr:`BUILT_IN_SKILLS`;
        subclasses do not need to include it. Return it explicitly only
        if overriding the schema or description.
        """
        return []

    def register_collaborations(self) -> list[Collaboration]:
        """Return collaborative flows this agent declares. Override in subclass."""
        return []

    # -- lifecycle --

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "BaseAgent":
        """Parse --id, --bus, --config from command line and construct."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--id", required=True)
        parser.add_argument("--bus", required=True)
        parser.add_argument("--config", default="")
        parser.add_argument("command", nargs="?")  # install / uninstall
        args = parser.parse_args(argv)

        agent = cls(
            agent_id=args.id,
            bus_url=args.bus,
            config_path=args.config,
        )

        if args.command == "install":
            agent._do_install()
            sys.exit(0)
        elif args.command == "uninstall":
            agent._do_uninstall()
            sys.exit(0)

        return agent

    async def start(self) -> None:
        """Connect to bus via WebSocket, register, and handle requests.

        No port binding, no HTTP server. The agent is a pure WebSocket
        client. The bus sends requests over the same connection.
        """
        skills = self._all_skills()
        collabs = self.register_collaborations()

        self._connector = BusConnector(
            bus_url=self.bus_url,
            agent_id=self.agent_id,
            on_request=self._dispatch_request,
        )

        # Connect and register (raises RuntimeError if bus is unreachable).
        # Wrap in try/finally so _http is cleaned up on failure.
        try:
            await self._connector.connect_and_register(
                agent_type=self.agent_type,
                version=self.version,
                pid=os.getpid(),
                skills=[
                    {
                        "name": s.name,
                        "description": s.description,
                        "parameters": s.parameters,
                        "since": s.since,
                    }
                    for s in skills
                ],
                collaborations=[
                    {
                        "name": c.name,
                        "description": c.description,
                        "requires": c.requires,
                        "steps": c.steps,
                    }
                    for c in collabs
                ],
            )

        except Exception:
            await self._http.aclose()
            raise

        logger.info(
            "Agent %s started (%d skills, WebSocket)",
            self.agent_id,
            len(skills),
        )

        # Handle signals for clean shutdown (not supported on all platforms)
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
        except NotImplementedError:
            pass

        # Run the WebSocket message loop (blocks until disconnect)
        try:
            await self._connector.run()
        finally:
            await self._http.aclose()

    async def shutdown(self) -> None:
        """Disconnect from bus."""
        if self._connector:
            await self._connector.disconnect()
        await self._http.aclose()
        logger.info("Agent %s shut down", self.agent_id)

    # -- request dispatch --

    async def _dispatch_request(self, msg: dict) -> Any:
        """Route a bus request to the matching @handler method.

        Raises exceptions for transport-level errors (unknown operation,
        handler crashes). The connector catches these and sends an error
        message to the bus. Handler return values — even dicts with an
        "error" key — are treated as legitimate payloads.
        """
        operation = msg.get("operation", "")
        args = msg.get("args", {})

        handler_fn = self._handlers.get(operation)
        if not handler_fn:
            raise ValueError(f"unknown operation: {operation}")

        return await handler_fn(args)

    # -- install/uninstall (HTTP — these are one-shot, not WebSocket) --

    def _do_install(self) -> None:
        """Synchronous install — called from from_cli when command is 'install'."""
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{self.bus_url}/v1/install", json={
                "agent_type": self.agent_type,
                "id": self.agent_id,
                "command": sys.executable,
                "args": ["-m", self.module_name],
                "cwd": os.getcwd(),
                "config": os.path.abspath(self.config_path) if self.config_path else "",
            })
            r.raise_for_status()
            try:
                print(json.dumps(r.json(), indent=2))
            except (json.JSONDecodeError, ValueError):
                print(r.text)

    def _do_uninstall(self) -> None:
        """Synchronous uninstall."""
        with httpx.Client(timeout=10.0) as client:
            r = client.request("DELETE", f"{self.bus_url}/v1/install/{self.agent_id}")
            r.raise_for_status()
            try:
                print(json.dumps(r.json(), indent=2))
            except (json.JSONDecodeError, ValueError):
                print(r.text)

    # -- pub/sub helpers --

    async def publish(self, topic: str, payload: Any) -> None:
        """Publish an event through the bus (via WebSocket).

        Raises:
            RuntimeError: If the agent is not connected to the bus.
        """
        if not self._connector:
            raise RuntimeError(
                f"Agent {self.agent_id} is not connected to the bus. "
                "Call start() before publishing."
            )
        await self._connector.publish(topic, payload)

    async def nack(self, message_id: str, topic: str, reason: str = "") -> None:
        """Negative-acknowledge a message for redelivery."""
        # NACK goes via HTTP since it's not part of the WebSocket protocol
        await self._http.post(
            f"{self.bus_url}/v1/nack",
            json={
                "subscriber_id": self.agent_id,
                "message_id": message_id,
                "topic": topic,
                "reason": reason,
            },
        )

    # -- session context --

    async def get_session_context(
        self, session_id: str, scope: str = "public"
    ) -> dict[str, Any]:
        """Read session context from the bus."""
        r = await self._http.get(
            f"{self.bus_url}/v1/session/{session_id}/context",
            params={"scope": scope},
        )
        return r.json()

    async def update_session_context(
        self,
        session_id: str,
        public: dict[str, Any] | None = None,
        private: dict[str, Any] | None = None,
    ) -> dict:
        """Write session context to the bus."""
        body: dict[str, Any] = {}
        if public is not None:
            body["public_ctx"] = public
        if private is not None:
            body["private_ctx"] = private
        r = await self._http.post(
            f"{self.bus_url}/v1/session/{session_id}/context",
            json=body,
        )
        return r.json()

    # -- bus request helper --

    async def request(
        self,
        agent_id: str | None = None,
        agent_type: str | None = None,
        operation: str = "",
        args: dict[str, Any] | None = None,
        timeout: float = 30.0,
        response_mode: str = "raw",
    ) -> dict:
        """Make a request to another agent via the bus."""
        payload: dict[str, Any] = {
            "operation": operation,
            "args": args or {},
            "timeout": timeout,
            "response_mode": response_mode,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        elif agent_type:
            payload["agent_type"] = agent_type
        r = await self._http.post(f"{self.bus_url}/v1/request", json=payload)
        return r.json()

    # -- gap reporting --

    async def report_gap(self, operation: str, reason: str, context: dict | None = None) -> None:
        """Report a capability gap through the bus (via WebSocket).

        Raises:
            RuntimeError: If the agent is not connected to the bus.
        """
        if not self._connector:
            raise RuntimeError(
                f"Agent {self.agent_id} is not connected to the bus. "
                "Call start() before reporting gaps."
            )
        await self._connector.report_gap(operation, reason, context)

    # -- from_mcp migration helper --

    @classmethod
    def from_mcp(
        cls,
        mcp_server,
        agent_type: str,
        agent_id: str = "",
        bus_url: str = "",
        config_path: str = "",
    ) -> "BaseAgent":
        """Wrap an existing FastMCP server's tools as bus handlers.

        Introspects the server's registered tools and creates a BaseAgent
        subclass with @handler for each tool. The tool functions stay
        identical — only the transport changes from stdio to bus WebSocket.

        This is a migration bridge. The end state is native @handler
        methods, not wrapped MCP tools.

        Note: calls ``asyncio.run(mcp_server.list_tools())`` to introspect
        tools. Must be called outside an existing event loop.
        """
        import asyncio as _aio

        tools = _aio.run(mcp_server.list_tools())

        handlers = {}
        skills = []

        for tool in tools:
            tool_name = tool.name
            skills.append(Skill(
                name=tool_name,
                description=tool.description or "",
                parameters={},
            ))

            def _make_handler(tn):
                async def _handler(self_inner, args):
                    result = await mcp_server.call_tool(tn, args)
                    if isinstance(result, tuple) and len(result) == 2:
                        meta = result[1]
                        if isinstance(meta, dict) and "result" in meta:
                            return {"result": meta["result"]}
                    if isinstance(result, list) and result:
                        first = result[0]
                        if hasattr(first, "text"):
                            return {"result": first.text}
                    return {"result": str(result)}
                return _handler

            handler_fn = _make_handler(tool_name)
            setattr(handler_fn, _HANDLER_ATTR, tool_name)
            handlers[tool_name] = handler_fn

        agent_cls = type(
            f"{agent_type.title()}Agent",
            (cls,),
            {
                "agent_type": agent_type,
                **{f"_handle_{n}": fn for n, fn in handlers.items()},
            },
        )

        def _register_skills(self_inner):
            return skills
        agent_cls.register_skills = _register_skills

        return agent_cls(
            agent_id=agent_id or f"{agent_type}-primary",
            bus_url=bus_url,
            config_path=config_path,
        )
