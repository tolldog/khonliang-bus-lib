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
from khonliang_bus.registry import (
    ExecutionProfile,
    OutputContract,
    RuntimeProfile,
    SkillAuthority,
    SkillDescriptor,
    SkillStatus,
)
from khonliang_bus.versioning import resolve_version

logger = logging.getLogger(__name__)


def _has_explicit_version(agent: "BaseAgent") -> bool:
    """Return True when the agent's ``version`` was set by the subclass.

    Checks, in order:
      1. ``self.__dict__`` — subclass assigned ``self.version = ...`` before
         calling ``super().__init__`` (instance-level override).
      2. Each class in the MRO from the subclass down to (but not
         including) ``BaseAgent`` — a subclass or intermediate base
         declared ``version`` as a class attribute.

    Value equality against ``BaseAgent.version`` is deliberately NOT
    used: a subclass that pins ``version = "0.0.0"`` on purpose must
    not be overwritten just because it coincides with the default.
    """
    if "version" in agent.__dict__:
        return True
    for klass in type(agent).__mro__:
        if klass is BaseAgent:
            break
        if "version" in klass.__dict__:
            return True
    return False


# ---------------------------------------------------------------------------
# Skill descriptor
# ---------------------------------------------------------------------------


@dataclass
class Skill:
    """A skill this agent can handle.

    The optional ``default_timeout_s`` field is an author-declared default
    timeout (seconds) for calls to this skill. It is consumed by the MCP
    adapter's timeout precedence ladder (fr_khonliang_a3dc662d) at step 2,
    used when no per-call ``_mcp_timeout`` hint is supplied. ``None`` means
    "not set"; the ladder falls through to the env/CLI default. Must be a
    positive number when set; zero and negative values are rejected.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    since: str = ""
    capability: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_contract: OutputContract | dict[str, Any] | None = None
    authority: SkillAuthority | str = SkillAuthority.AUTHORITATIVE
    status: SkillStatus | str = SkillStatus.ACTIVE
    aliases: list[str] = field(default_factory=list)
    execution_profiles: list[ExecutionProfile | dict[str, Any]] = field(default_factory=list)
    runtime_profile: RuntimeProfile | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Consumed by the MCP adapter's timeout precedence ladder
    # (fr_khonliang_a3dc662d) at step 2:
    #   per-call ``_mcp_timeout`` hint
    #     → Skill.default_timeout_s  ← this field
    #     → env / CLI adapter default
    #     → library fallback
    # Appended at the end of the field list to preserve positional-arg
    # compatibility for existing ``Skill(...)`` call sites.
    default_timeout_s: float | None = None

    def __post_init__(self) -> None:
        # Deep-copy so per-parameter nested dicts (type/default/description
        # objects, JSON-schema sub-trees) don't alias across Skill clones.
        # ``_all_skills`` rebuilds built-ins via ``Skill(**s.to_dict())``;
        # without deepcopy, a caller mutating ``params['detail']['default']``
        # on one clone would silently corrupt ``BUILT_IN_SKILLS`` for every
        # subsequent call (and every other agent in the process).
        from copy import deepcopy

        self.parameters = deepcopy(dict(self.parameters))
        self.input_schema = deepcopy(dict(self.input_schema or self.parameters))
        if self.output_contract is not None:
            self.output_contract = (
                self.output_contract
                if isinstance(self.output_contract, OutputContract)
                else OutputContract.from_dict(self.output_contract)
            )
        self.authority = SkillAuthority.coerce(self.authority)
        self.status = SkillStatus.coerce(self.status)
        self.aliases = list(self.aliases)
        self.execution_profiles = [
            profile
            if isinstance(profile, ExecutionProfile)
            else ExecutionProfile.from_dict(profile)
            for profile in self.execution_profiles
        ]
        if self.runtime_profile is not None:
            self.runtime_profile = (
                self.runtime_profile
                if isinstance(self.runtime_profile, RuntimeProfile)
                else RuntimeProfile.from_dict(self.runtime_profile)
            )
        if self.default_timeout_s is not None:
            # Accept ints gracefully; the ladder treats them as seconds.
            if isinstance(self.default_timeout_s, bool) or not isinstance(
                self.default_timeout_s, (int, float)
            ):
                raise TypeError(
                    "default_timeout_s must be a number (got "
                    f"{type(self.default_timeout_s).__name__})"
                )
            if self.default_timeout_s <= 0:
                raise ValueError(
                    "default_timeout_s must be > 0 (got "
                    f"{self.default_timeout_s})"
                )
            self.default_timeout_s = float(self.default_timeout_s)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the skill for bus registration.

        The legacy fields stay stable while richer registry fields are included
        only when the agent declared them.
        """
        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "since": self.since,
        }
        optional = {
            "capability": self.capability,
            "input_schema": self.input_schema if self.input_schema != self.parameters else {},
            "output_contract": (
                self.output_contract.to_dict()
                if isinstance(self.output_contract, OutputContract)
                else None
            ),
            "authority": self.authority.value,
            "status": self.status.value,
            "aliases": self.aliases,
            "execution_profiles": [
                profile.to_dict()
                for profile in self.execution_profiles
            ],
            "runtime_profile": (
                self.runtime_profile.to_dict()
                if isinstance(self.runtime_profile, RuntimeProfile)
                else None
            ),
            "default_timeout_s": self.default_timeout_s,
            "metadata": self.metadata,
        }
        payload.update({
            key: value
            for key, value in optional.items()
            if value not in (None, "", [], {})
        })
        return payload

    def descriptor(self, provider_id: str, skill_id: str | None = None) -> SkillDescriptor:
        """Convert the agent-facing skill into a registry descriptor."""
        return SkillDescriptor(
            skill_id=skill_id or f"{provider_id}.{self.name}",
            provider_id=provider_id,
            name=self.name,
            capability=self.capability or self.name,
            description=self.description,
            input_schema=self.input_schema,
            output_contract=self.output_contract or OutputContract(),
            authority=self.authority,
            status=self.status,
            aliases=self.aliases,
            execution_profiles=self.execution_profiles,
            runtime_profile=self.runtime_profile or RuntimeProfile(),
            metadata=self.metadata,
        )


@dataclass
class Collaboration:
    """A multi-agent flow this agent declares."""

    name: str
    description: str = ""
    requires: dict[str, str] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class WelcomeEntryPoint:
    """A canonical starting skill for a common cold-start path.

    ``skill`` is the bus-skill name (matches a Skill registration on
    this agent). ``when_to_use`` is a short phrase a cold-start LLM
    can match against an incoming request.

    Frozen so accidental ``ep.skill = "..."`` reassignment fails
    fast — see Welcome's class doc for the broader invariant.
    """

    skill: str
    when_to_use: str


@dataclass(frozen=True)
class Welcome:
    """Editorial agent introduction for the cold-start ``welcome`` skill.

    Subclasses populate this on the class via ``WELCOME = Welcome(...)``
    so every cold-start LLM session calling ``welcome`` gets a
    role-contextualized briefing without paying the LLM tokens to
    derive it from skill descriptions alone. See
    fr_khonliang-bus-lib_6a82732c.

    All fields are optional; missing fields are omitted from the
    welcome payload rather than producing placeholder text. An agent
    with no Welcome override still answers welcome — it just returns
    only the auto-derived fields (identity, version, skill catalog).

    Mutability invariant: the dataclass itself is frozen, but the
    ``not_responsible_for`` / ``delegates_to`` / ``entry_points``
    collections are still concrete list / dict / list — they could
    technically be mutated in-place. **Subclasses must replace the
    entire WELCOME class attribute via ``WELCOME = Welcome(...)``,
    not mutate fields on the inherited default**. The frozen=True
    above catches the most common mistake (attribute reassignment);
    convention catches the rest.
    """

    role: str = ""                                # 'development lifecycle authority'
    mission: str = ""                             # one-paragraph editorial — why this agent exists
    not_responsible_for: list[str] = field(default_factory=list)
    delegates_to: dict[str, str] = field(default_factory=dict)  # {agent: reason}
    entry_points: list[WelcomeEntryPoint] = field(default_factory=list)
    guide_skill: str = ""                         # name of a deeper-context skill (e.g. 'developer_guide')

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.role:
            out["role"] = self.role
        if self.mission:
            out["mission"] = self.mission
        if self.not_responsible_for or self.delegates_to:
            boundaries: dict[str, Any] = {}
            if self.not_responsible_for:
                boundaries["not_responsible_for"] = list(self.not_responsible_for)
            if self.delegates_to:
                boundaries["delegates_to"] = dict(self.delegates_to)
            out["boundaries"] = boundaries
        if self.entry_points:
            out["entry_points"] = [
                {"skill": ep.skill, "when_to_use": ep.when_to_use}
                for ep in self.entry_points
            ]
        if self.guide_skill:
            out["guide_skill"] = self.guide_skill
        return out


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
        self._started_at: float = time.monotonic()
        self._collect_handlers()
        # Auto-derive version from the distribution that owns the
        # subclass's module when the subclass hasn't set one explicitly.
        # Detect the override by presence in the instance or subclass
        # MRO rather than by value equality — a subclass that pins
        # ``version = "0.0.0"`` on purpose must not be overwritten just
        # because it matches BaseAgent's default sentinel.
        if not _has_explicit_version(self):
            resolved = resolve_version(type(self).__module__)
            if resolved is not None:
                self.version = resolved

    def _collect_handlers(self) -> None:
        """Discover @handler-decorated methods.

        Walks the MRO from most-base to most-derived so that a subclass
        handler for the same operation wins, even when its method name
        differs from the base's (e.g. ``handle_health_check`` overridden
        by a new ``custom_health`` method on the subclass).
        """
        for klass in reversed(type(self).__mro__):
            for attr_name, attr_value in vars(klass).items():
                if not callable(attr_value) or not hasattr(attr_value, _HANDLER_ATTR):
                    continue
                op = getattr(attr_value, _HANDLER_ATTR)
                self._handlers[op] = getattr(self, attr_name)

    # -- built-in skills --

    # Tuple (not list) so the class-level descriptors can't be accidentally
    # mutated; `_all_skills` also returns fresh Skill instances so callers
    # that mutate the result don't affect future calls.
    BUILT_IN_SKILLS: tuple[Skill, ...] = (
        Skill(
            name="health_check",
            description="Agent liveness + identity probe. Always available.",
            parameters={},
        ),
        Skill(
            name="welcome",
            description=(
                "Cold-start orientation: agent identity + role + mission "
                "+ skill catalog grouped by category prefix. Call this "
                "first to learn what this agent is for and where to "
                "drill in. Always available."
            ),
            parameters={
                "detail": {
                    "type": "string",
                    "default": "brief",
                    "description": "compact (identity + role) | brief (+ entry points + skill counts) | full (+ all skill names by category)",
                },
            },
        ),
    )

    # Subclasses override this class attribute to provide editorial
    # welcome content. The default is empty Welcome(); welcome() then
    # returns only auto-derived fields. See fr_khonliang-bus-lib_6a82732c.
    WELCOME: "Welcome" = Welcome()

    def _all_skills(self) -> list[Skill]:
        """Compose subclass skills with built-ins.

        Subclass names take precedence — a subclass can replace the
        built-in schema/description (e.g. to return a richer health
        payload) without losing the skill advertisement.

        Fresh Skill instances are constructed for built-ins so the
        class-level descriptors stay pristine across calls.
        """
        subclass_skills = self.register_skills()
        subclass_names = {s.name for s in subclass_skills}
        extras = [
            Skill(**s.to_dict())
            for s in self.BUILT_IN_SKILLS
            if s.name not in subclass_names
        ]
        return subclass_skills + extras

    @handler("welcome")
    async def handle_welcome(self, args: dict) -> dict:
        """Return cold-start orientation: identity + role + skill catalog.

        Auto-derived fields always present: agent_id, agent_type,
        version, skill_count. ``skill_categories`` is added at
        ``brief``+ detail; ``skills_by_category`` is added at ``full``
        only. Editorial fields (only when the subclass populates
        ``WELCOME``): role, mission, boundaries, entry_points,
        guide_skill.

        Detail levels:
        - ``compact``: identity + role + skill_count.
        - ``brief`` (default): + mission + boundaries + entry_points
          + skill_categories (counts per category).
        - ``full``: brief + skills_by_category (skill names grouped).

        Skills are categorized by their name's first underscore-
        separated prefix (e.g. ``git_*`` → ``git``, ``list_frs_local``
        → ``list``). Skills without a clear prefix fall under ``misc``.
        Built-ins (``health_check``, ``welcome``) get their own
        ``builtin`` bucket.
        """
        # ``args.get("detail")`` may return None (caller passed
        # ``{"detail": null}``); treat that as "not provided" and fall
        # back to the default rather than coercing to the string
        # ``'none'`` — which would produce a confusing "detail must be
        # one of …" error for what's effectively a missing arg.
        raw_detail = args.get("detail")
        if raw_detail is None:
            detail = "brief"
        else:
            detail = str(raw_detail).strip().lower() or "brief"
        if detail not in {"compact", "brief", "full"}:
            return {"error": f"detail must be one of compact|brief|full (got {detail!r})"}

        skills = self._all_skills()

        out: dict[str, Any] = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "skill_count": len(skills),
        }

        # Refuse silently-dropped editorial: a subclass that sets
        # WELCOME to anything other than a Welcome instance is a
        # programmer error and should fail loudly, not produce a
        # response missing role / mission / entry_points without
        # explanation.
        if not isinstance(self.WELCOME, Welcome):
            raise TypeError(
                f"{type(self).__name__}.WELCOME must be a Welcome instance "
                f"(got {type(self.WELCOME).__name__}). Replace the class "
                f"attribute with WELCOME = Welcome(...)."
            )
        editorial = self.WELCOME.to_dict()
        if "role" in editorial:
            out["role"] = editorial["role"]

        if detail == "compact":
            return out

        # Categorize only when the response will actually use it
        # (brief / full); compact returns above without paying the
        # sort + group cost.
        categories = self._categorize_skills(skills)

        # brief + full: add editorial mission + boundaries + entry_points
        for key in ("mission", "boundaries", "entry_points", "guide_skill"):
            if key in editorial:
                out[key] = editorial[key]

        # brief: per-category counts
        out["skill_categories"] = {
            name: len(group) for name, group in categories.items()
        }

        if detail == "full":
            out["skills_by_category"] = {
                name: [s.name for s in group]
                for name, group in categories.items()
            }
        return out

    @classmethod
    def _categorize_skills(cls, skills: list[Skill]) -> dict[str, list[Skill]]:
        """Group skills by name prefix.

        Heuristic: split on first underscore; the prefix is the
        category. Skills with no underscore land in ``misc``. Built-in
        skills (derived from ``BUILT_IN_SKILLS``) get their own
        ``builtin`` bucket for visibility — they're always present and
        shouldn't dilute a domain category's count. Sourcing the names
        from the tuple keeps this in lock-step with whatever the agent
        actually treats as a built-in.

        ``classmethod`` (not ``staticmethod``) so the lookup of
        ``cls.BUILT_IN_SKILLS`` honors any subclass override of the
        tuple — a subclass that adds a built-in via tuple extension
        gets it bucketed correctly without overriding this method.
        """
        groups: dict[str, list[Skill]] = {}
        builtin_names = {s.name for s in cls.BUILT_IN_SKILLS}
        for s in sorted(skills, key=lambda x: x.name):
            if s.name in builtin_names:
                groups.setdefault("builtin", []).append(s)
                continue
            if "_" in s.name:
                prefix = s.name.split("_", 1)[0]
            else:
                prefix = "misc"
            groups.setdefault(prefix, []).append(s)
        return dict(sorted(groups.items()))

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
            "uptime_seconds": round(time.monotonic() - self._started_at, 3),
            "bus_url": self.bus_url,
            "connected": bool(self._connector and self._connector.connected),
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
                skills=[s.to_dict() for s in skills],
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
        read_timeout = max(float(timeout), 0.0) + 5.0
        http_timeout = httpx.Timeout(
            connect=30.0,
            write=30.0,
            pool=30.0,
            read=read_timeout,
        )
        r = await self._http.post(
            f"{self.bus_url}/v1/request",
            json=payload,
            timeout=http_timeout,
        )
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

    # -- FastMCP migration bridge --

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

        This is a migration bridge. New agents should implement native @handler
        methods directly instead of wrapping MCP tools.

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
