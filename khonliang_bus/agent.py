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
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import httpx

from khonliang_bus.connector import BusConnector
from khonliang_bus.launch import capture_launch_info, capture_launch_spec
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
    # Multi-aspect introspection fields (fr_khonliang-bus-lib_6e42567d).
    # All optional; agents populate the ones that are useful for
    # cold-start LLM consumers and for sibling-agent calls. Each
    # aspect is exposed via the ``help(skills, aspect=…)`` skill
    # (fr_khonliang-bus-lib_42555320) so a consumer can ask for the
    # exact slice they need without paying for the full SkillEntry.
    #
    # ``kw_only=True`` so adding these fields doesn't shift the
    # positional-arg signature: existing call sites that passed
    # ``default_timeout_s`` (or any prior optional) positionally
    # still bind correctly. Any caller that wants the new fields
    # passes them by keyword, which is the only sensible shape for
    # editorial metadata anyway.
    prompt: str = field(default="", kw_only=True)
    examples: list[dict[str, Any]] = field(default_factory=list, kw_only=True)
    pairs_with: list[str] = field(default_factory=list, kw_only=True)
    not_appropriate_for: list[str] = field(default_factory=list, kw_only=True)
    # Mode B opt-in (fr_khonliang-bus-lib_6e42567d): when True, the bus
    # dispatcher validates incoming ``args`` against ``input_schema``
    # before invoking the handler — required-but-missing args fail
    # fast, unknown kwargs surface as errors with the accepted set
    # (closing the silent-arg-drop class structurally). Default False
    # to keep existing under-declared skill schemas working unchanged;
    # a follow-up FR audits the fleet and flips the default.
    strict_args: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        # Deep-copy so per-parameter nested dicts (type/default/description
        # objects, JSON-schema sub-trees) don't alias across Skill clones.
        # ``_all_skills`` rebuilds built-ins via ``Skill(**s.to_dict())``;
        # without deepcopy, a caller mutating ``params['detail']['default']``
        # on one clone would silently corrupt ``BUILT_IN_SKILLS`` for every
        # subsequent call (and every other agent in the process).
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
        # Aspect-field coercion. The type guards are deliberate: these
        # fields ride the registration payload and surface through the
        # ``help`` skill, so silent type drift (a stray ``False`` or
        # ``{}``) would corrupt the discoverability output for every
        # downstream consumer. Each guard fails loudly at construction
        # rather than letting a misuse propagate.
        #
        # ``prompt`` must be ``str`` (None coerces to "" — JSON null
        # for an unset aspect).
        if self.prompt is None:
            self.prompt = ""
        elif not isinstance(self.prompt, str):
            raise TypeError(
                f"Skill prompt must be a str (got "
                f"{type(self.prompt).__name__}: {self.prompt!r})."
            )

        # List-aspects must be ``list`` of the right element type
        # (None coerces to []). ``str`` gets a specific outer-container
        # hint because ``list('foo')`` silently splits into
        # ``['f','o','o']`` (the easy JSON / CLI mistake); every other
        # non-list type is rejected as the generic "expected list, got
        # X" case so dict / bool / int / etc. can't sneak through.
        # Element types are validated too — a stray non-string in
        # ``pairs_with`` or non-dict in ``examples`` would corrupt the
        # registration payload that downstream consumers (the ``help``
        # skill, sibling agents reading aspect=schema, the future
        # bus-side schema validator from fr_6e42567d Mode B) treat as
        # authoritative.
        _list_aspect_element_types: dict[str, tuple[type, ...]] = {
            "examples": (dict,),
            "pairs_with": (str,),
            "not_appropriate_for": (str,),
        }
        for aspect_name, allowed_elem_types in _list_aspect_element_types.items():
            value = getattr(self, aspect_name)
            if value is None:
                setattr(self, aspect_name, [])
                continue
            if isinstance(value, str):
                raise TypeError(
                    f"Skill {aspect_name!r} must be a list (got str). "
                    f"Wrap a single entry in a list: {aspect_name}=[{value!r}]."
                )
            if not isinstance(value, list):
                raise TypeError(
                    f"Skill {aspect_name!r} must be a list (got "
                    f"{type(value).__name__}: {value!r})."
                )
            elem_type_name = "/".join(t.__name__ for t in allowed_elem_types)
            for idx, elem in enumerate(value):
                if not isinstance(elem, allowed_elem_types):
                    raise TypeError(
                        f"Skill {aspect_name!r} entry {idx} must be "
                        f"{elem_type_name} (got "
                        f"{type(elem).__name__}: {elem!r})."
                    )

        # ``examples`` is deep-copied because each entry is a dict
        # with nested ``input_args`` / ``expected_output_shape``
        # shapes — a shallow copy would still alias the inner dicts
        # to the caller's literals. The simpler list-of-strings
        # aspects only need a shallow ``list(...)`` of the outer
        # list. After the type guards above, empty lists round-trip
        # cleanly through both calls (``deepcopy([]) == []``).
        self.examples = deepcopy(self.examples)
        self.pairs_with = list(self.pairs_with)
        self.not_appropriate_for = list(self.not_appropriate_for)
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
        # ``strict_args`` is a public flag that gates bus-side
        # validation; a truthy non-bool ('true', 1, etc.) would
        # silently turn validation on while ``to_dict()`` serializes
        # it as boolean True, masking the misuse from downstream
        # consumers reading the registration payload. Type-check
        # explicitly so misuse fails at construction.
        if not isinstance(self.strict_args, bool):
            raise TypeError(
                f"Skill strict_args must be a bool (got "
                f"{type(self.strict_args).__name__}: {self.strict_args!r})."
            )

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
            "prompt": self.prompt,
            "examples": self.examples,
            "pairs_with": self.pairs_with,
            "not_appropriate_for": self.not_appropriate_for,
        }
        payload.update({
            key: value
            for key, value in optional.items()
            if value not in (None, "", [], {})
        })
        # Omit ``strict_args`` when False so the registration payload
        # stays signal-dense (matches the existing optional-field
        # convention — the dict-comprehension filter above doesn't
        # cover bool because both True and False are outside its
        # ``(None, "", [], {})`` skip-set).
        if self.strict_args:
            payload["strict_args"] = True
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


_EMPTY_DELEGATES: Mapping[str, str] = MappingProxyType({})


# Sentinel for ``request_typed``'s remote-schema cache. A fetch
# failure stores this object so the cache hit short-circuits
# without retrying, while staying distinguishable from a
# legitimately empty schema ``{}`` (a zero-args contract under
# ``strict_args=True``). Using ``object()`` rather than ``None``
# or ``{}`` keeps "fetch failed" non-aliasable with any valid
# schema value a remote agent might declare.
_SCHEMA_FETCH_FAILED: Any = object()


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
    with no Welcome override still answers welcome — ``handle_welcome``
    detects the empty default and synthesizes an explicit
    "this agent is undocumented" response from the auto-derived
    skill catalog (see ``BaseAgent.handle_welcome``).

    Immutability: ``frozen=True`` blocks attribute reassignment, and
    ``__post_init__`` coerces the collection fields to truly immutable
    shapes (tuple / MappingProxyType). A shared default Welcome — even
    one accidentally aliased across multiple agents in the same
    process — cannot be mutated in place. Callers may still pass
    ordinary list / dict literals; the coercion happens at
    construction.
    """

    role: str = ""                                # 'development lifecycle authority'
    mission: str = ""                             # one-paragraph editorial — why this agent exists
    not_responsible_for: tuple[str, ...] = ()
    delegates_to: Mapping[str, str] = field(default_factory=lambda: _EMPTY_DELEGATES)
    entry_points: tuple[WelcomeEntryPoint, ...] = ()
    guide_skill: str = ""                         # name of a deeper-context skill (e.g. 'developer_guide')

    def __post_init__(self) -> None:
        # Coerce to immutable shapes regardless of what the caller
        # passed. ``object.__setattr__`` is the standard frozen-dataclass
        # idiom — direct ``self.x = ...`` would raise FrozenInstanceError.
        #
        # Always re-wrap ``delegates_to``: even when the caller passes an
        # existing ``MappingProxyType``, that proxy may wrap a mutable
        # dict the caller still holds — mutating that backing dict would
        # leak into the frozen Welcome. Copying via ``dict(...)`` then
        # rewrapping severs the reference.
        object.__setattr__(
            self,
            "not_responsible_for",
            tuple(self.not_responsible_for),
        )
        object.__setattr__(
            self,
            "delegates_to",
            MappingProxyType(dict(self.delegates_to)),
        )
        object.__setattr__(
            self,
            "entry_points",
            tuple(self.entry_points),
        )

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
                    "description": (
                        "compact (identity + version + skill_count + "
                        "role) | brief (+ mission + boundaries + "
                        "entry_points + skill_categories — when the "
                        "agent populates WELCOME; otherwise the "
                        "fallback emits documentation_gaps and a "
                        "synthesized missing-doc role/mission) | full "
                        "(+ skills_by_category, plus "
                        "skill_documentation_gaps for undocumented "
                        "agents)"
                    ),
                },
            },
        ),
        Skill(
            name="help",
            description=(
                "Per-skill introspection: arg schema + descriptive "
                "info + (when populated) prompt template / examples / "
                "pairs_with / not_appropriate_for. Pass ``skill_names`` "
                "for batch lookup or ``[]`` for the full catalog. Use "
                "``aspect=`` for fine-grained reads (one field per "
                "skill). Always available."
            ),
            parameters={
                "skill_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": (
                        "Names to look up. Empty list returns every "
                        "registered skill on this agent. Unknown "
                        "names appear in the response with "
                        "``found: false`` rather than being dropped."
                    ),
                },
                "detail": {
                    "type": "string",
                    "default": "brief",
                    "description": (
                        "compact (name + description) | brief (+ "
                        "parameters + input_schema + capability when "
                        "set) | full (+ aliases + since + populated "
                        "aspect fields: prompt / examples / "
                        "pairs_with / not_appropriate_for)"
                    ),
                },
                "aspect": {
                    "type": "string",
                    "default": "",
                    "description": (
                        "Optional aspect-mode read. One of: brief, "
                        "help, schema, prompt, examples, pairs_with, "
                        "not_appropriate_for. When set, response is a "
                        "flat list of ``{name, found, value}`` entries "
                        "carrying just the requested aspect — token-"
                        "efficient for callers that already know what "
                        "slice they need (e.g. an LLM asking for "
                        "``prompt`` to adapt a template)."
                    ),
                },
            },
        ),
    )

    # Subclasses override this class attribute to provide editorial
    # welcome content. The default is empty Welcome(); ``handle_welcome``
    # detects the empty case and synthesizes a "this agent is
    # undocumented — here is what we know, please document" response
    # from the auto-derived skill catalog. See
    # fr_khonliang-bus-lib_6a82732c.
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

        Undocumented-agent fallback: when ``WELCOME`` is the empty
        bus-lib default (subclass forgot to populate), the response
        substitutes synthesized ``role`` / ``mission`` markers
        announcing the missing editorial, plus a
        ``documentation_gaps`` list of agent-level fields that need
        filling and a ``skill_documentation_gaps`` map of per-skill
        gaps (full detail). The agent stays callable; the cold-start
        LLM can see what's there, what's missing, and ask for the
        agent / skills to be documented.
        """
        # ``args.get("detail")`` may return None (caller passed
        # ``{"detail": null}``); treat that as "not provided" and fall
        # back to the default rather than coercing to the string
        # ``'none'`` — which would produce a confusing "detail must be
        # one of …" error for what's effectively a missing arg. Per-
        # call wrapper normalization (None / non-dict args) lives in
        # ``_dispatch_request`` (fr_khonliang-bus-lib_d900f0b5); this
        # handler is guaranteed to receive a dict.
        raw_detail = args.get("detail")
        if raw_detail is None:
            detail = "brief"
        else:
            detail = str(raw_detail).strip().lower() or "brief"
        if detail not in {"compact", "brief", "full"}:
            return {"error": f"detail must be one of compact|brief|full (got {detail!r})"}

        # ``_skills`` is a reserved internal arg used by ``start()`` to pass
        # an already-computed skills list. Skipping the re-call into
        # ``_all_skills()`` avoids a subtle divergence: if ``register_skills()``
        # is dynamic / side-effecting, two calls could disagree, making the
        # auto-published welcome's catalog drift from what bus actually
        # registered. External callers don't supply this key; the bus's
        # request schema doesn't advertise it. Closes Copilot PR #25 R2 / R3
        # — by routing start() through ``handle_welcome`` (rather than the
        # private ``_compose_welcome``), an agent that overrides
        # ``handle_welcome`` to customize welcome shape sees its
        # customization reflected in the persisted-at-register welcome too.
        precomputed = args.get("_skills")
        skills = (
            precomputed
            if isinstance(precomputed, list)
            else self._all_skills()
        )
        return self._compose_welcome(skills, detail)

    def _compose_welcome(self, skills: list[Skill], detail: str) -> dict[str, Any]:
        """Build the welcome payload from a precomputed skills list.

        Factored out of ``handle_welcome`` so ``start()`` can compute
        ``skills = self._all_skills()`` exactly once and pass it to both
        the register handshake AND the auto-published welcome — avoiding
        the prior subtle bug where ``register_skills()`` overrides with
        side-effects could yield a welcome whose ``skill_count`` /
        ``skills_by_category`` disagreed with what bus actually
        registered. fr_khonliang-bus_f96722dd / Copilot PR #25 R2.

        ``detail`` is one of ``compact``/``brief``/``full`` —
        ``handle_welcome`` already validates the value.
        """
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
        is_undocumented = not editorial

        if is_undocumented:
            out["role"] = (
                "(undocumented agent — WELCOME not populated)"
            )
        elif "role" in editorial:
            out["role"] = editorial["role"]

        if detail == "compact":
            return out

        # Categorize only when the response will actually use it
        # (brief / full); compact returns above without paying the
        # sort + group cost.
        categories = self._categorize_skills(skills)

        if is_undocumented:
            out["mission"] = (
                "This agent has not declared a WELCOME editorial. "
                "The skill catalog below shows what it CAN do, but "
                "role / mission / boundaries / entry_points / "
                "guide_skill are unset. Populate "
                "``WELCOME = Welcome(role=..., mission=..., "
                "entry_points=[...])`` on the agent class so cold-"
                "start LLM sessions can orient without reading "
                "source. See fr_khonliang-bus-lib_6a82732c. Per-"
                "skill documentation gaps are listed at "
                "``detail=full``; please fill in any skill that "
                "appears there."
            )
            out["documentation_gaps"] = self._welcome_doc_gaps()
        else:
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
            if is_undocumented:
                gaps_by_skill = {
                    s.name: gaps
                    for s in skills
                    for gaps in (self._skill_doc_gaps(s),)
                    if gaps
                }
                if gaps_by_skill:
                    out["skill_documentation_gaps"] = gaps_by_skill
        return out

    @staticmethod
    def _welcome_doc_gaps() -> list[str]:
        """Agent-level WELCOME fields that should be populated.

        Returned verbatim in the fallback response so a tooling
        consumer (or a documenting LLM) can build a checklist
        without parsing prose. Order mirrors how the editorial
        fields appear in a populated response: orientation first
        (role, mission), then boundaries, then routing
        (entry_points, guide_skill).
        """
        return [
            "role",
            "mission",
            "boundaries (not_responsible_for + delegates_to)",
            "entry_points",
            "guide_skill",
        ]

    @staticmethod
    def _skill_doc_gaps(skill: Skill) -> list[str]:
        """Per-skill documentation gaps.

        Reports what's missing so cold-start consumers know which
        skills lack the editorial they'd need to call them
        confidently. Empty list = fully documented; the caller
        omits skills with zero gaps from the response.
        """
        gaps: list[str] = []
        if not skill.description.strip():
            gaps.append("description is empty")
        if not skill.parameters and not skill.input_schema:
            gaps.append("parameters / input_schema not declared")
        if not skill.capability:
            gaps.append("capability tag not set")
        return gaps

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

    # -- help skill (fr_khonliang-bus-lib_42555320 + 6e42567d) --

    # Aspect → Skill-attribute mapping consumed by ``handle_help``'s
    # aspect-mode short-circuit. Centralized so each aspect has one
    # definition shared between schema validation and response
    # construction. ``brief`` and ``help`` reuse ``description`` —
    # bus-lib v1 doesn't separate one-line / long-form prose; when
    # a future Phase parses docstring sections (``Notes:``, ``Raises:``,
    # etc.) the ``help`` aspect will route to that richer field.
    _ASPECT_FIELDS: dict[str, str] = {
        "brief": "description",
        "help": "description",
        "schema": "input_schema",
        "prompt": "prompt",
        "examples": "examples",
        "pairs_with": "pairs_with",
        "not_appropriate_for": "not_appropriate_for",
    }

    @handler("help")
    async def handle_help(self, args: dict) -> dict:
        """Per-skill introspection — arg schema + descriptive info.

        Two response modes:

        - **SkillEntry mode** (``aspect=`` empty): returns a list of
          ``SkillEntry`` dicts with the canonical name, description,
          arguments (parameters / input_schema), and any populated
          aspect fields. ``detail=compact|brief|full`` selects how
          much to include.

        - **Aspect mode** (``aspect=brief|help|schema|prompt|examples|
          pairs_with|not_appropriate_for``): returns a flat list of
          ``{name, found, value}`` per requested skill, carrying just
          the requested aspect. Token-efficient for callers that
          already know which slice they need (e.g. an LLM asking for
          ``prompt`` to adapt a template).

        Unknown skill names appear in the response with
        ``found: false`` rather than being dropped, so the caller
        learns which names missed.

        ``skill_names=[]`` is shorthand for "every registered skill on
        this agent" — equivalent to a full catalog dump.
        """
        raw_names = args.get("skill_names", [])
        if not isinstance(raw_names, list):
            return {
                "error": (
                    f"skill_names must be a list (got "
                    f"{type(raw_names).__name__})"
                )
            }
        skill_names = [str(n) for n in raw_names]

        # Defaulting: only treat ``None`` as "not provided". Falsey
        # values like ``0`` / ``False`` are caller-supplied bad values
        # and should hit the validation envelope below — falsey-coalesce
        # would silently substitute the default and mask the bug. The
        # post-strip ``or default`` is narrow: only kicks in when the
        # caller passed an explicit empty/whitespace string, matching
        # ``handle_welcome``'s convention.
        raw_detail = args.get("detail")
        if raw_detail is None:
            detail = "brief"
        else:
            detail = str(raw_detail).strip().lower() or "brief"
        if detail not in {"compact", "brief", "full"}:
            return {"error": f"detail must be one of compact|brief|full (got {detail!r})"}

        raw_aspect = args.get("aspect")
        if raw_aspect is None:
            aspect = ""
        else:
            aspect = str(raw_aspect).strip().lower()
        if aspect and aspect not in self._ASPECT_FIELDS:
            allowed = "|".join(sorted(self._ASPECT_FIELDS))
            return {"error": f"aspect must be one of {allowed} (got {aspect!r})"}

        skills = self._all_skills()
        by_name = {s.name: s for s in skills}

        # Empty list = full catalog. Sort alphabetically here for
        # determinism — ``_all_skills`` itself preserves subclass
        # registration order followed by ``BUILT_IN_SKILLS`` tuple
        # order, which would surface different orderings to different
        # agents. Sorting once at the response-shaping step gives a
        # stable, comparable catalog across the fleet without forcing
        # ``_all_skills`` callers (e.g. categorization, welcome) to pay
        # for ordering they don't need.
        if not skill_names:
            target_names = sorted(by_name)
        else:
            target_names = skill_names

        # Aspect-mode short-circuit: flat list keyed by skill name.
        if aspect:
            attr = self._ASPECT_FIELDS[aspect]
            results = []
            for name in target_names:
                skill = by_name.get(name)
                if skill is None:
                    results.append({
                        "name": name,
                        "found": False,
                        "reason": "no skill with that name on this agent",
                    })
                    continue
                value = getattr(skill, attr)
                # ``schema`` resolves to input_schema; fall back to
                # parameters when input_schema is empty (the default
                # for legacy Skill registrations that only set
                # parameters).
                if aspect == "schema" and not value:
                    value = skill.parameters
                results.append({
                    "name": name,
                    "found": True,
                    "value": value,
                })
            return {"aspect": aspect, "skills": results}

        # SkillEntry mode.
        entries = []
        for name in target_names:
            skill = by_name.get(name)
            if skill is None:
                entries.append({
                    "name": name,
                    "found": False,
                    "reason": "no skill with that name on this agent",
                })
                continue
            entry: dict[str, Any] = {
                "name": skill.name,
                "found": True,
                "description": skill.description,
            }
            if detail in {"brief", "full"}:
                entry["parameters"] = skill.parameters
                # Surface input_schema only when it diverges from
                # ``parameters`` (matches the registration-payload
                # convention in ``Skill.to_dict``).
                if skill.input_schema and skill.input_schema != skill.parameters:
                    entry["input_schema"] = skill.input_schema
                if skill.capability:
                    entry["capability"] = skill.capability
            if detail == "full":
                # Aspect fields are advisory; emit only when populated
                # so the response stays signal-dense for skills that
                # don't bother declaring them.
                if skill.prompt:
                    entry["prompt"] = skill.prompt
                if skill.examples:
                    entry["examples"] = skill.examples
                if skill.pairs_with:
                    entry["pairs_with"] = skill.pairs_with
                if skill.not_appropriate_for:
                    entry["not_appropriate_for"] = skill.not_appropriate_for
                if skill.aliases:
                    entry["aliases"] = skill.aliases
                if skill.since:
                    entry["since"] = skill.since
            entries.append(entry)
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "skills": entries,
        }

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

        # Capture launch metadata at startup. ``launch_spec`` carries the
        # declarative how-to-launch fields (executable, args, cwd, config) —
        # what the bus needs to spawn another process matching this agent_id.
        # ``launch_info`` carries the runtime snapshot (started_at, git
        # info) — what process is currently serving this agent_id; pid is
        # the top-level register payload field, not duplicated in
        # launch_info. Bus joins both to surface canonical-vs-ad-hoc
        # provenance. See fr_khonliang-bus-lib_2cfc0de6 (launch_spec) +
        # fr_khonliang-bus-lib_cccaa6a9 (launch_info).
        launch_spec = capture_launch_spec()
        launch_info = capture_launch_info()

        # Compute welcome at startup so the bus can persist it alongside the
        # registration. Same shape as ``handle_welcome`` returns (single
        # source of truth — no per-agent re-implementation). Bus stores this
        # in a survives-deregister catalog so ``bus_welcome`` super-skill +
        # ``GET /v1/agents/<id>/welcome`` work even after the agent process
        # exits. See fr_khonliang-bus_f96722dd.
        #
        # Route through ``handle_welcome`` (not the private ``_compose_welcome``
        # directly) so an agent that overrides ``handle_welcome`` to customize
        # welcome shape sees that customization reflected in the persisted-
        # at-register welcome too. The reserved ``_skills`` arg passes the
        # already-computed skills list — avoids a second ``_all_skills()``
        # call where a dynamic ``register_skills()`` override could otherwise
        # yield a different result and make the auto-published welcome's
        # catalog drift from what bus actually registered.
        # fr_khonliang-bus_f96722dd / Copilot PR #25 R2 + R3.
        #
        # Defensive: a malformed Welcome / handle_welcome override must not
        # block registration. Three independent failure modes get the same
        # treatment (log + welcome=None + continue), so a buggy welcome
        # never holds an agent off the bus:
        #   (1) the welcome call raises.
        #   (2) it returns a non-dict (bus's persist path expects dict).
        #   (3) it returns a dict containing non-JSON-serializable values —
        #       would later raise inside connect_and_register's json.dumps.
        welcome: dict | None = None
        try:
            candidate = await self.handle_welcome({"detail": "full", "_skills": skills})
        except Exception:
            logger.exception(
                "Agent %s: handle_welcome raised at start(); "
                "registering without welcome payload",
                self.agent_id,
            )
        else:
            if not isinstance(candidate, dict):
                logger.warning(
                    "Agent %s: handle_welcome returned %s, not dict; "
                    "registering without welcome payload",
                    self.agent_id, type(candidate).__name__,
                )
            else:
                try:
                    json.dumps(candidate)  # serializability check
                except (TypeError, ValueError):
                    logger.exception(
                        "Agent %s: handle_welcome returned a non-JSON-"
                        "serializable dict; registering without welcome "
                        "payload",
                        self.agent_id,
                    )
                else:
                    welcome = candidate

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
                launch_spec=launch_spec,
                launch_info=launch_info,
                welcome=welcome,
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

        Wrapper-shape normalization (fr_khonliang-bus-lib_d900f0b5): the
        bus may legitimately deliver ``args=null`` (JSON null) or, in a
        malformed-input case, a non-dict shape (list, scalar). Handlers
        should never have to defend against either — the dispatcher
        normalizes once so every @handler method receives a dict.

        Per-skill schema validation (fr_khonliang-bus-lib_6e42567d Mode
        B): when the registered Skill declares ``strict_args=True``,
        the dispatcher validates ``args`` against the skill's
        ``input_schema`` (or ``parameters`` when input_schema is empty)
        before invoking the handler. Required-but-missing args fail
        fast; unknown kwargs surface as errors with the accepted set —
        closing the silent-arg-drop class structurally
        (bug_developer_ad60dca4 / b5fd44ce / a349c77b). Skills with
        ``strict_args=False`` (the default) keep their existing
        permissive behavior; a follow-up FR audits the fleet and flips
        the default once schemas are caught up.
        """
        operation = msg.get("operation", "")
        raw_args = msg.get("args", {})

        if raw_args is None:
            args: dict = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return {
                "error": (
                    f"args must be an object (got {type(raw_args).__name__})"
                )
            }

        handler_fn = self._handlers.get(operation)
        if not handler_fn:
            raise ValueError(f"unknown operation: {operation}")

        # Strict-args validation runs only for skills that opted in.
        # Lookup is lazy-cached on first dispatch; fleet skill counts
        # are small (<100 per agent) so the dict-build cost is one-time.
        skill = self._skill_for_operation(operation)
        if skill is not None and skill.strict_args:
            error = self._validate_args_against_schema(skill, args)
            if error is not None:
                return {"error": error}

        return await handler_fn(args)

    def _skill_for_operation(self, operation: str) -> "Skill | None":
        """Lazy-cached skill lookup by operation name.

        Cache invalidation is intentionally absent: ``register_skills``
        runs once at agent construction and the result is stable for
        the lifetime of the process. If a future code path mutates
        the registered skills mid-run, that path is responsible for
        clearing ``self._skills_by_op_cache``.

        On first build, warn loudly when a skill declares
        ``strict_args=True`` but its name doesn't match any registered
        handler operation. Without the warning, the dispatcher would
        silently look up by operation, miss the skill, and skip the
        validation the author asked for — exactly the silent-bypass
        bug the strict_args opt-in is meant to close.
        """
        cache = getattr(self, "_skills_by_op_cache", None)
        if cache is None:
            cache = {}
            mismatched: list[str] = []
            for skill in self._all_skills():
                cache[skill.name] = skill
                if skill.strict_args and skill.name not in self._handlers:
                    mismatched.append(skill.name)
            if mismatched:
                logger.warning(
                    "Agent %s: skills declared strict_args=True but have "
                    "no matching @handler operation; strict-args "
                    "validation will not run for them. Skills: %s. "
                    "Either rename the Skill so its name == the @handler "
                    "operation, or remove strict_args until the handler "
                    "is registered.",
                    self.agent_id,
                    ", ".join(sorted(mismatched)),
                )
            self._skills_by_op_cache = cache
        return cache.get(operation)

    # JSON-Schema-style type-name → Python isinstance tuple. ``bool`` is
    # excluded from ``integer`` / ``number`` because it's a subclass of
    # ``int`` and accepting True/False where an integer is expected is
    # almost always a caller bug worth surfacing.
    _SCHEMA_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }

    @classmethod
    def _validate_args_against_schema(
        cls, skill: "Skill", args: dict
    ) -> str | None:
        """Validate ``args`` against the skill's declared schema.

        Returns an error string on the first failure (caller wraps in
        ``{"error": ...}``) or ``None`` when valid. Three failure
        classes, all of which closed silent-drop bugs in the past:

        - **Unknown kwargs**: ``args`` carries a key the schema
          doesn't declare. Lists the accepted-set so the caller sees
          what they should have used.
        - **Required-but-missing**: schema declares
          ``required=True`` and the key isn't present.
        - **Wrong type**: declared ``type`` doesn't match the runtime
          value's class. Surfaces the declared and actual types.

        Schema lookup falls back to ``parameters`` when
        ``input_schema`` is empty — legacy registrations that only
        set ``parameters`` still validate correctly.

        An explicitly-empty schema (``{}``) is treated as a valid
        zero-args contract under ``strict_args=True``: the skill
        author has declared "this takes nothing", so any kwarg the
        caller supplies is rejected as unknown. Health-check-shape
        skills that take no arguments use this path. (We can't tell
        ``Skill(parameters={})`` from the dataclass default-empty at
        runtime, but the strict_args opt-in makes the author's
        intent explicit either way.)
        """
        schema = skill.input_schema or skill.parameters
        accepted = set(schema)
        unknown = set(args) - accepted
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            accepted_list = (
                ", ".join(sorted(accepted))
                or "(none — skill declared empty schema)"
            )
            return (
                f"unknown args for {skill.name!r}: {unknown_list}. "
                f"Accepted: {accepted_list}."
            )

        for field_name, declared in schema.items():
            if not isinstance(declared, dict):
                continue
            present = field_name in args
            required = bool(declared.get("required", False))
            if not present:
                if required:
                    return (
                        f"missing required arg for {skill.name!r}: "
                        f"{field_name!r}."
                    )
                continue
            declared_type = declared.get("type")
            if not declared_type or declared_type not in cls._SCHEMA_TYPE_CHECKS:
                continue
            value = args[field_name]
            if not cls._SCHEMA_TYPE_CHECKS[declared_type](value):
                # Surface the declared and actual types only — NOT the
                # value itself. Validation errors flow through the bus
                # response envelope and into logs; echoing the offending
                # value would leak large payloads (the same args path
                # carries 50KB messages, paper PDFs, etc.) and any
                # sensitive content (API keys, paper text, user PII)
                # into downstream storage / context windows. The caller
                # already has the value; the error only needs to tell
                # them which arg + what shape was expected.
                return (
                    f"arg {field_name!r} for {skill.name!r} must be "
                    f"{declared_type} (got {type(value).__name__})."
                )
        return None

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

    async def request_typed(
        self,
        agent_type: str,
        operation: str,
        args: dict[str, Any] | None = None,
        timeout: float = 30.0,
        response_mode: str = "raw",
    ) -> dict:
        """Caller-side schema validation companion to ``request``
        (fr_khonliang-bus-lib_6e42567d Mode B caller side).

        Looks up the destination skill's input_schema (via the
        target's built-in ``help`` skill, lazy-cached per
        ``(agent_type, operation)``), validates ``args`` locally
        with the same logic the dispatcher uses, then dispatches
        only when the call shape is correct. Surfaces silent-drop
        bugs BEFORE the network round-trip — the caller sees the
        same error envelope a strict_args=True receiver would
        emit, but without paying the bus + remote hop.

        On schema-fetch failure (target down, help-skill missing,
        unknown skill) the call falls through to the existing
        ``request`` path with a one-time warning so the caller
        isn't blocked on validation infrastructure that's
        independently broken. Cache cleared via
        ``invalidate_schema_cache`` when a remote agent restarts
        with a changed schema.
        """
        # Normalize args once at the entry point — None becomes the
        # empty dict (the no-args contract), non-dict short-circuits
        # with the same error envelope ``_dispatch_request`` emits.
        # Without this normalization a list (or other unhashable /
        # non-mapping shape) would either crash the local validator
        # (``set(list_with_unhashables)``), validate against the
        # wrong shape (set-membership against schema field names),
        # or be silently coerced by ``request`` to ``{}`` — three
        # different ways to mask the caller bug. Use the normalized
        # dict for both validation AND the outgoing dispatch so the
        # local and remote paths agree on the call shape.
        if args is None:
            normalized_args: dict[str, Any] = {}
        elif isinstance(args, dict):
            normalized_args = args
        else:
            return {
                "error": (
                    f"args must be an object (got {type(args).__name__})"
                )
            }

        cache_key = (agent_type, operation)
        cache = getattr(self, "_remote_schema_cache", None)
        if cache is None:
            cache = {}
            self._remote_schema_cache = cache

        if cache_key not in cache:
            fetched = await self._fetch_remote_skill_schema(
                agent_type, operation, timeout=timeout
            )
            new_value = (
                fetched if fetched is not None else _SCHEMA_FETCH_FAILED
            )
            # Stampede guard: two coroutines on the same uncached key
            # can both miss and both fetch. The first to return wins
            # the slot, EXCEPT when the existing entry is the
            # failure sentinel and we have a real schema — in that
            # case overwrite, so a transient sibling failure can't
            # poison the cache for a successful fetch in flight.
            # Conversely, never overwrite a real schema with the
            # sentinel. ``None`` from the fetch means "couldn't
            # fetch" — cached as ``_SCHEMA_FETCH_FAILED`` so
            # subsequent calls short-circuit without retrying
            # (avoiding retry storms) while staying distinguishable
            # from a legitimately empty schema ``{}`` (a valid
            # zero-args contract under ``strict_args=True``).
            if cache_key not in cache:
                cache[cache_key] = new_value
            elif (
                cache[cache_key] is _SCHEMA_FETCH_FAILED
                and new_value is not _SCHEMA_FETCH_FAILED
            ):
                cache[cache_key] = new_value
        schema = cache[cache_key]

        # Validate whenever the schema was successfully fetched —
        # even ``{}`` (zero-args contract). Only the explicit
        # fetch-failure sentinel skips validation.
        if schema is not _SCHEMA_FETCH_FAILED:
            # Synthesize a minimal Skill so the existing validator
            # can be reused verbatim — same error shape as the
            # dispatcher-side path, same field-name semantics.
            # A malformed remote schema (unexpected shapes, types
            # the Skill dataclass rejects) shouldn't block the
            # caller — fall through to unvalidated dispatch the
            # same way a schema fetch failure does, with a
            # warning so the operator notices.
            try:
                stand_in = Skill(
                    name=operation, parameters=schema, strict_args=True,
                )
            except Exception as exc:
                logger.warning(
                    "request_typed: remote schema for %s.%s rejected by "
                    "Skill construction (%s) — falling back to "
                    "unvalidated dispatch.",
                    agent_type, operation, exc,
                )
            else:
                error = self._validate_args_against_schema(
                    stand_in, normalized_args,
                )
                if error is not None:
                    return {"error": error}

        return await self.request(
            agent_type=agent_type,
            operation=operation,
            args=normalized_args,
            timeout=timeout,
            response_mode=response_mode,
        )

    async def _fetch_remote_skill_schema(
        self,
        agent_type: str,
        operation: str,
        timeout: float = 30.0,
    ) -> dict[str, Any] | None:
        """Fetch the destination's declared schema for one operation.

        Uses the built-in ``help(skill_names=[op], aspect='schema')``
        round-trip — every fleet agent inherits help from bus-lib
        (fr_khonliang-bus-lib_42555320), so this works without
        per-agent cooperation. Returns the schema dict on success
        (including ``{}`` when the skill genuinely takes no args)
        or ``None`` on any of: help skill unavailable, operation
        unknown on target, transport error. The caller maps
        ``None`` to the ``_SCHEMA_FETCH_FAILED`` sentinel in the
        cache to short-circuit retries while staying distinct
        from a real empty schema.
        """
        try:
            response = await self.request(
                agent_type=agent_type,
                operation="help",
                args={"skill_names": [operation], "aspect": "schema"},
                timeout=timeout,
            )
        except Exception as exc:
            logger.warning(
                "request_typed: schema fetch failed for %s.%s: %s — "
                "falling back to unvalidated dispatch.",
                agent_type, operation, exc,
            )
            return None

        # ``self.request`` returns ``r.json()`` — could be a dict (the
        # normal envelope), a list, a scalar, or None for malformed /
        # unexpected responses. Reject anything that isn't a dict
        # before probing, so a downstream ``.get`` doesn't raise
        # AttributeError out of the schema-fetch path. Warn on each
        # non-dispatch outcome so the docstring's "fall through with
        # a warning" promise holds for every return-None path, not
        # just transport-exception. Spam is bounded by the caller's
        # negative-cache: at most one warning per (agent_type, op).
        if not isinstance(response, dict):
            logger.warning(
                "request_typed: schema fetch for %s.%s returned non-dict "
                "envelope (%s) — falling back to unvalidated dispatch.",
                agent_type, operation, type(response).__name__,
            )
            return None

        # The bus envelope wraps the actual response; the help skill's
        # aspect-mode payload nests under ``result`` (the standard
        # MCP shape) or appears at the top level depending on the
        # bus's response transform. Probe both.
        body = response.get("result", response)
        if not isinstance(body, dict):
            logger.warning(
                "request_typed: schema fetch for %s.%s returned non-dict "
                "result body — falling back to unvalidated dispatch.",
                agent_type, operation,
            )
            return None
        skills = body.get("skills")
        if not isinstance(skills, list) or not skills:
            logger.warning(
                "request_typed: schema fetch for %s.%s missing 'skills' "
                "list (target may not implement help skill) — falling "
                "back to unvalidated dispatch.",
                agent_type, operation,
            )
            return None
        entry = skills[0]
        if not isinstance(entry, dict) or not entry.get("found"):
            reason = (
                entry.get("reason", "unknown skill")
                if isinstance(entry, dict)
                else "malformed help entry"
            )
            logger.warning(
                "request_typed: schema fetch for %s.%s reports skill not "
                "found (%s) — falling back to unvalidated dispatch.",
                agent_type, operation, reason,
            )
            return None
        value = entry.get("value")
        if not isinstance(value, dict):
            logger.warning(
                "request_typed: schema fetch for %s.%s returned non-dict "
                "schema value — falling back to unvalidated dispatch.",
                agent_type, operation,
            )
            return None
        return value

    def invalidate_schema_cache(
        self,
        agent_type: str | None = None,
        operation: str | None = None,
    ) -> None:
        """Drop one or all entries from the typed-request schema cache.

        Call after a remote agent restarts with a changed schema, or
        when the cache is suspected stale. ``agent_type=None`` clears
        every entry; passing ``agent_type`` alone clears all skills
        on that agent; passing both clears the single entry.
        """
        cache = getattr(self, "_remote_schema_cache", None)
        if cache is None:
            return
        if agent_type is None:
            cache.clear()
            return
        if operation is None:
            keys = [k for k in cache if k[0] == agent_type]
        else:
            keys = [(agent_type, operation)]
        for key in keys:
            cache.pop(key, None)

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
