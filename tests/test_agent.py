"""Tests for BaseAgent: skills, handlers, connector-based lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from khonliang_bus import (
    BaseAgent,
    Collaboration,
    Skill,
    Welcome,
    WelcomeEntryPoint,
    handler,
)


class EchoAgent(BaseAgent):
    agent_type = "echo"
    module_name = "tests.test_agent"
    version = "0.2.0"

    def register_skills(self):
        return [
            Skill("echo", "Echo the input", {"text": {"type": "string"}}, since="0.1.0"),
            Skill("upper", "Uppercase the input", since="0.2.0"),
            Skill("fail", "Always fails"),
        ]

    def register_collaborations(self):
        return [
            Collaboration(
                "echo_and_research",
                "Echo then research",
                requires={"researcher": ">=0.5.0"},
                steps=[{"call": "echo.echo"}, {"call": "researcher.find_papers"}],
            ),
        ]

    @handler("echo")
    async def echo(self, args):
        return {"echoed": args.get("text", "")}

    @handler("upper")
    async def upper(self, args):
        return {"result": args.get("text", "").upper()}

    @handler("fail")
    async def fail(self, args):
        raise ValueError("intentional failure")


@pytest.fixture
def agent():
    return EchoAgent(agent_id="echo-test", bus_url="http://localhost:9999")


# -- skills --

def test_skills_registered(agent):
    skills = agent.register_skills()
    assert len(skills) == 3
    assert skills[0].name == "echo"
    assert skills[0].since == "0.1.0"


def test_skill_has_parameters(agent):
    skills = agent.register_skills()
    echo = next(s for s in skills if s.name == "echo")
    assert "text" in echo.parameters


# -- collaborations --

def test_collaborations_registered(agent):
    collabs = agent.register_collaborations()
    assert len(collabs) == 1
    assert collabs[0].requires == {"researcher": ">=0.5.0"}


# -- handlers --

def test_handlers_collected(agent):
    # EchoAgent declares echo/upper/fail; health_check + welcome + help
    # come for free from BaseAgent's BUILT_IN_SKILLS.
    assert set(agent._handlers) == {
        "echo", "upper", "fail", "health_check", "welcome", "help",
    }


@pytest.mark.asyncio
async def test_dispatch_success(agent):
    result = await agent._dispatch_request({"operation": "echo", "args": {"text": "hello"}})
    assert result == {"echoed": "hello"}


@pytest.mark.asyncio
async def test_dispatch_failure_raises(agent):
    """Handler exceptions propagate — connector catches them and sends error to bus."""
    with pytest.raises(ValueError, match="intentional"):
        await agent._dispatch_request({"operation": "fail", "args": {}})


@pytest.mark.asyncio
async def test_dispatch_unknown_operation_raises(agent):
    with pytest.raises(ValueError, match="unknown operation"):
        await agent._dispatch_request({"operation": "nope", "args": {}})


# -- dispatcher args-shape normalization (fr_khonliang-bus-lib_d900f0b5) --


@pytest.mark.asyncio
async def test_dispatch_normalizes_null_args_to_empty_dict(agent):
    """``args=null`` is a legitimate transport shape — the dispatcher
    normalizes to an empty dict so handlers never see None."""
    result = await agent._dispatch_request({"operation": "echo", "args": None})
    # echo's handler reads args.get("text", "") — None would have raised
    # AttributeError; with normalization it returns the empty default.
    assert result == {"echoed": ""}


@pytest.mark.asyncio
async def test_dispatch_normalizes_missing_args_to_empty_dict(agent):
    """A request envelope without an ``args`` key at all behaves like
    ``args={}`` — the dispatcher's ``msg.get('args', {})`` already
    handles this; pinning it so the contract is explicit."""
    result = await agent._dispatch_request({"operation": "echo"})
    assert result == {"echoed": ""}


@pytest.mark.asyncio
async def test_dispatch_rejects_non_dict_args_with_validation_envelope(agent):
    """Non-dict / non-None args (list, scalar, string) produce a clean
    validation envelope. The handler is never invoked — the dispatcher
    short-circuits with an ``error`` field naming the bad type."""
    for bad in (["not", "a", "dict"], "string", 42, True):
        result = await agent._dispatch_request(
            {"operation": "echo", "args": bad}
        )
        assert "error" in result
        assert "object" in result["error"]
        # The bad-type name must surface so the caller can see what
        # they sent without re-reading their own code.
        assert type(bad).__name__ in result["error"]


@pytest.mark.asyncio
async def test_dispatch_validates_before_unknown_op_check(agent):
    """Dispatcher rejects bad-shape args even when the operation
    itself is unknown — the validation envelope wins over the unknown-
    operation ValueError. Either ordering is defensible; pin the
    chosen one so future refactors don't accidentally invert it."""
    result = await agent._dispatch_request(
        {"operation": "nope", "args": ["bad"]}
    )
    assert "error" in result
    assert "object" in result["error"]


# -- CLI --

def test_from_cli():
    a = EchoAgent.from_cli(["--id", "my-echo", "--bus", "http://localhost:8787"])
    assert a.agent_id == "my-echo"
    assert a.bus_url == "http://localhost:8787"


# -- start fails when bus unreachable --

@pytest.mark.asyncio
async def test_start_fails_when_bus_unreachable():
    a = EchoAgent(agent_id="test-fatal", bus_url="http://localhost:1")
    with pytest.raises(RuntimeError, match="failed to connect"):
        await a.start()


# -- helpers exist --

def test_publish_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.publish)


@pytest.mark.asyncio
async def test_publish_raises_when_not_connected(agent):
    """publish() should raise RuntimeError before the agent connects to the bus."""
    with pytest.raises(RuntimeError, match="not connected"):
        await agent.publish("some.topic", {"data": 1})


def test_nack_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.nack)


def test_request_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.request)


@pytest.mark.asyncio
async def test_request_sets_http_timeout_from_bus_timeout(agent):
    captured = {}

    class FakeResponse:
        def json(self):
            return {"result": "ok"}

    class FakeHTTP:
        async def post(self, url, *, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

    agent._http = FakeHTTP()

    result = await agent.request(agent_type="researcher", operation="slow", timeout=90)

    assert result == {"result": "ok"}
    assert captured["url"] == "http://localhost:9999/v1/request"
    assert captured["json"]["timeout"] == 90
    assert captured["timeout"].read == 95.0
    assert captured["timeout"].connect == 30.0
    assert captured["timeout"].write == 30.0
    assert captured["timeout"].pool == 30.0


@pytest.mark.asyncio
async def test_request_buffer_preserves_bus_timeout_payload(agent):
    class FakeResponse:
        def json(self):
            return {"error": "timeout", "trace_id": "t-slow"}

    class FakeHTTP:
        async def post(self, url, *, json=None, timeout=None):
            assert json["timeout"] == 1
            assert timeout.read == 6.0
            await asyncio.sleep(0.01)
            return FakeResponse()

    agent._http = FakeHTTP()

    result = await agent.request(agent_type="researcher", operation="slow", timeout=1)

    assert result == {"error": "timeout", "trace_id": "t-slow"}


def test_session_context_helpers_exist(agent):
    assert asyncio.iscoroutinefunction(agent.get_session_context)
    assert asyncio.iscoroutinefunction(agent.update_session_context)


def test_report_gap_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.report_gap)


@pytest.mark.asyncio
async def test_report_gap_raises_when_not_connected(agent):
    """report_gap() should raise RuntimeError before the agent connects to the bus."""
    with pytest.raises(RuntimeError, match="not connected"):
        await agent.report_gap("do_something", "no handler available")


def test_version_attribute(agent):
    assert agent.version == "0.2.0"


def test_no_fastapi_dependency(agent):
    """BaseAgent should NOT use FastAPI/uvicorn — pure WebSocket client."""
    assert not hasattr(agent, "_build_app")
    assert agent._connector is None


# -- built-in health_check --

class BareAgent(BaseAgent):
    """Agent that declares no skills of its own."""

    agent_type = "bare"
    module_name = "tests.test_agent"
    version = "1.0.0"


def test_bare_agent_advertises_builtins():
    """An agent with no skills of its own still gets the built-ins."""
    a = BareAgent(agent_id="bare", bus_url="http://localhost:9999")
    names = {s.name for s in a._all_skills()}
    assert names == {"health_check", "welcome", "help"}


def test_subclass_skills_augmented_with_builtins(agent):
    """Subclass skills + the three built-ins are merged."""
    names = {s.name for s in agent._all_skills()}
    assert names == {
        "echo", "upper", "fail", "health_check", "welcome", "help",
    }


def test_register_skills_unchanged_by_builtins(agent):
    """register_skills() itself still returns only what the subclass declared."""
    assert {s.name for s in agent.register_skills()} == {"echo", "upper", "fail"}


@pytest.mark.asyncio
async def test_health_check_handler_returns_identity(agent):
    result = await agent._dispatch_request({"operation": "health_check", "args": {}})
    assert result["agent_id"] == "echo-test"
    assert result["agent_type"] == "echo"
    assert result["version"] == "0.2.0"
    assert result["bus_url"] == "http://localhost:9999"
    assert result["connected"] is False
    assert result["uptime_seconds"] >= 0
    assert isinstance(result["pid"], int)


class OverridingAgent(BaseAgent):
    """Subclass that replaces health_check with a richer payload."""

    agent_type = "overriding"
    module_name = "tests.test_agent"

    def register_skills(self):
        return [
            Skill(
                name="health_check",
                description="Custom health payload for overriding agent",
                parameters={"detail": {"type": "string"}},
            ),
        ]

    @handler("health_check")
    async def custom_health(self, args):
        base = await super().handle_health_check(args)
        return {**base, "custom": True, "detail_requested": args.get("detail", "")}


def test_subclass_can_override_health_check_schema():
    """Subclass-declared health_check replaces the built-in skill descriptor."""
    a = OverridingAgent(agent_id="ov", bus_url="http://localhost:9999")
    skills = a._all_skills()
    names = [s.name for s in skills]
    # Exactly one health_check — the subclass version, not duplicated
    assert names.count("health_check") == 1
    hc = next(s for s in skills if s.name == "health_check")
    assert "Custom health payload" in hc.description
    assert "detail" in hc.parameters


@pytest.mark.asyncio
async def test_subclass_can_extend_health_check_via_super():
    """Subclass handler can call super's payload and add fields."""
    a = OverridingAgent(agent_id="ov", bus_url="http://localhost:9999")
    result = await a._dispatch_request(
        {"operation": "health_check", "args": {"detail": "verbose"}}
    )
    assert result["agent_id"] == "ov"
    assert result["custom"] is True
    assert result["detail_requested"] == "verbose"


# -- auto-derived version from distribution metadata --


class _UnversionedAgent(BaseAgent):
    """Subclass that declares no ``version`` — should auto-derive."""

    agent_type = "unversioned"
    module_name = "fake_pkg.agent"


def test_unversioned_agent_auto_derives_version(monkeypatch):
    """When no subclass sets version, resolve from distribution metadata."""
    monkeypatch.setattr(_UnversionedAgent, "__module__", "fake_pkg.agent")

    def fake_packages_distributions():
        return {"fake_pkg": ["khonliang-fake"]}

    def fake_version(name):
        assert name == "khonliang-fake"
        return "9.9.9"

    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", fake_packages_distributions)
    monkeypatch.setattr(md, "version", fake_version)

    a = _UnversionedAgent(agent_id="u", bus_url="http://localhost:9999")
    assert a.version == "9.9.9"


def test_unversioned_agent_falls_back_when_no_distribution(monkeypatch):
    """Auto-derivation silently falls back to the BaseAgent default."""
    monkeypatch.setattr(_UnversionedAgent, "__module__", "ghost_pkg.agent")

    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})

    a = _UnversionedAgent(agent_id="u", bus_url="http://localhost:9999")
    assert a.version == BaseAgent.version  # "0.0.0"


def test_explicit_version_wins_over_autoderive(monkeypatch):
    """A subclass that sets ``version`` explicitly is never overwritten."""

    class Explicit(BaseAgent):
        agent_type = "explicit"
        module_name = "fake_pkg.agent"
        version = "3.2.1"

    # Even if auto-derivation would return something different, skip it.
    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"fake_pkg": ["khonliang-fake"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "9.9.9")

    a = Explicit(agent_id="e", bus_url="http://localhost:9999")
    assert a.version == "3.2.1"


def test_explicit_sentinel_version_is_not_overridden(monkeypatch):
    """Subclass pinning ``version = "0.0.0"`` on purpose is respected.

    Regression test for the value-equality bug: detection must be based
    on whether the subclass *declared* ``version``, not whether its
    value happens to equal ``BaseAgent.version``. Otherwise a legitimate
    explicit ``"0.0.0"`` pin silently gets upgraded to whatever
    ``packages_distributions`` happens to return.
    """

    class PinnedZero(BaseAgent):
        agent_type = "pinned-zero"
        module_name = "fake_pkg.agent"
        version = "0.0.0"

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"fake_pkg": ["khonliang-fake"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "9.9.9")

    a = PinnedZero(agent_id="p", bus_url="http://localhost:9999")
    assert a.version == "0.0.0"


def test_intermediate_base_class_version_is_respected(monkeypatch):
    """Version declared on an intermediate base in the MRO wins over auto-derive."""

    class Mid(BaseAgent):
        agent_type = "mid"
        module_name = "fake_pkg.agent"
        version = "5.0.0"

    class Leaf(Mid):
        # Inherits version from Mid; does not redeclare.
        agent_type = "leaf"

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"fake_pkg": ["khonliang-fake"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "9.9.9")

    a = Leaf(agent_id="l", bus_url="http://localhost:9999")
    assert a.version == "5.0.0"


def test_instance_assignment_before_super_is_respected(monkeypatch):
    """``self.version = ...`` set before super().__init__ is respected."""

    class PreSet(BaseAgent):
        agent_type = "preset"
        module_name = "fake_pkg.agent"

        def __init__(self, **kwargs):
            # Instance-level assignment lands in self.__dict__ before
            # the base's __init__ runs its auto-derive guard.
            self.version = "7.7.7"
            super().__init__(**kwargs)

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"fake_pkg": ["khonliang-fake"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "9.9.9")

    a = PreSet(agent_id="ps", bus_url="http://localhost:9999")
    assert a.version == "7.7.7"


def test_unversioned_agent_autoderives_when_launched_via_dash_m(monkeypatch):
    """End-to-end: a subclass whose __module__ is "__main__" still resolves."""
    import sys
    import types

    class DashMAgent(BaseAgent):
        agent_type = "dash-m"
        module_name = "fake_pkg.agent"

    monkeypatch.setattr(DashMAgent, "__module__", "__main__")

    fake_spec = types.SimpleNamespace(name="fake_pkg.agent")
    fake_main = types.SimpleNamespace(__spec__=fake_spec)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)

    import importlib.metadata as md

    monkeypatch.setattr(
        md, "packages_distributions", lambda: {"fake_pkg": ["khonliang-fake"]}
    )
    monkeypatch.setattr(md, "version", lambda name: "4.5.6")

    a = DashMAgent(agent_id="d", bus_url="http://localhost:9999")
    assert a.version == "4.5.6"


# -- Skill.default_timeout_s (fr_khonliang_1d7b5dd3) --


def test_skill_default_timeout_s_defaults_to_none():
    """Backwards-compat: existing callers don't have to opt in."""
    s = Skill("echo", "Echo the input")
    assert s.default_timeout_s is None


def test_skill_positional_args_preserve_metadata_slot():
    """Back-compat: appending ``default_timeout_s`` at the END of the
    dataclass field list must not shift the positional-arg slot of
    ``metadata`` (or any pre-existing field). A caller that passed
    ``metadata`` positionally before this field landed must still hit
    ``metadata`` — not ``default_timeout_s`` — after.

    Positional order for Skill is: name, description, parameters, since,
    capability, input_schema, output_contract, authority, status, aliases,
    execution_profiles, runtime_profile, metadata, default_timeout_s.
    """
    s = Skill(
        "name",
        "desc",
        {"handler": "schema"},
        "0.1.0",
        "cap",
        {"handler": "schema"},
        None,
        "authoritative",
        "active",
        [],
        [],
        None,
        {"foo": "bar"},
    )
    assert s.metadata == {"foo": "bar"}
    assert s.default_timeout_s is None


def test_skill_default_timeout_s_accepts_positive_float():
    s = Skill("review_pr", "Slow skill", default_timeout_s=120.0)
    assert s.default_timeout_s == 120.0
    assert isinstance(s.default_timeout_s, float)


def test_skill_default_timeout_s_coerces_int_to_float():
    """Ints are accepted and stored as floats for ladder-consumer simplicity."""
    s = Skill("review_pr", default_timeout_s=120)
    assert s.default_timeout_s == 120.0
    assert isinstance(s.default_timeout_s, float)


def test_skill_default_timeout_s_round_trips_through_to_dict():
    s = Skill("review_pr", "Slow skill", default_timeout_s=120.0)
    payload = s.to_dict()
    assert payload["default_timeout_s"] == 120.0
    # Reconstruct from the serialized payload — mirrors the
    # ``Skill(**s.to_dict())`` reconstruction path in ``_all_skills``.
    restored = Skill(**payload)
    assert restored.default_timeout_s == 120.0


def test_skill_default_timeout_s_omitted_when_none_in_to_dict():
    """None means unset; stay out of the serialized payload entirely.

    This matches the existing convention for every other optional field
    on Skill.to_dict (capability, output_contract, metadata, etc.).
    """
    s = Skill("echo", "Echo the input")
    payload = s.to_dict()
    assert "default_timeout_s" not in payload


def test_skill_absent_default_timeout_s_key_deserializes_to_none():
    """Older serialized payloads (no default_timeout_s key) must load cleanly."""
    legacy_payload = {
        "name": "legacy_skill",
        "description": "Predates default_timeout_s",
        "parameters": {},
        "since": "0.1.0",
    }
    s = Skill(**legacy_payload)
    assert s.default_timeout_s is None


def test_skill_default_timeout_s_rejects_zero():
    with pytest.raises(ValueError, match="default_timeout_s"):
        Skill("bad", default_timeout_s=0)


def test_skill_default_timeout_s_rejects_negative():
    with pytest.raises(ValueError, match="default_timeout_s"):
        Skill("bad", default_timeout_s=-5)


def test_skill_default_timeout_s_rejects_non_numeric():
    with pytest.raises(TypeError, match="default_timeout_s"):
        Skill("bad", default_timeout_s="120")


def test_skill_default_timeout_s_rejects_bool():
    """``True`` is technically an int in Python; reject it explicitly to
    avoid an author typo silently becoming a 1-second timeout."""
    with pytest.raises(TypeError, match="default_timeout_s"):
        Skill("bad", default_timeout_s=True)


# ---------------------------------------------------------------------------
# welcome (fr_khonliang-bus-lib_6a82732c)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_welcome_default_brief_announces_undocumented(agent):
    """Bare agent (no WELCOME override) returns a 'please document me'
    fallback: identity + auto-derived skill catalog + explicit
    missing-doc markers and a checklist of agent-level fields that
    need filling."""
    result = await agent._dispatch_request({"operation": "welcome", "args": {}})
    assert result["agent_id"] == "echo-test"
    assert result["agent_type"] == "echo"
    assert result["version"] == "0.2.0"
    assert result["skill_count"] == 6  # echo, upper, fail, health_check, welcome, help
    # Synthesized fallback editorial — the lib announces the undocumented
    # state instead of returning a sparse silent response.
    assert result["role"].startswith("(undocumented agent")
    assert "WELCOME" in result["mission"]
    assert "fr_khonliang-bus-lib_6a82732c" in result["mission"]
    # Agent-level checklist: every editorial field that should be filled.
    assert result["documentation_gaps"] == [
        "role",
        "mission",
        "boundaries (not_responsible_for + delegates_to)",
        "entry_points",
        "guide_skill",
    ]
    # No editorial sub-keys — these would mean the agent IS documented.
    assert "boundaries" not in result
    assert "entry_points" not in result
    # Categories still present at brief detail.
    assert "skill_categories" in result
    # health_check + welcome + help are the builtin category.
    assert result["skill_categories"]["builtin"] == 3


@pytest.mark.asyncio
async def test_welcome_default_compact_keeps_role_marker(agent):
    """compact still returns the missing-doc role marker so even the
    cheapest welcome variant tells the consumer the agent is
    undocumented; mission / categories / gaps stay absent at compact."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": "compact"}}
    )
    assert result["role"].startswith("(undocumented agent")
    assert "skill_categories" not in result
    assert "skills_by_category" not in result
    assert "mission" not in result
    assert "documentation_gaps" not in result
    assert "skill_documentation_gaps" not in result
    assert result["skill_count"] == 6


@pytest.mark.asyncio
async def test_welcome_default_full_lists_per_skill_documentation_gaps(agent):
    """At full detail, undocumented agents emit a per-skill gap map so
    a documenting LLM can target each skill's missing fields."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": "full"}}
    )
    gaps_by_skill = result["skill_documentation_gaps"]
    # ``upper`` and ``fail`` were registered without parameters → they
    # flag both "parameters / input_schema not declared" and
    # "capability tag not set"; ``echo`` only flags capability.
    assert "parameters / input_schema not declared" in gaps_by_skill["upper"]
    assert "capability tag not set" in gaps_by_skill["upper"]
    assert "capability tag not set" in gaps_by_skill["echo"]
    # ``echo`` HAS parameters, so no parameters-gap entry.
    assert "parameters / input_schema not declared" not in gaps_by_skill["echo"]
    # Built-ins also surface so the lib's own undocumented surface is
    # visible — operators see what THEY would need to populate to make
    # the platform fully self-documenting.
    assert "health_check" in gaps_by_skill
    assert "welcome" in gaps_by_skill


@pytest.mark.asyncio
async def test_welcome_full_lists_skills_per_category(agent):
    """full detail enumerates skill names per category."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": "full"}}
    )
    assert "skills_by_category" in result
    cats = result["skills_by_category"]
    # echo / upper / fail have no underscore prefix → 'misc'.
    assert set(cats["misc"]) == {"echo", "upper", "fail"}
    # builtins are in their own bucket.
    assert set(cats["builtin"]) == {"health_check", "welcome", "help"}


@pytest.mark.asyncio
async def test_welcome_invalid_detail_returns_error(agent):
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": "bogus"}}
    )
    assert "error" in result
    assert "compact|brief|full" in result["error"]


# -- editorial WELCOME override --


class CuratedAgent(BaseAgent):
    """Agent with a populated WELCOME override."""

    agent_type = "curated"
    module_name = "tests.test_agent"
    version = "1.0.0"

    WELCOME = Welcome(
        role="curated test agent",
        mission="Demonstrates the editorial fields surfaced via welcome.",
        not_responsible_for=["paper ingestion (researcher)"],
        delegates_to={"researcher": "evidence/context only"},
        entry_points=[
            WelcomeEntryPoint(skill="curated_action", when_to_use="for curated workflows"),
        ],
        guide_skill="curated_guide",
    )

    def register_skills(self):
        return [
            Skill("curated_action", "Run a curated workflow"),
            Skill("curated_guide", "Curated workflow guide"),
            Skill("git_status", "Stub git skill — exercises category prefix"),
        ]


@pytest.mark.asyncio
async def test_welcome_brief_includes_editorial_fields():
    a = CuratedAgent(agent_id="cur", bus_url="http://localhost:9999")
    result = await a._dispatch_request({"operation": "welcome", "args": {}})
    assert result["role"] == "curated test agent"
    assert result["mission"].startswith("Demonstrates")
    assert result["boundaries"]["not_responsible_for"] == ["paper ingestion (researcher)"]
    assert result["boundaries"]["delegates_to"] == {"researcher": "evidence/context only"}
    assert result["entry_points"] == [
        {"skill": "curated_action", "when_to_use": "for curated workflows"}
    ]
    assert result["guide_skill"] == "curated_guide"


@pytest.mark.asyncio
async def test_welcome_compact_keeps_role_drops_mission():
    a = CuratedAgent(agent_id="cur", bus_url="http://localhost:9999")
    result = await a._dispatch_request(
        {"operation": "welcome", "args": {"detail": "compact"}}
    )
    # compact: keep role for context, drop the longer-form fields.
    assert result["role"] == "curated test agent"
    assert "mission" not in result
    assert "boundaries" not in result
    assert "entry_points" not in result
    assert "skill_categories" not in result


@pytest.mark.asyncio
async def test_welcome_categorizes_by_underscore_prefix():
    """Skills with an underscore prefix go to that category bucket."""
    a = CuratedAgent(agent_id="cur", bus_url="http://localhost:9999")
    result = await a._dispatch_request(
        {"operation": "welcome", "args": {"detail": "full"}}
    )
    cats = result["skills_by_category"]
    # 'curated_action' / 'curated_guide' → 'curated' bucket.
    assert set(cats["curated"]) == {"curated_action", "curated_guide"}
    # 'git_status' → 'git' bucket.
    assert cats["git"] == ["git_status"]
    # builtins separately.
    assert set(cats["builtin"]) == {"health_check", "welcome", "help"}


def test_welcome_dataclass_to_dict_drops_empty_fields():
    """An empty Welcome serializes to {} so welcome's auto-derived
    fields are the only ones present."""
    assert Welcome().to_dict() == {}


def test_welcome_dataclass_to_dict_drops_individually_empty_fields():
    """Partial population: only populated fields appear."""
    w = Welcome(role="x")
    assert w.to_dict() == {"role": "x"}


def test_welcome_dataclass_collapses_boundaries():
    """boundaries dict only appears if at least one of its sub-fields is set."""
    w = Welcome(role="x", not_responsible_for=["y"])
    out = w.to_dict()
    assert out["boundaries"] == {"not_responsible_for": ["y"]}
    # delegates_to absent because it was empty.
    assert "delegates_to" not in out["boundaries"]


@pytest.mark.asyncio
async def test_welcome_detail_none_treated_as_default(agent):
    """JSON null / Python None for ``detail`` should fall back to the
    default rather than coerce to the string 'none' (which would fail
    the compact|brief|full check)."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": None}}
    )
    # No error; same shape as the default brief response.
    assert "error" not in result
    assert result["skill_count"] == 6
    assert "skill_categories" in result


def test_welcome_dataclass_is_frozen():
    """frozen=True catches the common 'mutate the shared default' error
    (welcome.role = 'x' raises FrozenInstanceError)."""
    import dataclasses
    w = Welcome(role="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.role = "y"


def test_welcome_entry_point_is_frozen():
    import dataclasses
    ep = WelcomeEntryPoint(skill="x", when_to_use="y")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ep.skill = "z"


def test_welcome_collections_are_immutable_after_construction():
    """Callers may pass list/dict literals for convenience, but the
    stored fields are coerced to truly-immutable shapes (tuple /
    MappingProxyType). Mutating the original literals doesn't leak
    into the Welcome instance, and the stored collections themselves
    cannot be mutated in place — addressing the shared-default
    leakage class structurally rather than via convention."""
    from types import MappingProxyType

    src_list = ["paper ingestion (researcher)"]
    src_dict = {"researcher": "evidence/context only"}
    src_eps = [WelcomeEntryPoint(skill="x", when_to_use="y")]
    w = Welcome(
        not_responsible_for=src_list,
        delegates_to=src_dict,
        entry_points=src_eps,
    )
    assert isinstance(w.not_responsible_for, tuple)
    assert isinstance(w.delegates_to, MappingProxyType)
    assert isinstance(w.entry_points, tuple)

    # Mutating the original literals after construction must not bleed
    # into ``w`` — coercion is by-value.
    src_list.append("leaked")
    src_dict["new"] = "leaked"
    src_eps.append(WelcomeEntryPoint(skill="leaked", when_to_use="leaked"))
    assert w.not_responsible_for == ("paper ingestion (researcher)",)
    assert dict(w.delegates_to) == {"researcher": "evidence/context only"}
    assert len(w.entry_points) == 1

    # Direct mutation of the stored collections fails.
    with pytest.raises((AttributeError, TypeError)):
        w.not_responsible_for.append("x")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        w.delegates_to["new"] = "x"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        w.entry_points.append(  # type: ignore[attr-defined]
            WelcomeEntryPoint(skill="x", when_to_use="y")
        )


def test_welcome_rewraps_existing_mappingproxy_to_sever_caller_alias():
    """If the caller passes a ``MappingProxyType`` that wraps a dict
    they still hold, ``__post_init__`` must copy + re-wrap so later
    mutation of the caller's backing dict doesn't bleed into the
    frozen Welcome."""
    from types import MappingProxyType

    backing = {"researcher": "evidence/context only"}
    caller_proxy = MappingProxyType(backing)
    w = Welcome(delegates_to=caller_proxy)

    backing["leaked"] = "should not appear"
    assert "leaked" not in w.delegates_to
    assert dict(w.delegates_to) == {"researcher": "evidence/context only"}


@pytest.mark.asyncio
async def test_welcome_handles_null_args_cleanly(agent):
    """``args=None`` (transport-level glitch / JSON null) must not
    raise — returns the same shape as an empty args dict."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": None}
    )
    assert "error" not in result
    assert result["agent_id"] == "echo-test"
    assert result["skill_count"] == 6


@pytest.mark.asyncio
async def test_welcome_rejects_non_dict_args(agent):
    """A non-dict / non-None args produces a clean validation error
    instead of an AttributeError leaking as a transport failure."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": ["unexpected", "list"]}
    )
    assert "error" in result
    assert "object" in result["error"]


def test_skill_doc_gaps_flags_missing_fields():
    """The per-skill gap helper reports description / parameters /
    capability gaps independently."""
    fully_documented = Skill(
        name="x",
        description="does the thing",
        parameters={"q": {"type": "string"}},
        capability="x.do",
    )
    assert BaseAgent._skill_doc_gaps(fully_documented) == []

    bare = Skill(name="y")
    gaps = BaseAgent._skill_doc_gaps(bare)
    assert "description is empty" in gaps
    assert "parameters / input_schema not declared" in gaps
    assert "capability tag not set" in gaps


# ---------------------------------------------------------------------------
# help skill (fr_khonliang-bus-lib_42555320 + fr_khonliang-bus-lib_6e42567d)
# ---------------------------------------------------------------------------


class AspectAgent(BaseAgent):
    """Agent with skills that populate the new aspect fields so the
    help-skill aspect-mode tests can check round-trip."""

    agent_type = "aspect"
    module_name = "tests.test_agent"
    version = "0.3.0"

    def register_skills(self):
        return [
            Skill(
                name="distill_paper",
                description="Run LLM distillation on a stored paper.",
                parameters={"paper_id": {"type": "string"}},
                capability="research.distill",
                prompt=(
                    "Use distill_paper to summarize a paper that's "
                    "already been ingested. Pass paper_id from the "
                    "ingest response. The skill returns "
                    "{summary, triples, applicability}."
                ),
                examples=[
                    {
                        "input_args": {"paper_id": "pp_abc123"},
                        "expected_output_shape": (
                            "{summary: str, triples: list, applicability: dict}"
                        ),
                        "narrative": "Distill a known paper id.",
                    }
                ],
                pairs_with=["fetch_paper", "find_relevant"],
                not_appropriate_for=[
                    "binary attachments — use stage_payload",
                    "freeform text without an ingested paper id",
                ],
            ),
            Skill(
                name="bare_skill",
                description="Skill with no aspect fields populated.",
                parameters={},
            ),
        ]


@pytest.fixture
def aspect_agent():
    return AspectAgent(agent_id="aspect-test", bus_url="http://localhost:9999")


@pytest.mark.asyncio
async def test_help_empty_list_returns_full_catalog(agent):
    """``help([])`` shorthand returns every registered skill."""
    result = await agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": []}}
    )
    names = {s["name"] for s in result["skills"]}
    # Subclass skills + the three built-ins.
    assert names == {
        "echo", "upper", "fail", "health_check", "welcome", "help",
    }
    # Every entry is found.
    assert all(s["found"] for s in result["skills"])


@pytest.mark.asyncio
async def test_help_unknown_name_marks_found_false_with_reason(agent):
    """Unknown skill names appear with ``found: false`` and a reason
    string — never silently dropped, so callers learn what missed."""
    result = await agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": ["nonexistent"]}}
    )
    assert len(result["skills"]) == 1
    entry = result["skills"][0]
    assert entry["name"] == "nonexistent"
    assert entry["found"] is False
    assert "no skill" in entry["reason"]


@pytest.mark.asyncio
async def test_help_mixed_known_and_unknown_names_all_surface(agent):
    """Mixed lookups preserve order: known entries get full info,
    unknown entries get found:false. Both appear so the caller can
    correlate by index/name."""
    result = await agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": ["echo", "ghost"]}}
    )
    assert [s["name"] for s in result["skills"]] == ["echo", "ghost"]
    assert result["skills"][0]["found"] is True
    assert result["skills"][1]["found"] is False


@pytest.mark.asyncio
async def test_help_compact_returns_only_name_and_description(agent):
    """``detail=compact`` is the cheapest read — name + description
    only; no parameters / input_schema / aspect fields."""
    result = await agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": ["echo"], "detail": "compact"}}
    )
    entry = result["skills"][0]
    assert entry["name"] == "echo"
    assert entry["description"] == "Echo the input"
    assert "parameters" not in entry
    assert "input_schema" not in entry
    assert "prompt" not in entry


@pytest.mark.asyncio
async def test_help_brief_includes_parameters(agent):
    """``detail=brief`` (default) adds parameters / input_schema /
    capability — the args contract the caller actually needs."""
    result = await agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": ["echo"]}}
    )
    entry = result["skills"][0]
    assert "parameters" in entry
    assert entry["parameters"] == {"text": {"type": "string"}}


@pytest.mark.asyncio
async def test_help_full_surfaces_populated_aspect_fields(aspect_agent):
    """``detail=full`` adds aspect fields ONLY when populated. Skills
    that didn't declare aspects don't get noisy empty entries."""
    result = await aspect_agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": [], "detail": "full"}}
    )
    by_name = {s["name"]: s for s in result["skills"]}
    distill = by_name["distill_paper"]
    assert distill["prompt"].startswith("Use distill_paper")
    assert distill["pairs_with"] == ["fetch_paper", "find_relevant"]
    assert distill["not_appropriate_for"][0].startswith("binary attachments")
    assert len(distill["examples"]) == 1
    bare = by_name["bare_skill"]
    # bare_skill declared no aspect fields — entry stays signal-dense.
    assert "prompt" not in bare
    assert "pairs_with" not in bare
    assert "examples" not in bare


@pytest.mark.asyncio
async def test_help_aspect_mode_returns_flat_value_list(aspect_agent):
    """Aspect-mode read returns ``{name, found, value}`` per skill —
    token-efficient for callers that already know which slice they
    want (LLM asking for prompt to adapt a template, sibling agent
    asking for schema to validate a call)."""
    result = await aspect_agent._dispatch_request(
        {
            "operation": "help",
            "args": {"skill_names": ["distill_paper"], "aspect": "prompt"},
        }
    )
    assert result["aspect"] == "prompt"
    assert len(result["skills"]) == 1
    entry = result["skills"][0]
    assert entry["name"] == "distill_paper"
    assert entry["found"] is True
    assert entry["value"].startswith("Use distill_paper")


@pytest.mark.asyncio
async def test_help_aspect_schema_falls_back_to_parameters(aspect_agent):
    """When ``input_schema`` is empty, ``aspect=schema`` falls back to
    ``parameters`` — legacy Skill registrations that only set
    parameters still answer the schema aspect correctly."""
    result = await aspect_agent._dispatch_request(
        {
            "operation": "help",
            "args": {"skill_names": ["distill_paper"], "aspect": "schema"},
        }
    )
    assert result["skills"][0]["value"] == {"paper_id": {"type": "string"}}


@pytest.mark.asyncio
async def test_help_aspect_unknown_skill_marks_found_false(aspect_agent):
    """Unknown skills in aspect mode also surface as found:false."""
    result = await aspect_agent._dispatch_request(
        {
            "operation": "help",
            "args": {"skill_names": ["ghost"], "aspect": "prompt"},
        }
    )
    entry = result["skills"][0]
    assert entry["found"] is False
    assert "no skill" in entry["reason"]
    # No 'value' key when not found.
    assert "value" not in entry


@pytest.mark.asyncio
async def test_help_rejects_invalid_aspect(aspect_agent):
    """An unknown aspect string returns a validation envelope listing
    the accepted values — caller learns the contract without re-reading
    the source."""
    result = await aspect_agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": [], "aspect": "bogus"}}
    )
    assert "error" in result
    assert "prompt" in result["error"]
    assert "schema" in result["error"]


@pytest.mark.asyncio
async def test_help_rejects_invalid_detail(aspect_agent):
    """An unknown detail level returns a validation envelope."""
    result = await aspect_agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": [], "detail": "bogus"}}
    )
    assert "error" in result
    assert "compact|brief|full" in result["error"]


@pytest.mark.asyncio
async def test_help_rejects_non_list_skill_names(aspect_agent):
    """``skill_names`` must be a list — a string here is the easy
    mistake (caller passing one name unwrapped)."""
    result = await aspect_agent._dispatch_request(
        {"operation": "help", "args": {"skill_names": "echo"}}
    )
    assert "error" in result
    assert "list" in result["error"]


@pytest.mark.asyncio
async def test_help_falsey_non_none_detail_hits_validation(aspect_agent):
    """``detail=0`` / ``detail=False`` are caller-supplied bad values,
    not "not provided" — they must hit the validation envelope rather
    than silently coalesce to the default. Only ``None`` and missing
    keys default to ``brief``."""
    for bad in (0, False):
        result = await aspect_agent._dispatch_request(
            {"operation": "help", "args": {"skill_names": [], "detail": bad}}
        )
        assert "error" in result, f"falsey detail={bad!r} silently defaulted"
        assert "compact|brief|full" in result["error"]


@pytest.mark.asyncio
async def test_help_falsey_non_none_aspect_hits_validation(aspect_agent):
    """``aspect=0`` / ``aspect=False`` likewise must surface as bad
    values rather than silently meaning "no aspect mode" (which is
    what ``aspect=None`` / missing should do)."""
    for bad in (0, False):
        result = await aspect_agent._dispatch_request(
            {"operation": "help", "args": {"skill_names": [], "aspect": bad}}
        )
        assert "error" in result, f"falsey aspect={bad!r} silently defaulted"
        # Validation envelope lists the accepted aspects.
        assert "prompt" in result["error"]


@pytest.mark.asyncio
async def test_help_explicit_none_detail_aspect_use_defaults(aspect_agent):
    """``detail=None`` and ``aspect=None`` (JSON null) are the
    legitimate "not provided" sentinel — they must default to the
    schema-declared defaults (brief / no-aspect-mode) rather than
    raising."""
    result = await aspect_agent._dispatch_request(
        {
            "operation": "help",
            "args": {"skill_names": [], "detail": None, "aspect": None},
        }
    )
    assert "error" not in result
    # Brief mode response shape — has the SkillEntry list, no
    # "aspect" envelope key.
    assert "skills" in result
    assert "aspect" not in result


def test_skill_rejects_non_str_prompt():
    """``prompt`` must be ``str`` (None coerces to ""). Other
    types — bool, int, list, dict — fail loudly at construction
    rather than serializing into the registration payload."""
    for bad in (False, 42, ["wrong"], {"wrong": "shape"}):
        with pytest.raises(TypeError) as exc_info:
            Skill(name="x", prompt=bad)  # type: ignore[arg-type]
        msg = str(exc_info.value)
        assert "prompt" in msg
        assert "str" in msg
        assert type(bad).__name__ in msg


def test_skill_rejects_non_list_aspect_values():
    """List-aspects must be ``list`` (str gets the specific
    character-splitting hint; everything else gets the generic
    'expected list, got X' message). bool / dict / int are all
    rejected — silent coercion would corrupt the registration
    payload."""
    for aspect in ("examples", "pairs_with", "not_appropriate_for"):
        for bad in (False, 42, {"wrong": "shape"}):
            with pytest.raises(TypeError) as exc_info:
                Skill(name="x", **{aspect: bad})
            msg = str(exc_info.value)
            assert aspect in msg
            assert "list" in msg
            assert type(bad).__name__ in msg


def test_skill_rejects_str_for_list_aspects_with_helpful_message():
    """A caller passing a string by mistake (common JSON/CLI shape)
    must fail loudly instead of getting silent character-splitting via
    ``list('foo')`` → ``['f', 'o', 'o']``. The error message points at
    the offending aspect and shows how to fix it."""
    for aspect in ("examples", "pairs_with", "not_appropriate_for"):
        with pytest.raises(TypeError) as exc_info:
            Skill(name="x", **{aspect: "single-string-by-mistake"})
        msg = str(exc_info.value)
        assert aspect in msg
        assert "list" in msg
        # Surfaces the offending value so the caller sees what they
        # passed without re-reading their own code.
        assert "single-string-by-mistake" in msg


def test_skill_examples_are_deep_copied_to_sever_caller_aliasing():
    """``examples`` entries are dicts with nested args/output shapes —
    a shallow ``list(...)`` would still alias the inner dicts to the
    caller's literals. Mutation of a caller-held example dict after
    construction must not bleed into the registered Skill."""
    backing = {"input_args": {"q": "x"}, "expected_output_shape": "ok"}
    s = Skill(name="x", examples=[backing])
    backing["input_args"]["q"] = "leaked"
    backing["expected_output_shape"] = "leaked"
    assert s.examples[0]["input_args"]["q"] == "x"
    assert s.examples[0]["expected_output_shape"] == "ok"


def test_skill_aspect_none_inputs_are_coerced_to_empty():
    """A caller passing ``None`` for an aspect (JSON null in the
    registration payload) must not raise — None coerces to the empty
    default rather than ``TypeError`` in __post_init__."""
    s = Skill(
        name="x",
        prompt=None,  # type: ignore[arg-type]
        examples=None,  # type: ignore[arg-type]
        pairs_with=None,  # type: ignore[arg-type]
        not_appropriate_for=None,  # type: ignore[arg-type]
    )
    assert s.prompt == ""
    assert s.examples == []
    assert s.pairs_with == []
    assert s.not_appropriate_for == []


def test_skill_aspect_fields_round_trip_through_to_dict():
    """The new aspect fields survive the registration-payload round
    trip and are dropped (not faked) when empty — matches the existing
    optional-field convention."""
    s = Skill(
        name="x",
        description="d",
        parameters={"q": {"type": "string"}},
        prompt="example template",
        examples=[{"input_args": {}, "expected_output_shape": "ok"}],
        pairs_with=["y"],
        not_appropriate_for=["z"],
    )
    payload = s.to_dict()
    assert payload["prompt"] == "example template"
    assert payload["pairs_with"] == ["y"]
    assert payload["not_appropriate_for"] == ["z"]
    assert len(payload["examples"]) == 1

    bare = Skill(name="bare", description="d")
    bare_payload = bare.to_dict()
    # Empty aspect fields stay out of the payload — bus registration
    # response stays signal-dense.
    assert "prompt" not in bare_payload
    assert "pairs_with" not in bare_payload
    assert "not_appropriate_for" not in bare_payload
    assert "examples" not in bare_payload
