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


def test_skill_aspect_fields_are_keyword_only_preserving_positional_order():
    """Pre-aspect call sites passed up to ``default_timeout_s``
    positionally. Adding new positional fields after it would
    silently bind the timeout's value to ``prompt``. Aspect fields
    are ``kw_only=True`` so existing positional calls still work."""
    # Reproduces a positional call that includes default_timeout_s
    # in its expected slot. If the new aspect fields were positional,
    # the 14th positional arg (5.0) would bind to ``prompt``
    # instead of ``default_timeout_s``.
    s = Skill(
        "x",                       # name
        "desc",                    # description
        {"q": {"type": "string"}}, # parameters
        "1.0.0",                   # since
        "x.do",                    # capability
        {},                        # input_schema
        None,                      # output_contract
        "authoritative",           # authority
        "active",                  # status
        ["alias_x"],               # aliases
        [],                        # execution_profiles
        None,                      # runtime_profile
        {},                        # metadata
        5.0,                       # default_timeout_s
    )
    assert s.default_timeout_s == 5.0
    # New aspect fields stay at their defaults; the positional 5.0
    # bound to default_timeout_s, not prompt.
    assert s.prompt == ""
    assert s.examples == []

    # And new aspect fields can only be passed by keyword.
    with pytest.raises(TypeError):
        Skill("x", "d", {}, "", "", {}, None, "authoritative", "active",
              [], [], None, {}, None, "would-be-prompt-positional")  # noqa: E501


def test_skill_rejects_non_str_pairs_with_elements():
    """``pairs_with: list[str]`` — element types are validated.
    A stray int or dict in the list would corrupt the registration
    payload that downstream ``help`` consumers treat as authoritative."""
    for bad_elem in (1, {"wrong": "shape"}, None, True):
        with pytest.raises(TypeError) as exc_info:
            Skill(name="x", pairs_with=["good", bad_elem])  # type: ignore[list-item]
        msg = str(exc_info.value)
        assert "pairs_with" in msg
        assert "entry 1" in msg
        assert "str" in msg


def test_skill_rejects_non_str_not_appropriate_for_elements():
    """Same per-element guard for ``not_appropriate_for``."""
    with pytest.raises(TypeError) as exc_info:
        Skill(name="x", not_appropriate_for=[42])  # type: ignore[list-item]
    assert "not_appropriate_for" in str(exc_info.value)
    assert "entry 0" in str(exc_info.value)


def test_skill_rejects_non_dict_examples_elements():
    """``examples: list[dict]`` — non-dict entries fail loudly."""
    for bad_elem in ("string-not-dict", 42, ["nested-list"]):
        with pytest.raises(TypeError) as exc_info:
            Skill(name="x", examples=[bad_elem])  # type: ignore[list-item]
        msg = str(exc_info.value)
        assert "examples" in msg
        assert "entry 0" in msg
        assert "dict" in msg


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


# ---------------------------------------------------------------------------
# Mode B — bus-side schema validation (fr_khonliang-bus-lib_6e42567d)
# ---------------------------------------------------------------------------


class StrictAgent(BaseAgent):
    """Agent with skills that opt into strict-args validation, exercising
    every failure class the dispatcher checks: unknown kwargs, missing
    required, wrong type."""

    agent_type = "strict"
    module_name = "tests.test_agent"
    version = "0.4.0"

    def register_skills(self):
        return [
            Skill(
                name="strict_echo",
                description="Echo with declared schema and strict_args.",
                parameters={
                    "text": {"type": "string", "required": True},
                    "loud": {"type": "boolean"},
                },
                strict_args=True,
            ),
            Skill(
                name="strict_count",
                description="Counts and casts to declared types.",
                parameters={
                    "n": {"type": "integer", "required": True},
                    "tags": {"type": "array"},
                    "extra": {"type": "object"},
                },
                strict_args=True,
            ),
            Skill(
                name="lenient_echo",
                description="Same shape, no strict_args (legacy default).",
                parameters={"text": {"type": "string", "required": True}},
            ),
        ]

    @handler("strict_echo")
    async def strict_echo(self, args):
        return {"echoed": args.get("text", ""), "loud": args.get("loud", False)}

    @handler("strict_count")
    async def strict_count(self, args):
        return {"n": args["n"], "tags": args.get("tags", [])}

    @handler("lenient_echo")
    async def lenient_echo(self, args):
        return {"echoed": args.get("text", "")}


@pytest.fixture
def strict_agent():
    return StrictAgent(agent_id="strict-test", bus_url="http://localhost:9999")


@pytest.mark.asyncio
async def test_strict_args_rejects_unknown_kwargs_with_accepted_set(strict_agent):
    """The silent-arg-drop class (bug_developer_ad60dca4 / b5fd44ce /
    a349c77b): caller passes ``txt`` instead of ``text``; today that's
    silently ignored and the handler runs with the wrong args. Strict
    validation rejects with a message naming the bad key + the
    accepted-set so the caller learns what they should have used."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_echo",
        "args": {"txt": "hello"},
    })
    assert "error" in result
    assert "txt" in result["error"]
    # accepted set surfaces both declared keys, alphabetically sorted
    assert "loud, text" in result["error"]


@pytest.mark.asyncio
async def test_strict_args_rejects_missing_required_arg(strict_agent):
    """Required-but-missing fails fast with a clear message — no silent
    handler call against an empty dict."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_echo",
        "args": {"loud": True},
    })
    assert "error" in result
    assert "missing required" in result["error"]
    assert "text" in result["error"]


@pytest.mark.asyncio
async def test_strict_args_rejects_wrong_type(strict_agent):
    """Type mismatch surfaces declared and actual types so the caller
    can correct without re-reading the schema. The OFFENDING VALUE is
    deliberately NOT in the error envelope — it could be a 50KB blob
    or sensitive content (API keys, paper text, user PII) and would
    otherwise leak into bus responses + downstream logs."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_echo",
        "args": {"text": 42},
    })
    assert "error" in result
    assert "string" in result["error"]
    assert "int" in result["error"]
    # The bad value (42) must NOT appear in the error — leakage guard.
    assert "42" not in result["error"]


@pytest.mark.asyncio
async def test_strict_args_does_not_leak_large_payload_in_error(strict_agent):
    """Real-world version of the leakage guard: caller passes a large
    string where a non-string type is expected. The error message stays
    bounded — only declared / actual type names — so a multi-KB
    payload never lands in the bus response or downstream logs."""
    big_value = "X" * 5000  # 5KB; could just as easily be 50KB
    result = await strict_agent._dispatch_request({
        "operation": "strict_count",
        "args": {"n": big_value},
    })
    assert "error" in result
    # Error length is bounded regardless of input value size.
    assert len(result["error"]) < 200, (
        f"error grew with value size — likely leaking value: "
        f"{result['error'][:200]}..."
    )
    assert "integer" in result["error"]
    assert big_value not in result["error"]


@pytest.mark.asyncio
async def test_strict_args_rejects_bool_for_integer(strict_agent):
    """``bool`` is a subclass of ``int`` in Python — accepting True/
    False where an integer is declared is almost always a caller bug.
    Validation rejects it explicitly."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_count",
        "args": {"n": True},
    })
    assert "error" in result
    assert "integer" in result["error"]
    assert "bool" in result["error"]


@pytest.mark.asyncio
async def test_strict_args_validates_array_and_object_types(strict_agent):
    """JSON-schema ``array`` requires list, ``object`` requires dict."""
    bad_array = await strict_agent._dispatch_request({
        "operation": "strict_count",
        "args": {"n": 3, "tags": "not-a-list"},
    })
    assert "error" in bad_array
    assert "array" in bad_array["error"]

    bad_object = await strict_agent._dispatch_request({
        "operation": "strict_count",
        "args": {"n": 3, "extra": ["not", "a", "dict"]},
    })
    assert "error" in bad_object
    assert "object" in bad_object["error"]


@pytest.mark.asyncio
async def test_strict_args_passes_well_formed_call(strict_agent):
    """Sanity: a well-formed call hits the handler and returns its
    result. Validation must not get in the way of correct callers."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_echo",
        "args": {"text": "hi", "loud": True},
    })
    assert result == {"echoed": "hi", "loud": True}


@pytest.mark.asyncio
async def test_strict_args_skips_validation_when_flag_false(strict_agent):
    """``lenient_echo`` shares the same schema shape but omits
    strict_args — the dispatcher must NOT validate, so the historical
    silent-pass-through behavior holds. This is the load-bearing
    backward-compat property: every existing skill in the fleet keeps
    working unchanged."""
    # Unknown kwarg silently passes through to the handler; handler
    # uses ``args.get("text", "")`` so it sees the empty default.
    result = await strict_agent._dispatch_request({
        "operation": "lenient_echo",
        "args": {"txt": "typo"},
    })
    assert "error" not in result
    assert result == {"echoed": ""}


@pytest.mark.asyncio
async def test_strict_args_falls_back_to_parameters_when_input_schema_empty():
    """Legacy registrations that only set ``parameters`` (without a
    separate input_schema) still validate — the schema lookup falls
    back to parameters when input_schema is empty. Most existing
    Skill(...) call sites in the fleet are this shape."""

    class LegacyShape(BaseAgent):
        agent_type = "legacy"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="legacy_skill",
                    description="d",
                    parameters={"q": {"type": "string", "required": True}},
                    strict_args=True,
                ),
            ]

        @handler("legacy_skill")
        async def legacy_skill(self, args):
            return {"q": args.get("q", "")}

    a = LegacyShape(agent_id="legacy-test", bus_url="http://localhost:9999")
    bad = await a._dispatch_request({
        "operation": "legacy_skill",
        "args": {"qq": "wrong-key"},
    })
    assert "error" in bad
    assert "qq" in bad["error"]
    good = await a._dispatch_request({
        "operation": "legacy_skill",
        "args": {"q": "ok"},
    })
    assert good == {"q": "ok"}


@pytest.mark.asyncio
async def test_strict_args_unknown_type_in_schema_skips_type_check():
    """A schema with a non-standard / unknown ``type`` value still
    validates presence + required, but skips type checking on that
    field — keeps forward-compat with future JSON-Schema additions
    (``date-time``, custom types, etc.) without rejecting them."""

    class ForwardCompat(BaseAgent):
        agent_type = "forward"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="future_skill",
                    description="d",
                    parameters={
                        "id": {"type": "uuid", "required": True},
                    },
                    strict_args=True,
                ),
            ]

        @handler("future_skill")
        async def future_skill(self, args):
            return {"id": args.get("id")}

    a = ForwardCompat(agent_id="forward", bus_url="http://localhost:9999")
    # Required-presence check still fires.
    missing = await a._dispatch_request({
        "operation": "future_skill",
        "args": {},
    })
    assert "error" in missing
    assert "missing required" in missing["error"]
    # But any value is accepted on the unrecognized type.
    ok = await a._dispatch_request({
        "operation": "future_skill",
        "args": {"id": "anything-goes"},
    })
    assert ok == {"id": "anything-goes"}


def test_strict_args_serializes_only_when_true():
    """strict_args=True surfaces in the registration payload (so the
    bus / consumers can see the contract); strict_args=False stays
    omitted to keep the payload signal-dense."""
    on = Skill(name="x", parameters={"q": {"type": "string"}}, strict_args=True)
    off = Skill(name="y", parameters={"q": {"type": "string"}})
    assert on.to_dict()["strict_args"] is True
    assert "strict_args" not in off.to_dict()


@pytest.mark.asyncio
async def test_strict_args_empty_schema_rejects_any_kwargs():
    """An explicitly-empty schema with strict_args=True is a valid
    zero-args contract — any kwarg the caller supplies is rejected
    as unknown. Health-check-shape skills that legitimately take no
    arguments use this path. The accepted-set message names the
    empty-schema case explicitly so the caller doesn't think the
    error message is broken."""

    class ZeroArgs(BaseAgent):
        agent_type = "zeroargs"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="ping",
                    description="Takes nothing.",
                    parameters={},
                    strict_args=True,
                ),
            ]

        @handler("ping")
        async def ping(self, args):
            return {"pong": True}

    a = ZeroArgs(agent_id="zero-test", bus_url="http://localhost:9999")
    # Empty args → handler runs.
    ok = await a._dispatch_request({"operation": "ping", "args": {}})
    assert ok == {"pong": True}
    # Any kwarg → rejected with the empty-schema marker.
    bad = await a._dispatch_request({
        "operation": "ping",
        "args": {"unexpected": "value"},
    })
    assert "error" in bad
    assert "unexpected" in bad["error"]
    assert "empty schema" in bad["error"]


def test_skill_rejects_non_bool_strict_args():
    """``strict_args`` must be a real ``bool``. Truthy non-bool values
    ('true', 1, [1]) silently enabling validation while ``to_dict()``
    serializes them as ``True`` would mask the misuse from downstream
    consumers reading the registration payload. Construction must
    fail loudly for the same reason str-rejection covers the list-
    aspects: a public flag deserves type-locked semantics."""
    for bad in ("true", "false", 1, 0, [True], {"strict": True}):
        with pytest.raises(TypeError) as exc_info:
            Skill(name="x", strict_args=bad)  # type: ignore[arg-type]
        msg = str(exc_info.value)
        assert "strict_args" in msg
        assert "bool" in msg
        assert type(bad).__name__ in msg
    # Real bools accepted on either side.
    Skill(name="ok_true", strict_args=True)
    Skill(name="ok_false", strict_args=False)


@pytest.mark.asyncio
async def test_strict_args_uses_input_schema_when_diverges_from_parameters():
    """When ``input_schema`` is non-empty, validation uses it — NOT
    ``parameters``. Pre-existing skills set parameters only and rely
    on the input_schema-or-parameters fallback; future skills can
    declare a richer input_schema (e.g. richer types in input_schema,
    legacy MCP-shaped params) and the validator must follow the
    richer one. A regression that swaps precedence to parameters-
    primary would silently bypass strict checks."""

    class DivergentSchema(BaseAgent):
        agent_type = "divergent"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="diverge_skill",
                    description="d",
                    # Legacy MCP-shaped parameters (lenient).
                    parameters={"loose_field": {"type": "string"}},
                    # Strict richer schema; validation must use this one.
                    input_schema={
                        "strict_field": {"type": "string", "required": True},
                    },
                    strict_args=True,
                ),
            ]

        @handler("diverge_skill")
        async def diverge_skill(self, args):
            return {"got": list(args.keys())}

    a = DivergentSchema(agent_id="div", bus_url="http://localhost:9999")
    # Passing the parameters-only key fails — input_schema doesn't
    # accept it.
    bad = await a._dispatch_request({
        "operation": "diverge_skill",
        "args": {"loose_field": "x"},
    })
    assert "error" in bad
    assert "loose_field" in bad["error"]
    assert "strict_field" in bad["error"]  # accepted-set names input_schema's key
    # Passing input_schema's key works.
    good = await a._dispatch_request({
        "operation": "diverge_skill",
        "args": {"strict_field": "x"},
    })
    assert good == {"got": ["strict_field"]}


@pytest.mark.asyncio
async def test_strict_args_required_field_set_to_none_fails_type_check(strict_agent):
    """``{"text": None}`` — present in args (so required-check passes),
    but ``isinstance(None, str)`` is False, so the type check rejects
    it. Pin the behavior so a future change to "treat None as missing"
    doesn't slip in unnoticed — callers that expect None to mean
    'pass-through default' need to omit the key, not pass null."""
    result = await strict_agent._dispatch_request({
        "operation": "strict_echo",
        "args": {"text": None},
    })
    assert "error" in result
    assert "string" in result["error"]
    assert "NoneType" in result["error"]


@pytest.mark.asyncio
async def test_strict_args_all_optional_empty_args_passes():
    """A skill with no required fields accepts ``args={}`` — the
    type checks only run for fields that are present, so an empty
    args is the all-defaults call and must not be rejected."""

    class AllOptional(BaseAgent):
        agent_type = "alloptional"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="opt_skill",
                    description="d",
                    parameters={
                        "a": {"type": "string"},  # no required flag
                        "b": {"type": "integer"},
                    },
                    strict_args=True,
                ),
            ]

        @handler("opt_skill")
        async def opt_skill(self, args):
            return {"a": args.get("a"), "b": args.get("b")}

    a = AllOptional(agent_id="opt", bus_url="http://localhost:9999")
    result = await a._dispatch_request({
        "operation": "opt_skill",
        "args": {},
    })
    assert "error" not in result
    assert result == {"a": None, "b": None}


@pytest.mark.asyncio
async def test_strict_args_warns_when_skill_name_doesnt_match_handler(caplog):
    """Silent-bypass guard: if a Skill declares ``strict_args=True``
    but its ``name`` doesn't match any registered handler operation,
    the dispatcher's lookup would miss and validation would silently
    skip. Loud-warn at first cache build so the agent author notices
    the misregistration instead of shipping a falsely-strict skill."""
    import logging

    class Misregistered(BaseAgent):
        agent_type = "misreg"
        module_name = "tests.test_agent"

        def register_skills(self):
            return [
                Skill(
                    name="declared_name",  # mismatch — handler is "actual_op"
                    description="d",
                    parameters={"q": {"type": "string"}},
                    strict_args=True,
                ),
            ]

        @handler("actual_op")
        async def actual_op(self, args):
            return {}

    a = Misregistered(agent_id="misreg-test", bus_url="http://localhost:9999")
    with caplog.at_level(logging.WARNING):
        # Trigger cache build via any dispatch.
        await a._dispatch_request({"operation": "actual_op", "args": {}})

    assert any(
        "strict_args=True" in rec.message and "declared_name" in rec.message
        for rec in caplog.records
    ), "expected warning naming the misregistered skill"


# ---------------------------------------------------------------------------
# request_typed — caller-side schema validation (Mode B caller side,
# fr_khonliang-bus-lib_6e42567d)
# ---------------------------------------------------------------------------


class _FakeBusHTTP:
    """Mock httpx-shaped client for ``BaseAgent.request`` / request_typed.

    Records every POST with its payload and replays scripted responses
    keyed on (operation). Lets the tests verify that request_typed
    actually shortcuts to a help() lookup on first call, caches it,
    and only dispatches the real op when the args validate.
    """

    def __init__(self, scripted_responses: dict[str, Any]):
        self.scripted = scripted_responses
        self.calls: list[dict[str, Any]] = []

    async def post(self, url, *, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        # Pull the response keyed on the operation in the request body.
        op = (json or {}).get("operation", "")
        body = self.scripted.get(op, {"error": f"no scripted response for {op}"})

        class _Resp:
            def __init__(self, data):
                self._data = data

            def json(self):
                return self._data

        return _Resp(body)

    async def aclose(self):
        pass

    @property
    def operations_called(self) -> list[str]:
        return [c["json"].get("operation", "") for c in self.calls]


def _help_schema_response(op_name: str, schema: dict) -> dict:
    """Build the bus-envelope response shape that ``_fetch_remote_skill_schema``
    expects from a remote help() call."""
    return {
        "result": {
            "aspect": "schema",
            "skills": [{"name": op_name, "found": True, "value": schema}],
        },
    }


@pytest.mark.asyncio
async def test_request_typed_validates_locally_before_remote_dispatch(agent):
    """The caller-side counterpart to PR#22's dispatcher validation:
    ``request_typed`` fetches the destination's schema, validates
    locally, and short-circuits with an error envelope WITHOUT making
    the real remote call. Saves the network round-trip for known-bad
    calls AND surfaces silent-drop bugs at the call site."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        # No scripted entry for review_diff — if validation worked,
        # the test never reaches the dispatch path.
    })
    agent._http = fake

    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"text": "wrong key"},  # caller meant 'diff'
    )

    assert "error" in result
    assert "text" in result["error"]
    assert "diff" in result["error"]
    # Exactly ONE call — the help() lookup. The dispatch was
    # short-circuited by local validation.
    assert fake.operations_called == ["help"]


@pytest.mark.asyncio
async def test_request_typed_dispatches_when_args_validate(agent):
    """A well-formed call dispatches normally. The schema fetch + the
    real dispatch produce two POSTs: help, then review_diff."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        "review_diff": {"result": {"findings": []}},
    })
    agent._http = fake

    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"diff": "valid diff content"},
    )

    assert result == {"result": {"findings": []}}
    assert fake.operations_called == ["help", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_caches_schema_across_calls(agent):
    """Schema fetch happens once per ``(agent_type, operation)``;
    subsequent typed calls reuse the cached schema. With three
    well-formed calls, we expect: 1 help + 3 review_diff = 4 POSTs.
    A naive implementation would call help on every dispatch."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        "review_diff": {"result": {"ok": True}},
    })
    agent._http = fake

    for _ in range(3):
        await agent.request_typed(
            agent_type="reviewer",
            operation="review_diff",
            args={"diff": "x"},
        )

    assert fake.operations_called == ["help", "review_diff", "review_diff", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_falls_back_when_schema_fetch_fails(agent, caplog):
    """If the help() lookup blows up (target down, transport error,
    help-skill missing on a legacy agent), validation skips with a
    warning and the call falls through to the existing ``request``
    path. Caller stays unblocked when validation infrastructure is
    independently broken."""
    import logging

    class _FailingHelp:
        def __init__(self):
            self.calls = []

        async def post(self, url, *, json=None, timeout=None):
            self.calls.append((json or {}).get("operation"))
            op = (json or {}).get("operation", "")
            if op == "help":
                raise RuntimeError("simulated transport failure")

            class _Resp:
                def json(self_inner):
                    return {"result": {"ok": True}}

            return _Resp()

        async def aclose(self):
            pass

    fake = _FailingHelp()
    agent._http = fake
    with caplog.at_level(logging.WARNING):
        result = await agent.request_typed(
            agent_type="reviewer",
            operation="review_diff",
            args={"anything": "goes"},
        )

    # Warning surfaced; dispatch still happened (no validation gate).
    assert any("schema fetch failed" in rec.message for rec in caplog.records)
    assert result == {"result": {"ok": True}}
    assert fake.calls == ["help", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_caches_negative_lookup_to_avoid_retry_storm(agent):
    """When schema fetch fails, the cache stores the
    ``_SCHEMA_FETCH_FAILED`` sentinel so subsequent typed calls don't
    retry help() on every dispatch. The caller must explicitly
    ``invalidate_schema_cache`` to force a fresh attempt. The sentinel
    is distinct from ``{}`` so a legitimately empty schema (zero-args
    contract) still validates locally — see
    ``test_request_typed_validates_against_empty_schema``."""
    fail_count = 0

    class _OneFailHelp:
        def __init__(self):
            self.calls = []

        async def post(self_inner, url, *, json=None, timeout=None):
            nonlocal fail_count
            op = (json or {}).get("operation", "")
            self_inner.calls.append(op)
            if op == "help":
                fail_count += 1
                raise RuntimeError("transient failure")

            class _Resp:
                def json(self):
                    return {"result": {"ok": True}}

            return _Resp()

        async def aclose(self):
            pass

    fake = _OneFailHelp()
    agent._http = fake
    for _ in range(3):
        await agent.request_typed(
            agent_type="reviewer", operation="review_diff", args={"x": 1},
        )
    # Help called once (first call, cached negative); subsequent
    # typed calls dispatched directly without re-fetching.
    assert fail_count == 1
    assert fake.calls == ["help", "review_diff", "review_diff", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_invalidate_schema_cache_forces_refetch(agent):
    """``invalidate_schema_cache(agent_type, operation)`` clears the
    cached schema for that one (agent, op) pair so the next typed
    call refetches. Use case: the destination agent restarted with a
    changed schema and the caller knows it."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        "review_diff": {"result": {}},
    })
    agent._http = fake

    await agent.request_typed(
        agent_type="reviewer", operation="review_diff", args={"diff": "x"},
    )
    assert fake.operations_called == ["help", "review_diff"]

    agent.invalidate_schema_cache("reviewer", "review_diff")
    await agent.request_typed(
        agent_type="reviewer", operation="review_diff", args={"diff": "y"},
    )
    # help called again after invalidation.
    assert fake.operations_called == [
        "help", "review_diff", "help", "review_diff",
    ]


@pytest.mark.asyncio
async def test_request_typed_invalidate_all_clears_every_entry(agent):
    """``invalidate_schema_cache()`` with no args drops every cached
    entry — useful when the bus restarts and every remote schema
    might be stale."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        "review_diff": {"result": {}},
    })
    agent._http = fake

    await agent.request_typed(
        agent_type="reviewer", operation="review_diff", args={"diff": "x"},
    )
    assert agent._remote_schema_cache  # populated
    agent.invalidate_schema_cache()
    assert agent._remote_schema_cache == {}


@pytest.mark.asyncio
async def test_request_typed_skips_validation_for_skill_without_schema(agent):
    """If the destination's help() returns ``found: false`` (skill
    unknown on target) or returns a non-dict schema value, fall
    through to unvalidated dispatch — same posture as the schema
    fetch failure path. The caller still gets a remote response;
    the dispatcher's strict_args (if set) handles the protection on
    the receiving side."""
    fake = _FakeBusHTTP({
        "help": {
            "result": {
                "aspect": "schema",
                "skills": [
                    {"name": "review_diff", "found": False,
                     "reason": "no skill with that name on this agent"},
                ],
            },
        },
        "review_diff": {"result": {"ok": True}},
    })
    agent._http = fake
    result = await agent.request_typed(
        agent_type="reviewer", operation="review_diff", args={"anything": "goes"},
    )
    assert result == {"result": {"ok": True}}
    assert fake.operations_called == ["help", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_caches_per_agent_type_no_cross_talk(agent):
    """Same operation name on two different agent_types must cache
    independently — ``reviewer.review_diff`` and ``developer.review_diff``
    have DIFFERENT schemas in the real fleet, and a regression that
    keys on operation alone (not the (agent_type, op) tuple) would
    silently apply one agent's schema to the other."""

    class _RouterFakeHTTP:
        """Routes help() responses by ``agent_type`` so each remote
        sees its own schema for the same operation name."""

        def __init__(self):
            self.calls: list[tuple[str, str]] = []
            self.schemas = {
                "reviewer": {"diff": {"type": "string", "required": True}},
                "developer": {"diff": {"type": "object", "required": True}},
            }

        async def post(self, url, *, json=None, timeout=None):
            payload = json or {}
            agent_type = payload.get("agent_type", "")
            op = payload.get("operation", "")
            self.calls.append((agent_type, op))
            if op == "help":
                schema = self.schemas[agent_type]
                resp = _help_schema_response("review_diff", schema)
            else:
                resp = {"result": {"agent": agent_type}}

            class _Resp:
                def json(self_inner):
                    return resp

            return _Resp()

        async def aclose(self):
            pass

    fake = _RouterFakeHTTP()
    agent._http = fake

    # reviewer expects str → str arg passes
    r1 = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"diff": "patch"},
    )
    assert r1 == {"result": {"agent": "reviewer"}}

    # developer expects object — same str arg should now FAIL because
    # the developer schema demands ``object``. If the cache were keyed
    # on operation alone, reviewer's str-schema would have been
    # reused and this call would have wrongly passed.
    r2 = await agent.request_typed(
        agent_type="developer",
        operation="review_diff",
        args={"diff": "patch"},
    )
    assert "error" in r2
    assert "object" in r2["error"]
    # Only the developer-flavored object call should pass:
    r3 = await agent.request_typed(
        agent_type="developer",
        operation="review_diff",
        args={"diff": {"v": 1}},
    )
    assert r3 == {"result": {"agent": "developer"}}


@pytest.mark.asyncio
async def test_request_typed_surfaces_missing_required_field(agent):
    """Required-field-missing surfaces locally as a clean error
    envelope before any remote dispatch — same shape as the
    dispatcher-side strict_args path."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
    })
    agent._http = fake
    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={},
    )
    assert "error" in result
    assert "missing required" in result["error"]
    assert "diff" in result["error"]
    # No remote dispatch — local validation short-circuited.
    assert fake.operations_called == ["help"]


@pytest.mark.asyncio
async def test_request_typed_invalidate_agent_type_only_clears_that_agent(agent):
    """``invalidate_schema_cache(agent_type)`` with no operation
    clears every cached entry for that agent but leaves other
    agents' cached schemas intact. Use case: a single remote agent
    restarted with new schemas; other agents are still in sync."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response(
            "review_diff", {"diff": {"type": "string", "required": True}},
        ),
        "review_diff": {"result": {}},
    })
    agent._http = fake

    # Populate cache entries for two agent types.
    await agent.request_typed(
        agent_type="reviewer", operation="review_diff", args={"diff": "x"},
    )
    await agent.request_typed(
        agent_type="developer", operation="review_diff", args={"diff": "y"},
    )
    assert ("reviewer", "review_diff") in agent._remote_schema_cache
    assert ("developer", "review_diff") in agent._remote_schema_cache

    # Invalidate only the reviewer entries.
    agent.invalidate_schema_cache("reviewer")
    assert ("reviewer", "review_diff") not in agent._remote_schema_cache
    # Developer's entry survives.
    assert ("developer", "review_diff") in agent._remote_schema_cache


@pytest.mark.asyncio
async def test_request_typed_handles_realistic_bus_envelope_with_extra_keys(agent):
    """The bus envelope can carry extra keys (``trace_id``, timing,
    routing metadata) alongside ``result``. The probe must extract
    the schema from inside ``result`` regardless of what else the
    envelope carries."""

    class _RealisticHTTP:
        def __init__(self):
            self.calls: list[str] = []

        async def post(self_inner, url, *, json=None, timeout=None):
            op = (json or {}).get("operation", "")
            self_inner.calls.append(op)
            if op == "help":
                resp = {
                    "result": {
                        "aspect": "schema",
                        "skills": [{
                            "name": "review_diff",
                            "found": True,
                            "value": {
                                "diff": {"type": "string", "required": True},
                            },
                        }],
                    },
                    "trace_id": "t-abc123",
                    "served_at": 1234567890.0,
                    "agent_type": "reviewer",
                    "served_by_pool": "gpu",
                }
            else:
                resp = {
                    "result": {"findings": []},
                    "trace_id": "t-def456",
                    "served_at": 1234567891.0,
                }

            class _R:
                def json(self):
                    return resp

            return _R()

        async def aclose(self):
            pass

    fake = _RealisticHTTP()
    agent._http = fake

    # Schema is correctly extracted from inside the realistic envelope;
    # validation runs and passes the well-formed call through.
    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"diff": "patch content"},
    )
    assert result["result"] == {"findings": []}
    assert result["trace_id"] == "t-def456"
    assert fake.calls == ["help", "review_diff"]


@pytest.mark.asyncio
async def test_request_typed_validates_against_empty_schema(agent):
    """An empty schema ``{}`` is a valid zero-args contract under
    ``strict_args=True`` — caller-side validation must reject any
    kwargs locally rather than treating ``{}`` as "no schema". This
    pins the negative-cache sentinel as distinct from a real empty
    schema (Copilot pass-1 finding on PR#23)."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response("ping", {}),
    })
    agent._http = fake
    result = await agent.request_typed(
        agent_type="reviewer", operation="ping", args={"unexpected": "kw"},
    )
    # Local rejection — empty-schema marker, no remote dispatch.
    assert "error" in result
    assert "empty schema" in result["error"]
    assert fake.operations_called == ["help"]


@pytest.mark.asyncio
async def test_request_typed_empty_schema_passes_zero_args_through(agent):
    """The empty-schema contract still allows the no-arg call: when
    args is ``None`` (or ``{}``), validation passes and the request
    is dispatched. This is the other side of the empty-schema
    contract — empty in, empty out."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response("ping", {}),
        "ping": {"result": {"pong": True}},
    })
    agent._http = fake
    result = await agent.request_typed(
        agent_type="reviewer", operation="ping", args=None,
    )
    assert result == {"result": {"pong": True}}
    assert fake.operations_called == ["help", "ping"]


@pytest.mark.asyncio
async def test_request_typed_negative_cache_sentinel_distinct_from_empty_schema(agent):
    """The "fetch failed" sentinel must not be ``{}`` — otherwise a
    legitimately empty schema would mask validation. After a failed
    fetch and a subsequent successful one (post-invalidation) for an
    empty-schema skill, kwargs should be rejected locally — proving
    the cache state was the failure sentinel, not an empty schema."""
    call_count = {"help": 0}

    class _FlipFlopHTTP:
        def __init__(self):
            self.calls: list[str] = []

        async def post(self_inner, url, *, json=None, timeout=None):
            op = (json or {}).get("operation", "")
            self_inner.calls.append(op)
            if op == "help":
                call_count["help"] += 1
                if call_count["help"] == 1:
                    raise RuntimeError("transient")
                resp = _help_schema_response("ping", {})
            else:
                resp = {"result": {"pong": True}}

            class _R:
                def json(self):
                    return resp

            return _R()

        async def aclose(self):
            pass

    fake = _FlipFlopHTTP()
    agent._http = fake

    # First call: fetch fails, sentinel cached, dispatch falls through.
    r1 = await agent.request_typed(
        agent_type="reviewer", operation="ping", args={"k": "v"},
    )
    assert r1 == {"result": {"pong": True}}

    # Invalidate so the next call refetches; empty schema returns this time.
    agent.invalidate_schema_cache("reviewer", "ping")

    # Second call: schema = {}, kwargs must be rejected locally.
    r2 = await agent.request_typed(
        agent_type="reviewer", operation="ping", args={"k": "v"},
    )
    assert "error" in r2
    assert "empty schema" in r2["error"]


@pytest.mark.asyncio
async def test_request_typed_stampede_real_schema_overwrites_failure_sentinel(agent):
    """Concurrent stampede recovery: when one sibling coroutine
    writes the failure sentinel before another's successful fetch
    lands, the successful fetch must overwrite the sentinel rather
    than silently losing the schema. Otherwise a transient sibling
    failure would poison the cache for an in-flight success until
    manual ``invalidate_schema_cache`` (Copilot pass-2 finding)."""
    # Pre-populate the cache with the failure sentinel by hand,
    # then call request_typed — the path should run a fresh fetch
    # because the key check uses ``cache_key not in cache``... but
    # the key IS in cache (with the sentinel). So this test
    # exercises the in-method overwrite branch by simulating two
    # interleaved fetches: we manually set the sentinel after the
    # first fetch starts but before its write lands.
    from khonliang_bus.agent import _SCHEMA_FETCH_FAILED

    schema = {"diff": {"type": "string", "required": True}}
    real_schema_response = _help_schema_response("review_diff", schema)

    # Wire up a fake HTTP that returns the real schema, but right
    # before request_typed writes the cache, a "sibling" injects
    # the failure sentinel into the cache slot. The fix should
    # detect the sentinel and overwrite it with the real schema.
    sibling_injected = False

    class _RaceyHTTP:
        def __init__(self):
            self.calls: list[str] = []

        async def post(self_inner, url, *, json=None, timeout=None):
            nonlocal sibling_injected
            op = (json or {}).get("operation", "")
            self_inner.calls.append(op)
            if op == "help" and not sibling_injected:
                # Inject the sibling's failure sentinel into the
                # cache before we return — simulates a concurrent
                # coroutine that lost the race to the failure side.
                # ``request_typed`` has already populated
                # ``_remote_schema_cache`` (line 1495-1496), so we
                # mutate it in place — assignment via
                # ``= getattr(...) or {}`` would silently swap the
                # dict because ``{}`` is falsy, breaking the
                # caller's local-variable identity.
                agent._remote_schema_cache[("reviewer", "review_diff")] = (
                    _SCHEMA_FETCH_FAILED
                )
                sibling_injected = True
                resp = real_schema_response
            else:
                resp = {"result": {"ok": True}}

            class _R:
                def json(self):
                    return resp

            return _R()

        async def aclose(self):
            pass

    fake = _RaceyHTTP()
    agent._http = fake

    # First call: our fetch succeeds, the "sibling" wrote the
    # sentinel mid-fetch. Our fix must overwrite it.
    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"diff": "hello"},
    )
    # Successful real-schema overwrite — call dispatched, no error.
    assert result == {"result": {"ok": True}}
    # Cache now holds the real schema, not the sentinel.
    cached = agent._remote_schema_cache[("reviewer", "review_diff")]
    assert cached == schema
    assert cached is not _SCHEMA_FETCH_FAILED


@pytest.mark.asyncio
async def test_request_typed_real_schema_not_overwritten_by_sentinel(agent):
    """Inverse of the stampede recovery test: when the cache already
    holds a real schema, a subsequent failed fetch must NOT
    overwrite it with the sentinel. This pins the asymmetric
    overwrite policy — sentinel→real upgrades, real→sentinel
    downgrades are blocked."""
    from khonliang_bus.agent import _SCHEMA_FETCH_FAILED

    schema = {"diff": {"type": "string", "required": True}}
    # Hand-prime the cache with a real schema, then simulate the
    # mid-fetch overwrite path: another fetch failing while the
    # cache already holds a successful schema.
    agent._remote_schema_cache = {("reviewer", "review_diff"): schema}

    # request_typed will see the key already in cache (real schema)
    # and skip the fetch entirely — that's the correct fast path.
    # To exercise the "real schema present + new failure" branch
    # explicitly, we drive the in-method overwrite logic by hand
    # via a second concurrent simulation: clear the entry, then
    # have the help fetch fail while a sibling has already written
    # a real schema before our write lands.
    sibling_wrote_real = False

    class _SiblingWritesRealHTTP:
        async def post(self_inner, url, *, json=None, timeout=None):
            nonlocal sibling_wrote_real
            op = (json or {}).get("operation", "")
            if op == "help" and not sibling_wrote_real:
                # Sibling injects a real schema before our failure lands.
                agent._remote_schema_cache[("reviewer", "review_diff")] = (
                    schema
                )
                sibling_wrote_real = True
                raise RuntimeError("our fetch fails")

            class _R:
                def json(self):
                    return {"result": {"ok": True}}

            return _R()

        async def aclose(self):
            pass

    # Clear cache so request_typed will fetch.
    agent._remote_schema_cache.clear()
    agent._http = _SiblingWritesRealHTTP()

    # Our fetch fails (raises) → returns None → caller would write
    # sentinel. But sibling wrote a real schema mid-flight. The
    # asymmetric-overwrite logic must keep the real schema.
    result = await agent.request_typed(
        agent_type="reviewer",
        operation="review_diff",
        args={"diff": "x"},
    )
    # Real schema validated locally → call dispatched cleanly.
    assert result == {"result": {"ok": True}}
    cached = agent._remote_schema_cache[("reviewer", "review_diff")]
    assert cached == schema
    assert cached is not _SCHEMA_FETCH_FAILED


@pytest.mark.asyncio
async def test_fetch_remote_schema_warns_on_skill_not_found(agent, caplog):
    """``_fetch_remote_skill_schema`` returning None silently for
    ``found:false`` was a docstring/behavior drift (Copilot pass-2
    finding). Now logs a single warning with the reason so operators
    notice when typed-call validation is silently disabled. Spam is
    bounded by the negative-cache: one warning per (agent_type, op)
    until ``invalidate_schema_cache``."""
    import logging

    fake = _FakeBusHTTP({
        "help": {
            "result": {
                "aspect": "schema",
                "skills": [{
                    "name": "review_diff", "found": False,
                    "reason": "no skill with that name on this agent",
                }],
            },
        },
        "review_diff": {"result": {"ok": True}},
    })
    agent._http = fake

    with caplog.at_level(logging.WARNING):
        await agent.request_typed(
            agent_type="reviewer", operation="review_diff",
            args={"x": 1},
        )

    msgs = [r.message for r in caplog.records if "skill not found" in r.message]
    assert msgs, "expected warning about skill not found"
    assert "reviewer" in msgs[0]
    assert "review_diff" in msgs[0]


@pytest.mark.asyncio
async def test_fetch_remote_schema_warns_on_non_dict_envelope(agent, caplog):
    """Non-dict bus envelope (a list / scalar / None) is a transport
    or upstream-bug signal. Log a warning rather than silently
    dropping validation."""
    import logging

    class _NonDictHTTP:
        async def post(self_inner, url, *, json=None, timeout=None):
            op = (json or {}).get("operation", "")

            class _R:
                def json(self):
                    if op == "help":
                        return ["unexpected", "list", "shape"]
                    return {"result": {"ok": True}}

            return _R()

        async def aclose(self):
            pass

    agent._http = _NonDictHTTP()

    with caplog.at_level(logging.WARNING):
        await agent.request_typed(
            agent_type="reviewer", operation="review_diff", args={"x": 1},
        )

    msgs = [r.message for r in caplog.records if "non-dict envelope" in r.message]
    assert msgs, "expected warning about non-dict envelope"
    assert "list" in msgs[0]


@pytest.mark.asyncio
async def test_request_typed_rejects_non_dict_args_with_envelope_error(agent):
    """Non-dict ``args`` (a list, a string, an int) is a caller bug.
    Reject it locally with the same error envelope the dispatcher
    emits — no schema fetch, no remote round-trip, no shape-mismatch
    crash inside the validator (Copilot pass-3 finding)."""
    fake = _FakeBusHTTP({})
    agent._http = fake

    for bad_args, expected_type in [
        (["not", "a", "dict"], "list"),
        ("string", "str"),
        (42, "int"),
        ((1, 2), "tuple"),
    ]:
        result = await agent.request_typed(
            agent_type="reviewer", operation="review_diff", args=bad_args,
        )
        assert "error" in result
        assert "must be an object" in result["error"]
        assert expected_type in result["error"]
    # No schema fetch, no remote dispatch — pure local rejection.
    assert fake.operations_called == []


@pytest.mark.asyncio
async def test_request_typed_normalizes_none_args_to_empty_dict(agent):
    """``args=None`` is the no-args contract; it must round-trip the
    same way ``args={}`` does — schema fetched, validated, dispatched
    with ``{}`` to the remote so the receiver sees a consistent
    shape regardless of which form the caller used."""
    fake = _FakeBusHTTP({
        "help": _help_schema_response("ping", {}),
        "ping": {"result": {"pong": True}},
    })
    agent._http = fake
    result = await agent.request_typed(
        agent_type="reviewer", operation="ping", args=None,
    )
    assert result == {"result": {"pong": True}}
    # Outgoing dispatch carried args={} (normalized from None).
    ping_call = next(
        c for c in fake.calls if c["json"]["operation"] == "ping"
    )
    assert ping_call["json"]["args"] == {}
