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
    # EchoAgent declares echo/upper/fail; health_check + welcome come for
    # free from BaseAgent's BUILT_IN_SKILLS.
    assert set(agent._handlers) == {"echo", "upper", "fail", "health_check", "welcome"}


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
    assert names == {"health_check", "welcome"}


def test_subclass_skills_augmented_with_builtins(agent):
    """Subclass skills + built-in health_check + welcome are merged."""
    names = {s.name for s in agent._all_skills()}
    assert names == {"echo", "upper", "fail", "health_check", "welcome"}


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
async def test_welcome_default_brief_returns_identity_and_categories(agent):
    """Bare agent (no WELCOME override) gets identity + skill_categories.

    No editorial fields when WELCOME is left at its empty default.
    """
    result = await agent._dispatch_request({"operation": "welcome", "args": {}})
    assert result["agent_id"] == "echo-test"
    assert result["agent_type"] == "echo"
    assert result["version"] == "0.2.0"
    assert result["skill_count"] == 5  # echo, upper, fail, health_check, welcome
    # Editorial fields absent because WELCOME is the empty default.
    assert "role" not in result
    assert "mission" not in result
    assert "boundaries" not in result
    assert "entry_points" not in result
    # Categories present at brief detail.
    assert "skill_categories" in result
    # health_check + welcome are the builtin category.
    assert result["skill_categories"]["builtin"] == 2


@pytest.mark.asyncio
async def test_welcome_compact_omits_categories(agent):
    """compact returns only identity + role; no skill catalog."""
    result = await agent._dispatch_request(
        {"operation": "welcome", "args": {"detail": "compact"}}
    )
    assert "skill_categories" not in result
    assert "skills_by_category" not in result
    assert "mission" not in result
    assert result["skill_count"] == 5


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
    assert set(cats["builtin"]) == {"health_check", "welcome"}


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
    assert set(cats["builtin"]) == {"health_check", "welcome"}


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
