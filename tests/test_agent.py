"""Tests for BaseAgent: skills, handlers, connector-based lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from khonliang_bus import BaseAgent, Skill, Collaboration, handler
from khonliang_bus import agent as agent_module


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
    # EchoAgent declares echo/upper/fail; health_check comes for free from BaseAgent.
    assert set(agent._handlers) == {"echo", "upper", "fail", "health_check"}


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


def test_bare_agent_advertises_health_check():
    """An agent with no skills of its own still gets health_check."""
    a = BareAgent(agent_id="bare", bus_url="http://localhost:9999")
    names = {s.name for s in a._all_skills()}
    assert names == {"health_check"}


def test_subclass_skills_augmented_with_health_check(agent):
    """Subclass skills + built-in health_check are merged."""
    names = {s.name for s in agent._all_skills()}
    assert names == {"echo", "upper", "fail", "health_check"}


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


def test_resolve_distribution_version_handles_missing_metadata(monkeypatch):
    """Helper returns None for modules with no installed distribution."""
    import importlib.metadata as md

    monkeypatch.setattr(md, "packages_distributions", lambda: {})
    assert agent_module._resolve_distribution_version("nope.mod") is None
    assert agent_module._resolve_distribution_version("") is None


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
