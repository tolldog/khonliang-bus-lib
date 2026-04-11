"""Tests for BaseAgent: skills, handlers, registration, session context, request helper."""

from __future__ import annotations

import asyncio

import pytest

from khonliang_bus import BaseAgent, Skill, Collaboration, handler


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
    assert set(agent._handlers) == {"echo", "upper", "fail"}


def test_handler_success(agent):
    app = agent._build_app()
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/v1/handle", json={
        "operation": "echo", "args": {"text": "hello"}, "correlation_id": "c1",
    }).json()
    assert r["result"] == {"echoed": "hello"}
    assert r["correlation_id"] == "c1"


def test_handler_failure(agent):
    app = agent._build_app()
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/v1/handle", json={
        "operation": "fail", "args": {}, "correlation_id": "c2",
    }).json()
    assert "intentional" in r["error"]
    assert r["retryable"] is True


def test_handler_unknown_operation(agent):
    app = agent._build_app()
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.post("/v1/handle", json={
        "operation": "nope", "args": {}, "correlation_id": "c3",
    }).json()
    assert "unknown operation" in r["error"]
    assert r["retryable"] is False


def test_health_endpoint(agent):
    app = agent._build_app()
    from fastapi.testclient import TestClient
    c = TestClient(app)
    r = c.get("/v1/health").json()
    assert r["agent_id"] == "echo-test"


# -- CLI --

def test_from_cli():
    a = EchoAgent.from_cli(["--id", "my-echo", "--bus", "http://localhost:8787"])
    assert a.agent_id == "my-echo"
    assert a.bus_url == "http://localhost:8787"


# -- registration failure --

def test_register_fails_fatally():
    a = EchoAgent(agent_id="test-fatal", bus_url="http://localhost:1")
    with pytest.raises(RuntimeError, match="failed to register"):
        a._build_app()
        asyncio.run(a._register())


def test_register_error_includes_bus_url():
    a = EchoAgent(agent_id="test-msg", bus_url="http://localhost:1")
    with pytest.raises(RuntimeError, match="localhost:1"):
        a._build_app()
        asyncio.run(a._register())


# -- helpers exist --

def test_publish_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.publish)


def test_nack_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.nack)


def test_request_is_async(agent):
    assert asyncio.iscoroutinefunction(agent.request)


def test_session_context_helpers_exist(agent):
    assert asyncio.iscoroutinefunction(agent.get_session_context)
    assert asyncio.iscoroutinefunction(agent.update_session_context)


def test_version_attribute(agent):
    assert agent.version == "0.2.0"
