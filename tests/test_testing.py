"""Tests for the testing harness itself."""

from __future__ import annotations

import pytest

from khonliang_bus import BaseAgent, Skill, Collaboration, handler
from khonliang_bus.testing import AgentTestHarness


class SampleAgent(BaseAgent):
    agent_type = "sample"
    module_name = "tests.test_testing"
    version = "1.0.0"

    def register_skills(self):
        return [
            Skill("greet", "Greet someone", {"name": {"type": "string"}}),
            Skill("add", "Add two numbers", {"a": {"type": "int"}, "b": {"type": "int"}}),
        ]

    def register_collaborations(self):
        return [
            Collaboration(
                "greet_and_log",
                "Greet then log",
                requires={"logger": ">=0.1.0"},
                steps=[{"call": "sample.greet"}, {"call": "logger.log"}],
            ),
        ]

    @handler("greet")
    async def greet(self, args):
        return {"greeting": f"hello {args.get('name', 'world')}"}

    @handler("add")
    async def add(self, args):
        return {"sum": args.get("a", 0) + args.get("b", 0)}


@pytest.fixture
def harness():
    return AgentTestHarness(SampleAgent)


# -- skills --

def test_skill_names(harness):
    assert harness.skill_names == {"greet", "add"}


def test_get_skill(harness):
    s = harness.get_skill("greet")
    assert s is not None
    assert s.description == "Greet someone"


def test_get_skill_not_found(harness):
    assert harness.get_skill("nonexistent") is None


def test_assert_skill_exists(harness):
    s = harness.assert_skill_exists("greet")
    assert s.name == "greet"


def test_assert_skill_exists_with_description(harness):
    harness.assert_skill_exists("greet", description="Greet")


def test_assert_skill_fails_when_missing(harness):
    with pytest.raises(AssertionError, match="not found"):
        harness.assert_skill_exists("ghost")


# -- collaborations --

def test_collaboration_names(harness):
    assert harness.collaboration_names == {"greet_and_log"}


def test_assert_collaboration_exists(harness):
    c = harness.assert_collaboration_exists("greet_and_log")
    assert c.requires == {"logger": ">=0.1.0"}


def test_assert_collaboration_with_requires(harness):
    harness.assert_collaboration_exists(
        "greet_and_log",
        requires={"logger": ">=0.1.0"},
    )


def test_assert_collaboration_fails_when_missing(harness):
    with pytest.raises(AssertionError, match="not found"):
        harness.assert_collaboration_exists("ghost")


def test_assert_collaboration_fails_on_wrong_requires(harness):
    with pytest.raises(AssertionError, match="requires mismatch"):
        harness.assert_collaboration_exists(
            "greet_and_log",
            requires={"logger": ">=9.9.9"},
        )


# -- handler dispatch --

@pytest.mark.asyncio
async def test_call_handler(harness):
    result = await harness.call("greet", {"name": "tim"})
    assert result == {"greeting": "hello tim"}


@pytest.mark.asyncio
async def test_call_handler_default_args(harness):
    result = await harness.call("greet")
    assert result == {"greeting": "hello world"}


@pytest.mark.asyncio
async def test_call_add(harness):
    result = await harness.call("add", {"a": 3, "b": 7})
    assert result == {"sum": 10}


@pytest.mark.asyncio
async def test_call_unknown_raises(harness):
    with pytest.raises(KeyError, match="no handler"):
        await harness.call("nonexistent")


# -- registration --

def test_registration_captured(harness):
    reg = harness.registration
    assert reg.agent_id == "sample-test"
    assert reg.agent_type == "sample"
    assert reg.version == "1.0.0"
    assert len(reg.skills) == 2
    assert len(reg.collaborations) == 1
