"""Testing utilities for bus agents.

Test agents without running the real bus. The :class:`AgentTestHarness`
creates an agent instance, collects its skills and collaborations, and
dispatches directly to ``@handler`` methods — no HTTP, no bus, no FastAPI.

Usage::

    from khonliang_bus.testing import AgentTestHarness

    class TestMyAgent:
        def setup_method(self):
            self.harness = AgentTestHarness(MyAgent)

        async def test_find_papers(self):
            result = await self.harness.call("find_papers", {"query": "consensus"})
            assert "papers" in result

        def test_skills_registered(self):
            assert "find_papers" in self.harness.skill_names

        def test_collaboration_requires(self):
            self.harness.assert_collaboration_exists(
                "evaluate_spec",
                requires={"researcher": ">=0.5.0"},
            )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from khonliang_bus.agent import BaseAgent, Collaboration, Skill


@dataclass
class MockRegistration:
    """Captured registration payload from an agent."""

    agent_id: str
    agent_type: str
    version: str
    skills: list[dict[str, Any]]
    collaborations: list[dict[str, Any]]


class AgentTestHarness:
    """Test harness for bus agents. Exercises handlers without a real bus.

    Creates the agent, collects its skills and collaborations, and
    provides a :meth:`call` method that dispatches directly to the
    agent's ``@handler`` methods — no HTTP, no bus, no FastAPI.

    Args:
        agent_cls: The BaseAgent subclass to test.
        agent_id: Override agent ID (default: ``"{agent_type}-test"``).
        config_path: Optional config path to pass to the agent.
        **kwargs: Extra kwargs passed to the agent constructor.
    """

    def __init__(
        self,
        agent_cls: type[BaseAgent],
        agent_id: str = "",
        config_path: str = "",
        **kwargs: Any,
    ):
        self.agent_cls = agent_cls
        agent_id = agent_id or f"{agent_cls.agent_type}-test"
        self.agent = agent_cls(
            agent_id=agent_id,
            bus_url="http://mock-bus:0",
            config_path=config_path,
            **kwargs,
        )
        self._skills = self.agent.register_skills()
        self._collaborations = self.agent.register_collaborations()

        self.registration = MockRegistration(
            agent_id=agent_id,
            agent_type=agent_cls.agent_type,
            version=getattr(self.agent, "version", "0.0.0"),
            skills=[
                {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                    "since": s.since,
                }
                for s in self._skills
            ],
            collaborations=[
                {
                    "name": c.name,
                    "description": c.description,
                    "requires": c.requires,
                    "steps": c.steps,
                }
                for c in self._collaborations
            ],
        )

    @property
    def skill_names(self) -> set[str]:
        """Set of registered skill names — subclass register_skills only."""
        return {s.name for s in self._skills}

    @property
    def skills(self) -> list[Skill]:
        """Registered Skill objects — subclass register_skills only."""
        return list(self._skills)

    @property
    def all_skill_names(self) -> set[str]:
        """Set of every skill name the agent advertises — subclass
        ``register_skills`` plus :class:`BaseAgent` built-ins like the
        default ``health_check``. Use this rather than
        :attr:`skill_names` when asserting symmetry against
        :attr:`handler_names`, since BaseAgent's built-in handlers are
        always present even when the subclass omits them from
        ``register_skills``.
        """
        return {s.name for s in self.agent._all_skills()}

    @property
    def handler_names(self) -> set[str]:
        """Set of every ``@handler`` operation name on the agent.

        Sourced from the same internal handler registry that
        :meth:`call` dispatches against, so what you see here is what
        the agent will actually answer. Pair with :attr:`all_skill_names`
        for set-symmetry registration checks.
        """
        return set(self.agent._handlers)

    @property
    def collaboration_names(self) -> set[str]:
        """Set of declared collaboration names."""
        return {c.name for c in self._collaborations}

    @property
    def collaborations(self) -> list[Collaboration]:
        """Declared Collaboration objects."""
        return list(self._collaborations)

    async def call(
        self,
        operation: str,
        args: dict[str, Any] | None = None,
    ) -> Any:
        """Call an agent handler directly. Returns the handler's result.

        Dispatches to the ``@handler``-decorated method matching
        ``operation``. Raises ``KeyError`` if no handler exists.
        Raises whatever the handler raises (no error wrapping).

        Args:
            operation: The skill name to invoke.
            args: Arguments dict (passed to the handler).

        Returns:
            Whatever the handler returns.
        """
        handler_fn = self.agent._handlers.get(operation)
        if handler_fn is None:
            raise KeyError(
                f"no handler for operation {operation!r}. "
                f"Available: {sorted(self.agent._handlers)}"
            )
        return await handler_fn(args or {})

    def get_skill(self, name: str) -> Skill | None:
        """Look up a skill by name. Returns None if not found."""
        for s in self._skills:
            if s.name == name:
                return s
        return None

    def get_collaboration(self, name: str) -> Collaboration | None:
        """Look up a collaboration by name. Returns None if not found."""
        for c in self._collaborations:
            if c.name == name:
                return c
        return None

    def assert_skill_exists(
        self, name: str, description: str | None = None
    ) -> Skill:
        """Assert a skill is registered. Optionally check description contains text."""
        skill = self.get_skill(name)
        assert skill is not None, (
            f"skill {name!r} not found. Registered: {sorted(self.skill_names)}"
        )
        if description is not None:
            assert description in skill.description, (
                f"skill {name!r} description mismatch: "
                f"expected {description!r} in {skill.description!r}"
            )
        return skill

    def assert_collaboration_exists(
        self,
        name: str,
        requires: dict[str, str] | None = None,
    ) -> Collaboration:
        """Assert a collaboration is declared. Optionally check requirements."""
        collab = self.get_collaboration(name)
        assert collab is not None, (
            f"collaboration {name!r} not found. "
            f"Declared: {sorted(self.collaboration_names)}"
        )
        if requires is not None:
            assert collab.requires == requires, (
                f"collaboration {name!r} requires mismatch: "
                f"expected {requires}, got {collab.requires}"
            )
        return collab
