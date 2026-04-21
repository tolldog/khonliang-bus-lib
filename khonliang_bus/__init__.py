"""khonliang-bus-lib: agent library for the khonliang-bus platform.

Everything an agent author needs to register with the bus, handle
requests, publish events, manage session context, and test handlers.

Quick start::

    from khonliang_bus import BaseAgent, Skill, handler

    class MyAgent(BaseAgent):
        agent_type = "my_agent"
        module_name = "my_agent.agent"
        version = "0.1.0"

        def register_skills(self):
            return [Skill("do_thing", "Does a thing", {"x": {"type": "string"}})]

        @handler("do_thing")
        async def do_thing(self, args):
            return {"result": args.get("x", "")}

    if __name__ == "__main__":
        import asyncio
        agent = MyAgent.from_cli()
        asyncio.run(agent.start())

For testing::

    from khonliang_bus.testing import AgentTestHarness

    harness = AgentTestHarness(MyAgent)
    result = await harness.call("do_thing", {"x": "hello"})
"""

__version__ = "0.3.0"

from khonliang_bus.agent import BaseAgent, Collaboration, Skill, handler
from khonliang_bus.client import BusClient, Message
from khonliang_bus.versioning import resolve_version
from khonliang_bus.registry import (
    AggregationMethod,
    CapabilityRoute,
    ContextNeed,
    CostLevel,
    ExecutionMode,
    ExecutionProfile,
    ExecutionRun,
    LatencyClass,
    Locality,
    ModelSize,
    OutputContract,
    OutputMode,
    ProviderDescriptor,
    ProviderStatus,
    ProviderType,
    ReasoningLevel,
    RegistryValue,
    RunTier,
    RuntimeProfile,
    SkillAuthority,
    SkillDescriptor,
    SkillStatus,
)

__all__ = [
    "AggregationMethod",
    "BaseAgent",
    "BusClient",
    "CapabilityRoute",
    "Collaboration",
    "ContextNeed",
    "CostLevel",
    "ExecutionMode",
    "ExecutionProfile",
    "ExecutionRun",
    "LatencyClass",
    "Locality",
    "Message",
    "ModelSize",
    "OutputContract",
    "OutputMode",
    "ProviderDescriptor",
    "ProviderStatus",
    "ProviderType",
    "ReasoningLevel",
    "RegistryValue",
    "RunTier",
    "RuntimeProfile",
    "Skill",
    "SkillAuthority",
    "SkillDescriptor",
    "SkillStatus",
    "handler",
    "resolve_version",
]
