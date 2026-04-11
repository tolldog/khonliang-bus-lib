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

__version__ = "0.1.0"

from khonliang_bus.agent import BaseAgent, Collaboration, Skill, handler
from khonliang_bus.client import BusClient, Message

__all__ = [
    "BaseAgent",
    "BusClient",
    "Collaboration",
    "Message",
    "Skill",
    "handler",
]
