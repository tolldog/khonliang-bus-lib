# khonliang-bus-lib

Importable agent SDK for the khonliang bus.

Use this library from agent/app repos that need to register with
`khonliang-bus`, expose skills, handle requests, publish/subscribe to topics,
or test agent behavior without booting the full bus service.

## What Belongs Here

- `BaseAgent` for HTTP callback agents.
- `Skill` and `Collaboration` registration descriptors.
- `@handler` for mapping agent methods to bus request operations.
- `BusClient` for low-level bus HTTP/WebSocket operations.
- `BusConnector` for registration, request handling, and heartbeat plumbing.
- `AgentTestHarness` for unit-testing agents without a running bus.

## What Does Not Belong Here

- Bus server routes, database schema, sessions, artifacts, or flow execution.
  Those live in `khonliang-bus`.
- Domain behavior such as researcher paper ingestion or developer FR lifecycle.
  Those live in their app/agent repos.
- MCP adapter tool generation. The Claude-facing adapter lives in
  `khonliang-bus`.

## Typical Consumer

```python
from khonliang_bus import BaseAgent, Skill, handler


class ExampleAgent(BaseAgent):
    agent_id = "example-primary"
    agent_type = "example"

    def skills(self):
        return [
            Skill(
                name="ping",
                description="Return a ping response.",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    @handler("ping")
    async def ping(self, args):
        return {"result": "pong"}
```

The bus service starts or discovers the agent, the agent registers its skills,
and the bus MCP adapter exposes those skills to Claude.

## Development

```bash
pip install -e ".[test]"
pytest
```

Keep this package lightweight. If adding a dependency would only help one app,
put that code in the app repo instead.
