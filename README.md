# khonliang-bus-lib

Importable agent SDK for the khonliang bus.

Use this library from agent/app repos that need to register with
`khonliang-bus`, expose skills, handle requests, publish/subscribe to topics,
or test agent behavior without booting the full bus service.

## What Belongs Here

- `BaseAgent` for agents that connect to the bus via `BusConnector`.
- `Skill` and `Collaboration` registration descriptors.
- Skill registry contracts for capability ownership, runtime profile, execution
  shape, and output contracts.
- `@handler` for mapping agent methods to bus request operations.
- `BusClient` for low-level bus HTTP/WebSocket operations.
- `BusConnector` for registration, request handling, and heartbeat plumbing.
- `AgentTestHarness` for unit-testing agents without a running bus.

## What Does Not Belong Here

- Bus server routes, database schema, sessions, artifacts, or flow execution.
  Those live in `khonliang-bus`.
- Domain behavior such as researcher paper ingestion or developer FR lifecycle.
  Those live in their app/agent repos.
- Bus-side MCP adapter tool generation. The Claude-facing adapter lives in
  `khonliang-bus`. This library only includes an optional `from_mcp` migration
  helper for wrapping existing FastMCP tools as bus agent handlers.

## Typical Consumer

```python
from khonliang_bus import BaseAgent, Skill, handler


class ExampleAgent(BaseAgent):
    agent_id = "example-primary"
    agent_type = "example"

    def register_skills(self):
        return [
            Skill(
                name="ping",
                description="Return a ping response.",
                parameters={"type": "object", "properties": {}},
            )
        ]

    @handler("ping")
    async def ping(self, args):
        return {"result": "pong"}
```

The bus service starts or discovers the agent, the agent registers its skills,
and the bus MCP adapter exposes those skills to Claude.

## Registry Metadata

Agents can keep using the minimal `Skill(name, description, parameters)` form.
When a skill needs routing metadata, declare the capability contract at the same
boundary:

```python
from khonliang_bus import ExecutionProfile, OutputContract, Skill


Skill(
    name="next_work_unit",
    description="Return the next ready FR bundle.",
    parameters={"target": {"type": "string"}},
    capability="fr.bundle.next",
    output_contract=OutputContract(
        output_mode="artifact+summary",
        artifact_kind="work_unit",
        summary_fields=["frs", "suggested_next_actions"],
    ),
    execution_profiles=[
        ExecutionProfile(
            profile_id="three-medium-one-large",
            mode="workflow",
            runs=[
                {"tier": "medium", "count": 3, "role": "candidate"},
                {"tier": "large", "count": 1, "role": "adjudicator"},
            ],
            aggregation="rank_merge_adjudicate",
        )
    ],
    runtime_profile={"model_size": "small", "latency": "fast", "cost": "low"},
)
```

The registry dataclasses are dependency-free and round-trip through plain
dictionaries so the bus service can persist them or route by capability later.

## Development

```bash
pip install -e ".[test]"
pytest
```

Keep this package lightweight. If adding a dependency would only help one app,
put that code in the app repo instead.
