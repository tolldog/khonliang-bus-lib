# khonliang-bus-lib

Agent SDK for the khonliang-bus platform. This is what agents import to connect to and interact with the bus.

## What this library provides

- `BaseAgent` — base class for building agents that register with the bus
- `Skill`, `Collaboration` — typed registration descriptors
- `@handler` — decorator marking methods as skill handlers
- `BusClient` — low-level client for pub/sub, ack, nack, deregister
- `Message` — message type received from subscriptions
- `AgentTestHarness` — test agents without a running bus

## What this library does NOT provide

- The bus server itself (that's `khonliang-bus`)
- The MCP adapter (that's `khonliang-bus/bus/mcp_adapter.py`)
- Flow orchestration engine (bus-side concern)
- Database schema (bus-side concern)

## Ecosystem position

```
LIBRARIES
├─ khonliang          — agent primitives, stores, MCP transport
├─ khonliang-bus-lib  — agent SDK for bus interaction  ← THIS REPO
└─ researcher-lib     — evaluation primitives

APPS (agents)
├─ researcher         — ingests the world → corpus → FR ideas
└─ developer          — consumes corpus → specs, milestones, dispatch

INFRASTRUCTURE
├─ khonliang-bus      — agent orchestration platform (depends on bus-lib)
└─ khonliang-scheduler— LLM inference scheduling
```

## Running tests

```bash
pip install -e ".[test]"
pytest
```
