---
applyTo: "**"
---

# Review Instructions

Review this repository as a lightweight shared SDK for khonliang agents.

Prioritize findings in this order:

1. Backward compatibility for public imports, dataclass constructors, serialized
   payload keys, and wire protocols used by existing agents.
2. Behavioral bugs, validation gaps, and schema drift between agent-facing
   contracts and bus-facing registration payloads.
3. Dependency discipline. This package should stay lightweight; do not add
   dependencies for behavior that can be handled with the standard library.
4. Test coverage for compatibility, validation failures, serialization
   round-trips, and async agent behavior.
5. Documentation accuracy for public SDK usage.

Do not leave actionable correctness issues as vague future work. If a change is
needed for correctness or compatibility, call it out directly with the affected
file and line.

When reviewing registry or contract changes, check that:

- Legacy `Skill(name, description, parameters, since)` usage still works.
- New metadata round-trips through plain dictionaries.
- Optional fields do not break older bus servers that only read
  `name`, `description`, `parameters`, and `since`.
- Runtime and execution profile validation rejects invalid values.
- Tests cover both the old minimal path and the new richer metadata path.

When reviewing agent lifecycle code, check that:

- Agents remain pure WebSocket clients unless the change explicitly targets a
  one-shot install or migration helper path.
- Connection, reconnect, heartbeat, and request errors are surfaced clearly.
- Long-running or artifact-producing work can be summarized instead of forcing
  large outputs into the model context.
