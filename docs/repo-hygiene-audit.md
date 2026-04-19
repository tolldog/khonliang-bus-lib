# Repo Hygiene Audit

Generated: 1776579546.2963605
Repo: `khonliang-bus-lib`

## Summary

- 1 docs drift findings, 2 stale/deprecated findings, 3 proposed actions, 0 applied changes
- Python files: 12
- Test files: 6
- Docs files: 3

## Cleanup Plan

- **docs-refresh** [low] Refresh README/CLAUDE/config documentation (`README.md`)
  - Docs drift findings indicate setup or architecture guidance is stale or incomplete.
- **review-stale-references** [low] Review stale wording in docs and source comments (`README.md`, `khonliang_bus/agent.py`)
  - Stale terms may be historical, but current guidance should not point at retired milestones or runtimes.
- **write-hygiene-artifact** [low] Write compact repo hygiene artifact (`docs/repo-hygiene-audit.md`)
  - Persist the audit so future sessions can resume without rereading raw files.

## Docs Drift

- [medium] `README.md`: README does not mention 'config'. Action: document local config/example boundaries

## Deprecated Or Stale Paths

- [low] `README.md`: Found stale marker 'from_mcp'. Action: review whether this is historical context or current guidance
- [low] `khonliang_bus/agent.py`: Found stale marker 'from_mcp'. Action: review whether this is historical context or current guidance

## Test Plan

- `.venv/bin/python -m pytest -q`
- `.venv/bin/python -m compileall .`
