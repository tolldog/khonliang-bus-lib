"""Tests for shared skill registry contracts."""

from __future__ import annotations

import pytest

from khonliang_bus import (
    AggregationMethod,
    CapabilityRoute,
    ExecutionProfile,
    ExecutionRun,
    OutputContract,
    OutputMode,
    ProviderDescriptor,
    ProviderType,
    RunTier,
    RuntimeProfile,
    Skill,
    SkillAuthority,
    SkillDescriptor,
    SkillStatus,
)


def test_provider_descriptor_round_trips():
    provider = ProviderDescriptor(
        provider_id="developer-primary",
        provider_type=ProviderType.AGENT,
        version="0.4.0",
        declared_profile=RuntimeProfile(
            model_size="medium",
            reasoning="high",
            context_need="medium",
            latency="normal",
            cost="medium",
            locality="local",
        ),
    )

    restored = ProviderDescriptor.from_dict(provider.to_dict())

    assert restored.provider_id == "developer-primary"
    assert restored.provider_type == ProviderType.AGENT
    assert restored.declared_profile.reasoning.value == "high"


def test_execution_profile_supports_multi_run_shapes():
    profile = ExecutionProfile(
        profile_id="three-medium-one-large",
        mode="workflow",
        runs=[
            ExecutionRun(RunTier.MEDIUM, count=3, role="candidate"),
            {"tier": "large", "count": 1, "role": "adjudicator"},
        ],
        aggregation=AggregationMethod.RANK_MERGE_ADJUDICATE,
        output_mode=OutputMode.ARTIFACT_SUMMARY,
        supports_async=True,
        supports_resume=True,
    )

    payload = profile.to_dict()

    assert payload["runs"][0]["count"] == 3
    assert payload["runs"][1]["tier"] == "large"
    assert payload["aggregation"] == "rank_merge_adjudicate"
    assert payload["output_mode"] == "artifact+summary"
    assert ExecutionProfile.from_dict(payload).runs[1].role == "adjudicator"


def test_skill_descriptor_round_trips_contracts():
    descriptor = SkillDescriptor(
        skill_id="developer-primary.next_work_unit",
        provider_id="developer-primary",
        name="next_work_unit",
        capability="fr.bundle.next",
        description="Return the next ready FR bundle.",
        input_schema={"target": {"type": "string"}},
        output_contract=OutputContract(
            schema={"type": "object", "required": ["frs"]},
            summary_fields=["frs", "suggested_next_actions"],
            output_mode="artifact+summary",
            artifact_kind="work_unit",
        ),
        authority=SkillAuthority.AUTHORITATIVE,
        status=SkillStatus.ACTIVE,
        aliases=["next_bundle"],
        runtime_profile={"model_size": "small", "latency": "fast", "cost": "low"},
    )

    restored = SkillDescriptor.from_dict(descriptor.to_dict())

    assert restored.capability == "fr.bundle.next"
    assert restored.output_contract.artifact_kind == "work_unit"
    assert restored.runtime_profile.model_size.value == "small"
    assert restored.aliases == ["next_bundle"]


def test_capability_route_supports_owner_fallbacks_and_aliases():
    route = CapabilityRoute(
        capability="git.pr.review",
        owner="developer-primary.pr_review",
        fallbacks=["github.pr_view"],
        deprecated_aliases={"review_pr": "git.pr.review"},
    )

    restored = CapabilityRoute.from_dict(route.to_dict())

    assert restored.owner == "developer-primary.pr_review"
    assert restored.fallbacks == ["github.pr_view"]
    assert restored.deprecated_aliases["review_pr"] == "git.pr.review"


def test_skill_keeps_legacy_payload_shape():
    skill = Skill("echo", "Echo input", {"text": {"type": "string"}}, since="0.1.0")

    assert skill.parameters == {"text": {"type": "string"}}
    assert skill.input_schema == skill.parameters
    assert skill.to_dict() == {
        "name": "echo",
        "description": "Echo input",
        "parameters": {"text": {"type": "string"}},
        "since": "0.1.0",
        "authority": "authoritative",
        "status": "active",
    }


def test_skill_serializes_registry_metadata_and_descriptor():
    skill = Skill(
        name="paper_context",
        description="Return distilled paper context.",
        parameters={"query": {"type": "string"}},
        capability="research.paper.context",
        output_contract=OutputContract(
            summary_fields=["summary", "refs"],
            output_mode="artifact+summary",
        ),
        aliases=["get_paper_context"],
        execution_profiles=[
            {
                "profile_id": "single-small",
                "runs": [{"tier": "small", "count": 1}],
                "output_mode": "artifact+summary",
            }
        ],
        runtime_profile={"model_size": "small", "context_need": "medium"},
        metadata={"owner": "researcher"},
    )

    payload = skill.to_dict()
    descriptor = skill.descriptor("researcher-primary")

    assert payload["capability"] == "research.paper.context"
    assert payload["output_contract"]["output_mode"] == "artifact+summary"
    assert payload["execution_profiles"][0]["runs"][0]["tier"] == "small"
    assert descriptor.skill_id == "researcher-primary.paper_context"
    assert descriptor.capability == "research.paper.context"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: RuntimeProfile(model_size="huge"), "invalid ModelSize"),
        (lambda: ExecutionRun("small", count=0), "count must be at least 1"),
        (lambda: ProviderDescriptor(provider_id="", provider_type="agent"), "provider_id"),
        (lambda: SkillDescriptor(skill_id="x", provider_id="", name="n"), "provider_id"),
        (lambda: CapabilityRoute(capability="", owner="x"), "capability"),
    ],
)
def test_registry_validation_errors(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
