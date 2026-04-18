"""Shared skill registry contracts for bus agents.

These dataclasses describe what a provider can do, how a skill is called,
and what runtime shape is expected. They are intentionally dependency-free so
agent packages can import the contracts without pulling in the bus server.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, TypeVar


class RegistryValue(str, Enum):
    """String enum with a compact validation helper."""

    @classmethod
    def coerce(cls, value: str | "RegistryValue") -> "RegistryValue":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(item.value for item in cls)
            raise ValueError(f"invalid {cls.__name__}: {value!r}; expected one of {valid}") from exc


class ProviderType(RegistryValue):
    AGENT = "agent"
    LLM = "llm"
    SERVICE = "service"
    HUMAN = "human"
    TOOL = "tool"


class ProviderStatus(RegistryValue):
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    DEPRECATED = "deprecated"


class SkillStatus(RegistryValue):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    MIGRATION_ONLY = "migration_only"
    REMOVED = "removed"
    OFFLINE = "offline"


class SkillAuthority(RegistryValue):
    AUTHORITATIVE = "authoritative"
    PROXY = "proxy"
    FALLBACK = "fallback"
    ADVISORY = "advisory"
    COMPATIBILITY = "compatibility"


class ModelSize(RegistryValue):
    NONE = "none"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ReasoningLevel(RegistryValue):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContextNeed(RegistryValue):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class CostLevel(RegistryValue):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LatencyClass(RegistryValue):
    FAST = "fast"
    NORMAL = "normal"
    SLOW = "slow"


class Locality(RegistryValue):
    LOCAL = "local"
    LAN = "lan"
    REMOTE = "remote"


class ExecutionMode(RegistryValue):
    SINGLE = "single"
    JOB = "job"
    WORKFLOW = "workflow"


class OutputMode(RegistryValue):
    INLINE = "inline"
    ARTIFACT = "artifact"
    STREAM = "stream"
    ARTIFACT_SUMMARY = "artifact+summary"


class RunTier(RegistryValue):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class AggregationMethod(RegistryValue):
    NONE = "none"
    MAJORITY_VOTE = "majority_vote"
    RANK = "rank"
    MERGE = "merge"
    ADJUDICATE = "adjudicate"
    RANK_MERGE = "rank_merge"
    RANK_MERGE_ADJUDICATE = "rank_merge_adjudicate"


EnumT = TypeVar("EnumT", bound=RegistryValue)


def _coerce_enum(enum_type: type[EnumT], value: str | EnumT) -> EnumT:
    return enum_type.coerce(value)  # type: ignore[return-value]


def _coerce_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"expected dict, got {type(value).__name__}")
    return dict(value)


def _clean_dict(value: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in value.items() if v not in (None, "", [], {})}


def _serialize(value: Any) -> Any:
    if isinstance(value, RegistryValue):
        return value.value
    if is_dataclass(value):
        return value.to_dict()  # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _coerce_dataclass(cls: type[Any], value: Any) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls.from_dict(value)
    raise TypeError(f"expected {cls.__name__} or dict, got {type(value).__name__}")


@dataclass
class RuntimeProfile:
    """Expected cost, model, and locality characteristics for a skill."""

    model_size: ModelSize | str = ModelSize.NONE
    reasoning: ReasoningLevel | str = ReasoningLevel.NONE
    context_need: ContextNeed | str = ContextNeed.SMALL
    latency: LatencyClass | str = LatencyClass.NORMAL
    cost: CostLevel | str = CostLevel.LOW
    locality: Locality | str = Locality.LOCAL
    budget_usd: float | None = None
    deadline_ms: int | None = None
    prefer_local: bool = True
    allow_external: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model_size = _coerce_enum(ModelSize, self.model_size)
        self.reasoning = _coerce_enum(ReasoningLevel, self.reasoning)
        self.context_need = _coerce_enum(ContextNeed, self.context_need)
        self.latency = _coerce_enum(LatencyClass, self.latency)
        self.cost = _coerce_enum(CostLevel, self.cost)
        self.locality = _coerce_enum(Locality, self.locality)
        self.metadata = _coerce_mapping(self.metadata)
        if self.budget_usd is not None and self.budget_usd < 0:
            raise ValueError("budget_usd must be non-negative")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "model_size": _serialize(self.model_size),
            "reasoning": _serialize(self.reasoning),
            "context_need": _serialize(self.context_need),
            "latency": _serialize(self.latency),
            "cost": _serialize(self.cost),
            "locality": _serialize(self.locality),
            "budget_usd": self.budget_usd,
            "deadline_ms": self.deadline_ms,
            "prefer_local": self.prefer_local,
            "allow_external": self.allow_external,
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeProfile":
        return cls(**_coerce_mapping(data))


@dataclass
class ExecutionRun:
    """One tier of a skill execution profile."""

    tier: RunTier | str
    count: int = 1
    role: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tier = _coerce_enum(RunTier, self.tier)
        self.metadata = _coerce_mapping(self.metadata)
        if self.count < 1:
            raise ValueError("execution run count must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "tier": _serialize(self.tier),
            "count": self.count,
            "role": self.role,
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionRun":
        return cls(**_coerce_mapping(data))


@dataclass
class ExecutionProfile:
    """Supported execution shape for a skill."""

    profile_id: str
    mode: ExecutionMode | str = ExecutionMode.SINGLE
    runs: list[ExecutionRun | dict[str, Any]] = field(default_factory=list)
    aggregation: AggregationMethod | str = AggregationMethod.NONE
    output_mode: OutputMode | str = OutputMode.INLINE
    supports_async: bool = False
    supports_resume: bool = False
    supports_cancel: bool = False
    idempotent: bool = True
    max_duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        self.mode = _coerce_enum(ExecutionMode, self.mode)
        self.runs = [_coerce_dataclass(ExecutionRun, run) for run in self.runs]
        self.aggregation = _coerce_enum(AggregationMethod, self.aggregation)
        self.output_mode = _coerce_enum(OutputMode, self.output_mode)
        self.metadata = _coerce_mapping(self.metadata)
        if self.max_duration_ms is not None and self.max_duration_ms <= 0:
            raise ValueError("max_duration_ms must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "profile_id": self.profile_id,
            "mode": _serialize(self.mode),
            "runs": _serialize(self.runs),
            "aggregation": _serialize(self.aggregation),
            "output_mode": _serialize(self.output_mode),
            "supports_async": self.supports_async,
            "supports_resume": self.supports_resume,
            "supports_cancel": self.supports_cancel,
            "idempotent": self.idempotent,
            "max_duration_ms": self.max_duration_ms,
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionProfile":
        return cls(**_coerce_mapping(data))


@dataclass
class OutputContract:
    """Describes a skill result payload."""

    content_type: str = "application/json"
    schema: dict[str, Any] = field(default_factory=dict)
    summary_fields: list[str] = field(default_factory=list)
    output_mode: OutputMode | str = OutputMode.INLINE
    artifact_kind: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_type:
            raise ValueError("content_type is required")
        self.schema = _coerce_mapping(self.schema)
        self.summary_fields = list(self.summary_fields)
        self.output_mode = _coerce_enum(OutputMode, self.output_mode)
        self.metadata = _coerce_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "content_type": self.content_type,
            "schema": _serialize(self.schema),
            "summary_fields": self.summary_fields,
            "output_mode": _serialize(self.output_mode),
            "artifact_kind": self.artifact_kind,
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputContract":
        return cls(**_coerce_mapping(data))


@dataclass
class ProviderDescriptor:
    """A provider that can own skills in the registry."""

    provider_id: str
    provider_type: ProviderType | str = ProviderType.AGENT
    status: ProviderStatus | str = ProviderStatus.ACTIVE
    version: str = ""
    locality: Locality | str = Locality.LOCAL
    declared_profile: RuntimeProfile | dict[str, Any] = field(default_factory=RuntimeProfile)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id is required")
        self.provider_type = _coerce_enum(ProviderType, self.provider_type)
        self.status = _coerce_enum(ProviderStatus, self.status)
        self.locality = _coerce_enum(Locality, self.locality)
        self.declared_profile = _coerce_dataclass(RuntimeProfile, self.declared_profile)
        self.metadata = _coerce_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "provider_id": self.provider_id,
            "provider_type": _serialize(self.provider_type),
            "status": _serialize(self.status),
            "version": self.version,
            "locality": _serialize(self.locality),
            "declared_profile": _serialize(self.declared_profile),
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderDescriptor":
        return cls(**_coerce_mapping(data))


@dataclass
class SkillDescriptor:
    """Registry entry for one callable capability."""

    skill_id: str
    provider_id: str
    name: str
    capability: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_contract: OutputContract | dict[str, Any] = field(default_factory=OutputContract)
    authority: SkillAuthority | str = SkillAuthority.AUTHORITATIVE
    status: SkillStatus | str = SkillStatus.ACTIVE
    aliases: list[str] = field(default_factory=list)
    execution_profiles: list[ExecutionProfile | dict[str, Any]] = field(default_factory=list)
    runtime_profile: RuntimeProfile | dict[str, Any] = field(default_factory=RuntimeProfile)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.skill_id:
            raise ValueError("skill_id is required")
        if not self.provider_id:
            raise ValueError("provider_id is required")
        if not self.name:
            raise ValueError("name is required")
        self.input_schema = _coerce_mapping(self.input_schema)
        self.output_contract = _coerce_dataclass(OutputContract, self.output_contract)
        self.authority = _coerce_enum(SkillAuthority, self.authority)
        self.status = _coerce_enum(SkillStatus, self.status)
        self.aliases = list(self.aliases)
        self.execution_profiles = [
            _coerce_dataclass(ExecutionProfile, profile)
            for profile in self.execution_profiles
        ]
        self.runtime_profile = _coerce_dataclass(RuntimeProfile, self.runtime_profile)
        self.metadata = _coerce_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "skill_id": self.skill_id,
            "provider_id": self.provider_id,
            "name": self.name,
            "capability": self.capability,
            "description": self.description,
            "input_schema": _serialize(self.input_schema),
            "output_contract": _serialize(self.output_contract),
            "authority": _serialize(self.authority),
            "status": _serialize(self.status),
            "aliases": self.aliases,
            "execution_profiles": _serialize(self.execution_profiles),
            "runtime_profile": _serialize(self.runtime_profile),
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDescriptor":
        return cls(**_coerce_mapping(data))


@dataclass
class CapabilityRoute:
    """Preferred owner and fallbacks for a capability."""

    capability: str
    owner: str
    fallbacks: list[str] = field(default_factory=list)
    deprecated_aliases: dict[str, str] = field(default_factory=dict)
    status: SkillStatus | str = SkillStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability:
            raise ValueError("capability is required")
        if not self.owner:
            raise ValueError("owner is required")
        self.fallbacks = list(self.fallbacks)
        self.deprecated_aliases = _coerce_mapping(self.deprecated_aliases)
        self.status = _coerce_enum(SkillStatus, self.status)
        self.metadata = _coerce_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return _clean_dict({
            "capability": self.capability,
            "owner": self.owner,
            "fallbacks": self.fallbacks,
            "deprecated_aliases": self.deprecated_aliases,
            "status": _serialize(self.status),
            "metadata": _serialize(self.metadata),
        })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapabilityRoute":
        return cls(**_coerce_mapping(data))
