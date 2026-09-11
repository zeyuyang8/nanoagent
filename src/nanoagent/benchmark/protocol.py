"""The stable contract between benchmark datasets, runners, and scorers.

An adapter owns benchmark-specific data and scoring. The engine sees a public
``PreparedCase`` and passes only that to the model; the private ``oracle`` remains
on ``BenchmarkCase`` and is available only to the scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from nanoagent.runtime.runner import RunnerResult

BENCHMARK_PROTOCOL = "nanoagent.benchmark.v1"
BENCHMARK_RESULT_PROTOCOL = "nanoagent.benchmark.result.v1"
BENCHMARK_CASE_PROTOCOL = "nanoagent.benchmark.case.v1"


@dataclass(frozen=True)
class BenchmarkDescriptor:
    name: str
    label: str
    version: str
    dataset_fingerprint: str
    primary_metric: str
    metrics: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "version": self.version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "primary_metric": self.primary_metric,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    input: str
    instructions: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    oracle: Any = None


@dataclass(frozen=True)
class PreparedCase:
    input: str
    instructions: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkScore:
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("benchmark score must contain at least one metric")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("benchmark metric names must be non-empty strings")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"benchmark metric {name!r} must be numeric")


class BenchmarkAdapter(Protocol):
    """A benchmark implementation. Implementations may be built in or loaded by name."""

    descriptor: BenchmarkDescriptor

    def cases(self) -> list[BenchmarkCase]: ...

    async def prepare(self, case: BenchmarkCase) -> PreparedCase: ...

    async def score(self, case: BenchmarkCase, result: RunnerResult) -> BenchmarkScore: ...

    def aggregate(self, scores: list[BenchmarkScore]) -> dict[str, float]: ...


class BaseBenchmarkAdapter:
    """Convenient defaults for adapters that only need to load and score cases."""

    descriptor: BenchmarkDescriptor

    async def prepare(self, case: BenchmarkCase) -> PreparedCase:
        return PreparedCase(
            input=case.input,
            instructions=case.instructions,
            metadata=dict(case.metadata),
        )

    def aggregate(self, scores: list[BenchmarkScore]) -> dict[str, float]:
        if not scores:
            return {}
        names = sorted({name for score in scores for name in score.metrics})
        return {
            name: sum(float(score.metrics[name]) for score in scores if name in score.metrics)
            / sum(1 for score in scores if name in score.metrics)
            for name in names
        }
