"""Typed experiment-registry records and status validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


EXPERIMENT_STATUSES = (
    "pending",
    "running",
    "completed",
    "failed",
    "partial",
)


@dataclass
class ExperimentRecord:
    study_id: str
    method: str
    variant: str
    seed: int
    model: str
    budget: int
    config_hash: str
    git_commit: str
    budget_protocol: str = "question_matched"
    status: str = "pending"
    experiment_dir: str = ""
    command: list[str] = field(default_factory=list)
    fingerprints: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    return_code: int | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EXPERIMENT_STATUSES:
            raise ValueError(f"Unsupported experiment status: {self.status}")
        if not self.study_id.strip():
            raise ValueError("study_id must not be empty")
        if self.budget <= 0:
            raise ValueError("budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentRecord":
        required = {
            "study_id",
            "method",
            "variant",
            "seed",
            "model",
            "budget",
            "config_hash",
            "git_commit",
            "status",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(
                f"Experiment record is missing fields: {', '.join(missing)}"
            )
        supported = set(cls.__dataclass_fields__)
        return cls(
            **{
                key: value
                for key, value in payload.items()
                if key in supported
            }
        )


def validate_registry(payload: Mapping[str, Any]) -> None:
    if str(payload.get("schema_version")) != "1.0":
        raise ValueError("Unsupported experiment registry schema_version")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise ValueError("Registry experiments must be a list")
    ids = []
    for item in experiments:
        record = ExperimentRecord.from_dict(item)
        ids.append(record.study_id)
    if len(ids) != len(set(ids)):
        raise ValueError("Registry contains duplicate study_id values")
