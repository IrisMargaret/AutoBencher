"""Explicit total-question budgets for comparable study runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSpec:
    total_questions: int
    num_iterations: int
    max_cycles: int

    def __post_init__(self) -> None:
        if self.total_questions <= 0:
            raise ValueError("total_questions must be positive")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.total_questions % self.stage_count:
            raise ValueError(
                "The total question budget must be divisible by "
                "num_iterations * max_cycles so every method receives exactly "
                "the declared budget."
            )

    @property
    def stage_count(self) -> int:
        return self.num_iterations * self.max_cycles

    @property
    def questions_per_iteration(self) -> int:
        return self.total_questions // self.stage_count

    def overrides(self) -> list[str]:
        return [
            f"experiment.num_iterations={self.num_iterations}",
            f"experiment.max_cycles={self.max_cycles}",
            (
                "experiment.questions_per_iteration="
                f"{self.questions_per_iteration}"
            ),
        ]


def parse_budget(
    value: int | dict[str, int],
    default_num_iterations: int,
    default_max_cycles: int,
) -> BudgetSpec:
    if isinstance(value, dict):
        total = int(value["total_questions"])
        iterations = int(
            value.get("num_iterations", default_num_iterations)
        )
        cycles = int(value.get("max_cycles", default_max_cycles))
        return BudgetSpec(total, iterations, cycles)
    return BudgetSpec(
        int(value),
        int(default_num_iterations),
        int(default_max_cycles),
    )
