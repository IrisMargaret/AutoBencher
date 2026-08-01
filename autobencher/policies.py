"""Generation policies for baseline, adaptive, and ablation studies.

The policy layer owns question-budget allocation only.  Coverage statistics and
the adaptive posterior calculations remain in :mod:`autobencher.coverage` so
that their existing public API stays available.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol

from .coverage import (
    adaptive_priority,
    beta_binomial_state,
    coverage_metrics,
    history_weight,
    largest_remainder,
    previous_round_accuracy_state,
    taxonomy_items,
)
from .difficulty import target_difficulty_profile


POLICY_VERSION = "2.0"

DEFAULT_COMPONENT_STATE = {
    "adaptive_allocation": True,
    "global_difficulty": True,
    "difficulty_module": True,
    "observed_difficulty": True,
    "coverage_priority": True,
    "uncertainty_priority": True,
    "persistent_error_priority": True,
    "retention_priority": True,
    "hard_pool_variants": True,
    "error_type_targeting": True,
}


@dataclass(frozen=True)
class PolicyContext:
    """Immutable inputs used to create one deterministic generation plan."""

    config: Mapping[str, Any]
    history_records: tuple[Mapping[str, Any], ...]
    previous_round_records: tuple[Mapping[str, Any], ...]
    hard_pool_records: tuple[Mapping[str, Any], ...]
    global_iteration: int
    cycle: int
    seed: int
    question_budget: int
    # Kept separately because the legacy compatibility entry point allowed a
    # caller to provide a pool size without materializing the records.
    hard_pool_size: int | None = None


@dataclass
class GenerationPlan:
    """Serializable plan shared by every sampling policy."""

    policy_name: str
    policy_version: str
    variant: str
    question_budget: int
    allocations: list[dict[str, Any]]
    source_budget: dict[str, int]
    adaptive_sampler_state: list[dict[str, Any]]
    previous_round_accuracy_state: dict[str, Any]
    coverage_before_generation: dict[str, Any]
    hard_pool_injection_enabled: bool
    component_state: dict[str, bool]
    history_mode: str
    decay_lambda: float
    component_evidence: dict[str, Any]
    diagnostics: dict[str, Any]
    global_iteration: int
    cycle: int
    seed: int
    quota_feasible: bool = True
    cumulative_quota_status: str = "complete"
    cumulative_quota_completion_possible_this_iteration: bool = True
    minimum_questions_for_full_quota: int = 0
    remaining_questions_for_full_quota_before_iteration: int = 0
    unsatisfied_subcategories: list[str] = field(default_factory=list)
    hard_pool_reference_count: int = 0
    directed_generation_question_count: int = 0
    coverage_repair_question_count: int = 0
    retention_question_count: int = 0
    fallbacks: list[dict[str, Any] | str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a mutable, recursively copied plain dictionary."""

        return copy.deepcopy(asdict(self))


class SamplingPolicy(Protocol):
    """Protocol implemented by every generation policy."""

    policy_name: str
    policy_version: str
    variant: str
    component_state: dict[str, bool]

    def descriptor(self) -> dict[str, Any]:
        """Describe the effective policy and components."""

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        """Create one generation plan."""


def _record_subcategory(record: Mapping[str, Any]) -> str:
    return str(
        record.get(
            "sub_category",
            record.get("subcategory", record.get("target_subcategory", "")),
        )
    )


def _record_is_correct(record: Mapping[str, Any]) -> bool:
    value = record.get("is_correct", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _disabled_accuracy_state(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "observation_count": 0,
        "accuracy": None,
        "band": "disabled",
        "difficulty_delta": 0,
        "reason": reason,
    }


def _study(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("study", {})
    return value if isinstance(value, Mapping) else {}


def _configured_components(config: Mapping[str, Any]) -> dict[str, bool]:
    components = dict(DEFAULT_COMPONENT_STATE)
    configured = _study(config).get("components", {})
    if isinstance(configured, Mapping):
        for name in components:
            if name in configured:
                components[name] = bool(configured[name])
    return components


def _actual_components(
    config: Mapping[str, Any],
    policy_name: str,
    variant: str,
) -> dict[str, bool]:
    if policy_name == "base":
        return {name: False for name in DEFAULT_COMPONENT_STATE}
    if policy_name in {"random", "uniform"}:
        components = {name: False for name in DEFAULT_COMPONENT_STATE}
        # These policies do not adapt from the score, but generated records
        # still retain the objective observed-difficulty diagnostics.
        components["observed_difficulty"] = bool(
            _configured_components(config)["observed_difficulty"]
        )
        components["difficulty_module"] = True
        return components
    if policy_name == "error_only":
        components = {name: False for name in DEFAULT_COMPONENT_STATE}
        components["adaptive_allocation"] = True
        components["observed_difficulty"] = bool(
            _configured_components(config)["observed_difficulty"]
        )
        components["difficulty_module"] = True
        components["persistent_error_priority"] = True
        return components

    components = _configured_components(config)
    if variant == "full_no_hard_pool":
        components["hard_pool_variants"] = False
    if variant == "full_no_error_targeting":
        components["error_type_targeting"] = False
    if variant == "full_no_observed_difficulty_sampling":
        components["observed_difficulty"] = False
    if variant == "full_no_difficulty_module":
        components["difficulty_module"] = False
        components["observed_difficulty"] = False
        components["global_difficulty"] = False
    return components


def _policy_seed(context: PolicyContext, policy_name: str) -> int:
    material = (
        f"{int(context.seed)}:{int(context.cycle)}:"
        f"{int(context.global_iteration)}:{policy_name}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _chunked_allocations(
    counts: Mapping[tuple[str, str, int], int],
    config: Mapping[str, Any],
    *,
    generation_source: str,
    generation_strategy: str,
    priority_scores: Mapping[tuple[str, str], float] | None = None,
) -> list[dict[str, Any]]:
    maximum = int(config["generation"]["max_questions_per_prompt"])
    allocations: list[dict[str, Any]] = []
    for (category, subcategory, difficulty), count in counts.items():
        remaining = int(count)
        while remaining > 0:
            chunk = min(remaining, maximum)
            allocations.append(
                {
                    "category": category,
                    "sub_category": subcategory,
                    "question_count": chunk,
                    "difficulty": int(difficulty),
                    "target_difficulty_profile": target_difficulty_profile(
                        int(difficulty),
                        config,
                    ),
                    "generation_source": generation_source,
                    "generation_strategy": generation_strategy,
                    "priority_score": float(
                        (priority_scores or {}).get(
                            (category, subcategory),
                            1.0,
                        )
                    ),
                }
            )
            remaining -= chunk
    return allocations


def _quota_fields(
    context: PolicyContext,
    allocations: list[dict[str, Any]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    items = taxonomy_items(context.config)
    allocated: Counter[str] = Counter()
    for allocation in allocations:
        allocated[str(allocation["sub_category"])] += int(
            allocation["question_count"]
        )
    required = sum(int(metadata["min_quota"]) for _, _, metadata in items)
    remaining = sum(
        max(
            0,
            int(metadata["min_quota"])
            - int(metrics["subcategory_counts"][subcategory]),
        )
        for _, subcategory, metadata in items
    )
    unsatisfied = [
        subcategory
        for _, subcategory, metadata in items
        if (
            int(metrics["subcategory_counts"][subcategory])
            + allocated[subcategory]
            < int(metadata["min_quota"])
        )
    ]
    if remaining == 0:
        status = "complete"
    elif int(context.question_budget) >= remaining:
        status = "scheduled_this_iteration"
    else:
        status = "multi_iteration_progress"
    return {
        "quota_feasible": True,
        "cumulative_quota_status": status,
        "cumulative_quota_completion_possible_this_iteration": (
            int(context.question_budget) >= remaining
        ),
        "minimum_questions_for_full_quota": required,
        "remaining_questions_for_full_quota_before_iteration": remaining,
        "unsatisfied_subcategories": unsatisfied,
    }


def _make_plan(
    policy: "_PolicyBase",
    context: PolicyContext,
    *,
    allocations: list[dict[str, Any]],
    source_budget: dict[str, int],
    adaptive_state: list[dict[str, Any]] | None = None,
    accuracy_state: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    hard_pool_enabled: bool = False,
    hard_pool_reference_count: int = 0,
    diagnostics: dict[str, Any] | None = None,
    fallbacks: list[dict[str, Any] | str] | None = None,
) -> GenerationPlan:
    metrics = metrics or coverage_metrics(
        context.history_records,
        context.config,
    )
    expected = int(context.question_budget)
    realized = sum(int(item["question_count"]) for item in allocations)
    if realized != expected:
        raise RuntimeError(
            "generation allocation did not conserve the question budget"
        )
    quota = _quota_fields(context, allocations, metrics)
    return GenerationPlan(
        **policy.descriptor(),
        question_budget=expected,
        allocations=allocations,
        source_budget=dict(source_budget),
        adaptive_sampler_state=list(adaptive_state or []),
        previous_round_accuracy_state=(
            accuracy_state
            or _disabled_accuracy_state(
                f"{policy.policy_name}_does_not_use_global_accuracy"
            )
        ),
        coverage_before_generation=dict(metrics),
        hard_pool_injection_enabled=bool(hard_pool_enabled),
        diagnostics=dict(diagnostics or {}),
        global_iteration=int(context.global_iteration),
        cycle=int(context.cycle),
        seed=int(context.seed),
        hard_pool_reference_count=int(hard_pool_reference_count),
        directed_generation_question_count=int(
            source_budget.get("hard_pool_variant", 0)
        ),
        coverage_repair_question_count=int(
            source_budget.get("coverage_deficit", 0)
        ),
        retention_question_count=int(
            source_budget.get("retention_known", 0)
        ),
        fallbacks=list(fallbacks or []),
        **quota,
    )


class _PolicyBase:
    policy_name = ""
    policy_version = POLICY_VERSION

    def __init__(
        self,
        config: Mapping[str, Any],
        variant: str | None = None,
    ) -> None:
        self.config = config
        study = _study(config)
        configured_variant = (
            study.get("variant")
            if study.get("policy") == self.policy_name
            else None
        )
        self.variant = str(
            variant or configured_variant or self.policy_name
        )
        self.component_state = _actual_components(
            config,
            self.policy_name,
            self.variant,
        )

    def descriptor(self) -> dict[str, Any]:
        adaptive = self.config["adaptive_sampling"]
        difficulty = self.config["difficulty"]
        evidence = {
            "posterior_history_scope": str(adaptive["history_mode"]),
            "coverage_history_scope": (
                "cumulative_generation_coverage"
                if self.component_state["coverage_priority"]
                else "disabled"
            ),
            "adaptive_weights": {
                "coverage_weight": (
                    float(adaptive["coverage_weight"])
                    if self.component_state["coverage_priority"]
                    else 0.0
                ),
                "uncertainty_weight": (
                    float(adaptive["uncertainty_weight"])
                    if self.component_state["uncertainty_priority"]
                    else 0.0
                ),
                "persistent_error_weight": (
                    float(adaptive["persistent_error_weight"])
                    if self.component_state["persistent_error_priority"]
                    else 0.0
                ),
                "retention_weight": (
                    float(adaptive["retention_weight"])
                    if self.component_state["retention_priority"]
                    else 0.0
                ),
            },
            "global_accuracy_adjustment_enabled": bool(
                self.component_state["global_difficulty"]
            ),
            "observed_difficulty_sampling_enabled": bool(
                self.component_state["observed_difficulty"]
                and self.component_state["difficulty_module"]
            ),
            "difficulty_rejection_enabled": bool(
                self.component_state["difficulty_module"]
                and difficulty["reject_outside_generation_bounds"]
            ),
            "difficulty_mismatch_action": (
                str(difficulty["mismatch_action"])
                if self.component_state["difficulty_module"]
                else "disabled"
            ),
            "hard_pool_variants_enabled": bool(
                self.component_state["hard_pool_variants"]
            ),
            "error_type_targeting_enabled": bool(
                self.component_state["error_type_targeting"]
            ),
        }
        return {
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "variant": self.variant,
            "component_state": copy.deepcopy(self.component_state),
            "history_mode": str(adaptive["history_mode"]),
            "decay_lambda": float(adaptive["decay_lambda"]),
            "component_evidence": evidence,
        }

    # A short alias is useful to external experiment drivers while
    # ``build_plan`` remains the protocol's explicit operation.
    def plan(self, context: PolicyContext) -> GenerationPlan:
        return self.build_plan(context)


class BasePolicy(_PolicyBase):
    """Evaluation-only policy; the runner should bypass generation entirely."""

    policy_name = "base"

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        base_context = PolicyContext(
            config=context.config,
            history_records=context.history_records,
            previous_round_records=context.previous_round_records,
            hard_pool_records=context.hard_pool_records,
            global_iteration=context.global_iteration,
            cycle=context.cycle,
            seed=context.seed,
            question_budget=0,
            hard_pool_size=context.hard_pool_size,
        )
        return _make_plan(
            self,
            base_context,
            allocations=[],
            source_budget={
                "hard_pool_variant": 0,
                "coverage_deficit": 0,
                "retention_known": 0,
                "base": 0,
            },
            diagnostics={"eval_only": True, "training_disabled": True},
        )


class RandomPolicy(_PolicyBase):
    """Seeded random taxonomy and difficulty sampling."""

    policy_name = "random"

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        items = taxonomy_items(context.config)
        rng = random.Random(_policy_seed(context, self.policy_name))
        lower = int(context.config["generation"]["minimum_difficulty"])
        upper = int(context.config["generation"]["maximum_difficulty"])
        # Draw subcategories independently per question, then batch questions
        # from the same subcategory into one prompt. Sampling difficulty once
        # per batch preserves a seeded random baseline while avoiding dozens
        # of one-question API calls caused by subcategory x difficulty cells.
        subcategory_counts: Counter[tuple[str, str]] = Counter()
        for _ in range(int(context.question_budget)):
            category, subcategory, _ = rng.choice(items)
            subcategory_counts[(category, subcategory)] += 1
        counts = {
            (category, subcategory, rng.randint(lower, upper)): count
            for (category, subcategory), count in subcategory_counts.items()
        }
        allocations = _chunked_allocations(
            counts,
            context.config,
            generation_source="random",
            generation_strategy="seeded_random",
        )
        return _make_plan(
            self,
            context,
            allocations=allocations,
            source_budget={
                "hard_pool_variant": 0,
                "coverage_deficit": 0,
                "retention_known": 0,
                "random": int(context.question_budget),
            },
            diagnostics={
                "rng": "python_random_mt19937",
                "derived_seed": _policy_seed(context, self.policy_name),
                "batching_unit": "subcategory",
                "difficulty_sampling_unit": "subcategory_batch",
            },
        )


class UniformPolicy(_PolicyBase):
    """Strictly uniform allocation over all configured subcategories."""

    policy_name = "uniform"

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        items = taxonomy_items(context.config)
        keys = {
            f"{category}|||{subcategory}": 1.0
            for category, subcategory, _ in items
        }
        distributed = largest_remainder(int(context.question_budget), keys)
        study = _study(context.config)
        difficulty = int(
            study.get(
                "uniform_difficulty",
                context.config["adaptive_sampling"]["initial_difficulty"],
            )
        )
        counts = {
            (category, subcategory, difficulty): distributed[
                f"{category}|||{subcategory}"
            ]
            for category, subcategory, _ in items
            if distributed[f"{category}|||{subcategory}"] > 0
        }
        allocations = _chunked_allocations(
            counts,
            context.config,
            generation_source="uniform",
            generation_strategy="uniform_taxonomy",
        )
        return _make_plan(
            self,
            context,
            allocations=allocations,
            source_budget={
                "hard_pool_variant": 0,
                "coverage_deficit": 0,
                "retention_known": 0,
                "uniform": int(context.question_budget),
            },
            diagnostics={"uniform_difficulty": difficulty},
        )


class ErrorOnlyPolicy(_PolicyBase):
    """Allocate by smoothed historical subcategory error rate."""

    policy_name = "error_only"

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        items = taxonomy_items(context.config)
        evaluated = [
            record
            for record in context.history_records
            if "is_correct" in record
        ]
        if not evaluated:
            uniform = UniformPolicy(context.config, self.variant)
            plan = uniform.build_plan(context)
            initial_difficulty = int(
                context.config["adaptive_sampling"][
                    "initial_difficulty"
                ]
            )
            plan.policy_name = self.policy_name
            plan.component_state = copy.deepcopy(self.component_state)
            plan.source_budget = {
                "hard_pool_variant": 0,
                "coverage_deficit": 0,
                "retention_known": 0,
                "error_only": int(context.question_budget),
            }
            plan.diagnostics.pop("uniform_difficulty", None)
            plan.diagnostics.update(
                {
                    "fallback": "uniform_no_history",
                    "fallback_difficulty": initial_difficulty,
                    "error_only_epsilon": float(
                        _study(context.config).get(
                            "error_only_epsilon",
                            0.05,
                        )
                    ),
                }
            )
            plan.fallbacks.append("error_only_no_history_uniform")
            plan.previous_round_accuracy_state = _disabled_accuracy_state(
                "error_only_does_not_use_global_accuracy"
            )
            for allocation in plan.allocations:
                allocation["difficulty"] = initial_difficulty
                allocation["target_difficulty_profile"] = (
                    target_difficulty_profile(
                        initial_difficulty,
                        context.config,
                    )
                )
                allocation["generation_source"] = "error_only"
                allocation["generation_strategy"] = (
                    "uniform_no_error_history"
                )
            return plan

        adaptive = context.config["adaptive_sampling"]
        alpha = float(adaptive["beta_prior_alpha"])
        beta = float(adaptive["beta_prior_beta"])
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in evaluated:
            grouped.setdefault(_record_subcategory(record), []).append(record)
        epsilon = float(
            _study(context.config).get("error_only_epsilon", 0.05)
        )
        weights: dict[str, float] = {}
        sampler_state: list[dict[str, Any]] = []
        for category, subcategory, _ in items:
            group = grouped.get(subcategory, [])
            weighted = [
                (
                    record,
                    history_weight(
                        record,
                        context.config,
                        current_cycle=context.cycle,
                    ),
                )
                for record in group
            ]
            weighted = [
                (record, weight)
                for record, weight in weighted
                if weight > 0
            ]
            correct = sum(
                weight
                for record, weight in weighted
                if _record_is_correct(record)
            )
            effective_count = sum(weight for _, weight in weighted)
            posterior_mean = (alpha + correct) / (
                alpha + beta + effective_count
            )
            weight = epsilon + (1.0 - posterior_mean)
            key = f"{category}|||{subcategory}"
            weights[key] = weight
            sampler_state.append(
                {
                    "category": category,
                    "subcategory": subcategory,
                    "correct_count": correct,
                    "incorrect_count": effective_count - correct,
                    "observation_count": effective_count,
                    "raw_observation_count": len(group),
                    "active_observation_count": len(weighted),
                    "history_mode": str(adaptive["history_mode"]),
                    "decay_lambda": float(adaptive["decay_lambda"]),
                    "posterior_accuracy": posterior_mean,
                    "priority_score": weight,
                    "sampling_reason": ["historical_error_rate_only"],
                }
            )
        distributed = largest_remainder(
            int(context.question_budget),
            weights,
        )
        initial = int(
            context.config["adaptive_sampling"]["initial_difficulty"]
        )
        lower = int(context.config["generation"]["minimum_difficulty"])
        upper = int(context.config["generation"]["maximum_difficulty"])
        latest: dict[str, int] = {}
        for record in context.history_records:
            if history_weight(
                record,
                context.config,
                current_cycle=context.cycle,
            ) <= 0:
                continue
            subcategory = _record_subcategory(record)
            try:
                difficulty = int(record.get("difficulty"))
            except (TypeError, ValueError):
                continue
            if subcategory and lower <= difficulty <= upper:
                latest[subcategory] = difficulty
        counts: dict[tuple[str, str, int], int] = {}
        scores: dict[tuple[str, str], float] = {}
        for category, subcategory, _ in items:
            count = distributed[f"{category}|||{subcategory}"]
            if count <= 0:
                continue
            counts[
                (category, subcategory, latest.get(subcategory, initial))
            ] = count
            scores[(category, subcategory)] = weights[
                f"{category}|||{subcategory}"
            ]
        allocations = _chunked_allocations(
            counts,
            context.config,
            generation_source="error_only",
            generation_strategy="historical_error_rate",
            priority_scores=scores,
        )
        return _make_plan(
            self,
            context,
            allocations=allocations,
            source_budget={
                "hard_pool_variant": 0,
                "coverage_deficit": 0,
                "retention_known": 0,
                "error_only": int(context.question_budget),
            },
            adaptive_state=sampler_state,
            diagnostics={"error_only_epsilon": epsilon},
        )


def _requested_difficulty(record: Mapping[str, Any]) -> int:
    value = record.get("target_difficulty")
    profile = record.get("difficulty_profile")
    if value is None and isinstance(profile, Mapping):
        value = profile.get("requested_score")
    if value is None:
        value = record.get("difficulty", 5)
    return int(value)


def _project_requested_difficulty(
    records: tuple[Mapping[str, Any], ...],
) -> list[Mapping[str, Any]]:
    projected: list[Mapping[str, Any]] = []
    for record in records:
        item = dict(record)
        try:
            item["difficulty"] = _requested_difficulty(record)
        except (TypeError, ValueError):
            pass
        projected.append(item)
    return projected


class FullAdaptivePolicy(_PolicyBase):
    """The existing full adaptive algorithm and its component ablations."""

    policy_name = "full"

    def _effective_config(self) -> dict[str, Any]:
        effective = dict(self.config)
        adaptive = dict(self.config["adaptive_sampling"])
        if not self.component_state["global_difficulty"]:
            adaptive["global_accuracy_enabled"] = False
        if not self.component_state["coverage_priority"]:
            adaptive["coverage_weight"] = 0.0
        if not self.component_state["uncertainty_priority"]:
            adaptive["uncertainty_weight"] = 0.0
        if not self.component_state["persistent_error_priority"]:
            adaptive["persistent_error_weight"] = 0.0
        if not self.component_state["retention_priority"]:
            adaptive["retention_weight"] = 0.0
        effective["adaptive_sampling"] = adaptive
        return effective

    def build_plan(self, context: PolicyContext) -> GenerationPlan:
        records: list[Mapping[str, Any]]
        if not self.component_state["difficulty_module"]:
            records = []
            initial = int(
                context.config["adaptive_sampling"]["initial_difficulty"]
            )
            for source in context.history_records:
                item = dict(source)
                item["difficulty"] = initial
                records.append(item)
        elif self.component_state["observed_difficulty"]:
            records = list(context.history_records)
        else:
            records = _project_requested_difficulty(
                context.history_records
            )
        previous = list(context.previous_round_records)
        hard_records = list(context.hard_pool_records)
        items = taxonomy_items(context.config)
        metrics = coverage_metrics(records, context.config)
        effective_config = self._effective_config()
        state = beta_binomial_state(
            records,
            effective_config,
            current_cycle=context.cycle,
        )
        global_state = previous_round_accuracy_state(
            previous,
            effective_config,
        )
        latest_difficulty: dict[str, int] = {}
        for record in records:
            if history_weight(
                record,
                effective_config,
                current_cycle=context.cycle,
            ) <= 0:
                continue
            subcategory = _record_subcategory(record)
            if subcategory:
                latest_difficulty[subcategory] = int(
                    record.get(
                        "difficulty",
                        context.config["adaptive_sampling"][
                            "initial_difficulty"
                        ],
                    )
                )

        priorities: list[dict[str, Any]] = []
        for category, subcategory, metadata in items:
            difficulty = latest_difficulty.get(
                subcategory,
                int(
                    context.config["adaptive_sampling"][
                        "initial_difficulty"
                    ]
                ),
            )
            item = adaptive_priority(
                subcategory,
                difficulty,
                int(metrics["subcategory_counts"][subcategory]),
                int(metadata["min_quota"]),
                state,
                effective_config,
            )
            if not self.component_state["difficulty_module"]:
                item["selected_difficulty"] = difficulty
                item["sampling_reason"].append(
                    "difficulty_module_disabled_keep_initial_difficulty"
                )
            if not self.component_state["adaptive_allocation"]:
                item["priority_score"] = 1.0
                item["sampling_reason"].append(
                    "adaptive_allocation_disabled"
                )
            local_selected = int(item["selected_difficulty"])
            combined_delta = (
                local_selected
                - difficulty
                + int(global_state["difficulty_delta"])
            )
            maximum_change = int(
                context.config["adaptive_sampling"][
                    "max_difficulty_change_per_iteration"
                ]
            )
            combined_delta = max(
                -maximum_change,
                min(maximum_change, combined_delta),
            )
            item["local_selected_difficulty"] = local_selected
            item["global_difficulty_delta"] = int(
                global_state["difficulty_delta"]
            )
            item["combined_difficulty_delta"] = combined_delta
            item["sampling_reason"].append(global_state["reason"])
            item["selected_difficulty"] = difficulty + combined_delta
            item["selected_difficulty"] = max(
                int(context.config["generation"]["minimum_difficulty"]),
                min(
                    int(
                        context.config["generation"][
                            "maximum_difficulty"
                        ]
                    ),
                    int(item["selected_difficulty"]),
                ),
            )
            item["category"] = category
            item["base_weight"] = float(
                metadata.get("base_weight", 1.0)
            )
            item["current_count"] = int(
                metrics["subcategory_counts"][subcategory]
            )
            item["min_quota"] = int(metadata["min_quota"])
            priorities.append(item)

        budget = int(context.question_budget)
        hard_pool_size = (
            len(hard_records)
            if context.hard_pool_size is None
            else int(context.hard_pool_size)
        )
        injection_start = int(
            context.config["hard_pool"]["injection_start_iteration"]
        )
        hard_component = self.component_state["hard_pool_variants"]
        injection_enabled = (
            hard_component
            and int(context.global_iteration) >= injection_start
            and hard_pool_size > 0
        )
        hard_disabled_by_ablation = not hard_component
        if hard_disabled_by_ablation:
            redistributed = largest_remainder(
                budget,
                {
                    "coverage_deficit": float(
                        context.config["generation_mix"][
                            "coverage_deficit"
                        ]
                    ),
                    "retention_known": float(
                        context.config["generation_mix"][
                            "retention_known"
                        ]
                    ),
                },
            )
            source_budget = {
                "hard_pool_variant": 0,
                "coverage_deficit": redistributed["coverage_deficit"],
                "retention_known": redistributed["retention_known"],
            }
        elif injection_enabled:
            source_budget = largest_remainder(
                budget,
                {
                    "hard_pool_variant": float(
                        context.config["generation_mix"][
                            "hard_pool_variants"
                        ]
                    ),
                    "coverage_deficit": float(
                        context.config["generation_mix"][
                            "coverage_deficit"
                        ]
                    ),
                    "retention_known": float(
                        context.config["generation_mix"][
                            "retention_known"
                        ]
                    ),
                },
            )
        else:
            source_budget = {
                "hard_pool_variant": 0,
                "coverage_deficit": budget,
                "retention_known": 0,
            }

        eligible_hard_keys = {
            (
                str(record.get("category", "")),
                _record_subcategory(record),
            )
            for record in hard_records
            if record.get("sample_grade") in {None, "train_eligible"}
        }
        if (
            not hard_disabled_by_ablation
            and hard_records
            and not eligible_hard_keys
        ):
            injection_enabled = False
            source_budget = {
                "hard_pool_variant": 0,
                "coverage_deficit": budget,
                "retention_known": 0,
            }

        allocations: list[dict[str, Any]] = []
        actual_subcategory_budget: Counter[str] = Counter()
        maximum = int(
            context.config["generation"]["max_questions_per_prompt"]
        )
        for source in (
            "hard_pool_variant",
            "coverage_deficit",
            "retention_known",
        ):
            source_total = int(source_budget[source])
            if source_total <= 0:
                continue
            source_items = priorities
            if source == "hard_pool_variant" and eligible_hard_keys:
                source_items = [
                    item
                    for item in priorities
                    if (
                        item["category"],
                        item["subcategory"],
                    )
                    in eligible_hard_keys
                ]
            source_weights = {
                f"{item['category']}|||{item['subcategory']}": max(
                    1e-9,
                    float(item["priority_score"])
                    * float(item["base_weight"]),
                )
                for item in source_items
            }
            source_allocation = largest_remainder(
                source_total,
                source_weights,
            )
            by_key = {
                f"{item['category']}|||{item['subcategory']}": item
                for item in source_items
            }
            for key, source_count in source_allocation.items():
                if not source_count:
                    continue
                item = by_key[key]
                actual_subcategory_budget[
                    item["subcategory"]
                ] += source_count
                remaining = int(source_count)
                while remaining:
                    chunk = min(remaining, maximum)
                    difficulty = int(item["selected_difficulty"])
                    allocations.append(
                        {
                            "category": item["category"],
                            "sub_category": item["subcategory"],
                            "question_count": chunk,
                            "difficulty": difficulty,
                            "target_difficulty_profile": (
                                target_difficulty_profile(
                                    difficulty,
                                    context.config,
                                )
                            ),
                            "generation_source": source,
                            "generation_strategy": (
                                "numeric_structure_variant"
                                if source == "hard_pool_variant"
                                else (
                                    "quota_repair"
                                    if source == "coverage_deficit"
                                    else "retention_probe"
                                )
                            ),
                            "priority_score": item["priority_score"],
                        }
                    )
                    remaining -= chunk

        if int(context.global_iteration) < injection_start and (
            source_budget["hard_pool_variant"] != 0
        ):
            raise RuntimeError(
                "hard pool injection occurred during warmup"
            )
        if (
            injection_enabled
            and source_budget["hard_pool_variant"] <= 0
        ):
            raise RuntimeError(
                "directed generation budget must be positive after injection"
            )
        diagnostics: dict[str, Any] = {
            "observed_difficulty_used_for_sampling": bool(
                self.component_state["observed_difficulty"]
            ),
            "error_type_targeting_enabled": bool(
                self.component_state["error_type_targeting"]
            ),
            "difficulty_module_enabled": bool(
                self.component_state["difficulty_module"]
            ),
            "history_mode": str(
                context.config["adaptive_sampling"]["history_mode"]
            ),
            "decay_lambda": float(
                context.config["adaptive_sampling"]["decay_lambda"]
            ),
        }
        if hard_disabled_by_ablation:
            diagnostics.update(
                {
                    "hard_pool_disabled_by_ablation": True,
                    "events": ["hard_pool_disabled_by_ablation"],
                }
            )
        runtime_checks = [
            {
                "check": "hard_pool_budget_zero_when_disabled",
                "passed": (
                    self.component_state["hard_pool_variants"]
                    or int(source_budget["hard_pool_variant"]) == 0
                ),
                "observed": int(source_budget["hard_pool_variant"]),
            },
            {
                "check": "coverage_weight_zero_when_disabled",
                "passed": (
                    self.component_state["coverage_priority"]
                    or float(
                        effective_config["adaptive_sampling"][
                            "coverage_weight"
                        ]
                    )
                    == 0.0
                ),
                "observed": float(
                    effective_config["adaptive_sampling"]["coverage_weight"]
                ),
            },
            {
                "check": "uncertainty_weight_zero_when_disabled",
                "passed": (
                    self.component_state["uncertainty_priority"]
                    or float(
                        effective_config["adaptive_sampling"][
                            "uncertainty_weight"
                        ]
                    )
                    == 0.0
                ),
                "observed": float(
                    effective_config["adaptive_sampling"][
                        "uncertainty_weight"
                    ]
                ),
            },
            {
                "check": "retention_weight_zero_when_disabled",
                "passed": (
                    self.component_state["retention_priority"]
                    or float(
                        effective_config["adaptive_sampling"][
                            "retention_weight"
                        ]
                    )
                    == 0.0
                ),
                "observed": float(
                    effective_config["adaptive_sampling"]["retention_weight"]
                ),
            },
            {
                "check": "global_difficulty_zero_when_disabled",
                "passed": (
                    self.component_state["global_difficulty"]
                    or (
                        not bool(global_state["enabled"])
                        and int(global_state["difficulty_delta"]) == 0
                    )
                ),
                "observed": int(global_state["difficulty_delta"]),
            },
            {
                "check": "difficulty_module_has_no_sampling_effect_when_disabled",
                "passed": (
                    self.component_state["difficulty_module"]
                    or all(
                        int(item["previous_difficulty"])
                        == int(
                            context.config["adaptive_sampling"][
                                "initial_difficulty"
                            ]
                        )
                        for item in priorities
                    )
                ),
                "observed": sorted(
                    {
                        int(item["previous_difficulty"])
                        for item in priorities
                    }
                ),
            },
        ]
        if not all(item["passed"] for item in runtime_checks):
            raise RuntimeError("A disabled study component remained active")
        diagnostics["component_runtime_checks"] = runtime_checks
        return _make_plan(
            self,
            context,
            allocations=allocations,
            source_budget=source_budget,
            adaptive_state=priorities,
            accuracy_state=global_state,
            metrics=metrics,
            hard_pool_enabled=injection_enabled,
            hard_pool_reference_count=(
                min(
                    hard_pool_size,
                    int(
                        context.config["hard_pool"][
                            "max_reference_samples"
                        ]
                    ),
                )
                if injection_enabled
                else 0
            ),
            diagnostics=diagnostics,
        )


def create_policy(config: Mapping[str, Any]) -> SamplingPolicy:
    """Create the effective policy selected by the resolved study config."""

    study = _study(config)
    policy_name = str(study.get("policy", "full"))
    variant = str(study.get("variant", policy_name))
    if variant in {
        "full",
        "full_no_hard_pool",
        "full_no_error_targeting",
        "full_no_observed_difficulty_sampling",
        "full_no_difficulty_module",
        "full_no_coverage_priority",
        "full_no_uncertainty_priority",
        "full_no_global_difficulty",
        "full_no_retention_priority",
    }:
        if policy_name != "full":
            raise ValueError(
                f"study.policy={policy_name!r} conflicts with "
                f"study.variant={variant!r}"
            )
        return FullAdaptivePolicy(config, variant)
    expected_variant = policy_name
    if variant != expected_variant:
        raise ValueError(
            f"study.policy={policy_name!r} conflicts with "
            f"study.variant={variant!r}"
        )
    policies: dict[str, type[_PolicyBase]] = {
        "base": BasePolicy,
        "random": RandomPolicy,
        "uniform": UniformPolicy,
        "error_only": ErrorOnlyPolicy,
    }
    try:
        policy_class = policies[policy_name]
    except KeyError as exc:
        raise ValueError(f"unsupported study policy: {policy_name!r}") from exc
    return policy_class(config, variant)


def policy_runtime_descriptor(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the actual policy/component state used by the runtime."""

    return create_policy(config).descriptor()


__all__ = [
    "BasePolicy",
    "ErrorOnlyPolicy",
    "FullAdaptivePolicy",
    "GenerationPlan",
    "PolicyContext",
    "RandomPolicy",
    "SamplingPolicy",
    "UniformPolicy",
    "create_policy",
    "policy_runtime_descriptor",
]
