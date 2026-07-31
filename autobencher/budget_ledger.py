"""Auditable runtime cost ledger and hard fairness-budget enforcement."""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping


PROTOCOLS = ("question_matched", "data_matched", "cost_matched")


class BudgetExhausted(RuntimeError):
    """Raised before starting more generation after a hard cost cap."""


def estimate_tokens(text: Any) -> int:
    """Deterministic fallback when a provider does not expose token usage."""
    value = str(text or "")
    return 0 if not value else max(1, math.ceil(len(value.encode("utf-8")) / 4))


def training_record_tokens(records: list[Mapping[str, Any]]) -> int:
    return sum(
        estimate_tokens(record.get("instruction"))
        + estimate_tokens(record.get("input"))
        + estimate_tokens(record.get("output"))
        for record in records
    )


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class BudgetLedger:
    def __init__(
        self,
        path: str | Path,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ):
        self.path = Path(path)
        self.config = copy.deepcopy(dict(config))
        protocol = str(self.config.get("protocol", "question_matched"))
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unsupported budget protocol: {protocol}")
        self._lock = threading.RLock()
        self._started = time.monotonic()
        self.data: dict[str, Any] = {
            "schema_version": "1.0",
            **dict(metadata),
            "status": "running",
            "protocol": {
                "name": protocol,
                "data_matched_target_samples": self.config.get(
                    "data_matched_target_samples"
                ),
                "max_generation_tokens": self.config.get(
                    "max_generation_tokens"
                ),
                "max_total_api_calls": self.config.get(
                    "max_total_api_calls"
                ),
                "max_gpu_hours": self.config.get("max_gpu_hours"),
                "exhausted": False,
                "exhausted_reason": None,
            },
            "generation": {
                "requested_question_count": 0,
                "generator_output_count": 0,
                "generator_accepted_count": 0,
                "retry_count": 0,
                "failed_generation_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "api_call_count": 0,
                "wall_time_seconds": 0.0,
                "token_count_exact_calls": 0,
                "token_count_estimated_calls": 0,
            },
            "validation": {
                "sympy_validation_count": 0,
                "judge_call_count": 0,
                "validation_failure_count": 0,
                "difficulty_rejection_count": 0,
                "dedup_rejection_count": 0,
                "semantic_filter_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "api_call_count": 0,
                "wall_time_seconds": 0.0,
            },
            "training": {
                "final_training_sample_count": 0,
                "dataset_token_count": 0,
                "final_training_token_count": 0,
                "optimizer_steps": 0,
                "gpu_hours": 0.0,
                "peak_gpu_memory_gb": None,
                "wall_time_seconds": 0.0,
                "training_attempt_count": 0,
                "completed_training_cycles": 0,
            },
            "evaluation": {
                "baseline_accuracy": None,
                "final_accuracy": None,
                "accuracy_gain": None,
            },
            "totals": {},
            "efficiency": {},
            "estimated_cost": {
                "currency": str(
                    self.config.get("pricing", {}).get("currency", "USD")
                ),
                "amount": None,
                "complete": False,
                "reason": "pricing_not_configured",
            },
        }
        self.flush()

    @property
    def protocol(self) -> str:
        return str(self.data["protocol"]["name"])

    def _derive(self) -> None:
        generation = self.data["generation"]
        validation = self.data["validation"]
        training = self.data["training"]
        evaluation = self.data["evaluation"]
        total_tokens = (
            int(generation["input_tokens"])
            + int(generation["output_tokens"])
            + int(validation["input_tokens"])
            + int(validation["output_tokens"])
            + int(training["final_training_token_count"])
        )
        total_api_calls = (
            int(generation["api_call_count"])
            + int(validation["api_call_count"])
        )
        self.data["totals"] = {
            "total_tokens": total_tokens,
            "total_api_calls": total_api_calls,
            "total_gpu_hours": float(training["gpu_hours"]),
            "total_wall_time_seconds": round(
                time.monotonic() - self._started,
                6,
            ),
        }
        generated_tokens = (
            int(generation["input_tokens"])
            + int(generation["output_tokens"])
        )
        training_tokens = int(training["final_training_token_count"])
        accuracy_gain = evaluation.get("accuracy_gain")
        self.data["efficiency"] = {
            "accepted_samples_per_1k_generated_tokens": (
                float(generation["generator_accepted_count"])
                / generated_tokens
                * 1000
                if generated_tokens
                else None
            ),
            "accuracy_gain_per_1k_training_tokens": (
                float(accuracy_gain) / training_tokens * 1000
                if accuracy_gain is not None and training_tokens
                else None
            ),
            "accuracy_gain_per_gpu_hour": (
                float(accuracy_gain) / float(training["gpu_hours"])
                if accuracy_gain is not None and training["gpu_hours"]
                else None
            ),
        }
        pricing = self.config.get("pricing", {})
        rates = (
            pricing.get("input_per_million_tokens"),
            pricing.get("output_per_million_tokens"),
            pricing.get("gpu_hour"),
        )
        if all(value is not None for value in rates):
            input_tokens = (
                int(generation["input_tokens"])
                + int(validation["input_tokens"])
            )
            output_tokens = (
                int(generation["output_tokens"])
                + int(validation["output_tokens"])
            )
            amount = (
                input_tokens / 1_000_000 * float(rates[0])
                + output_tokens / 1_000_000 * float(rates[1])
                + float(training["gpu_hours"]) * float(rates[2])
            )
            self.data["estimated_cost"] = {
                "currency": str(pricing.get("currency", "USD")),
                "amount": amount,
                "complete": True,
                "reason": None,
            }

    def flush(self) -> None:
        with self._lock:
            self._derive()
            _atomic_json(self.data, self.path)

    def assert_generation_available(self) -> None:
        with self._lock:
            if self.protocol != "cost_matched":
                return
            generation = self.data["generation"]
            used_tokens = (
                int(generation["input_tokens"])
                + int(generation["output_tokens"])
            )
            token_cap = self.data["protocol"].get("max_generation_tokens")
            call_cap = self.data["protocol"].get("max_total_api_calls")
            reason = None
            if token_cap is not None and used_tokens >= int(token_cap):
                reason = (
                    f"generation token budget exhausted: "
                    f"{used_tokens}/{int(token_cap)}"
                )
            elif call_cap is not None and int(
                self.data["totals"].get("total_api_calls", 0)
            ) >= int(call_cap):
                reason = (
                    f"API call budget exhausted: "
                    f"{self.data['totals']['total_api_calls']}/{int(call_cap)}"
                )
            if reason:
                self.data["protocol"]["exhausted"] = True
                self.data["protocol"]["exhausted_reason"] = reason
                self.flush()
                raise BudgetExhausted(reason)

    def record_generation_call(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
        retry: bool = False,
        api_calls: int = 1,
        exact_tokens: bool = False,
    ) -> None:
        with self._lock:
            section = self.data["generation"]
            section["input_tokens"] += int(input_tokens)
            section["output_tokens"] += int(output_tokens)
            section["wall_time_seconds"] += float(wall_time_seconds)
            section["api_call_count"] += int(api_calls)
            section["retry_count"] += int(bool(retry))
            key = (
                "token_count_exact_calls"
                if exact_tokens
                else "token_count_estimated_calls"
            )
            section[key] += 1
            self.flush()

    def assert_training_available(self) -> None:
        with self._lock:
            if self.protocol != "cost_matched":
                return
            gpu_cap = self.data["protocol"].get("max_gpu_hours")
            used = float(self.data["training"]["gpu_hours"])
            if gpu_cap is not None and used >= float(gpu_cap):
                reason = (
                    f"GPU-hour budget exhausted: {used:g}/{float(gpu_cap):g}"
                )
                self.data["protocol"]["exhausted"] = True
                self.data["protocol"]["exhausted_reason"] = reason
                self.flush()
                raise BudgetExhausted(reason)

    def record_generation_batch(
        self,
        *,
        requested: int,
        output: int,
        accepted: int,
        failed: int,
        repair_retries: int = 0,
        difficulty_rejections: int = 0,
        sympy_validations: int = 0,
        validation_failures: int = 0,
    ) -> None:
        with self._lock:
            section = self.data["generation"]
            section["requested_question_count"] += int(requested)
            section["generator_output_count"] += int(output)
            section["generator_accepted_count"] += int(accepted)
            section["failed_generation_count"] += int(failed)
            section["retry_count"] += int(repair_retries)
            self.data["validation"]["difficulty_rejection_count"] += int(
                difficulty_rejections
            )
            self.data["validation"]["sympy_validation_count"] += int(
                sympy_validations
            )
            self.data["validation"]["validation_failure_count"] += int(
                validation_failures
            )
            self.flush()

    def record_judge(
        self,
        *,
        success: bool,
        attempts: int,
        input_tokens: int,
        output_tokens: int,
        wall_time_seconds: float,
        api_calls: int | None = None,
    ) -> None:
        with self._lock:
            section = self.data["validation"]
            section["judge_call_count"] += int(attempts)
            section["api_call_count"] += int(
                attempts if api_calls is None else api_calls
            )
            section["input_tokens"] += int(input_tokens) * int(attempts)
            section["output_tokens"] += int(output_tokens)
            section["wall_time_seconds"] += float(wall_time_seconds)
            if not success:
                section["validation_failure_count"] += 1
            self.flush()

    def record_dataset(
        self,
        manifest: Mapping[str, Any],
        records: list[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            reasons = manifest.get("rejection_reasons", {})
            dedup_names = {
                "exact_duplicate",
                "near_duplicate",
                "template_duplicate",
            }
            semantic_names = {
                "near_duplicate",
                "holdout_semantic_duplicate",
                "holdout_near_duplicate",
            }
            self.data["validation"]["dedup_rejection_count"] += sum(
                int(reasons.get(name, 0)) for name in dedup_names
            )
            self.data["validation"]["semantic_filter_count"] += sum(
                int(reasons.get(name, 0)) for name in semantic_names
            )
            training = self.data["training"]
            dataset_tokens = training_record_tokens(records)
            training["final_training_sample_count"] += len(records)
            training["dataset_token_count"] += dataset_tokens
            self.flush()

    def record_training(self, summary: Mapping[str, Any]) -> None:
        with self._lock:
            section = self.data["training"]
            section["final_training_token_count"] += int(
                summary.get("training_token_count", 0) or 0
            )
            section["optimizer_steps"] += int(
                summary.get("optimizer_steps", 0) or 0
            )
            section["gpu_hours"] += float(
                summary.get("gpu_hours", 0.0) or 0.0
            )
            peak = summary.get("peak_gpu_memory_gb")
            if peak is not None:
                section["peak_gpu_memory_gb"] = max(
                    float(section["peak_gpu_memory_gb"] or 0.0),
                    float(peak),
                )
            section["wall_time_seconds"] += float(
                summary.get("training_wall_time_seconds", 0.0) or 0.0
            )
            section["training_attempt_count"] += 1
            if summary.get("status") != "failed":
                section["completed_training_cycles"] += 1
            self.flush()

    def record_accuracy(
        self,
        baseline_accuracy: float | None,
        final_accuracy: float | None,
    ) -> None:
        with self._lock:
            evaluation = self.data["evaluation"]
            evaluation["baseline_accuracy"] = baseline_accuracy
            evaluation["final_accuracy"] = final_accuracy
            evaluation["accuracy_gain"] = (
                float(final_accuracy) - float(baseline_accuracy)
                if baseline_accuracy is not None and final_accuracy is not None
                else None
            )
            self.flush()

    def finalize(self, status: str) -> dict[str, Any]:
        with self._lock:
            self.data["status"] = str(status)
            self.flush()
            return copy.deepcopy(self.data)


_ACTIVE_LEDGER: BudgetLedger | None = None


def set_active_ledger(ledger: BudgetLedger | None) -> None:
    global _ACTIVE_LEDGER
    _ACTIVE_LEDGER = ledger


def active_ledger() -> BudgetLedger | None:
    return _ACTIVE_LEDGER
