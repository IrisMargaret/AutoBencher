"""Central configuration loading, validation, precedence, and snapshots."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REQUIRED_DATA_ROOT = (
    "/vepfs-mlp2/queue010/20262202597/math_flywheel"
)


class ConfigurationError(RuntimeError):
    """A configuration error with a precise dotted field path."""

    def __init__(self, path: str, message: str, value: Any = None):
        detail = f"ConfigurationError: {path}: {message}"
        if value is not None:
            detail += f" (current value: {value!r})"
        super().__init__(detail)
        self.path = path
        self.value = value


class ProjectConfig(dict):
    """Recursively immutable, attribute-accessible runtime configuration."""

    def __init__(self, values: Mapping[str, Any]):
        dict.__init__(
            self,
            {
                key: _freeze_config(value)
                for key, value in values.items()
            },
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("ProjectConfig is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return ProjectConfig(value)
    if isinstance(value, list):
        return tuple(_freeze_config(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_config(item) for item in value)
    return value


def thaw_config(value: Any) -> Any:
    """Return a mutable plain-data copy suitable for serialization or tests."""
    if isinstance(value, Mapping):
        return {
            key: thaw_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [thaw_config(item) for item in value]
    return copy.deepcopy(value)


def resolve_sensitive_environment(
    config: Mapping[str, Any],
) -> dict[str, str | None]:
    """Read only the centrally declared sensitive environment variables."""
    declarations = config.get("sensitive_environment", {})
    return {
        str(logical_name): os.environ.get(str(variable_name))
        for logical_name, variable_name in declarations.items()
    }


FIXED_MATH_CATEGORIES = (
    "Arithmetic",
    "Algebra",
    "Geometry & Trigonometry",
    "Probability & Statistics",
    "Word Problems",
    "Number Theory",
    "Calculus",
    "Linear Algebra",
    "Composite Comprehensive",
)

DEFAULT_TAXONOMY = {
    "Arithmetic": (
        "Integer Operations",
        "Fraction and Decimal Operations",
        "Ratio and Percentage",
    ),
    "Algebra": (
        "Linear Equations",
        "Systems of Equations",
        "Polynomials and Inequalities",
    ),
    "Geometry & Trigonometry": (
        "Plane Geometry",
        "Solid Geometry",
        "Trigonometric Reasoning",
    ),
    "Probability & Statistics": (
        "Basic Probability",
        "Combinatorics",
        "Descriptive Statistics",
    ),
    "Word Problems": (
        "Rate and Distance",
        "Work and Mixture",
        "Financial Applications",
    ),
    "Number Theory": (
        "Divisibility and Factors",
        "Prime Factorization",
        "Modular Arithmetic",
    ),
    "Calculus": (
        "Limits and Continuity",
        "Differentiation",
        "Integration",
    ),
    "Linear Algebra": (
        "Matrix Operations",
        "Linear Systems",
        "Vectors and Vector Spaces",
    ),
    "Composite Comprehensive": (
        "Cross-Domain Multi-Step Problems",
        "Proof and Mathematical Reasoning",
        "Constraint Synthesis",
    ),
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        "boolean",
        "expected true/false, yes/no, on/off, or 1/0",
        value,
    )


SAFE_DEFAULTS: dict[str, Any] = {
    "schema_version": "1.0",
    "experiment": {
        "name": "math_flywheel",
        "seed": 42,
        "mode": "eval",
        "exp_mode": "autobencher",
        "num_iterations": 2,
        "max_cycles": 1,
        "questions_per_iteration": 450,
        "export_interval": 1,
        "clean_cycle_cache": True,
        "resume": True,
        "disk_warning_threshold_gb": 10,
    },
    "compatibility": {
        "use_helm": False,
        "preserve_existing_cli": True,
        "preserve_existing_cache": True,
        "preserve_existing_output_fields": True,
    },
    "models": {
        "evaluator": {
            "model_name": "deepseek-v4-pro",
            "use_privileged_tools": True,
            "temperature": 0.2,
            "max_tokens": 4096,
            "top_p": 0.9,
            "request_timeout_seconds": 120,
            "max_retries": 3,
            "retry_delay_seconds": 5,
        },
        "test_taker": {
            "model_path": "qwen2.5:7b-instruct",
            "use_external_tools": False,
            "temperature": 0.0,
            "max_new_tokens": 1024,
            "top_p": 1.0,
            "do_sample": False,
            "request_timeout_seconds": 300,
            "max_retries": 3,
        },
        "judge": {
            "model_name": None,
        },
    },
    "paths": {
        "output_root": "math_flywheel",
        "outfile_prefix": "qwen7b_dsagent",
        "allowed_data_root": None,
        "enforce_data_root": False,
        "cache_dir": None,
        "model_output_dir": None,
        "log_dir": None,
        "temp_dir": None,
        "dataset_dir": None,
        "checkpoint_dir": None,
        "review_dir": None,
    },
    "sensitive_environment": {
        "deepseek_api_key": "DEEPSEEK_API_KEY",
        "deepseek_base_url": "DEEPSEEK_BASE_URL",
        "openai_api_key": "OPENAI_API_KEY",
        "openai_organization": "OPENAI_ORG_ID",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "vllm_api_key": "VLLM_API_KEY",
        "vllm_base_url": "VLLM_BASE_URL",
        "ollama_api_key": "OLLAMA_API_KEY",
        "ollama_base_url": "OLLAMA_BASE_URL",
    },
    "coverage": {
        "default_min_quota": 20,
        "enforce_min_quota": True,
        "repair_rounds": 2,
        "raw_coverage_enabled": True,
        "effective_coverage_enabled": True,
        "balance_metrics": [
            "normalized_entropy",
            "js_divergence",
            "coefficient_of_variation",
            "max_min_count_ratio",
        ],
    },
    "taxonomy": {
        category: {
            subcategory: {"min_quota": 20, "base_weight": 1.0}
            for subcategory in subcategories
        }
        for category, subcategories in DEFAULT_TAXONOMY.items()
    },
    "generation_mix": {
        "warmup_iterations": 2,
        "hard_pool_variants": 0.45,
        "coverage_deficit": 0.40,
        "retention_known": 0.15,
    },
    "generation": {
        "max_questions_per_prompt": 50,
        "temperature": 0.0,
        "top_p": 0.1,
        "minimum_difficulty": 2,
        "maximum_difficulty": 6,
        "maximum_reasoning_steps": 6,
        "prohibit_competition_level": True,
        "generator_max_retry": 3,
        "max_quota_repair_rounds": 3,
        "allow_partial_question_budget": True,
        "minimum_verified_questions": 1,
        "gold_solver_backend": "sympy",
        "truth_solver_timeout_seconds": 10,
        "truth_solver_max_retry": 1,
        "subcategory_failure_cooldown_threshold": 3,
        "subcategory_cooldown_iterations": 2,
        "require_gold_answer_validation": True,
        "gold_validation_attempts": 2,
        "gold_validation_temperature": 0.0,
        "gold_validation_max_tokens": 4096,
        "gold_validation_timeout_seconds": 10,
    },
    "evaluator_pipeline": {
        "enabled": True,
        "max_parallel_questions": 4,
        "solver_prompt_path": "prompts/evaluator_python_solver.txt",
        "solver_strategy_path": "prompts/tora_evaluator_strategy.txt",
        "independent_solver_prompt_path": (
            "prompts/evaluator_independent_solver.txt"
        ),
        "postcheck_prompt_path": "prompts/evaluator_postcheck.txt",
        "semantic_judge_prompt_path": "prompts/semantic_answer_judge.txt",
        "temperature": 0.0,
        "solver_max_tokens": 4096,
        "independent_solver_max_tokens": 4096,
        "postcheck_max_tokens": 1024,
        "semantic_judge_max_tokens": 768,
        "code_generation_attempts": 2,
        "independent_solver_attempts": 2,
        "postcheck_attempts": 2,
        "semantic_judge_attempts": 2,
        "code_execution_timeout_seconds": 10,
        "max_code_chars": 12000,
        "max_ast_nodes": 4000,
        "max_integer_literal": 1000000,
        "max_output_chars": 16000,
        "max_answer_chars": 512,
        "minimum_difficulty": 2,
        "maximum_difficulty": 6,
        "semantic_judge_confidence_threshold": 0.80,
        "require_runtime_verification": True,
        "require_semantic_judge": True,
    },
    "hard_pool": {
        "enabled": True,
        "injection_start_iteration": 3,
        "max_reference_samples": 40,
        "max_prompt_tokens": 12000,
        "deduplicate_before_injection": True,
        "stratify_by_subcategory": True,
        "stratify_by_error_type": True,
        "hide_reference_answers": True,
    },
    "adaptive_sampling": {
        "enabled": True,
        "initial_difficulty": 4,
        "global_accuracy_enabled": True,
        "global_accuracy_low": 0.35,
        "global_accuracy_high": 0.70,
        "global_accuracy_min_observations": 10,
        "global_difficulty_step": 1,
        "max_difficulty_change_per_iteration": 1,
        "target_accuracy_low": 0.10,
        "target_accuracy_high": 0.30,
        "target_accuracy_mid": 0.20,
        "beta_prior_alpha": 1.0,
        "beta_prior_beta": 1.0,
        "boundary_weight": 0.40,
        "coverage_weight": 0.25,
        "uncertainty_weight": 0.15,
        "persistent_error_weight": 0.15,
        "retention_weight": 0.05,
        "temperature": 0.15,
        "min_observations_before_adjustment": 5,
    },
    "test_taker_prompt": {
        "require_reasoning_summary": True,
        "min_reasoning_steps": 1,
        "max_reasoning_steps": 8,
        "max_chars_per_step": 120,
        "require_json_output": True,
        "prohibit_prompt_echo": True,
        "prohibit_irrelevant_content": True,
        "prohibit_external_tools": True,
        "format_repair_attempts": 2,
        "stop_sequences": [
            "\nHuman:",
            "\nUser:",
            "\nSystem:",
            "\nAssistant:",
            "Human:",
            "User:",
            "System:",
            "Assistant:",
            "<|im_end|>",
            "<|endoftext|>",
        ],
    },
    "answer_normalization": {
        "absolute_tolerance": 1.0e-6,
        "relative_tolerance": 1.0e-5,
        "allow_fraction_decimal_equivalence": True,
        "normalize_units": True,
        "normalize_boolean_text": True,
        "symbolic_equivalence": True,
        "set_order_sensitive": False,
        "math_verify_enabled": True,
    },
    "error_attribution": {
        "enabled": True,
        "confidence_threshold": 0.70,
        "allow_multiple_tags": True,
        "require_primary_tag": True,
        "require_evidence": True,
        "low_confidence_tag": "unknown_error",
    },
    "dataset": {
        "exact_dedup": True,
        "text_near_dedup": True,
        "template_dedup": True,
        "semantic_dedup": True,
        "near_duplicate_threshold": 0.90,
        "semantic_similarity_threshold": 0.92,
        "evaluator_confidence_threshold": 0.80,
        "filter_ambiguous_samples": True,
        "filter_invalid_answers": True,
        "filter_tool_violations": True,
        "require_gold_reasoning_steps": True,
        "require_concrete_gold_reasoning": True,
        "require_gold_reasoning_verification": True,
        "require_gold_answer_in_reasoning": True,
        "min_gold_reasoning_steps": 2,
        "max_gold_reasoning_steps": 6,
        "min_gold_reasoning_chars_per_step": 8,
        "max_gold_reasoning_chars_per_step": 300,
        "max_samples_per_template_cluster": 5,
        "holdout_exact_template_rejection": True,
        "holdout_near_duplicate_threshold": 0.85,
        "holdout_semantic_similarity_threshold": 0.82,
        "text_dedup_enabled": True,
        "text_dedup_num_perm": 128,
        "text_dedup_ngram_size": 2,
        "text_dedup_seed": 42,
        "text_dedup_similarity_threshold": 0.78,
        "holdout_text_dedup_similarity_threshold": 0.72,
        "datasketch_enabled": True,
        "datasketch_required": True,
        "datasketch_lsh_enabled": True,
        "sentence_transformers_enabled": True,
        "sentence_transformers_required": True,
        "sentence_transformers_model": (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "sentence_transformers_revision": None,
        "sentence_transformers_device": "cpu",
        "sentence_transformers_cache_dir": None,
        "sentence_transformers_local_files_only": False,
        "sentence_transformers_batch_size": 64,
    },
    "training_mix": {
        "incorrect_boundary_samples": 0.55,
        "correct_retention_samples": 0.25,
        "coverage_repair_samples": 0.15,
        "format_instruction_samples": 0.05,
        "strict_correct_incorrect_ratio": True,
        "scope": "current_cycle",
        "minimum_samples": 4,
    },
    "fixed_test": {
        "enabled": True,
        "dataset_path": "benchmarks/fixed_math_test_set.json",
        "require_all_subcategories": True,
        "evaluate_baseline": True,
        "evaluate_after_each_training_cycle": True,
        "fail_on_training_leakage": True,
    },
    "finetune": {
        "enabled": True,
        "gpu": "0",
        "epochs": 3,
        "batch_size": 8,
        "lora_rank": 6,
        "new_local_model_suffix": "math_lora",
        "max_seq_length": 2048,
        "learning_rate": 2.0e-4,
        "save_step_metrics": True,
        "merge_adapter": True,
    },
    "structured_output": {
        "enabled": True,
        "local_backend": "outlines",
        "fallback_backend": "guidance",
        "required": False,
        "use_for_evaluator": True,
        "use_for_test_taker": True,
    },
    "tracking": {
        "wandb": {
            "enabled": True,
            "mode": "offline",
            "project": "autobencher-math-flywheel",
            "entity": None,
            "group": "adaptive-math-flywheel",
            "tags": ["autobencher", "math", "data-flywheel"],
            "log_model": False,
        },
    },
    "export": {
        "save_iteration_json": True,
        "save_raw_responses": True,
        "save_hard_pool_snapshot": True,
        "save_training_dataset": True,
        "save_finetune_metrics": True,
        "atomic_write": True,
        "schema_version": "1.0",
    },
    "logging": {
        "level": "INFO",
        "console_enabled": True,
        "file_enabled": True,
        "jsonl_enabled": True,
        "progress_enabled": True,
        "progress_style": "single",
        "disable_library_progress": True,
        "progress_leave": False,
        "progress_dynamic_ncols": True,
        "show_stage_summary": True,
        "suppress_duplicate_messages": True,
    },
}


CLI_CONFIG_MAPPING = (
    ("agent_modelname", "models.evaluator.model_name"),
    ("test_taker_modelname", "models.test_taker.model_path"),
    ("tool_modelname", "models.judge.model_name"),
    ("exp_mode", "experiment.exp_mode"),
    ("num_iters", "experiment.num_iterations"),
    ("mode", "experiment.mode"),
    ("export_interval", "experiment.export_interval"),
    ("max_cycle", "experiment.max_cycles"),
    ("clean_cycle_cache", "experiment.clean_cycle_cache"),
    ("disk_warning_threshold", "experiment.disk_warning_threshold_gb"),
    ("finetune_gpu", "finetune.gpu"),
    ("finetune_epoch", "finetune.epochs"),
    ("finetune_batch", "finetune.batch_size"),
    ("lora_rank", "finetune.lora_rank"),
    ("new_local_model_suffix", "finetune.new_local_model_suffix"),
    ("temperature", "models.evaluator.temperature"),
    ("top_p", "models.evaluator.top_p"),
)


def cli_config_overrides(
    namespace: Any,
    explicit_options: Iterable[str],
    include_implicit_defaults: bool,
) -> dict[str, Any]:
    """Map the backward-compatible CLI into centralized config fields."""
    explicit = set(explicit_options)
    overlay: dict[str, Any] = {}
    for attribute, path in CLI_CONFIG_MAPPING:
        value = getattr(namespace, attribute, None)
        if value is None:
            continue
        option_names = {
            f"--{attribute}",
            f"--{attribute.replace('_', '-')}",
        }
        if include_implicit_defaults or option_names & explicit:
            set_dotted(overlay, path, value)

    use_helm = getattr(namespace, "use_helm", None)
    if include_implicit_defaults or {"--use_helm", "--use-helm"} & explicit:
        set_dotted(
            overlay,
            "compatibility.use_helm",
            str(use_helm).strip().lower() == "yes",
        )

    acc_target = getattr(namespace, "acc_target", None)
    if acc_target is not None and (
        include_implicit_defaults
        or {"--acc_target", "--acc-target"} & explicit
    ):
        pieces = [
            piece
            for piece in re.split(r"\s*(?:--|,)\s*", str(acc_target))
            if piece
        ]
        if len(pieces) != 2:
            raise ConfigurationError(
                "acc_target",
                "expected low,high or low--high",
                acc_target,
            )
        low, high = map(float, pieces)
        set_dotted(overlay, "adaptive_sampling.target_accuracy_low", low)
        set_dotted(overlay, "adaptive_sampling.target_accuracy_high", high)
        set_dotted(
            overlay,
            "adaptive_sampling.target_accuracy_mid",
            (low + high) / 2,
        )

    outfile_prefix = getattr(namespace, "outfile_prefix1", None)
    if outfile_prefix and (
        include_implicit_defaults
        or {"--outfile_prefix1", "--outfile-prefix1"} & explicit
    ):
        raw_prefix = os.path.abspath(os.fspath(outfile_prefix))
        set_dotted(
            overlay,
            "paths.output_root",
            os.path.dirname(raw_prefix) or os.getcwd(),
        )
        set_dotted(
            overlay,
            "paths.outfile_prefix",
            os.path.basename(raw_prefix).rstrip(".") or "math",
        )

    resume = getattr(namespace, "resume", None)
    if resume is not None:
        set_dotted(overlay, "experiment.resume", bool(resume))
    return overlay


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_dotted(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    keys = [part for part in dotted_path.split(".") if part]
    if not keys:
        raise ConfigurationError("override", "override path cannot be empty")
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            cursor[key] = {}
        elif not isinstance(existing, dict):
            raise ConfigurationError(
                dotted_path,
                f"cannot descend through non-object field {key}",
            )
        cursor = cursor[key]
    cursor[keys[-1]] = value


def parse_overrides(overrides: Iterable[str] | None) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigurationError(
                "override",
                "each override must use dotted.path=value",
                item,
            )
        path, raw_value = item.split("=", 1)
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigurationError(path, f"invalid YAML scalar: {exc}") from exc
        set_dotted(parsed, path.strip(), value)
    return parsed


LEGACY_FIELD_MIGRATIONS = {
    "num_iters": "experiment.num_iterations",
    "max_cycle": "experiment.max_cycles",
    "export_interval": "experiment.export_interval",
    "clean_cycle_cache": "experiment.clean_cycle_cache",
    "finetune_gpu": "finetune.gpu",
    "finetune_epoch": "finetune.epochs",
    "finetune_batch": "finetune.batch_size",
    "lora_rank": "finetune.lora_rank",
    "new_local_model_suffix": "finetune.new_local_model_suffix",
}


def _migrate_legacy_fields(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    migrated = copy.deepcopy(dict(payload))
    migrations = []
    for legacy_path, current_path in LEGACY_FIELD_MIGRATIONS.items():
        if legacy_path not in migrated:
            continue
        value = migrated.pop(legacy_path)
        set_dotted(migrated, current_path, value)
        migrations.append(
            {
                "legacy_path": legacy_path,
                "current_path": current_path,
            }
        )
        warnings.warn(
            (
                f"Configuration field {legacy_path} is deprecated; "
                f"use {current_path}"
            ),
            DeprecationWarning,
            stacklevel=3,
        )
    return migrated, migrations


def _load_yaml_layers(
    path: Path,
    seen: set[Path] | None = None,
) -> list[tuple[str, dict[str, Any], list[dict[str, str]]]]:
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        raise ConfigurationError("extends", "configuration inheritance cycle", str(path))
    if not resolved.is_file():
        raise ConfigurationError("config", "configuration file does not exist", str(path))
    seen.add(resolved)
    try:
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError("config", f"invalid YAML: {exc}", str(path)) from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("config", "top-level YAML value must be an object")
    parent = payload.pop("extends", None)
    payload, migrations = _migrate_legacy_fields(payload)
    if not parent:
        return [(str(resolved), payload, migrations)]
    parent_path = (resolved.parent / str(parent)).resolve()
    return _load_yaml_layers(parent_path, seen) + [
        (str(resolved), payload, migrations)
    ]


def _load_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for _, payload, _ in _load_yaml_layers(path, seen):
        merged = deep_merge(merged, payload)
    return merged


def _flatten_leaves(
    value: Any,
    prefix: str = "",
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_leaves(item, path))
        return flattened
    return {prefix: value}


def _record_sources(
    field_sources: dict[str, dict[str, Any]],
    overlay: Mapping[str, Any],
    source: str,
) -> None:
    for path, value in _flatten_leaves(overlay).items():
        field_sources[path] = {
            "value": thaw_config(value),
            "source": source,
        }


def _validate_known_fields(
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    prefix: str = "",
) -> None:
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            raise ConfigurationError(
                path,
                "unknown field; remove it or add it to the configuration schema",
                value,
            )
        expected = schema[key]
        if path == "taxonomy":
            continue
        if isinstance(value, Mapping) and isinstance(expected, Mapping):
            _validate_known_fields(value, expected, path)


def _get(config: Mapping[str, Any], path: str) -> Any:
    cursor: Any = config
    for part in path.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ConfigurationError(path, "required field is missing")
        cursor = cursor[part]
    return cursor


def _validate_ratio_group(config: Mapping[str, Any], path: str, fields: list[str]) -> None:
    values = []
    for field in fields:
        value = _get(config, f"{path}.{field}")
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise ConfigurationError(
                f"{path}.{field}",
                "ratio must be between 0 and 1",
                value,
            )
        values.append(float(value))
    total = sum(values)
    if abs(total - 1.0) > 1e-9:
        detail = ", ".join(
            f"{path}.{field}={value:g}"
            for field, value in zip(fields, values)
        )
        raise ConfigurationError(
            path,
            f"ratios must sum to 1.0; current sum is {total:g}; {detail}",
        )


def validate_config(config: Mapping[str, Any], validate_paths: bool = False) -> None:
    _validate_known_fields(config, SAFE_DEFAULTS)
    for path in (
        "schema_version",
        "experiment.questions_per_iteration",
        "models.evaluator.model_name",
        "models.test_taker.model_path",
        "paths.output_root",
        "taxonomy",
    ):
        _get(config, path)
    if str(_get(config, "schema_version")) != "1.0":
        raise ConfigurationError(
            "schema_version",
            "unsupported configuration version; expected 1.0",
            _get(config, "schema_version"),
        )
    quota = _get(config, "coverage.default_min_quota")
    if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
        raise ConfigurationError(
            "coverage.default_min_quota",
            "minimum quota must be a positive integer",
            quota,
        )
    budget = _get(config, "experiment.questions_per_iteration")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ConfigurationError(
            "experiment.questions_per_iteration",
            "question budget must be a positive integer",
            budget,
        )
    mode = _get(config, "experiment.mode")
    if mode not in {"eval", "data_flywheel"}:
        raise ConfigurationError(
            "experiment.mode",
            "must be eval or data_flywheel",
            mode,
        )
    enforce_data_root = _get(config, "paths.enforce_data_root")
    if not isinstance(enforce_data_root, bool):
        raise ConfigurationError(
            "paths.enforce_data_root",
            "must be a Boolean",
            enforce_data_root,
        )
    if enforce_data_root:
        allowed_raw = _get(config, "paths.allowed_data_root")
        if not str(allowed_raw or "").strip():
            raise ConfigurationError(
                "paths.allowed_data_root",
                "is required when paths.enforce_data_root is true",
                allowed_raw,
            )
        if str(allowed_raw).replace("\\", "/").rstrip("/") != (
            REQUIRED_DATA_ROOT
        ):
            raise ConfigurationError(
                "paths.allowed_data_root",
                f"must be exactly {REQUIRED_DATA_ROOT}",
                allowed_raw,
            )
        allowed = os.path.normcase(
            os.path.realpath(
                os.path.abspath(os.path.expanduser(str(allowed_raw)))
            )
        )
        writable_fields = (
            "output_root",
            "cache_dir",
            "model_output_dir",
            "log_dir",
            "temp_dir",
            "dataset_dir",
            "checkpoint_dir",
            "review_dir",
        )
        for field in writable_fields:
            raw_value = _get(config, f"paths.{field}")
            if raw_value in {None, ""}:
                candidate = os.path.join(
                    os.path.abspath(
                        os.path.expanduser(
                            str(_get(config, "paths.output_root"))
                        )
                    ),
                    field.removesuffix("_dir"),
                )
            else:
                candidate = os.path.abspath(
                    os.path.expanduser(str(raw_value))
                )
            candidate = os.path.normcase(os.path.realpath(candidate))
            try:
                contained = os.path.commonpath(
                    [allowed, candidate]
                ) == allowed
            except ValueError:
                contained = False
            if not contained:
                raise ConfigurationError(
                    f"paths.{field}",
                    (
                        "writable path must stay under "
                        f"paths.allowed_data_root={allowed_raw}"
                    ),
                    raw_value,
                )
    for path in (
        "experiment.num_iterations",
        "experiment.max_cycles",
        "experiment.export_interval",
    ):
        value = _get(config, path)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(path, "must be a positive integer", value)
    disk_warning = _get(config, "experiment.disk_warning_threshold_gb")
    if (
        not isinstance(disk_warning, (int, float))
        or isinstance(disk_warning, bool)
        or disk_warning < 0
    ):
        raise ConfigurationError(
            "experiment.disk_warning_threshold_gb",
            "must be a non-negative number",
            disk_warning,
        )
    injection = _get(config, "hard_pool.injection_start_iteration")
    if not isinstance(injection, int) or injection < 1:
        raise ConfigurationError(
            "hard_pool.injection_start_iteration",
            "must be an integer greater than or equal to 1",
            injection,
        )
    prompt_batch = _get(config, "generation.max_questions_per_prompt")
    if (
        not isinstance(prompt_batch, int)
        or isinstance(prompt_batch, bool)
        or prompt_batch < 1
    ):
        raise ConfigurationError(
            "generation.max_questions_per_prompt",
            "must be a positive integer",
            prompt_batch,
        )
    gold_solver_backend = _get(
        config,
        "generation.gold_solver_backend",
    )
    if gold_solver_backend not in {"sympy", "llm_python"}:
        raise ConfigurationError(
            "generation.gold_solver_backend",
            "must be sympy or llm_python",
            gold_solver_backend,
        )
    if (
        gold_solver_backend == "llm_python"
        and not bool(_get(config, "evaluator_pipeline.enabled"))
    ):
        raise ConfigurationError(
            "evaluator_pipeline.enabled",
            "must be true when generation.gold_solver_backend is llm_python",
            False,
        )
    for path in (
        "generation.generator_max_retry",
        "generation.max_quota_repair_rounds",
        "generation.minimum_verified_questions",
        "generation.truth_solver_max_retry",
        "generation.subcategory_failure_cooldown_threshold",
        "generation.subcategory_cooldown_iterations",
        "generation.gold_validation_attempts",
        "generation.gold_validation_max_tokens",
        "models.evaluator.max_retries",
        "models.test_taker.max_retries",
        "evaluator_pipeline.max_parallel_questions",
    ):
        value = _get(config, path)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(path, "must be a positive integer", value)
    for path in (
        "generation.minimum_difficulty",
        "generation.maximum_difficulty",
        "generation.maximum_reasoning_steps",
        "evaluator_pipeline.solver_max_tokens",
        "evaluator_pipeline.independent_solver_max_tokens",
        "evaluator_pipeline.postcheck_max_tokens",
        "evaluator_pipeline.semantic_judge_max_tokens",
        "evaluator_pipeline.code_generation_attempts",
        "evaluator_pipeline.independent_solver_attempts",
        "evaluator_pipeline.postcheck_attempts",
        "evaluator_pipeline.semantic_judge_attempts",
        "evaluator_pipeline.max_code_chars",
        "evaluator_pipeline.max_ast_nodes",
        "evaluator_pipeline.max_integer_literal",
        "evaluator_pipeline.max_output_chars",
        "evaluator_pipeline.max_answer_chars",
        "dataset.text_dedup_num_perm",
        "dataset.text_dedup_ngram_size",
        "dataset.min_gold_reasoning_steps",
        "dataset.max_gold_reasoning_steps",
        "dataset.min_gold_reasoning_chars_per_step",
        "dataset.max_gold_reasoning_chars_per_step",
        "dataset.sentence_transformers_batch_size",
        "adaptive_sampling.initial_difficulty",
        "adaptive_sampling.global_accuracy_min_observations",
        "adaptive_sampling.global_difficulty_step",
        "adaptive_sampling.max_difficulty_change_per_iteration",
        "evaluator_pipeline.minimum_difficulty",
        "evaluator_pipeline.maximum_difficulty",
        "training_mix.minimum_samples",
    ):
        value = _get(config, path)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(path, "must be a positive integer", value)
    text_dedup_seed = _get(config, "dataset.text_dedup_seed")
    if (
        not isinstance(text_dedup_seed, int)
        or isinstance(text_dedup_seed, bool)
        or text_dedup_seed < 0
    ):
        raise ConfigurationError(
            "dataset.text_dedup_seed",
            "must be a non-negative integer",
            text_dedup_seed,
        )
    generation_difficulty = (
        int(_get(config, "generation.minimum_difficulty")),
        int(_get(config, "generation.maximum_difficulty")),
    )
    evaluator_difficulty = (
        int(_get(config, "evaluator_pipeline.minimum_difficulty")),
        int(_get(config, "evaluator_pipeline.maximum_difficulty")),
    )
    for path, (minimum, maximum) in (
        ("generation", generation_difficulty),
        ("evaluator_pipeline", evaluator_difficulty),
    ):
        if not 1 <= minimum <= maximum <= 10:
            raise ConfigurationError(
                f"{path}.minimum_difficulty",
                "difficulty bounds must satisfy 1 <= minimum <= maximum <= 10",
                [minimum, maximum],
            )
    validation_timeout = _get(
        config,
        "generation.truth_solver_timeout_seconds",
    )
    if (
        not isinstance(validation_timeout, (int, float))
        or isinstance(validation_timeout, bool)
        or validation_timeout <= 0
    ):
        raise ConfigurationError(
            "generation.truth_solver_timeout_seconds",
            "must be a positive number",
            validation_timeout,
        )
    for path in (
        "models.evaluator.request_timeout_seconds",
        "models.test_taker.request_timeout_seconds",
    ):
        request_timeout = _get(config, path)
        if (
            not isinstance(request_timeout, (int, float))
            or isinstance(request_timeout, bool)
            or request_timeout <= 0
        ):
            raise ConfigurationError(
                path,
                "must be a positive number",
                request_timeout,
            )
    evaluator_retry_delay = _get(
        config,
        "models.evaluator.retry_delay_seconds",
    )
    if (
        not isinstance(evaluator_retry_delay, (int, float))
        or isinstance(evaluator_retry_delay, bool)
        or evaluator_retry_delay < 0
    ):
        raise ConfigurationError(
            "models.evaluator.retry_delay_seconds",
            "must be a non-negative number",
            evaluator_retry_delay,
        )
    generation_temperature = _get(config, "generation.temperature")
    generation_top_p = _get(config, "generation.top_p")
    for path, value in (
        ("generation.temperature", generation_temperature),
        ("generation.top_p", generation_top_p),
    ):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfigurationError(
                path,
                "must be a numeric sampling value",
                value,
            )
    if float(generation_temperature) != 0.0:
        raise ConfigurationError(
            "generation.temperature",
            "math generation temperature must be 0.0",
            generation_temperature,
        )
    if float(generation_top_p) != 0.1:
        raise ConfigurationError(
            "generation.top_p",
            "math generation top_p must be 0.1",
            generation_top_p,
        )
    for path in (
        "generation.allow_partial_question_budget",
        "generation.require_gold_answer_validation",
        "generation.prohibit_competition_level",
        "evaluator_pipeline.enabled",
        "evaluator_pipeline.require_runtime_verification",
        "evaluator_pipeline.require_semantic_judge",
        "answer_normalization.math_verify_enabled",
        "dataset.holdout_exact_template_rejection",
        "dataset.require_gold_reasoning_steps",
        "dataset.require_concrete_gold_reasoning",
        "dataset.require_gold_reasoning_verification",
        "dataset.require_gold_answer_in_reasoning",
        "dataset.text_dedup_enabled",
        "dataset.datasketch_enabled",
        "dataset.datasketch_required",
        "dataset.datasketch_lsh_enabled",
        "dataset.sentence_transformers_enabled",
        "dataset.sentence_transformers_required",
        "dataset.sentence_transformers_local_files_only",
        "structured_output.enabled",
        "structured_output.required",
        "structured_output.use_for_evaluator",
        "structured_output.use_for_test_taker",
        "tracking.wandb.enabled",
        "tracking.wandb.log_model",
        "adaptive_sampling.enabled",
        "adaptive_sampling.global_accuracy_enabled",
        "training_mix.strict_correct_incorrect_ratio",
        "fixed_test.enabled",
        "fixed_test.require_all_subcategories",
        "fixed_test.evaluate_baseline",
        "fixed_test.evaluate_after_each_training_cycle",
        "fixed_test.fail_on_training_leakage",
    ):
        if not isinstance(_get(config, path), bool):
            raise ConfigurationError(path, "must be a Boolean")
    minimum_verified = _get(
        config,
        "generation.minimum_verified_questions",
    )
    if minimum_verified > budget:
        raise ConfigurationError(
            "generation.minimum_verified_questions",
            "must not exceed experiment.questions_per_iteration",
            minimum_verified,
        )
    reasoning_min = int(_get(config, "dataset.min_gold_reasoning_steps"))
    reasoning_max = int(_get(config, "dataset.max_gold_reasoning_steps"))
    if reasoning_min > reasoning_max:
        raise ConfigurationError(
            "dataset.min_gold_reasoning_steps",
            "must not exceed dataset.max_gold_reasoning_steps",
            reasoning_min,
        )
    test_taker_tools = _get(config, "models.test_taker.use_external_tools")
    if not isinstance(test_taker_tools, bool):
        raise ConfigurationError(
            "models.test_taker.use_external_tools",
            "must be a Boolean",
            test_taker_tools,
        )
    if test_taker_tools:
        raise ConfigurationError(
            "models.test_taker.use_external_tools",
            "test-taker tools must be disabled",
            True,
        )
    if not isinstance(_get(config, "models.evaluator.use_privileged_tools"), bool):
        raise ConfigurationError(
            "models.evaluator.use_privileged_tools",
            "must be a Boolean",
        )
    _validate_ratio_group(
        config,
        "generation_mix",
        ["hard_pool_variants", "coverage_deficit", "retention_known"],
    )
    _validate_ratio_group(
        config,
        "training_mix",
        [
            "incorrect_boundary_samples",
            "correct_retention_samples",
            "coverage_repair_samples",
            "format_instruction_samples",
        ],
    )
    correct_fraction = float(
        _get(config, "training_mix.correct_retention_samples")
    )
    if abs(correct_fraction - 0.25) > 1e-9:
        raise ConfigurationError(
            "training_mix.correct_retention_samples",
            "must be exactly 0.25 for the fixed 25% correct / 75% wrong mix",
            correct_fraction,
        )
    if _get(config, "training_mix.scope") != "current_cycle":
        raise ConfigurationError(
            "training_mix.scope",
            "must be current_cycle to prevent prior-cycle sampling drift",
            _get(config, "training_mix.scope"),
        )
    _validate_ratio_group(
        config,
        "adaptive_sampling",
        [
            "boundary_weight",
            "coverage_weight",
            "uncertainty_weight",
            "persistent_error_weight",
            "retention_weight",
        ],
    )
    accuracy_low = float(_get(config, "adaptive_sampling.target_accuracy_low"))
    accuracy_mid = float(_get(config, "adaptive_sampling.target_accuracy_mid"))
    accuracy_high = float(_get(config, "adaptive_sampling.target_accuracy_high"))
    if not 0 <= accuracy_low <= accuracy_mid <= accuracy_high <= 1:
        raise ConfigurationError(
            "adaptive_sampling.target_accuracy_low",
            "target accuracies must satisfy 0 <= low <= mid <= high <= 1",
            [accuracy_low, accuracy_mid, accuracy_high],
        )
    global_accuracy_low = float(
        _get(config, "adaptive_sampling.global_accuracy_low")
    )
    global_accuracy_high = float(
        _get(config, "adaptive_sampling.global_accuracy_high")
    )
    if not 0 <= global_accuracy_low <= global_accuracy_high <= 1:
        raise ConfigurationError(
            "adaptive_sampling.global_accuracy_low",
            "global accuracy bounds must satisfy 0 <= low <= high <= 1",
            [global_accuracy_low, global_accuracy_high],
        )
    initial_difficulty = int(
        _get(config, "adaptive_sampling.initial_difficulty")
    )
    if not generation_difficulty[0] <= initial_difficulty <= generation_difficulty[1]:
        raise ConfigurationError(
            "adaptive_sampling.initial_difficulty",
            "must be within the configured generation difficulty bounds",
            initial_difficulty,
        )
    for path in (
        "answer_normalization.absolute_tolerance",
        "answer_normalization.relative_tolerance",
        "dataset.near_duplicate_threshold",
        "dataset.semantic_similarity_threshold",
        "dataset.evaluator_confidence_threshold",
        "dataset.holdout_near_duplicate_threshold",
        "dataset.holdout_semantic_similarity_threshold",
        "dataset.text_dedup_similarity_threshold",
        "dataset.holdout_text_dedup_similarity_threshold",
        "error_attribution.confidence_threshold",
        "evaluator_pipeline.semantic_judge_confidence_threshold",
    ):
        value = _get(config, path)
        if not isinstance(value, (int, float)) or float(value) < 0:
            raise ConfigurationError(path, "threshold must be non-negative", value)
        if path.startswith(
            ("dataset.", "error_attribution.", "evaluator_pipeline.")
        ) and float(value) > 1:
            raise ConfigurationError(path, "threshold must not exceed 1", value)
    for path in (
        "evaluator_pipeline.temperature",
        "evaluator_pipeline.code_execution_timeout_seconds",
    ):
        value = _get(config, path)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or (
                float(value) < 0
                if path.endswith("temperature")
                else float(value) <= 0
            )
        ):
            raise ConfigurationError(
                path,
                (
                    "must be a non-negative number"
                    if path.endswith("temperature")
                    else "must be a positive number"
                ),
                value,
            )
    if evaluator_difficulty != generation_difficulty:
        raise ConfigurationError(
            "evaluator_pipeline.minimum_difficulty",
            "evaluator difficulty bounds must match generation bounds",
            list(evaluator_difficulty),
        )
    for path in (
        "evaluator_pipeline.solver_prompt_path",
        "evaluator_pipeline.solver_strategy_path",
        "evaluator_pipeline.independent_solver_prompt_path",
        "evaluator_pipeline.postcheck_prompt_path",
        "evaluator_pipeline.semantic_judge_prompt_path",
        "fixed_test.dataset_path",
    ):
        if not str(_get(config, path)).strip():
            raise ConfigurationError(path, "must be a non-empty path")
    for path in (
        "dataset.sentence_transformers_model",
        "dataset.sentence_transformers_device",
    ):
        if not str(_get(config, path)).strip():
            raise ConfigurationError(path, "must be a non-empty string")
    if _get(
        config,
        "dataset.sentence_transformers_required",
    ) and not _get(config, "dataset.sentence_transformers_enabled"):
        raise ConfigurationError(
            "dataset.sentence_transformers_required",
            "cannot be true when sentence_transformers_enabled is false",
            True,
        )
    if _get(
        config,
        "dataset.datasketch_required",
    ) and not _get(config, "dataset.datasketch_enabled"):
        raise ConfigurationError(
            "dataset.datasketch_required",
            "cannot be true when datasketch_enabled is false",
            True,
        )
    if (
        mode == "data_flywheel"
        and _get(config, "fixed_test.enabled")
        and _get(config, "fixed_test.fail_on_training_leakage")
        and not _get(config, "dataset.sentence_transformers_required")
    ):
        raise ConfigurationError(
            "dataset.sentence_transformers_required",
            "must be true for fail-closed holdout leakage protection in "
            "data_flywheel mode",
            False,
        )
    if (
        mode == "data_flywheel"
        and _get(config, "dataset.text_dedup_enabled")
        and not _get(config, "dataset.datasketch_required")
    ):
        raise ConfigurationError(
            "dataset.datasketch_required",
            "must be true for the production data-flywheel MinHash backend",
            False,
        )
    structured_backends = {"none", "outlines", "guidance"}
    for path in (
        "structured_output.local_backend",
        "structured_output.fallback_backend",
    ):
        value = str(_get(config, path)).strip().lower()
        if value not in structured_backends:
            raise ConfigurationError(
                path,
                "must be one of: none, outlines, guidance",
                value,
            )
    if (
        _get(config, "structured_output.required")
        and not _get(config, "structured_output.enabled")
    ):
        raise ConfigurationError(
            "structured_output.required",
            "cannot be true when structured_output.enabled is false",
            True,
        )
    wandb_mode = str(_get(config, "tracking.wandb.mode")).strip().lower()
    if wandb_mode not in {"online", "offline", "disabled"}:
        raise ConfigurationError(
            "tracking.wandb.mode",
            "must be one of: online, offline, disabled",
            wandb_mode,
        )
    if not str(_get(config, "tracking.wandb.project")).strip():
        raise ConfigurationError(
            "tracking.wandb.project",
            "must be a non-empty string",
        )
    wandb_tags = _get(config, "tracking.wandb.tags")
    if (
        not isinstance(wandb_tags, list)
        or any(not str(tag).strip() for tag in wandb_tags)
    ):
        raise ConfigurationError(
            "tracking.wandb.tags",
            "must be a list of non-empty strings",
            wandb_tags,
        )
    for path in ("finetune.epochs", "finetune.batch_size", "finetune.lora_rank"):
        value = _get(config, path)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ConfigurationError(path, "must be a positive integer", value)
    taxonomy = _get(config, "taxonomy")
    if not isinstance(taxonomy, Mapping) or not taxonomy:
        raise ConfigurationError("taxonomy", "must define at least one category")
    if tuple(taxonomy) != FIXED_MATH_CATEGORIES:
        raise ConfigurationError(
            "taxonomy",
            "must contain the fixed nine math categories in canonical order",
            list(taxonomy),
        )
    if sum(len(subcategories) for subcategories in taxonomy.values()) != 27:
        raise ConfigurationError(
            "taxonomy",
            "must contain exactly 27 subcategories",
        )
    for category, subcategories in taxonomy.items():
        if not isinstance(subcategories, Mapping) or not subcategories:
            raise ConfigurationError(
                f"taxonomy.{category}",
                "must define at least one subcategory",
            )
        for subcategory, metadata in subcategories.items():
            if not isinstance(metadata, Mapping):
                raise ConfigurationError(
                    f"taxonomy.{category}.{subcategory}",
                    "must be an object",
                )
            unknown_metadata = set(metadata) - {"min_quota", "base_weight"}
            if unknown_metadata:
                unknown = sorted(unknown_metadata)[0]
                raise ConfigurationError(
                    f"taxonomy.{category}.{subcategory}.{unknown}",
                    "unknown taxonomy metadata field",
                    metadata[unknown],
                )
            min_quota = metadata.get("min_quota", quota)
            if not isinstance(min_quota, int) or min_quota <= 0:
                raise ConfigurationError(
                    f"taxonomy.{category}.{subcategory}.min_quota",
                    "must be a positive integer",
                    min_quota,
                )
    if validate_paths:
        model_path = str(_get(config, "models.test_taker.model_path"))
        drive, _ = os.path.splitdrive(model_path)
        looks_like_path = (
            bool(drive)
            or model_path.startswith(("/", "./", "../"))
            or "/" in model_path
            or "\\" in model_path
        )
        if looks_like_path and not Path(model_path).expanduser().exists():
            raise ConfigurationError(
                "models.test_taker.model_path",
                "local model path does not exist",
                model_path,
            )
        output_root = Path(str(_get(config, "paths.output_root"))).expanduser()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            probe = output_root / ".autobencher_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise ConfigurationError(
                "paths.output_root",
                f"directory is not writable: {exc}",
                str(output_root),
            ) from exc
        project_root = Path(__file__).resolve().parents[1]
        for path in (
            "evaluator_pipeline.solver_prompt_path",
            "evaluator_pipeline.solver_strategy_path",
            "evaluator_pipeline.independent_solver_prompt_path",
            "evaluator_pipeline.postcheck_prompt_path",
            "evaluator_pipeline.semantic_judge_prompt_path",
            "fixed_test.dataset_path",
        ):
            configured = Path(str(_get(config, path))).expanduser()
            candidates = (
                [configured]
                if configured.is_absolute()
                else [Path.cwd() / configured, project_root / configured]
            )
            if not any(candidate.is_file() for candidate in candidates):
                raise ConfigurationError(
                    path,
                    "configured file does not exist",
                    str(configured),
                )


def config_hash(config: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_resolved_config(
    config_path: str | os.PathLike[str] | None,
    environment_path: str | os.PathLike[str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    temporary_overrides: Iterable[str] | None = None,
    validate_paths: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_layers: list[
        tuple[str, dict[str, Any], list[dict[str, str]]]
    ] = []
    if config_path:
        config_layers = _load_yaml_layers(Path(config_path))
    environment_layers: list[
        tuple[str, dict[str, Any], list[dict[str, str]]]
    ] = []
    if environment_path:
        environment_layers = _load_yaml_layers(Path(environment_path))

    resolved = copy.deepcopy(SAFE_DEFAULTS)
    field_sources: dict[str, dict[str, Any]] = {}
    _record_sources(field_sources, SAFE_DEFAULTS, "schema_defaults")
    migrations: list[dict[str, str]] = []

    # Parent/base configuration is lower priority than the environment. The
    # leaf experiment remains higher priority than the environment profile.
    base_layers = config_layers[:-1] if config_layers else []
    experiment_layers = config_layers[-1:] if config_layers else []
    ordered_layers = base_layers + environment_layers + experiment_layers
    for source, overlay, layer_migrations in ordered_layers:
        resolved = deep_merge(resolved, overlay)
        _record_sources(field_sources, overlay, source)
        migrations.extend(layer_migrations)

    cli_overlay = copy.deepcopy(dict(cli_overrides or {}))
    resolved = deep_merge(resolved, cli_overlay)
    _record_sources(field_sources, cli_overlay, "cli")
    parsed_temporary = parse_overrides(temporary_overrides)
    resolved = deep_merge(resolved, parsed_temporary)
    _record_sources(field_sources, parsed_temporary, "explicit_override")
    validate_config(resolved, validate_paths=validate_paths)
    digest = config_hash(resolved)
    sensitive_environment = {}
    for logical_name, variable_name in resolved["sensitive_environment"].items():
        sensitive_environment[logical_name] = {
            "environment_variable": variable_name,
            "present": bool(os.environ.get(str(variable_name))),
            "value": "***" if os.environ.get(str(variable_name)) else None,
            "source": "environment",
        }
    provenance = {
        "schema_version": str(resolved.get("schema_version", "1.0")),
        "config_hash": digest,
        "field_sources": field_sources,
        "sensitive_environment": sensitive_environment,
        "migrations": migrations,
        "validation": {
            "status": "passed",
            "validate_paths": bool(validate_paths),
            "unknown_fields_rejected": True,
            "ratios_validated": True,
            "cross_field_rules_validated": True,
        },
        "sources": {
            "safe_defaults": True,
            "config_layers": [
                source
                for source, _, _ in config_layers
            ],
            "environment_layers": [
                source
                for source, _, _ in environment_layers
            ],
            "cli_overrides": cli_overlay,
            "temporary_overrides": list(temporary_overrides or []),
        },
    }
    return resolved, provenance


def load_project_config(
    config_path: str | os.PathLike[str] | None,
    environment_path: str | os.PathLike[str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    temporary_overrides: Iterable[str] | None = None,
    validate_paths: bool = False,
) -> tuple[ProjectConfig, dict[str, Any]]:
    """Load the single immutable runtime configuration used by entry points."""
    resolved, provenance = load_resolved_config(
        config_path,
        environment_path=environment_path,
        cli_overrides=cli_overrides,
        temporary_overrides=temporary_overrides,
        validate_paths=validate_paths,
    )
    return ProjectConfig(resolved), provenance
