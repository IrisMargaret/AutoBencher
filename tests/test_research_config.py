from pathlib import Path

import pytest

from autobencher.config import (
    ConfigurationError,
    config_hash,
    load_project_config,
    load_resolved_config,
    str2bool,
)
from run_scripts import (
    _resolved_execution_mode,
    _strip_implicit_config_options,
    build_command,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "math_flywheel.yaml"
SMOKE_CONFIG = ROOT / "configs" / "math_flywheel_smoke_test.yaml"
EXPERIMENT_CONFIG = ROOT / "configs" / "experiments" / "math_flywheel.yaml"
QUICK_FLYWHEEL_CONFIG = (
    ROOT / "configs" / "experiments" / "quick_flywheel_27.yaml"
)
MINI_FLYWHEEL_CONFIG = (
    ROOT / "configs" / "experiments" / "mini_flywheel_8.yaml"
)
VOLCENGINE_CONFIG = ROOT / "configs" / "environments" / "volcengine.yaml"
STUDY_ROOT = ROOT / "configs" / "studies"
STUDY_CONFIGS = {
    "base": STUDY_ROOT / "base.yaml",
    "random": STUDY_ROOT / "random.yaml",
    "uniform": STUDY_ROOT / "uniform.yaml",
    "error_only": STUDY_ROOT / "error_only.yaml",
    "full": STUDY_ROOT / "full.yaml",
    "full_no_hard_pool": (
        STUDY_ROOT / "ablations" / "full_no_hard_pool.yaml"
    ),
    "full_no_observed_difficulty": (
        STUDY_ROOT / "ablations" / "full_no_observed_difficulty.yaml"
    ),
}


def test_default_config_loads_and_contains_nine_categories():
    config, provenance = load_resolved_config(DEFAULT_CONFIG)
    assert len(config["taxonomy"]) == 9
    assert provenance["config_hash"] == config_hash(config)
    assert config["difficulty"]["rubric_version"] == "observable_math_v1"
    assert sum(config["difficulty"]["dimension_weights"].values()) == (
        pytest.approx(1.0)
    )


def test_yaml_inheritance_overrides_parent():
    config, _ = load_resolved_config(SMOKE_CONFIG)
    assert config["experiment"]["questions_per_iteration"] == 27
    assert config["finetune"]["enabled"] is False
    assert config["models"]["evaluator"]["model_name"] == "deepseek-v4-pro"


def test_quick_flywheel_profile_runs_one_complete_27_question_cycle():
    config, _ = load_resolved_config(QUICK_FLYWHEEL_CONFIG)
    assert config["experiment"]["mode"] == "data_flywheel"
    assert config["experiment"]["questions_per_iteration"] == 27
    assert config["experiment"]["num_iterations"] == 1
    assert config["experiment"]["max_cycles"] == 1
    assert config["finetune"]["enabled"] is True
    assert config["finetune"]["epochs"] == 1
    assert config["fixed_test"]["evaluate_baseline"] is True
    assert config["fixed_test"]["evaluate_after_each_training_cycle"] is True
    assert config["dataset"]["datasketch_required"] is True
    assert config["dataset"]["sentence_transformers_required"] is True
    assert config["evaluator_pipeline"]["max_parallel_questions"] == 4
    assert (
        config["difficulty"]["reject_outside_generation_bounds"]
        is True
    )


def test_mini_flywheel_profile_runs_the_full_chain_with_sympy_gold():
    config, _ = load_resolved_config(MINI_FLYWHEEL_CONFIG)
    assert config["experiment"]["questions_per_iteration"] == 8
    assert config["generation"]["max_questions_per_prompt"] == 1
    assert config["generation"]["gold_solver_backend"] == "sympy"
    assert config["evaluator_pipeline"]["enabled"] is False
    assert config["finetune"]["enabled"] is True
    assert config["finetune"]["epochs"] == 1
    assert config["training_mix"]["minimum_samples"] == 1
    assert config["dataset"]["allow_optional_backend_fallback"] is True
    assert config["dataset"]["datasketch_enabled"] is True
    assert config["dataset"]["datasketch_required"] is False
    assert config["dataset"]["sentence_transformers_enabled"] is False
    assert config["dataset"]["sentence_transformers_required"] is False
    assert config["fixed_test"]["evaluate_baseline"] is True
    assert config["fixed_test"]["evaluate_after_each_training_cycle"] is True
    assert (
        config["difficulty"]["reject_outside_generation_bounds"]
        is False
    )


def test_volcengine_writable_paths_are_confined_to_required_data_root():
    config, _ = load_resolved_config(
        EXPERIMENT_CONFIG,
        environment_path=VOLCENGINE_CONFIG,
    )
    root = config["paths"]["allowed_data_root"].rstrip("/")
    assert config["paths"]["enforce_data_root"] is True
    for field in (
        "output_root",
        "cache_dir",
        "model_output_dir",
        "log_dir",
        "temp_dir",
        "dataset_dir",
        "checkpoint_dir",
        "review_dir",
    ):
        assert config["paths"][field].startswith(root + "/") or (
            config["paths"][field] == root
        )


def test_volcengine_rejects_system_disk_output_override():
    with pytest.raises(ConfigurationError, match="allowed_data_root"):
        load_resolved_config(
            EXPERIMENT_CONFIG,
            environment_path=VOLCENGINE_CONFIG,
            temporary_overrides=["paths.output_root=/root/code/output"],
        )


def test_precedence_is_defaults_then_yaml_then_cli_then_temporary():
    config, _ = load_resolved_config(
        SMOKE_CONFIG,
        cli_overrides={"experiment": {"num_iterations": 3}},
        temporary_overrides=["experiment.num_iterations=4"],
    )
    assert config["experiment"]["num_iterations"] == 4


def test_environment_is_between_base_and_experiment_layers():
    config, provenance = load_resolved_config(
        EXPERIMENT_CONFIG,
        environment_path=VOLCENGINE_CONFIG,
    )
    expected = "/vepfs-mlp2/queue010/20262202597/Qwen2.5-7B-Instruct"
    assert config["models"]["test_taker"]["model_path"] == expected
    source = provenance["field_sources"]["models.test_taker.model_path"]
    assert source["source"].endswith("volcengine.yaml")


def test_unknown_override_field_fails_fast():
    with pytest.raises(ConfigurationError, match="unknown field"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["experiment.unknown_switch=true"],
        )


def test_runtime_project_config_is_recursively_immutable():
    config, _ = load_project_config(SMOKE_CONFIG)
    with pytest.raises(TypeError, match="immutable"):
        config["experiment"]["num_iterations"] = 99
    assert config.experiment.num_iterations == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("false", False),
        ("0", False),
        ("no", False),
        ("true", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_str2bool_is_reliable(raw, expected):
    assert str2bool(raw) is expected


def test_invalid_boolean_fails_with_field_context():
    with pytest.raises(ConfigurationError, match="expected true/false"):
        str2bool("maybe")


def test_invalid_generation_ratio_fails_fast():
    with pytest.raises(ConfigurationError, match="generation_mix"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["generation_mix.hard_pool_variants=0.99"],
        )


def test_math_generator_sampling_contract_is_fixed():
    with pytest.raises(
        ConfigurationError,
        match="generation.temperature",
    ):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["generation.temperature=0.2"],
        )
    with pytest.raises(ConfigurationError, match="generation.top_p"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["generation.top_p=0.9"],
        )


@pytest.mark.parametrize(
    "override",
    [
        "models.evaluator.request_timeout_seconds=0",
        "models.test_taker.request_timeout_seconds=-1",
        "models.evaluator.max_retries=0",
        "models.test_taker.max_retries=0",
        "models.evaluator.retry_delay_seconds=-0.1",
        "evaluator_pipeline.max_parallel_questions=0",
    ],
)
def test_invalid_model_request_controls_fail_fast(override):
    with pytest.raises(ConfigurationError):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=[override],
        )


def test_test_taker_tools_cannot_be_enabled():
    with pytest.raises(ConfigurationError, match="use_external_tools"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["models.test_taker.use_external_tools=true"],
        )


def test_global_accuracy_bounds_must_be_ordered():
    with pytest.raises(ConfigurationError, match="global accuracy bounds"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=[
                "adaptive_sampling.global_accuracy_low=0.8",
                "adaptive_sampling.global_accuracy_high=0.4",
            ],
        )


def test_difficulty_weights_must_sum_to_one():
    with pytest.raises(ConfigurationError, match="dimension_weights"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=[
                "difficulty.dimension_weights.reasoning_steps=0.50"
            ],
        )


def test_difficulty_mismatch_action_is_validated():
    with pytest.raises(ConfigurationError, match="mismatch_action"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["difficulty.mismatch_action=guess"],
        )


def test_data_flywheel_requires_semantic_leakage_backend():
    with pytest.raises(
        ConfigurationError,
        match="fail-closed holdout leakage protection",
    ):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["experiment.mode=data_flywheel"],
        )


def test_config_hash_is_deterministic():
    left, _ = load_resolved_config(SMOKE_CONFIG)
    right, _ = load_resolved_config(SMOKE_CONFIG)
    assert config_hash(left) == config_hash(right)


def test_launcher_config_does_not_leak_implicit_defaults():
    command = build_command(
        "math",
        "implicit-model",
        8,
        config="configs/math_flywheel_smoke_test.yaml",
    )
    stripped = _strip_implicit_config_options(
        command,
        ["math", "--config", "configs/math_flywheel_smoke_test.yaml"],
    )
    assert "--config" in stripped
    assert "--agent_modelname" not in stripped
    assert "--mode" not in stripped


def test_launcher_explicit_cli_still_overrides_yaml():
    command = build_command(
        "math",
        "implicit-model",
        8,
        agent_modelname="explicit-agent",
        config="configs/math_flywheel_smoke_test.yaml",
    )
    stripped = _strip_implicit_config_options(
        command,
        [
            "math",
            "--config",
            "configs/math_flywheel_smoke_test.yaml",
            "--agent_modelname",
            "explicit-agent",
        ],
    )
    assert stripped[stripped.index("--agent_modelname") + 1] == "explicit-agent"


def test_launcher_forwards_preflight_without_creating_legacy_output():
    command = build_command(
        "math",
        "implicit-model",
        1,
        config="configs/experiments/mini_flywheel_8.yaml",
        preflight_only=True,
    )
    assert "--preflight-only" in command
    assert "--outfile_prefix1" not in command


def test_launcher_announces_data_flywheel_mode_from_yaml():
    args = type(
        "Args",
        (),
        {
            "execution_mode": "eval",
            "config": str(MINI_FLYWHEEL_CONFIG),
            "environment": None,
            "override": [],
        },
    )()

    mode, source = _resolved_execution_mode(args, ["--config", args.config])

    assert mode == "data_flywheel"
    assert source == "config"


@pytest.mark.parametrize(
    ("variant", "policy"),
    [
        ("base", "base"),
        ("random", "random"),
        ("uniform", "uniform"),
        ("error_only", "error_only"),
        ("full", "full"),
        ("full_no_hard_pool", "full"),
        ("full_no_observed_difficulty", "full"),
    ],
)
def test_study_profiles_resolve_strict_runtime_identity(variant, policy):
    config, _ = load_resolved_config(STUDY_CONFIGS[variant])
    assert config["study"]["policy"] == policy
    assert config["study"]["variant"] == variant
    if variant == "base":
        assert config["experiment"]["mode"] == "eval"
        assert config["finetune"]["enabled"] is False


@pytest.mark.parametrize(
    ("config_path", "override", "message"),
    [
        (
            STUDY_CONFIGS["full"],
            "study.policy=unknown",
            "study.policy",
        ),
        (
            STUDY_CONFIGS["full"],
            "study.variant=random",
            "requires study.policy",
        ),
        (
            STUDY_CONFIGS["uniform"],
            "study.uniform_difficulty=10",
            "uniform_difficulty",
        ),
        (
            STUDY_CONFIGS["error_only"],
            "study.error_only_epsilon=0",
            "error_only_epsilon",
        ),
        (
            STUDY_CONFIGS["full_no_hard_pool"],
            "study.components.error_type_targeting=true",
            "error_type_targeting",
        ),
        (
            STUDY_CONFIGS["base"],
            "finetune.enabled=true",
            "finetune.enabled",
        ),
        (
            STUDY_CONFIGS["full"],
            "experiment.seed=-1",
            "experiment.seed",
        ),
    ],
)
def test_invalid_study_configuration_fails_closed(
    config_path,
    override,
    message,
):
    with pytest.raises(ConfigurationError, match=message):
        load_resolved_config(
            config_path,
            temporary_overrides=[override],
        )
