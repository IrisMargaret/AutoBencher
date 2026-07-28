from pathlib import Path

import pytest

from autobencher.config import (
    ConfigurationError,
    config_hash,
    load_project_config,
    load_resolved_config,
    str2bool,
)
from run_scripts import _strip_implicit_config_options, build_command


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "math_flywheel.yaml"
SMOKE_CONFIG = ROOT / "configs" / "math_flywheel_smoke_test.yaml"
EXPERIMENT_CONFIG = ROOT / "configs" / "experiments" / "math_flywheel.yaml"
VOLCENGINE_CONFIG = ROOT / "configs" / "environments" / "volcengine.yaml"


def test_default_config_loads_and_contains_nine_categories():
    config, provenance = load_resolved_config(DEFAULT_CONFIG)
    assert len(config["taxonomy"]) == 9
    assert provenance["config_hash"] == config_hash(config)


def test_yaml_inheritance_overrides_parent():
    config, _ = load_resolved_config(SMOKE_CONFIG)
    assert config["experiment"]["questions_per_iteration"] == 27
    assert config["finetune"]["enabled"] is False
    assert config["models"]["evaluator"]["model_name"] == "deepseek-v4-pro"


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


def test_test_taker_tools_cannot_be_enabled():
    with pytest.raises(ConfigurationError, match="use_external_tools"):
        load_resolved_config(
            SMOKE_CONFIG,
            temporary_overrides=["models.test_taker.use_external_tools=true"],
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
