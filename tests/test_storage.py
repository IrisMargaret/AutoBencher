import os
import tempfile
from pathlib import Path

from autobencher.config import load_resolved_config
from autobencher.storage import configure_runtime_storage


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_caches_and_temp_are_routed_under_output_root(
    tmp_path,
):
    config, _ = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml",
        temporary_overrides=[f"paths.output_root={tmp_path.as_posix()}"],
    )
    previous_tempdir = tempfile.tempdir
    names = (
        "TMPDIR",
        "TEMP",
        "TMP",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "WANDB_DIR",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "PYTHONPYCACHEPREFIX",
    )
    previous_environment = {
        name: os.environ.get(name)
        for name in names
    }
    try:
        resolved = configure_runtime_storage(config)
        output_root = Path(resolved["output_root"])
        for name in names:
            configured = Path(os.environ[name]).resolve()
            assert configured.is_relative_to(output_root)
            assert configured.is_dir()
    finally:
        tempfile.tempdir = previous_tempdir
        for name, previous in previous_environment.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
