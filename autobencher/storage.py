"""Runtime storage isolation for generated data, caches, and temporary files."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


_DIRECTORY_DEFAULTS = {
    "cache_dir": "cache",
    "model_output_dir": "models",
    "log_dir": "logs",
    "temp_dir": "temp",
    "dataset_dir": "datasets",
    "checkpoint_dir": "checkpoints",
    "review_dir": "review",
}


def resolved_storage_paths(
    config: Mapping[str, Any],
) -> dict[str, Path]:
    """Resolve every writable runtime directory beneath the output root."""
    paths = config["paths"]
    output_root = Path(str(paths["output_root"])).expanduser().resolve()
    resolved = {"output_root": output_root}
    for field, default_name in _DIRECTORY_DEFAULTS.items():
        configured = paths.get(field)
        candidate = (
            Path(str(configured)).expanduser()
            if configured
            else output_root / default_name
        )
        resolved[field] = candidate.resolve()
    if bool(paths.get("enforce_data_root")):
        allowed = Path(
            str(paths["allowed_data_root"])
        ).expanduser().resolve()
        for field, candidate in resolved.items():
            if not candidate.is_relative_to(allowed):
                raise RuntimeError(
                    f"paths.{field} escapes allowed data root {allowed}"
                )
    return resolved


def configure_runtime_storage(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Route libraries and subprocesses to the configured data filesystem."""
    paths = resolved_storage_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    cache_dir = paths["cache_dir"]
    temp_dir = paths["temp_dir"]
    log_dir = paths["log_dir"]
    environment = {
        "TMPDIR": temp_dir,
        "TMP": temp_dir,
        "TEMP": temp_dir,
        "HF_HOME": cache_dir / "huggingface",
        "HF_DATASETS_CACHE": cache_dir / "huggingface" / "datasets",
        "TRANSFORMERS_CACHE": cache_dir / "huggingface" / "transformers",
        "TORCH_HOME": cache_dir / "torch",
        "XDG_CACHE_HOME": cache_dir / "xdg",
        "WANDB_DIR": log_dir / "wandb",
        "MPLCONFIGDIR": cache_dir / "matplotlib",
        "NUMBA_CACHE_DIR": cache_dir / "numba",
        "PYTHONPYCACHEPREFIX": cache_dir / "pycache",
    }
    for name, path in environment.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = os.fspath(path)

    # tempfile caches the selected directory after its first use.
    tempfile.tempdir = os.fspath(temp_dir)
    return {
        field: path.as_posix()
        for field, path in paths.items()
    }
