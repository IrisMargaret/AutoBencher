"""Build a reproducible, multi-source open mathematics holdout suite.

The builder deliberately writes no dataset into the source tree.  Hugging Face
downloads, transformed records, and manifests must all live below the configured
data root (normally the project VEPFS allocation).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import REQUIRED_DATA_ROOT
from .difficulty import analyze_difficulty
from .experiment import atomic_json
from .structured import normalize_answer_type


SOURCE_LICENSES = {
    "openai/gsm8k": {
        "license": "MIT",
        "upstream": "https://github.com/openai/grade-school-math",
    },
    "EleutherAI/hendrycks_math": {
        "license": "MIT",
        "upstream": "https://github.com/hendrycks/math",
    },
    "cais/mmlu": {
        "license": "MIT",
        "upstream": "https://github.com/hendrycks/test",
    },
    "google-deepmind/mathematics_dataset": {
        "license": "Apache-2.0",
        "upstream": (
            "https://github.com/google-deepmind/mathematics_dataset"
        ),
    },
}

MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

MMLU_MATH_CONFIGS = (
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
)

MATH_TAXONOMY = {
    "algebra": ("Algebra", "Polynomials and Inequalities"),
    "counting_and_probability": (
        "Probability & Statistics",
        "Combinatorics",
    ),
    "geometry": ("Geometry & Trigonometry", "Plane Geometry"),
    "intermediate_algebra": ("Algebra", "Systems of Equations"),
    "number_theory": ("Number Theory", "Divisibility and Factors"),
    "prealgebra": ("Arithmetic", "Fraction and Decimal Operations"),
    "precalculus": (
        "Geometry & Trigonometry",
        "Trigonometric Reasoning",
    ),
}

MMLU_TAXONOMY = {
    "abstract_algebra": ("Algebra", "Polynomials and Inequalities"),
    "college_mathematics": (
        "Composite Comprehensive",
        "Proof and Mathematical Reasoning",
    ),
    "elementary_mathematics": ("Arithmetic", "Integer Operations"),
    "high_school_mathematics": (
        "Composite Comprehensive",
        "Cross-Domain Multi-Step Problems",
    ),
    "high_school_statistics": (
        "Probability & Statistics",
        "Descriptive Statistics",
    ),
}

STRICT_TEST_SOURCE_TYPES = frozenset({"gsm8k", "math", "mmlu_math"})


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _require_test_split(source_type: str, split: str) -> str:
    normalized = str(split).strip().lower()
    if source_type in STRICT_TEST_SOURCE_TYPES and normalized != "test":
        raise ValueError(
            f"Fixed benchmark source {source_type!r} must use split='test'; "
            f"received {split!r}. Training and validation splits are forbidden."
        )
    return normalized


def _stable_select(
    records: Iterable[Mapping[str, Any]],
    count: int,
    seed: int,
    identity_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Select records deterministically without depending on source order."""
    keyed = []
    for record in records:
        identity = "\x1f".join(
            str(record.get(field, "")) for field in identity_fields
        )
        key = _sha256_text(f"{seed}\x1e{identity}")
        keyed.append((key, dict(record)))
    keyed.sort(key=lambda item: item[0])
    return [record for _, record in keyed[: max(0, int(count))]]


def _require_selected_count(
    source_name: str,
    records: list[dict[str, Any]],
    expected: int,
) -> None:
    if len(records) != int(expected):
        raise RuntimeError(
            f"{source_name} produced {len(records)} usable fixed-test records; "
            f"expected exactly {int(expected)}. Refusing a silently shrunken "
            "benchmark."
        )


def _boxed_answer(solution: Any) -> str | None:
    """Extract the last balanced ``\boxed{...}`` answer from MATH."""
    text = str(solution or "")
    positions = [
        match.end()
        for match in re.finditer(r"\\boxed\s*\{", text)
    ]
    for content_start in reversed(positions):
        depth = 1
        for index in range(content_start, len(text)):
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answer = text[content_start:index].strip()
                    return answer or None
    return None


def _gsm8k_answer(value: Any) -> str | None:
    matches = re.findall(r"####\s*([^\r\n]+)", str(value or ""))
    if not matches:
        return None
    return matches[-1].strip().replace(",", "")


def _gsm8k_taxonomy(question: str) -> tuple[str, str]:
    lowered = question.lower()
    if any(
        token in lowered
        for token in (
            "$",
            "cost",
            "price",
            "profit",
            "paid",
            "earn",
            "dollar",
            "budget",
        )
    ):
        return "Word Problems", "Financial Applications"
    if any(
        token in lowered
        for token in (
            "speed",
            "distance",
            "mile",
            "kilometer",
            "km",
            "per hour",
            "travel",
        )
    ):
        return "Word Problems", "Rate and Distance"
    if any(
        token in lowered
        for token in (
            "together",
            "worker",
            "work ",
            "fill",
            "mixture",
            "recipe",
        )
    ):
        return "Word Problems", "Work and Mixture"
    if "%" in question or "percent" in lowered or "ratio" in lowered:
        return "Arithmetic", "Ratio and Percentage"
    return (
        "Composite Comprehensive",
        "Cross-Domain Multi-Step Problems",
    )


def _difficulty_from_level(value: Any, default: int) -> int:
    match = re.search(r"(\d+)", str(value or ""))
    if not match:
        return default
    # MATH levels 1..5 are mapped to the project's 2..6 generation band.
    return max(1, min(10, int(match.group(1)) + 1))


def _canonical_record(
    *,
    source_dataset: str,
    source_config: str,
    source_split: str,
    source_index: int,
    question: str,
    answer: str,
    category: str,
    sub_category: str,
    difficulty: int,
    answer_type: str | None = None,
    source_solution: str | None = None,
    choices: list[str] | None = None,
) -> dict[str, Any]:
    source_key = (
        f"{source_dataset}:{source_config}:{source_split}:{source_index}:"
        f"{_sha256_text(question)[:16]}"
    )
    normalized_type = normalize_answer_type(
        answer_type or "auto",
        answer,
    )
    record = {
        "question_id": "open-" + _sha256_text(source_key)[:24],
        "category": category,
        "sub_category": sub_category,
        "difficulty": int(difficulty),
        "question": " ".join(str(question).split()),
        "answer_type": normalized_type,
        "canonical_answer": str(answer).strip(),
        "source_dataset": source_dataset,
        "source_config": source_config,
        "source_split": source_split,
        "source_index": int(source_index),
        "source_question_sha256": _sha256_text(
            " ".join(str(question).split())
        ),
        # Keep provenance without retaining chain-of-thought or worked
        # solutions in the runtime benchmark artifact.
        "source_solution_sha256": (
            _sha256_text(source_solution)
            if source_solution is not None
            else None
        ),
        "choices": list(choices or []),
    }
    profile = analyze_difficulty(
        record["question"],
        normalized_type,
    )
    profile.update(
        {
            "declared_source_score": int(difficulty),
            "effective_score": int(difficulty),
            "profile_role": (
                "cross_source_diagnostic_only; source score remains the "
                "fixed benchmark stratum"
            ),
        }
    )
    record.update(
        {
            "target_difficulty": int(difficulty),
            "observed_difficulty": int(profile["score"]),
            "difficulty_profile": profile,
        }
    )
    return record


def _load_hf_dataset(
    dataset_name: str,
    config_name: str,
    split: str,
    *,
    cache_dir: Path,
    revision: str | None,
    local_files_only: bool,
):
    try:
        from datasets import DownloadConfig, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to build the open benchmark"
        ) from exc
    download_config = DownloadConfig(
        cache_dir=str(cache_dir),
        local_files_only=bool(local_files_only),
    )
    kwargs = {
        "path": dataset_name,
        "name": config_name,
        "split": split,
        "cache_dir": str(cache_dir),
        "download_config": download_config,
    }
    if revision:
        kwargs["revision"] = revision
    return load_dataset(**kwargs)


def _build_gsm8k(
    source: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = _require_test_split(
        "gsm8k",
        str(source.get("split", "test")),
    )
    dataset = _load_hf_dataset(
        "openai/gsm8k",
        str(source.get("config", "main")),
        split,
        cache_dir=cache_dir,
        revision=source.get("revision"),
        local_files_only=local_files_only,
    )
    indexed = [
        {"source_index": index, **dict(record)}
        for index, record in enumerate(dataset)
    ]
    selected = _stable_select(
        indexed,
        int(source["samples"]),
        int(source["seed"]),
        ("question", "answer"),
    )
    output = []
    for record in selected:
        answer = _gsm8k_answer(record.get("answer"))
        if not answer:
            continue
        category, subcategory = _gsm8k_taxonomy(record["question"])
        output.append(
            _canonical_record(
                source_dataset="openai/gsm8k",
                source_config="main",
                source_split=split,
                source_index=record["source_index"],
                question=record["question"],
                answer=answer,
                category=category,
                sub_category=subcategory,
                difficulty=3,
                source_solution=str(record.get("answer", "")),
            )
        )
    _require_selected_count(
        "GSM8K",
        output,
        int(source["samples"]),
    )
    return output, {
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "available_count": len(dataset),
    }


def _build_math(
    source: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = _require_test_split(
        "math",
        str(source.get("split", "test")),
    )
    per_config = int(source["samples_per_config"])
    output = []
    fingerprints = {}
    available = 0
    for offset, config_name in enumerate(
        source.get("configs", MATH_CONFIGS)
    ):
        dataset = _load_hf_dataset(
            "EleutherAI/hendrycks_math",
            str(config_name),
            split,
            cache_dir=cache_dir,
            revision=source.get("revision"),
            local_files_only=local_files_only,
        )
        fingerprints[str(config_name)] = getattr(
            dataset,
            "_fingerprint",
            None,
        )
        available += len(dataset)
        indexed = [
            {"source_index": index, **dict(record)}
            for index, record in enumerate(dataset)
        ]
        selected = _stable_select(
            indexed,
            per_config,
            int(source["seed"]) + offset,
            ("problem", "solution"),
        )
        category, subcategory = MATH_TAXONOMY[str(config_name)]
        for record in selected:
            answer = _boxed_answer(record.get("solution"))
            if not answer:
                continue
            output.append(
                _canonical_record(
                    source_dataset="EleutherAI/hendrycks_math",
                    source_config=str(config_name),
                    source_split=split,
                    source_index=record["source_index"],
                    question=record["problem"],
                    answer=answer,
                    category=category,
                    sub_category=subcategory,
                    difficulty=_difficulty_from_level(
                        record.get("level"),
                        5,
                    ),
                    source_solution=str(record.get("solution", "")),
                )
            )
    _require_selected_count(
        "MATH",
        output,
        per_config * len(tuple(source.get("configs", MATH_CONFIGS))),
    )
    return output, {
        "dataset_fingerprints": fingerprints,
        "available_count": available,
    }


def _build_mmlu(
    source: Mapping[str, Any],
    cache_dir: Path,
    local_files_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = _require_test_split(
        "mmlu_math",
        str(source.get("split", "test")),
    )
    per_config = int(source["samples_per_config"])
    output = []
    fingerprints = {}
    available = 0
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for offset, config_name in enumerate(
        source.get("configs", MMLU_MATH_CONFIGS)
    ):
        dataset = _load_hf_dataset(
            "cais/mmlu",
            str(config_name),
            split,
            cache_dir=cache_dir,
            revision=source.get("revision"),
            local_files_only=local_files_only,
        )
        fingerprints[str(config_name)] = getattr(
            dataset,
            "_fingerprint",
            None,
        )
        available += len(dataset)
        indexed = [
            {"source_index": index, **dict(record)}
            for index, record in enumerate(dataset)
        ]
        selected = _stable_select(
            indexed,
            per_config,
            int(source["seed"]) + offset,
            ("question", "answer"),
        )
        category, subcategory = MMLU_TAXONOMY[str(config_name)]
        for record in selected:
            choices = [str(item) for item in record["choices"]]
            try:
                answer_index = int(record["answer"])
            except (TypeError, ValueError):
                answer_text = str(record["answer"]).strip().upper()
                answer_index = labels.index(answer_text)
            if not 0 <= answer_index < len(choices):
                continue
            rendered_choices = "\n".join(
                f"{labels[index]}. {choice}"
                for index, choice in enumerate(choices)
            )
            question = (
                f"{record['question']}\n\n{rendered_choices}\n"
                "Return the letter of the correct option."
            )
            output.append(
                _canonical_record(
                    source_dataset="cais/mmlu",
                    source_config=str(config_name),
                    source_split=split,
                    source_index=record["source_index"],
                    question=question,
                    answer=labels[answer_index],
                    category=category,
                    sub_category=subcategory,
                    difficulty=5,
                    answer_type="multiple_choice",
                    choices=choices,
                )
            )
    _require_selected_count(
        "MMLU mathematics",
        output,
        per_config
        * len(tuple(source.get("configs", MMLU_MATH_CONFIGS))),
    )
    return output, {
        "dataset_fingerprints": fingerprints,
        "available_count": available,
    }


def _load_deepmind_line_pairs(
    source: Mapping[str, Any],
    allowed_data_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load optional pre-generated DeepMind Mathematics question/answer files."""
    root = Path(str(source["local_root"])).expanduser().resolve()
    allowed = Path(allowed_data_root).expanduser().resolve()
    if not root.is_relative_to(allowed):
        raise RuntimeError(
            "DeepMind Mathematics local_root must remain below allowed data "
            f"root {allowed}: {root}"
        )
    if not root.is_dir():
        if bool(source.get("optional", True)):
            return [], {"status": "skipped_missing_local_root"}
        raise FileNotFoundError(
            f"DeepMind Mathematics root does not exist: {root}"
        )
    patterns = tuple(
        str(pattern)
        for pattern in source.get(
            "patterns",
            ("interpolate/*.txt", "extrapolate/*.txt"),
        )
    )
    if any(
        pattern.replace("\\", "/").lower().startswith("train")
        for pattern in patterns
    ):
        raise ValueError(
            "DeepMind Mathematics training files are forbidden in the fixed "
            "benchmark; use interpolate/ or extrapolate/ patterns only."
        )
    candidates = sorted(
        path
        for pattern in patterns
        for path in root.glob(str(pattern))
        if path.is_file()
    )
    raw_records = []
    for path in candidates:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for index in range(0, len(lines) - 1, 2):
            raw_records.append(
                {
                    "question": lines[index],
                    "answer": lines[index + 1],
                    "path": path.relative_to(root).as_posix(),
                    "source_index": index // 2,
                }
            )
    selected = _stable_select(
        raw_records,
        int(source.get("samples", 0)),
        int(source.get("seed", 42)),
        ("path", "question", "answer"),
    )
    output = []
    for record in selected:
        module = record["path"].split("/")[-1].split("__", 1)[0].lower()
        category, subcategory = {
            "algebra": ("Algebra", "Linear Equations"),
            "arithmetic": ("Arithmetic", "Integer Operations"),
            "calculus": ("Calculus", "Differentiation"),
            "numbers": ("Number Theory", "Modular Arithmetic"),
            "polynomials": ("Algebra", "Polynomials and Inequalities"),
            "probability": (
                "Probability & Statistics",
                "Basic Probability",
            ),
        }.get(
            module,
            (
                "Composite Comprehensive",
                "Constraint Synthesis",
            ),
        )
        output.append(
            _canonical_record(
                source_dataset="google-deepmind/mathematics_dataset",
                source_config=record["path"],
                source_split=record["path"].split("/", 1)[0],
                source_index=record["source_index"],
                question=record["question"],
                answer=record["answer"],
                category=category,
                sub_category=subcategory,
                difficulty=5,
            )
        )
    _require_selected_count(
        "DeepMind Mathematics",
        output,
        int(source.get("samples", 0)),
    )
    return output, {
        "local_root": root.as_posix(),
        "file_count": len(candidates),
        "available_count": len(raw_records),
    }


def build_open_fixed_suite(
    manifest_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    *,
    allowed_data_root: str | Path = REQUIRED_DATA_ROOT,
    local_files_only: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    allowed = Path(allowed_data_root).expanduser().resolve()
    for label, path in (
        ("output_path", output_path),
        ("cache_dir", cache_dir),
    ):
        if not path.is_relative_to(allowed):
            raise RuntimeError(
                f"{label} must remain below allowed data root {allowed}: {path}"
            )
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = yaml.safe_load(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Open benchmark manifest must be a YAML object")
    if output_path.exists() and not allow_overwrite:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_manifest = existing.get("builder_manifest_sha256")
        if existing_manifest != manifest_sha256:
            raise RuntimeError(
                "The fixed benchmark already exists and was built from a "
                "different manifest. Preserve it and choose a versioned output "
                "path, or pass --allow-overwrite explicitly."
            )
        return {
            "status": "already_exists",
            "output_path": output_path.as_posix(),
            "output_sha256": hashlib.sha256(
                output_path.read_bytes()
            ).hexdigest(),
            "question_count": int(existing.get("question_count", 0)),
            "source_counts": dict(existing.get("source_counts", {})),
            "source_metadata": list(existing.get("sources", [])),
        }
    cache_dir.mkdir(parents=True, exist_ok=True)
    questions = []
    source_metadata = []
    builders = {
        "gsm8k": lambda source: _build_gsm8k(
            source,
            cache_dir,
            local_files_only,
        ),
        "math": lambda source: _build_math(
            source,
            cache_dir,
            local_files_only,
        ),
        "mmlu_math": lambda source: _build_mmlu(
            source,
            cache_dir,
            local_files_only,
        ),
        "deepmind_mathematics_local": lambda source: (
            _load_deepmind_line_pairs(source, allowed)
        ),
    }
    for source in manifest.get("sources", []):
        if not bool(source.get("enabled", True)):
            continue
        source_type = str(source.get("type", ""))
        if source_type not in builders:
            raise ValueError(
                f"Unsupported open benchmark source type: {source_type}"
            )
        records, metadata = builders[source_type](source)
        questions.extend(records)
        source_name = (
            records[0]["source_dataset"]
            if records
            else str(source.get("dataset", source_type))
        )
        source_metadata.append(
            {
                "type": source_type,
                "source_dataset": source_name,
                "selected_count": len(records),
                "requested_revision": source.get("revision"),
                "selection_request": {
                    key: source.get(key)
                    for key in (
                        "split",
                        "config",
                        "configs",
                        "samples",
                        "samples_per_config",
                        "seed",
                        "patterns",
                    )
                    if key in source
                },
                **SOURCE_LICENSES.get(source_name, {}),
                **metadata,
            }
        )
    deduplicated = {}
    for record in questions:
        question_hash = record["source_question_sha256"]
        deduplicated.setdefault(question_hash, record)
    questions = sorted(
        deduplicated.values(),
        key=lambda record: (
            record["source_dataset"],
            record["source_config"],
            record["question_id"],
        ),
    )
    identifiers = set()
    for record in questions:
        identifier = record["question_id"]
        if identifier in identifiers:
            raise RuntimeError(
                f"Duplicate open benchmark question_id: {identifier}"
            )
        identifiers.add(identifier)
    payload = {
        "schema_version": "2.0",
        "name": str(manifest.get("name", "open_math_fixed_suite")),
        "description": str(manifest.get("description", "")),
        "builder_manifest_sha256": manifest_sha256,
        "selection_policy": {
            "method": "seeded_sha256_order_statistic",
            "training_splits_forbidden": True,
            "source_solution_excluded_from_model_prompt": True,
        },
        "sources": source_metadata,
        "question_count": len(questions),
        "source_counts": dict(
            sorted(Counter(
                record["source_dataset"] for record in questions
            ).items())
        ),
        "subcategory_counts": dict(
            sorted(Counter(
                f"{record['category']} / {record['sub_category']}"
                for record in questions
            ).items())
        ),
        "questions": questions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, output_path)
    result = {
        "status": "completed",
        "output_path": output_path.as_posix(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "question_count": len(questions),
        "source_counts": payload["source_counts"],
        "source_metadata": source_metadata,
    }
    atomic_json(
        result,
        output_path.with_suffix(output_path.suffix + ".manifest.json"),
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the multi-source open mathematics fixed benchmark",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--allowed-data-root",
        default=os.environ.get(
            "AUTOBENCHER_ALLOWED_DATA_ROOT",
            REQUIRED_DATA_ROOT,
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "replace an existing fixed artifact; prefer a new versioned path "
            "for scientific experiments"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = build_open_fixed_suite(
        args.manifest,
        args.output,
        args.cache_dir,
        allowed_data_root=args.allowed_data_root,
        local_files_only=args.local_files_only,
        allow_overwrite=args.allow_overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
