"""Offline, provenance-preserving import of open-source math benchmarks.

The output of this module is a *candidate* pool.  It deliberately carries
pending validation status and cannot pass the existing official-set release
gate until independent solving or human adjudication has been completed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import DEFAULT_TAXONOMY
from .evaluation_audit import normalize_question_text, template_signature
from .experiment import atomic_json
from .structured import normalize_generated_gold_contract


ADAPTERS = frozenset(
    {
        "gsm8k_jsonl",
        "hendrycks_math_json",
        "mmlu_csv",
        "deepmind_text",
    }
)
DIFFICULTY_BANDS = ("basic", "intermediate", "advanced")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _source_files(path: Path, adapter: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Open-source input does not exist: {path}")
    suffix = ".json" if adapter == "hendrycks_math_json" else ".csv"
    if adapter == "deepmind_text":
        suffix = ".txt"
    files = sorted(item for item in path.rglob(f"*{suffix}") if item.is_file())
    if not files:
        raise FileNotFoundError(f"No {suffix} files found beneath {path}")
    return files


def _extract_boxed(solution: str) -> str | None:
    marker = "\\boxed{"
    start = solution.rfind(marker)
    if start < 0:
        marker = "\\fbox{"
        start = solution.rfind(marker)
    if start < 0:
        return None
    index = start + len(marker)
    depth = 1
    for end in range(index, len(solution)):
        if solution[end] == "{":
            depth += 1
        elif solution[end] == "}":
            depth -= 1
            if depth == 0:
                return solution[index:end].strip()
    return None


def _infer_answer_type(answer: str, *, multiple_choice: bool = False) -> str:
    value = answer.strip()
    if multiple_choice:
        return "multiple_choice"
    if value.lower() in {"true", "false", "yes", "no"}:
        return "boolean"
    if re.fullmatch(r"[-+]?\d+", value.replace(",", "")):
        return "integer"
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)", value.replace(",", "")):
        return "decimal"
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
        return "rational"
    if value.startswith("(") and value.endswith(")"):
        return "ordered_tuple"
    if value.startswith("{") and value.endswith("}"):
        return "set"
    return "symbolic_expression"


def _band(difficulty: int) -> str:
    if difficulty <= 3:
        return "basic"
    if difficulty <= 6:
        return "intermediate"
    return "advanced"


def _complexity_difficulty(question: str, base: int) -> int:
    operations = len(re.findall(r"[+\-*/^=]", question))
    length_bonus = min(2, len(question.split()) // 35)
    return max(1, min(10, base + min(2, operations // 3) + length_bonus))


def _classify(question: str, source: str, source_type: str) -> tuple[str, str, str]:
    """Assign a reviewable taxonomy candidate using deterministic rules."""
    text = question.lower()
    if any(token in text for token in ("matrix", "determinant", "eigenvalue")):
        return "Linear Algebra", "Matrix Operations", "matrix_keyword"
    if any(token in text for token in ("vector", "dot product", "cross product")):
        return "Linear Algebra", "Vectors and Vector Spaces", "vector_keyword"
    if any(token in text for token in ("integral", "integrate", "antiderivative")):
        return "Calculus", "Integration", "integration_keyword"
    if any(token in text for token in ("derivative", "differentiate", "d/dx")):
        return "Calculus", "Differentiation", "differentiation_keyword"
    if "limit" in text or "continuous" in text:
        return "Calculus", "Limits and Continuity", "limit_keyword"
    if any(token in text for token in ("modulo", "remainder", " mod ")):
        return "Number Theory", "Modular Arithmetic", "modular_keyword"
    if any(token in text for token in ("prime factor", "product of primes")):
        return "Number Theory", "Prime Factorization", "prime_keyword"
    if any(token in text for token in ("divisor", "divisible", "gcd", "lcm", "factor")):
        return "Number Theory", "Divisibility and Factors", "divisibility_keyword"
    if any(token in text for token in ("sin", "cos", "tan", "trigonometric")):
        return "Geometry & Trigonometry", "Trigonometric Reasoning", "trig_keyword"
    if any(token in text for token in ("sphere", "cube", "prism", "cylinder", "volume")):
        return "Geometry & Trigonometry", "Solid Geometry", "solid_keyword"
    if any(token in text for token in ("triangle", "circle", "angle", "perimeter", "area")):
        return "Geometry & Trigonometry", "Plane Geometry", "plane_keyword"
    if any(token in text for token in ("mean", "median", "standard deviation", "variance")):
        return "Probability & Statistics", "Descriptive Statistics", "statistics_keyword"
    if any(token in text for token in ("arrange", "committee", "permutation", "combination")):
        return "Probability & Statistics", "Combinatorics", "combinatorics_keyword"
    if any(token in text for token in ("probability", "random", "drawn", "dice", "coin")):
        return "Probability & Statistics", "Basic Probability", "probability_keyword"
    if any(token in text for token in ("interest", "profit", "discount", "price", "cost")):
        return "Word Problems", "Financial Applications", "finance_keyword"
    if any(token in text for token in ("work together", "mixture", "solution", "machine")):
        return "Word Problems", "Work and Mixture", "work_mixture_keyword"
    if any(token in text for token in ("speed", "distance", "miles", "km/h", "travel")):
        return "Word Problems", "Rate and Distance", "rate_keyword"
    if any(token in text for token in ("system of", "simultaneous", "and solve for")):
        return "Algebra", "Systems of Equations", "system_keyword"
    if any(token in text for token in ("polynomial", "inequality", "roots", "quadratic")):
        return "Algebra", "Polynomials and Inequalities", "polynomial_keyword"
    if any(token in text for token in ("solve for", "equation", "unknown")):
        return "Algebra", "Linear Equations", "equation_keyword"
    if any(token in text for token in ("percent", "%", "ratio", "proportion")):
        return "Arithmetic", "Ratio and Percentage", "ratio_keyword"
    if any(token in text for token in ("fraction", "decimal", "/")):
        return "Arithmetic", "Fraction and Decimal Operations", "fraction_keyword"
    if source_type == "hendrycks_math_json":
        type_name = source.lower().replace(" ", "_")
        if "number_theory" in type_name:
            return "Number Theory", "Divisibility and Factors", "math_type_number_theory"
        if "geometry" in type_name or "precalculus" in type_name:
            return "Geometry & Trigonometry", "Plane Geometry", "math_type_geometry"
        if "counting" in type_name or "probability" in type_name:
            return "Probability & Statistics", "Combinatorics", "math_type_counting"
        return "Algebra", "Polynomials and Inequalities", "math_type_algebra"
    if source_type == "gsm8k_jsonl":
        return "Composite Comprehensive", "Cross-Domain Multi-Step Problems", "gsm8k_fallback"
    if source_type == "mmlu_csv":
        return "Composite Comprehensive", "Proof and Mathematical Reasoning", "mmlu_fallback"
    return "Arithmetic", "Integer Operations", "arithmetic_fallback"


def _gsm8k_records(path: Path) -> Iterable[dict[str, Any]]:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        item = json.loads(line)
        solution = str(item["answer"])
        marker = solution.rfind("####")
        if marker < 0:
            continue
        yield {
            "source_record_id": f"line-{index + 1}",
            "question": str(item["question"]),
            "answer": solution[marker + 4 :].strip().replace(",", ""),
            "source_type": "gsm8k_jsonl",
            "source_label": "gsm8k",
            "difficulty": _complexity_difficulty(str(item["question"]), 3),
            "reasoning_structure": "multi_step_word_problem",
        }


def _math_records(files: Iterable[Path], root: Path) -> Iterable[dict[str, Any]]:
    for path in files:
        item = json.loads(path.read_text(encoding="utf-8"))
        answer = _extract_boxed(str(item.get("solution", "")))
        if not answer:
            continue
        level_match = re.search(r"\d+", str(item.get("level", "")))
        level = int(level_match.group()) if level_match else 3
        yield {
            "source_record_id": path.relative_to(root).as_posix(),
            "question": str(item["problem"]),
            "answer": answer,
            "source_type": "hendrycks_math_json",
            "source_label": str(item.get("type", path.parent.name)),
            "difficulty": max(1, min(10, level * 2)),
            "reasoning_structure": (
                "math_"
                + str(item.get("type", path.parent.name))
                .lower()
                .replace(" ", "_")
            ),
        }


def _mmlu_records(
    files: Iterable[Path],
    root: Path,
    subjects: set[str],
) -> Iterable[dict[str, Any]]:
    for path in files:
        subject = re.sub(r"_(?:test|dev|val)$", "", path.stem)
        if subjects and subject not in subjects:
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.reader(handle)):
                if len(row) < 6 or row[5].strip().upper() not in {
                    "A",
                    "B",
                    "C",
                    "D",
                }:
                    continue
                choices = row[1:5]
                question = str(row[0]).strip() + "\n" + "\n".join(
                    f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)
                )
                yield {
                    "source_record_id": (
                        f"{path.relative_to(root).as_posix()}:{index + 1}"
                    ),
                    "question": question,
                    "answer": row[5].strip().upper(),
                    "source_type": "mmlu_csv",
                    "source_label": subject,
                    "difficulty": _complexity_difficulty(question, 5),
                    "reasoning_structure": f"multiple_choice_{subject}",
                    "multiple_choice": True,
                }


def _deepmind_records(
    files: Iterable[Path],
    root: Path,
) -> Iterable[dict[str, Any]]:
    for path in files:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(lines) % 2:
            raise ValueError(
                "DeepMind file must contain alternating question/answer "
                f"lines: {path}"
            )
        module = path.stem
        for index in range(0, len(lines), 2):
            question, answer = lines[index : index + 2]
            yield {
                "source_record_id": f"{path.relative_to(root).as_posix()}:{index // 2 + 1}",
                "question": question,
                "answer": answer,
                "source_type": "deepmind_text",
                "source_label": module,
                "difficulty": _complexity_difficulty(question, 4),
                "reasoning_structure": f"deepmind_{module}",
            }


def _load_source_records(
    source: Mapping[str, Any],
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = str(source.get("adapter", ""))
    if adapter not in ADAPTERS:
        raise ValueError(f"Unsupported open-source adapter: {adapter}")
    path = (source_root / str(source["path"])).resolve()
    if not path.is_relative_to(source_root):
        raise ValueError(f"Source path escapes source root: {path}")
    files = _source_files(path, adapter)
    file_manifest = [
        {
            "path": item.relative_to(source_root).as_posix(),
            "sha256": _sha256_bytes(item.read_bytes()),
            "size": item.stat().st_size,
        }
        for item in files
    ]
    if adapter == "gsm8k_jsonl":
        raw = _gsm8k_records(files[0])
    elif adapter == "hendrycks_math_json":
        raw = _math_records(files, path)
    elif adapter == "mmlu_csv":
        raw = _mmlu_records(files, path, set(source.get("subjects", [])))
    else:
        raw = _deepmind_records(files, path)
    records = []
    for item in raw:
        question = " ".join(str(item["question"]).split())
        answer = str(item["answer"]).strip()
        answer_type = _infer_answer_type(
            answer,
            multiple_choice=bool(item.get("multiple_choice")),
        )
        try:
            contract = normalize_generated_gold_contract(
                question,
                answer,
                answer_type,
            )
        except Exception:
            continue
        category, subcategory, rule = _classify(
            question,
            str(item["source_label"]),
            str(item["source_type"]),
        )
        upstream_key = f"{source['id']}:{item['source_record_id']}"
        record_hash = _sha256_text(
            json.dumps(
                {"question": question, "answer": answer},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        difficulty = int(item["difficulty"])
        records.append(
            {
                "question_id": f"oss-{_sha256_text(upstream_key)[:16]}",
                "category": category,
                "sub_category": subcategory,
                "difficulty": difficulty,
                "difficulty_band": _band(difficulty),
                "question": question,
                "answer_type": contract["answer_type"],
                "canonical_answer": contract["canonical_answer"],
                "display_answer": contract["display_answer"],
                "tolerance": contract["tolerance"],
                "reasoning_structure": str(item["reasoning_structure"]),
                "template_cluster": _sha256_text(template_signature(question)),
                "source_dataset": str(source["id"]),
                "source_split": str(source["split"]),
                "source_record_id": str(item["source_record_id"]),
                "source_revision": str(source["revision"]),
                "source_repository": str(source["repository"]),
                "source_license": str(source["license"]),
                "source_record_sha256": record_hash,
                "taxonomy_assignment": {
                    "status": "requires_human_review",
                    "rule": rule,
                },
                "validation": {
                    "status": "pending_independent_verification",
                    "sources": [
                        {
                            "source_id": f"upstream:{upstream_key}",
                            "method": "upstream_published_answer",
                            "solver_id": str(source["id"]),
                            "answer": contract["canonical_answer"],
                        }
                    ],
                },
            }
        )
    return records, file_manifest


def _select_bucket(records: list[dict[str, Any]], quota: int, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: _sha256_text(
            f"{seed}:{item['source_dataset']}:{item['question_id']}"
        ),
    )
    selected: list[dict[str, Any]] = []
    bands: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    templates: Counter[str] = Counter()
    while ordered and len(selected) < quota:
        eligible = [item for item in ordered if templates[item["template_cluster"]] < 2]
        if not eligible:
            break
        item = max(
            eligible,
            key=lambda candidate: (
                bands[candidate["difficulty_band"]] == 0,
                answer_types[candidate["answer_type"]] == 0,
                sources[candidate["source_dataset"]] == 0,
                -bands[candidate["difficulty_band"]],
                -answer_types[candidate["answer_type"]],
                -sources[candidate["source_dataset"]],
                _sha256_text(f"{seed}:{candidate['question_id']}"),
            ),
        )
        ordered.remove(item)
        selected.append(item)
        bands[item["difficulty_band"]] += 1
        answer_types[item["answer_type"]] += 1
        sources[item["source_dataset"]] += 1
        templates[item["template_cluster"]] += 1
    return selected


def build_open_source_candidate_payload(
    catalog_path: str | Path,
    source_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog_file = Path(catalog_path).resolve()
    catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))
    if str(catalog.get("schema_version")) != "1.0":
        raise ValueError("Open-source catalog schema_version must be 1.0")
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Open-source source root does not exist: {root}")
    candidate_spec = catalog["candidate_set"]
    seed = int(candidate_spec["seed"])
    quota = int(candidate_spec["questions_per_subcategory"])
    all_records = []
    source_manifests = []
    for source in catalog["sources"]:
        revision = str(source.get("revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"Source revision must be a 40-character git SHA: {source.get('id')}")
        records, files = _load_source_records(source, root)
        all_records.extend(records)
        source_manifests.append(
            {
                "id": source["id"],
                "repository": source["repository"],
                "revision": revision,
                "license": source["license"],
                "split": source["split"],
                "imported_record_count": len(records),
                "files": files,
            }
        )
    unique = {}
    for record in all_records:
        key = normalize_question_text(record["question"])
        unique.setdefault(key, record)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in unique.values():
        buckets[(record["category"], record["sub_category"])].append(record)
    selected = []
    deficits = []
    for category, subcategories in DEFAULT_TAXONOMY.items():
        for subcategory in subcategories:
            chosen = _select_bucket(buckets[(category, subcategory)], quota, seed)
            if len(chosen) < quota:
                deficits.append(
                    {
                        "category": category,
                        "sub_category": subcategory,
                        "available": len(chosen),
                        "required": quota,
                    }
                )
            selected.extend(chosen)
    if deficits:
        summary = ", ".join(
            f"{item['category']}/{item['sub_category']}={item['available']}/{item['required']}"
            for item in deficits[:12]
        )
        raise ValueError(f"Open-source candidate coverage is incomplete: {summary}")
    payload = {
        "schema_version": "1.0",
        "name": str(candidate_spec["name"]),
        "version": str(candidate_spec["version"]),
        "role": "official_candidate_pool",
        "release_status": "pending_independent_verification_and_human_taxonomy_review",
        "method_selection_prohibited": True,
        "training_use_prohibited": True,
        "questions": selected,
    }
    manifest = {
        "schema_version": "1.0",
        "name": payload["name"],
        "version": payload["version"],
        "seed": seed,
        "question_count": len(selected),
        "subcategory_count": sum(
            len(value) for value in DEFAULT_TAXONOMY.values()
        ),
        "questions_per_subcategory": quota,
        "catalog_path": catalog_file.as_posix(),
        "catalog_sha256": _sha256_bytes(catalog_file.read_bytes()),
        "source_manifests": source_manifests,
        "source_counts": dict(
            sorted(
                Counter(item["source_dataset"] for item in selected).items()
            )
        ),
        "release_ready": False,
        "required_next_step": (
            "independent verification, taxonomy review, leakage audit, "
            "and official assembly"
        ),
    }
    return payload, manifest


def write_open_source_candidate_set(
    catalog_path: str | Path,
    source_root: str | Path,
    output_path: str | Path,
    allowed_data_root: str | Path,
) -> dict[str, Any]:
    root = Path(allowed_data_root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_relative_to(root):
        raise ValueError("Open-source inputs must remain beneath allowed_data_root")
    if not output.is_relative_to(root):
        raise ValueError("Candidate output must remain beneath allowed_data_root")
    if output.exists():
        raise FileExistsError(f"Immutable candidate set already exists: {output}")
    payload, manifest = build_open_source_candidate_payload(catalog_path, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, output)
    manifest.update(
        {
            "path": output.as_posix(),
            "sha256": _sha256_bytes(output.read_bytes()),
        }
    )
    atomic_json(manifest, output.with_suffix(output.suffix + ".manifest.json"))
    return manifest
