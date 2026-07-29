"""Isolated evaluator-side Python solving and semantic answer judging."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping

from util import gen_from_prompt

from .output_schemas import (
    EvaluatorPostcheck,
    SemanticAnswerJudgment,
    SolverProposal,
)
from .reasoning import validate_training_reasoning
from .structured import (
    ANSWER_TYPES,
    answers_equivalent,
    normalize_answer,
    normalize_answer_type,
)


SOLVER_PROMPT_VERSION = "evaluator_python_solver_v3_blind_consensus"
INDEPENDENT_SOLVER_PROMPT_VERSION = "evaluator_independent_solver_v1"
POSTCHECK_PROMPT_VERSION = "evaluator_python_postcheck_v3_training_derivation"
SEMANTIC_JUDGE_PROMPT_VERSION = "semantic_answer_judge_v2_cross_format"

_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()

_FORBIDDEN_AST_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.Yield,
    ast.YieldFrom,
)
_FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "print",
    "quit",
    "type",
    "vars",
}
_SAFE_NAMED_CALLS = {
    "Decimal",
    "Fraction",
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "pow",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_SAFE_METHOD_CALLS = {
    "append",
    "as_dict",
    "count",
    "det",
    "diff",
    "doit",
    "evalf",
    "factor",
    "integrate",
    "items",
    "limit",
    "n",
    "reshape",
    "simplify",
    "solve",
    "subs",
    "tolist",
    "values",
}
_SAFE_MODULE_ROOTS = {"sp", "math", "statistics", "itertools"}

_RUNNER = r"""
import base64
import json
import math
import statistics
import itertools
import sys
from decimal import Decimal
from fractions import Fraction
import sympy as sp

safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "len": len,
    "list": list, "max": max, "min": min, "pow": pow, "range": range,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip,
}
scope = {
    "__builtins__": safe_builtins,
    "sp": sp, "math": math, "statistics": statistics,
    "itertools": itertools, "Fraction": Fraction, "Decimal": Decimal,
}
source = base64.b64decode(sys.argv[1]).decode("utf-8")
exec(compile(source, "<evaluator-solution>", "exec"), scope, scope)
result = scope.get("result")
if not isinstance(result, dict):
    raise TypeError("generated code must assign a dictionary to result")
print(json.dumps(result, ensure_ascii=False, default=str))
"""


class EvaluatorProtocolError(RuntimeError):
    """Raised when a privileged evaluator response violates its contract."""


_NUMERIC_ANSWER_TYPES = {"integer", "decimal", "rational", "percentage"}
_SUSPICIOUS_ANSWER_TEXT = (
    "```",
    "<|",
    "|>",
    "question_json",
    "runtime_result",
    "primary_result",
    "independent_result",
    "analysis_summary",
    "python_code",
    "system:",
    "assistant:",
    "user:",
    "human:",
    "ignore previous",
    "as an ai",
)


def _answer_types_compatible(left: Any, right: Any) -> bool:
    left_type = normalize_answer_type(left)
    right_type = normalize_answer_type(right)
    return (
        left_type == right_type
        or {
            left_type,
            right_type,
        }.issubset(_NUMERIC_ANSWER_TYPES)
    )


def _validate_answer_payload(
    answer: Any,
    answer_type: Any,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """Reject prose, prompt fragments, control text, and unparsable answers."""
    text = str(answer or "").strip()
    normalized_type = normalize_answer_type(answer_type, text)
    maximum = int(config["evaluator_pipeline"]["max_answer_chars"])
    if not text:
        raise EvaluatorProtocolError("canonical answer is empty")
    if len(text) > maximum:
        raise EvaluatorProtocolError(
            f"canonical answer exceeds {maximum} characters"
        )
    if any(
        ord(character) < 32 and character not in {" "}
        for character in text
    ) or "\ufffd" in text:
        raise EvaluatorProtocolError(
            "canonical answer contains control or replacement characters"
        )
    lowered = text.lower()
    if any(marker in lowered for marker in _SUSPICIOUS_ANSWER_TEXT):
        raise EvaluatorProtocolError(
            "canonical answer contains prompt, role, code, or Markdown text"
        )
    if normalized_type not in ANSWER_TYPES:
        raise EvaluatorProtocolError(
            f"unsupported canonical answer type {normalized_type!r}"
        )
    normalized = normalize_answer(text, normalized_type, config)
    if not normalized["success"]:
        raise EvaluatorProtocolError(
            "canonical answer cannot be parsed for its declared answer type"
        )
    if normalized_type != "text" and not answers_equivalent(
        text,
        text,
        normalized_type,
        config,
    )["equivalent"]:
        raise EvaluatorProtocolError(
            "canonical answer is not a valid mathematical value of its type"
        )
    if normalized_type == "text":
        words = text.split()
        if len(words) > 8 or "\n" in text or "\r" in text:
            raise EvaluatorProtocolError(
                "text answer must be a short standalone mathematical label"
            )
    return text, normalized_type


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pipeline_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config["evaluator_pipeline"]


def _resolve_prompt(path_value: Any) -> str:
    path = Path(str(path_value)).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        _project_root() / path,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Evaluator prompt file not found: {path_value}")


def _render_prompt(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(
            "{{" + key + "}}",
            json.dumps(value, ensure_ascii=False, indent=2),
        )
    return rendered


def _strict_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip().lstrip("\ufeff")
    if raw.startswith("```") or raw.endswith("```"):
        raise EvaluatorProtocolError("Markdown fences are not allowed")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvaluatorProtocolError(f"response is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluatorProtocolError("response must be one JSON object")
    return value


def _model_json(
    model_info: tuple[Any, Any, Any],
    prompt: str,
    *,
    temperature: float,
    max_tokens: int,
    config: Mapping[str, Any] | None = None,
    schema: Any | None = None,
) -> dict[str, Any]:
    model, tokenizer, service = model_info
    structured = (
        config.get("structured_output", {})
        if isinstance(config, Mapping)
        else {}
    )
    use_structured = bool(
        schema is not None
        and structured.get("enabled")
        and structured.get("use_for_evaluator")
    )
    evaluator_model_config = (
        config.get("models", {}).get("evaluator", {})
        if isinstance(config, Mapping)
        else {}
    )
    request_result = gen_from_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=[prompt],
        echo_prompt=False,
        temperature=temperature,
        max_tokens=max_tokens,
        process_func=None,
        service=service,
        terminate_by_linebreak="no",
        verbose=False,
        structured_schema=schema if use_structured else None,
        structured_backend=structured.get("local_backend", "none"),
        structured_fallback_backend=structured.get(
            "fallback_backend",
            "none",
        ),
        structured_required=bool(structured.get("required", False)),
        request_timeout_seconds=evaluator_model_config.get(
            "request_timeout_seconds"
        ),
        max_num_retries=int(
            evaluator_model_config.get("max_retries", 3)
        ),
        retry_delay_seconds=float(
            evaluator_model_config.get("retry_delay_seconds", 5)
        ),
    )
    if not request_result.completions:
        raise EvaluatorProtocolError("evaluator returned no completion")
    return _strict_json_object(request_result.completions[0].text)


def validate_generated_python(
    source: Any,
    max_chars: int,
    *,
    max_ast_nodes: int = 4000,
    max_integer_literal: int = 1_000_000,
) -> str:
    """Validate evaluator-generated code before it enters a subprocess."""
    code = str(source or "").strip()
    if not code:
        raise EvaluatorProtocolError("python_code is empty")
    if len(code) > int(max_chars):
        raise EvaluatorProtocolError(
            f"python_code exceeds {int(max_chars)} characters"
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise EvaluatorProtocolError(f"python_code is invalid: {exc}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > int(max_ast_nodes):
        raise EvaluatorProtocolError(
            f"python_code exceeds {int(max_ast_nodes)} AST nodes"
        )
    if any(isinstance(node, _FORBIDDEN_AST_NODES) for node in nodes):
        raise EvaluatorProtocolError(
            "python_code contains imports, definitions, context managers, "
            "exception control, or asynchronous constructs"
        )
    assigns_result = False
    assignments: dict[str, ast.AST] = {}
    for node in nodes:
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _FORBIDDEN_NAMES:
                raise EvaluatorProtocolError(
                    f"python_code uses forbidden name {node.id!r}"
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise EvaluatorProtocolError(
                f"python_code uses private attribute {node.attr!r}"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if abs(node.value) > int(max_integer_literal):
                raise EvaluatorProtocolError(
                    "python_code contains an excessive integer literal"
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_NAMED_CALLS:
                    raise EvaluatorProtocolError(
                        f"python_code calls forbidden function {node.func.id!r}"
                    )
            elif isinstance(node.func, ast.Attribute):
                root = node.func.value
                while isinstance(root, (ast.Attribute, ast.Call, ast.Subscript)):
                    if isinstance(root, ast.Attribute):
                        root = root.value
                    elif isinstance(root, ast.Call):
                        root = root.func
                    else:
                        root = root.value
                root_name = root.id if isinstance(root, ast.Name) else None
                if (
                    root_name not in _SAFE_MODULE_ROOTS
                    and node.func.attr not in _SAFE_METHOD_CALLS
                ):
                    raise EvaluatorProtocolError(
                        f"python_code calls forbidden method {node.func.attr!r}"
                    )
            else:
                raise EvaluatorProtocolError(
                    "python_code contains an indirect callable"
                )
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
            if any(
                isinstance(target, ast.Name) and target.id == "result"
                for target in targets
            ):
                assigns_result = True
    if not assigns_result:
        raise EvaluatorProtocolError("python_code must assign result")
    required_computation_names = {
        "primary_answer",
        "independent_answer",
        "verification_passed",
        "substitution_passed",
    }
    missing = required_computation_names - set(assignments)
    if missing:
        raise EvaluatorProtocolError(
            "python_code is missing computed variables: "
            + ", ".join(sorted(missing))
        )
    primary_ast = assignments["primary_answer"]
    independent_ast = assignments["independent_answer"]
    if (
        isinstance(independent_ast, ast.Name)
        and independent_ast.id == "primary_answer"
    ) or ast.dump(primary_ast, include_attributes=False) == ast.dump(
        independent_ast,
        include_attributes=False,
    ):
        raise EvaluatorProtocolError(
            "independent_answer must use a distinct computation"
        )
    for name in ("verification_passed", "substitution_passed"):
        expression = assignments[name]
        if isinstance(expression, ast.Constant):
            raise EvaluatorProtocolError(
                f"{name} must be computed, not a constant"
            )
    result_expression = assignments.get("result")
    if not isinstance(result_expression, ast.Dict):
        raise EvaluatorProtocolError("result must be a dictionary literal")
    result_fields = {
        key.value: value
        for key, value in zip(
            result_expression.keys,
            result_expression.values,
        )
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    for name in ("verification_passed", "substitution_passed"):
        value = result_fields.get(name)
        if not isinstance(value, ast.Name) or value.id != name:
            raise EvaluatorProtocolError(
                f"result.{name} must reference the computed {name} variable"
            )
    return code


def execute_generated_python(
    source: str,
    *,
    timeout_seconds: float,
    max_output_chars: int,
) -> dict[str, Any]:
    """Execute validated code with isolated cwd, argv, env, and built-ins."""
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="autobencher-evaluator-") as temp_dir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER, encoded],
                cwd=temp_dir,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluatorProtocolError(
                "generated Python exceeded the configured timeout"
            ) from exc
    stdout = completed.stdout[: int(max_output_chars)]
    stderr = completed.stderr[: int(max_output_chars)]
    if completed.returncode != 0:
        raise EvaluatorProtocolError(
            "generated Python failed: "
            + (stderr.strip() or f"exit code {completed.returncode}")
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise EvaluatorProtocolError(
            f"generated Python returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(result, dict):
        raise EvaluatorProtocolError("generated Python result must be an object")
    required = {
        "canonical_answer",
        "answer_type",
        "verification_passed",
        "substitution_passed",
        "verification_details",
    }
    if set(result) != required:
        raise EvaluatorProtocolError(
            "generated Python result fields do not match the contract"
        )
    if not isinstance(result["verification_passed"], bool):
        raise EvaluatorProtocolError("verification_passed must be Boolean")
    if not isinstance(result["substitution_passed"], bool):
        raise EvaluatorProtocolError("substitution_passed must be Boolean")
    if not isinstance(result["verification_details"], list):
        raise EvaluatorProtocolError("verification_details must be a list")
    if not str(result["canonical_answer"]).strip():
        raise EvaluatorProtocolError("canonical_answer is empty")
    result["canonical_answer"] = str(result["canonical_answer"]).strip()
    result["answer_type"] = normalize_answer_type(
        result["answer_type"],
        result["canonical_answer"],
    )
    return result


def _execute_model_solution(
    *,
    evaluator_info: tuple[Any, Any, Any],
    prompt: str,
    pipeline: Mapping[str, Any],
    config: Mapping[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    proposal = _model_json(
        evaluator_info,
        prompt,
        temperature=float(pipeline["temperature"]),
        max_tokens=int(max_tokens),
        config=config,
        schema=SolverProposal,
    )
    if set(proposal) != {"analysis_summary", "python_code"}:
        raise EvaluatorProtocolError(
            "solver response fields do not match the contract"
        )
    if (
        not isinstance(proposal["analysis_summary"], list)
        or not proposal["analysis_summary"]
        or any(
            not isinstance(step, str) or not step.strip()
            for step in proposal["analysis_summary"]
        )
    ):
        raise EvaluatorProtocolError(
            "analysis_summary must be a non-empty string list"
        )
    maximum_steps = min(
        int(config["generation"]["maximum_reasoning_steps"]),
        int(config["dataset"]["max_gold_reasoning_steps"]),
    )
    maximum_step_chars = int(
        config["dataset"]["max_gold_reasoning_chars_per_step"]
    )
    if len(proposal["analysis_summary"]) > maximum_steps:
        raise EvaluatorProtocolError(
            f"analysis_summary exceeds {maximum_steps} reasoning steps"
        )
    reasoning_markers = (
        "```",
        "<|",
        "|>",
        "system:",
        "assistant:",
        "user:",
        "human:",
        "question_json",
        "python_code",
        "ignore previous",
    )
    for step in proposal["analysis_summary"]:
        normalized_step = " ".join(step.split())
        if (
            len(normalized_step) > maximum_step_chars
            or any(
                marker in normalized_step.lower()
                for marker in reasoning_markers
            )
        ):
            raise EvaluatorProtocolError(
                "analysis_summary contains excessive, role, prompt, or "
                "code-fence content"
            )
    proposal["analysis_summary"] = [
        " ".join(step.split())
        for step in proposal["analysis_summary"]
    ]
    code = validate_generated_python(
        proposal["python_code"],
        int(pipeline["max_code_chars"]),
        max_ast_nodes=int(pipeline["max_ast_nodes"]),
        max_integer_literal=int(pipeline["max_integer_literal"]),
    )
    runtime_result = execute_generated_python(
        code,
        timeout_seconds=float(
            pipeline["code_execution_timeout_seconds"]
        ),
        max_output_chars=int(pipeline["max_output_chars"]),
    )
    if not runtime_result["verification_passed"]:
        raise EvaluatorProtocolError(
            "generated Python did not verify its answer"
        )
    if not runtime_result["substitution_passed"]:
        raise EvaluatorProtocolError(
            "generated Python failed substitution/recomputation"
        )
    if (
        not runtime_result["verification_details"]
        or any(
            not isinstance(detail, str)
            or not detail.strip()
            or len(detail) > 500
            for detail in runtime_result["verification_details"]
        )
    ):
        raise EvaluatorProtocolError(
            "verification_details must contain short concrete checks"
        )
    (
        runtime_result["canonical_answer"],
        runtime_result["answer_type"],
    ) = _validate_answer_payload(
        runtime_result["canonical_answer"],
        runtime_result["answer_type"],
        config,
    )
    return proposal, code, runtime_result


def _cache_read(path: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not path or not Path(path).is_file():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _cache_write(
    path: str | os.PathLike[str] | None,
    payload: Mapping[str, Any],
) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _cache_store_entry(
    path: str | os.PathLike[str] | None,
    key: str,
    value: Mapping[str, Any],
) -> None:
    """Merge one cache entry atomically across parallel question workers."""
    if not path:
        return
    normalized_path = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _CACHE_LOCKS_GUARD:
        lock = _CACHE_LOCKS.setdefault(normalized_path, threading.Lock())
    with lock:
        latest = _cache_read(path)
        latest[key] = dict(value)
        _cache_write(path, latest)


def solve_with_privileged_python(
    question: str,
    evaluator_info: tuple[Any, Any, Any],
    config: Mapping[str, Any],
    *,
    cache_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Generate, execute, and post-check one isolated evaluator solution."""
    pipeline = _pipeline_config(config)
    question_text = str(question).strip()
    solver_template = _resolve_prompt(pipeline["solver_prompt_path"])
    solver_strategy = _resolve_prompt(pipeline["solver_strategy_path"])
    independent_solver_template = _resolve_prompt(
        pipeline["independent_solver_prompt_path"]
    )
    postcheck_template = _resolve_prompt(pipeline["postcheck_prompt_path"])
    question_hash = hashlib.sha256(question_text.encode("utf-8")).hexdigest()
    prompt_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "solver": solver_template,
                "strategy": solver_strategy,
                "independent_solver": independent_solver_template,
                "postcheck": postcheck_template,
                "minimum_difficulty": int(pipeline["minimum_difficulty"]),
                "maximum_difficulty": int(pipeline["maximum_difficulty"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_key = (
        f"{SOLVER_PROMPT_VERSION}:{prompt_fingerprint}:{question_hash}"
    )
    cache = _cache_read(cache_path)
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("status") == "passed":
        return dict(cached)

    solver_prompt = _render_prompt(
        solver_template,
        SOLVER_STRATEGY=solver_strategy,
        QUESTION_JSON={"question": question_text},
    )
    independent_solver_prompt = _render_prompt(
        independent_solver_template,
        QUESTION_JSON={"question": question_text},
    )
    last_error: Exception | None = None
    attempts = int(pipeline["code_generation_attempts"])
    for attempt in range(1, attempts + 1):
        try:
            primary_prompt = solver_prompt
            if last_error is not None:
                primary_prompt += (
                    "\n\nTRUSTED_RETRY_FEEDBACK_JSON:\n"
                    + json.dumps(
                        {
                            "previous_failure": (
                                f"{type(last_error).__name__}: {last_error}"
                            ),
                            "required_action": (
                                "Use a different derivation and verification "
                                "route. Return only the required JSON."
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
            proposal, code, runtime_result = _execute_model_solution(
                evaluator_info=evaluator_info,
                prompt=primary_prompt,
                pipeline=pipeline,
                config=config,
                max_tokens=int(pipeline["solver_max_tokens"]),
            )

            independent_proposal = None
            independent_code = None
            independent_result = None
            independent_error: Exception | None = None
            for independent_attempt in range(
                1,
                int(pipeline["independent_solver_attempts"]) + 1,
            ):
                blind_prompt = independent_solver_prompt
                if independent_error is not None:
                    blind_prompt += (
                        "\n\nTRUSTED_RETRY_FEEDBACK_JSON:\n"
                        + json.dumps(
                            {
                                "previous_failure": (
                                    f"{type(independent_error).__name__}: "
                                    f"{independent_error}"
                                ),
                                "required_action": (
                                    "Start over with another independent "
                                    "method and verify every condition."
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )
                try:
                    (
                        independent_proposal,
                        independent_code,
                        independent_result,
                    ) = _execute_model_solution(
                        evaluator_info=evaluator_info,
                        prompt=blind_prompt,
                        pipeline=pipeline,
                        config=config,
                        max_tokens=int(
                            pipeline["independent_solver_max_tokens"]
                        ),
                    )
                    break
                except (
                    EvaluatorProtocolError,
                    OSError,
                    RuntimeError,
                    subprocess.SubprocessError,
                    TypeError,
                    ValueError,
                ) as exc:
                    independent_error = exc
            if (
                independent_proposal is None
                or independent_code is None
                or independent_result is None
            ):
                raise EvaluatorProtocolError(
                    f"blind independent solver failed: {independent_error}"
                )
            independent_equivalence = answers_equivalent(
                runtime_result["canonical_answer"],
                independent_result["canonical_answer"],
                runtime_result["answer_type"],
                config,
            )
            independent_type_consistent = _answer_types_compatible(
                runtime_result["answer_type"],
                independent_result["answer_type"],
            )
            if (
                not independent_equivalence["equivalent"]
                or not independent_type_consistent
            ):
                raise EvaluatorProtocolError(
                    "primary and blind independent Python solvers disagree"
                )

            postcheck_prompt = _render_prompt(
                postcheck_template,
                DIFFICULTY_JSON={
                    "minimum": int(pipeline["minimum_difficulty"]),
                    "maximum": int(pipeline["maximum_difficulty"]),
                },
                REASONING_REQUIREMENTS_JSON={
                    "minimum_steps": int(
                        config["dataset"]["min_gold_reasoning_steps"]
                    ),
                    "maximum_steps": int(
                        config["dataset"]["max_gold_reasoning_steps"]
                    ),
                    "minimum_chars_per_step": int(
                        config["dataset"][
                            "min_gold_reasoning_chars_per_step"
                        ]
                    ),
                    "maximum_chars_per_step": int(
                        config["dataset"][
                            "max_gold_reasoning_chars_per_step"
                        ]
                    ),
                    "must_show_concrete_derivation": bool(
                        config["dataset"][
                            "require_concrete_gold_reasoning"
                        ]
                    ),
                    "must_end_with_verification": bool(
                        config["dataset"][
                            "require_gold_reasoning_verification"
                        ]
                    ),
                    "must_state_verified_answer": bool(
                        config["dataset"][
                            "require_gold_answer_in_reasoning"
                        ]
                    ),
                },
                QUESTION_JSON={"question": question_text},
                PRIMARY_RESULT_JSON=runtime_result,
                INDEPENDENT_RESULT_JSON=independent_result,
            )
            postcheck = None
            postcheck_error = None
            for _ in range(int(pipeline["postcheck_attempts"])):
                try:
                    postcheck = _model_json(
                        evaluator_info,
                        postcheck_prompt,
                        temperature=0.0,
                        max_tokens=int(pipeline["postcheck_max_tokens"]),
                        config=config,
                        schema=EvaluatorPostcheck,
                    )
                    break
                except (EvaluatorProtocolError, ValueError, TypeError) as exc:
                    postcheck_error = exc
            if postcheck is None:
                raise EvaluatorProtocolError(
                    f"postcheck failed: {postcheck_error}"
                )
            required = {
                "accepted",
                "verified_answer",
                "answer_type",
                "reasoning_summary",
                "substitution_passed",
                "difficulty_acceptable",
                "estimated_difficulty",
                "reason",
            }
            if set(postcheck) != required:
                raise EvaluatorProtocolError(
                    "postcheck response fields do not match the contract"
                )
            boolean_fields = (
                "accepted",
                "substitution_passed",
                "difficulty_acceptable",
            )
            if any(not isinstance(postcheck[field], bool) for field in boolean_fields):
                raise EvaluatorProtocolError(
                    "postcheck status fields must be JSON booleans"
                )
            estimated_difficulty = int(postcheck["estimated_difficulty"])
            if not 1 <= estimated_difficulty <= 10:
                raise EvaluatorProtocolError(
                    "estimated_difficulty must be between 1 and 10"
                )
            verified_answer = str(postcheck["verified_answer"]).strip()
            verified_answer, verified_type = _validate_answer_payload(
                verified_answer,
                postcheck["answer_type"],
                config,
            )
            training_reasoning_summary, reasoning_error = (
                validate_training_reasoning(
                    postcheck["reasoning_summary"],
                    config,
                    verified_answer=verified_answer,
                )
            )
            if reasoning_error:
                raise EvaluatorProtocolError(
                    "postcheck reasoning_summary is not a concrete verified "
                    f"derivation: {reasoning_error}"
                )
            equivalence = answers_equivalent(
                runtime_result["canonical_answer"],
                verified_answer,
                runtime_result["answer_type"],
                config,
            )
            type_consistent = _answer_types_compatible(
                verified_type,
                runtime_result["answer_type"],
            )
            passed = bool(
                postcheck["accepted"]
                and postcheck["substitution_passed"]
                and postcheck["difficulty_acceptable"]
                and equivalence["equivalent"]
                and type_consistent
            )
            result = {
                "status": "passed" if passed else "failed",
                "source_question_sha256": question_hash,
                "solver_question_sha256": question_hash,
                "solver_prompt_version": SOLVER_PROMPT_VERSION,
                "independent_solver_prompt_version": (
                    INDEPENDENT_SOLVER_PROMPT_VERSION
                ),
                "postcheck_prompt_version": POSTCHECK_PROMPT_VERSION,
                "prompt_fingerprint": prompt_fingerprint,
                "solver_strategy": "microsoft_tora_single_round_adaptation",
                "attempt": attempt,
                "analysis_summary": training_reasoning_summary,
                "training_reasoning_summary": training_reasoning_summary,
                "solver_analysis_summary": proposal["analysis_summary"],
                "python_code": code,
                "python_code_sha256": hashlib.sha256(
                    code.encode("utf-8")
                ).hexdigest(),
                "independent_analysis_summary": independent_proposal[
                    "analysis_summary"
                ],
                "independent_python_code": independent_code,
                "independent_python_code_sha256": hashlib.sha256(
                    independent_code.encode("utf-8")
                ).hexdigest(),
                "independent_python_answer": independent_result[
                    "canonical_answer"
                ],
                "independent_answer_type": independent_result[
                    "answer_type"
                ],
                "independent_answer_equivalent": bool(
                    independent_equivalence["equivalent"]
                ),
                "independent_answer_type_consistent": (
                    independent_type_consistent
                ),
                "python_answer": runtime_result["canonical_answer"],
                "canonical_answer": runtime_result["canonical_answer"],
                "answer_type": runtime_result["answer_type"],
                "verification_passed": runtime_result[
                    "verification_passed"
                ],
                "substitution_passed": runtime_result[
                    "substitution_passed"
                ],
                "verification_details": runtime_result[
                    "verification_details"
                ],
                "postcheck": postcheck,
                "answer_equivalent": bool(equivalence["equivalent"]),
                "answer_type_consistent": type_consistent,
                "estimated_difficulty": estimated_difficulty,
                "difficulty_acceptable": bool(
                    postcheck["difficulty_acceptable"]
                ),
                "failure_reason": (
                    None
                    if passed
                    else "post-execution verification rejected the answer"
                ),
            }
            _cache_store_entry(cache_path, cache_key, result)
            return result
        except (
            EvaluatorProtocolError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TypeError,
            ValueError,
        ) as exc:
            last_error = exc
    failed = {
        "status": "failed",
        "source_question_sha256": question_hash,
        "solver_question_sha256": question_hash,
        "solver_prompt_version": SOLVER_PROMPT_VERSION,
        "independent_solver_prompt_version": (
            INDEPENDENT_SOLVER_PROMPT_VERSION
        ),
        "postcheck_prompt_version": POSTCHECK_PROMPT_VERSION,
        "prompt_fingerprint": prompt_fingerprint,
        "solver_strategy": "microsoft_tora_single_round_adaptation",
        "attempt": attempts,
        "failure_reason": (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "evaluator solver failed"
        ),
    }
    _cache_store_entry(cache_path, cache_key, failed)
    return failed


def judge_answer_semantics(
    *,
    question: str,
    gold_answer: Any,
    predicted_answer: Any,
    answer_type: str,
    evaluator_info: tuple[Any, Any, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Ask the isolated LLM judge whether two normalized answers agree."""
    pipeline = _pipeline_config(config)
    template = _resolve_prompt(pipeline["semantic_judge_prompt_path"])
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    equivalence = answers_equivalent(
        gold_answer,
        predicted_answer,
        answer_type,
        config,
    )
    prompt = _render_prompt(
        template,
        INPUT_JSON={
            "question": str(question),
            "reference_answer": str(gold_answer),
            "candidate_answer": str(predicted_answer),
            "answer_type": normalize_answer_type(answer_type),
            "absolute_tolerance": float(
                config["answer_normalization"]["absolute_tolerance"]
            ),
            "relative_tolerance": float(
                config["answer_normalization"]["relative_tolerance"]
            ),
            "deterministic_normalization": {
                "reference": equivalence["gold_normalized"],
                "candidate": equivalence["predicted_normalized"],
            },
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, int(pipeline["semantic_judge_attempts"]) + 1):
        try:
            result = _model_json(
                evaluator_info,
                prompt,
                temperature=0.0,
                max_tokens=int(pipeline["semantic_judge_max_tokens"]),
                config=config,
                schema=SemanticAnswerJudgment,
            )
            required = {
                "semantically_equivalent",
                "confidence",
                "reason",
                "format_only_difference",
            }
            if set(result) != required:
                raise EvaluatorProtocolError(
                    "semantic judge fields do not match the contract"
                )
            if not isinstance(result["semantically_equivalent"], bool):
                raise EvaluatorProtocolError(
                    "semantically_equivalent must be Boolean"
                )
            if not isinstance(result["format_only_difference"], bool):
                raise EvaluatorProtocolError(
                    "format_only_difference must be Boolean"
                )
            confidence = float(result["confidence"])
            if not 0 <= confidence <= 1:
                raise EvaluatorProtocolError(
                    "semantic judge confidence must be within [0, 1]"
                )
            return {
                **result,
                "confidence": confidence,
                "status": "success",
                "attempt": attempt,
                "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
                "prompt_sha256": prompt_sha256,
                "deterministic_equivalent": bool(equivalence["equivalent"]),
            }
        except (EvaluatorProtocolError, TypeError, ValueError) as exc:
            last_error = exc
    return {
        "semantically_equivalent": False,
        "confidence": 0.0,
        "reason": (
            f"judge protocol failure: {type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "judge protocol failure"
        ),
        "format_only_difference": False,
        "status": "failed",
        "attempt": int(pipeline["semantic_judge_attempts"]),
        "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
        "prompt_sha256": prompt_sha256,
        "deterministic_equivalent": bool(equivalence["equivalent"]),
    }
