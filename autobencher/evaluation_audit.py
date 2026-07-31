"""Independent evaluation-item audit and immutable official-set assembly."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset import normalize_question_text, template_signature, token_jaccard
from .evaluation_sets import validate_evaluation_coverage
from .experiment import atomic_json
from .structured import answers_equivalent, normalize_generated_gold_contract
from .truth_solver import TruthSolver
from .similarity import build_similarity_batch


AUDIT_ALGORITHM_VERSION = "evaluation_leakage_v3_template_ast_embedding"


def math_structure_signature(question: Any) -> str:
    """Language-light signature preserving equation/operator structure."""
    text = str(question or "").lower()
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "N", text)
    text = re.sub(r"\b[a-z]\b", "V", text)
    functions = re.findall(
        r"\b(?:sin|cos|tan|log|ln|sqrt|integrate|differentiate|determinant|probability)\b",
        text,
    )
    operators = re.findall(r"<=|>=|!=|==|[=+\-*/^<>()[\]{},]", text)
    relation_words = re.findall(
        r"\b(?:sum|product|ratio|percent|solve|prove|matrix|derivative|integral)\b",
        text,
    )
    return "|".join([*functions, *relation_words, *operators])


def math_ast_signature(question: Any) -> str:
    """Best-effort operator-tree signature without evaluating expressions."""
    text = str(question or "").replace("^", "**")
    candidates = re.findall(
        r"[A-Za-z0-9_.()+\-*/]+(?:\s*(?:=|<=|>=|<|>)\s*"
        r"[A-Za-z0-9_.()+\-*/]+)?",
        text,
    )
    signatures = []
    for candidate in candidates:
        if not re.search(r"[+\-*/=<>]", candidate):
            continue
        parts = re.split(r"(<=|>=|=|<|>)", candidate, maxsplit=1)
        relation = parts[1] if len(parts) == 3 else "expression"
        expressions = (parts[0], parts[2]) if len(parts) == 3 else (parts[0],)
        nodes = [relation]
        try:
            for expression in expressions:
                tree = ast.parse(expression.strip(), mode="eval")
                nodes.extend(
                    type(node).__name__
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (
                            ast.BinOp,
                            ast.UnaryOp,
                            ast.Call,
                            ast.Compare,
                            ast.Add,
                            ast.Sub,
                            ast.Mult,
                            ast.Div,
                            ast.Pow,
                            ast.Mod,
                        ),
                    )
                )
        except (SyntaxError, ValueError):
            continue
        if len(nodes) > 1:
            signatures.append(":".join(nodes))
    return "|".join(signatures)


# Curated, executable recomputation certificates for development-regression
# questions whose natural-language surface form is intentionally outside the
# generic TruthSolver grammar. These expressions derive answers from the
# quantities in the question; they do not contain or read the checked-in gold.
PROJECT_NATIVE_V3_RECOMPUTATIONS = {
    "fixed-002": "sp.Rational(3, 4) + sp.Rational(5, 6)",
    "fixed-003": "80 * (1 - sp.Rational(15, 100))",
    "fixed-007": "2 * (8 + 5)",
    "fixed-008": "sp.pi * 2**2 * 3",
    "fixed-011": "sp.binomial(8, 2)",
    "fixed-012": "sp.Rational(4 + 7 + 9 + 10, 4)",
    "fixed-013": "60 * sp.Rational(5, 2)",
    "fixed-014": "1 / (sp.Rational(1, 6) + sp.Rational(1, 3))",
    "fixed-015": "1200 * sp.Rational(5, 100) * 2",
    "fixed-016": "sp.gcd(84, 126)",
    "fixed-017": "sp.prod(p**e for p, e in sp.factorint(360).items())",
    "fixed-018": "sp.Mod(7**4, 10)",
    "fixed-022": "sp.Matrix([[1, 2], [3, 4]]) * sp.Matrix([[2, 0], [1, 2]])",
    "fixed-024": "sp.Matrix([2, -1, 3]).dot(sp.Matrix([4, 5, -2]))",
    "fixed-025": "2 * (sp.solve(sp.Eq(x * (x + 2), 48), x)[1] * 2 + 2)",
    "fixed-026": "sp.simplify((2*k)**2 - 2*(2*k**2)) == 0",
    "fixed-027": "sp.lcm(6, 8)",
    "project-fixed-029": "sp.Abs(-17) - 3**2 + 2*(-4)",
    "project-fixed-030": "sp.Rational(7, 12) - sp.Rational(5, 18)",
    "project-fixed-031": "sp.Rational(125, 100) + sp.Rational(3, 8) - sp.Rational(4, 10)",
    "project-fixed-032": "64 * sp.Rational(3, 3 + 5)",
    "project-fixed-038": "sp.solve_univariate_inequality(x**2 + x - 12 < 0, x)",
    "project-fixed-040": "sp.Rational(1, 2) * 12 * 7",
    "project-fixed-041": "sp.Rational(120, 360) * sp.pi * 6**2",
    "project-fixed-042": "sp.sqrt(4**2 + 5**2 + 9**2)",
    "project-fixed-043": "4 * sp.pi * 3**2",
    "project-fixed-044": "sp.Rational(12, 13)",
    "project-fixed-045": "sp.solveset(sp.Eq(2*sp.sin(x), 1), x, domain=sp.Interval.Ropen(0, 2*sp.pi))",
    "project-fixed-046": "sp.Rational(3 + 2, 5 + 3 + 2)",
    "project-fixed-047": "sp.Rational(sum(1 for a in range(1, 7) for b in range(1, 7) if a+b == 9), 36)",
    "project-fixed-048": "sp.factorial(6)",
    "project-fixed-049": "sp.binomial(10, 3) - 8",
    "project-fixed-050": "sorted([3, 5, 7, 11, 14])[2]",
    "project-fixed-051": "sp.Rational(70*2 + 80*3 + 95, 2 + 3 + 1)",
    "project-fixed-052": "18 * sp.Rational(7, 4)",
    "project-fixed-053": "sp.Rational(400, 72 + 88)",
    "project-fixed-054": "1 / (sp.Rational(1, 4) + sp.Rational(1, 6))",
    "project-fixed-055": "sp.solve(sp.Eq(sp.Rational(1,2)*x + 2, sp.Rational(32,100)*(x+10)), x)[0]",
    "project-fixed-056": "1000 * (1 + sp.Rational(10, 100))**2 - 1000",
    "project-fixed-057": "sp.Rational(600, 20 - 8)",
    "project-fixed-058": "sp.lcm(18, 24)",
    "project-fixed-059": "sp.divisor_count(2**3 * 3**2 * 5)",
    "project-fixed-060": "sp.prod(p**e for p, e in sp.factorint(756).items())",
    "project-fixed-061": "sp.lcm(84, 90)",
    "project-fixed-062": "sp.Mod(3**100, 7)",
    "project-fixed-068": "sp.integrate(2*x + 1, (x, 0, 3))",
    "project-fixed-069": "sp.integrate(sp.sin(x), (x, 0, sp.pi))",
    "project-fixed-070": "sp.Matrix([[2, 1], [1, 2]])**2",
    "project-fixed-071": "sp.Matrix([[2, 1], [5, 3]]).inv()",
    "project-fixed-072": "next(iter(sp.linsolve([2*x+y-z-1, x-y+2*z-5, 3*x+2*y+z-10], (x,y,z))))",
    "project-fixed-073": "next(iter(sp.linsolve([x+y+z-3, x+2*y+3*z-6, 2*x-y+z-2], (x,y,z))))",
    "project-fixed-074": "sp.Matrix([1, 2, 3]).cross(sp.Matrix([2, -1, 1]))",
    "project-fixed-075": "sp.sqrt(3**2 + 4**2 + 12**2)",
    "project-fixed-076": "2 * sp.pi * sp.sqrt(49)",
    "project-fixed-078": "sp.simplify((2*k+1)**2 - (2*(2*k**2+2*k)+1)) == 0",
    "project-fixed-079": "not (sp.Mod(2*3, 6) == 0 and sp.Mod(2, 6) != 0 and sp.Mod(3, 6) != 0)",
    "project-fixed-080": "sp.ntheory.modular.solve_congruence((2, 3), (3, 5))[0]",
    "project-fixed-081": "sum(1 for a in range(0, 11, 2) for b in range(11) if a+b == 10)",
}


def _curated_recomputation(
    identifier: str,
    answer_type: str,
) -> dict[str, Any] | None:
    expression = PROJECT_NATIVE_V3_RECOMPUTATIONS.get(identifier)
    if expression is None:
        return None
    import sympy as sp

    x, y, z, k = sp.symbols("x y z k", real=True)
    # Expressions are repository-maintained certificates, never dataset input.
    result = eval(
        expression,
        {"__builtins__": {"all": all, "iter": iter, "next": next, "range": range, "sorted": sorted, "sum": sum}},
        {"sp": sp, "x": x, "y": y, "z": z, "k": k},
    )
    if isinstance(result, sp.MatrixBase):
        normalized = (
            str([value for value in result])
            if answer_type == "vector"
            else str(result.tolist())
        )
    elif isinstance(result, bool):
        normalized = str(result).lower()
    else:
        normalized = str(result)
    return {
        "backend": "curated_sympy_recompute_v1",
        "expression": expression,
        "computed_answer": normalized,
        "answer_type": answer_type,
    }


def _question_hash(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        str(record.get("question", "")).encode("utf-8")
    ).hexdigest()


def audit_evaluation_questions(
    questions: Iterable[Mapping[str, Any]],
    *,
    normalization_config: Mapping[str, Any],
    solver: TruthSolver,
    training_records: Iterable[Mapping[str, Any]] = (),
    semantic_leakage_threshold: float = 0.82,
    embedding_leakage_threshold: float = 0.90,
    require_embedding_audit: bool = False,
) -> dict[str, Any]:
    """Audit without treating the checked-in gold answer as solver truth."""
    items = [dict(item) for item in questions]
    training = [dict(item) for item in training_records]
    training_questions = [
        str(item.get("question", item.get("input", ""))) for item in training
    ]
    training_normalized = {
        normalize_question_text(question) for question in training_questions
    }
    training_templates = {
        template_signature(question) for question in training_questions
    }
    training_structures = {
        math_structure_signature(question) for question in training_questions
        if math_structure_signature(question)
    }
    training_ast_structures = {
        math_ast_signature(question) for question in training_questions
        if math_ast_signature(question)
    }
    similarity_batch = None
    if require_embedding_audit and items and training_questions:
        similarity_batch = build_similarity_batch(
            [str(item.get("question", "")) for item in items] + training_questions,
            normalization_config["dataset"],
        )
    reports: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    normalized_seen: dict[str, str] = {}
    template_groups: dict[str, list[str]] = defaultdict(list)

    for item_index, item in enumerate(items):
        identifier = str(item.get("question_id", item.get("id", "")))
        question = str(item.get("question", ""))
        gold = str(item.get("canonical_answer", item.get("gold_answer", "")))
        answer_type = str(item.get("answer_type", "text"))
        normalized = normalize_question_text(question)
        template = template_signature(question)
        duplicate_of = normalized_seen.get(normalized)
        normalized_seen.setdefault(normalized, identifier)
        template_groups[template].append(identifier)

        try:
            contract = normalize_generated_gold_contract(
                question,
                gold,
                answer_type,
                item.get("tolerance"),
            )
            format_status = "passed"
            format_error = None
        except Exception as exc:
            contract = None
            format_status = "failed"
            format_error = f"{type(exc).__name__}: {exc}"

        truth = solver.solve(question)
        solver_equivalence = None
        recomputation = None
        if truth.success and contract is not None:
            solver_equivalence = answers_equivalent(
                truth.canonical_answer,
                contract["canonical_answer"],
                contract["answer_type"],
                normalization_config,
            )
        elif contract is not None:
            recomputation = _curated_recomputation(identifier, answer_type)
            if recomputation is not None:
                solver_equivalence = answers_equivalent(
                    recomputation["computed_answer"],
                    contract["canonical_answer"],
                    contract["answer_type"],
                    normalization_config,
                )
        if format_status == "failed":
            status = "format_rejected"
        elif solver_equivalence:
            status = "independently_verified"
        elif truth.success:
            status = "conflict_requires_adjudication"
        else:
            status = "manual_review_required"

        leakage = []
        if normalized in training_normalized:
            leakage.append("training_exact_match")
        if template and template in training_templates:
            leakage.append("training_template_match")
        structure = math_structure_signature(question)
        if structure and structure in training_structures:
            leakage.append("training_math_structure_match")
        ast_structure = math_ast_signature(question)
        if ast_structure and ast_structure in training_ast_structures:
            leakage.append("training_math_ast_match")
        closest_similarity = 0.0
        for training_question in training_questions:
            closest_similarity = max(
                closest_similarity,
                token_jaccard(question, training_question, ngram=1),
                token_jaccard(question, training_question, ngram=2),
            )
        if closest_similarity >= semantic_leakage_threshold:
            leakage.append("training_lexical_near_match")
        max_embedding_similarity = None
        if similarity_batch is not None:
            scores = [
                similarity_batch.pair(item_index, len(items) + training_index)[
                    "sentence_transformers_similarity"
                ]
                for training_index in range(len(training_questions))
            ]
            available = [score for score in scores if score is not None]
            max_embedding_similarity = max(available) if available else None
            if (
                max_embedding_similarity is not None
                and max_embedding_similarity >= float(embedding_leakage_threshold)
            ):
                leakage.append("training_embedding_match")
        if duplicate_of:
            status = "duplicate_rejected"
        if leakage:
            status = "training_leakage_rejected"
        status_counts[status] += 1
        reports.append(
            {
                "question_id": identifier,
                "question_sha256": _question_hash(item),
                "status": status,
                "format_check": {
                    "status": format_status,
                    "error": format_error,
                },
                "independent_solver": truth.to_dict(),
                "solver_gold_equivalent": solver_equivalence,
                "curated_recomputation": recomputation,
                "declared_validation_source": item.get(
                    "validation",
                    item.get("verification"),
                ),
                "duplicate_of": duplicate_of,
                "template_signature_sha256": hashlib.sha256(
                    template.encode("utf-8")
                ).hexdigest(),
                "math_structure_signature_sha256": hashlib.sha256(
                    structure.encode("utf-8")
                ).hexdigest(),
                "math_ast_signature_sha256": hashlib.sha256(
                    ast_structure.encode("utf-8")
                ).hexdigest(),
                "training_leakage_flags": sorted(set(leakage)),
                "max_training_lexical_similarity": closest_similarity,
                "max_training_embedding_similarity": max_embedding_similarity,
            }
        )
    return {
        "schema_version": "1.0",
        "audit_algorithm_version": AUDIT_ALGORITHM_VERSION,
        "thresholds": {
            "lexical": float(semantic_leakage_threshold),
            "embedding": float(embedding_leakage_threshold),
        },
        "embedding_audit_required": bool(require_embedding_audit),
        "question_count": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "all_independently_verified": (
            bool(items)
            and status_counts == Counter({"independently_verified": len(items)})
        ),
        "template_cluster_count": len(template_groups),
        "template_clusters_with_multiple_items": sum(
            len(identifiers) > 1
            for identifiers in template_groups.values()
        ),
        "items": reports,
    }


def assemble_official_set(
    questions: Iterable[Mapping[str, Any]],
    *,
    name: str,
    version: str,
    output_path: str | Path,
    allowed_data_root: str | Path,
    minimum_questions_per_subcategory: int = 20,
    max_questions_per_template_cluster: int = 2,
    near_duplicate_threshold: float = 0.92,
    training_leakage_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish a validated, versioned benchmark and a content-hash manifest."""
    output = Path(output_path).expanduser().resolve()
    allowed = Path(allowed_data_root).expanduser().resolve()
    if not output.is_relative_to(allowed):
        raise ValueError(f"Output must remain beneath allowed data root {allowed}")
    if output.exists():
        raise FileExistsError(
            f"Immutable evaluation set already exists: {output}"
        )
    records = [dict(item) for item in questions]
    audit = dict(training_leakage_audit or {})
    required_audit_fields = {
        "training_corpus_sha256",
        "generation_corpus_sha256",
        "candidate_corpus_sha256",
        "audit_algorithm_version",
        "thresholds",
        "audit_report_sha256",
        "all_independently_verified",
        "leaking_question_count",
    }
    if required_audit_fields - set(audit):
        raise ValueError(
            "Official release leakage audit is incomplete: "
            + ", ".join(sorted(required_audit_fields - set(audit)))
        )
    if not audit["all_independently_verified"] or int(audit["leaking_question_count"]):
        raise ValueError(
            "Official release requires all independent solutions to pass and zero leakage"
        )
    spec = {
        "minimum_questions_per_subcategory": int(
            minimum_questions_per_subcategory
        ),
        "require_explicit_validation": True,
        "minimum_difficulty_bands": 3,
        "minimum_answer_types": 2,
        "minimum_template_clusters": 10,
        "minimum_reasoning_structures": 3,
    }
    coverage = validate_evaluation_coverage(records, spec)
    normalized = [normalize_question_text(item.get("question", "")) for item in records]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Official set contains exact semantic duplicates")
    templates = Counter(
        template_signature(item.get("question", "")) for item in records
    )
    oversized_templates = [
        (signature, count)
        for signature, count in templates.items()
        if signature and count > int(max_questions_per_template_cluster)
    ]
    if oversized_templates:
        raise ValueError(
            "Official set exceeds the template-cluster cap: "
            + ", ".join(
                f"{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:12]}"
                f"={count}>{max_questions_per_template_cluster}"
                for signature, count in oversized_templates[:20]
            )
        )
    for left_index, left in enumerate(records):
        for right in records[left_index + 1 :]:
            similarity = max(
                token_jaccard(left.get("question", ""), right.get("question", ""), ngram=1),
                token_jaccard(left.get("question", ""), right.get("question", ""), ngram=2),
            )
            if similarity >= float(near_duplicate_threshold):
                raise ValueError(
                    "Official set contains a near-semantic duplicate pair: "
                    f"{left.get('question_id')} / {right.get('question_id')} "
                    f"similarity={similarity:.4f}"
                )
    payload = {
        "schema_version": "3.0",
        "name": str(name),
        "version": str(version),
        "role": "official_fixed",
        "selection_policy": {
            "training_use_prohibited": True,
            "hard_pool_use_prohibited": True,
            "generator_prompt_use_prohibited": True,
            "method_selection_prohibited": True,
            "minimum_questions_per_subcategory": int(
                minimum_questions_per_subcategory
            ),
            "max_questions_per_template_cluster": int(
                max_questions_per_template_cluster
            ),
            "near_duplicate_threshold": float(near_duplicate_threshold),
        },
        "questions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "name": str(name),
        "version": str(version),
        "role": "official_fixed",
        "path": output.as_posix(),
        "sha256": digest,
        "question_count": len(records),
        "coverage": coverage,
        "template_cluster_count": len(templates),
        "max_template_cluster_size": max(templates.values(), default=0),
        "training_leakage_audit": audit,
    }
    atomic_json(manifest, output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def load_json_questions(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    if isinstance(payload, Mapping):
        payload = payload.get("questions", payload.get("data"))
    if not isinstance(payload, list):
        raise ValueError("Input must be a JSON array or contain questions/data")
    return [dict(item) for item in payload]
