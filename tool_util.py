import copy
import contextlib
import hashlib
import re
import requests
import os, argparse, ast, json, tqdm
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote
from time import sleep
from collections import defaultdict
import numpy as np
from util import gen_from_prompt
from autobencher.output_schemas import TestTakerOutput
from autobencher.structured import parse_test_taker_output, test_taker_prompt


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_HEADERS = {
    "User-Agent": (
        "AutoBencher/1.0 "
        "(https://github.com/XiangLi1999/AutoBencher; research benchmark)"
    )
}


DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""

DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."


# [ADDED] Fixed English math taxonomy and error-tag vocabulary.
MATH_CATEGORIES = (
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

ERROR_TAGS = (
    "calculation_error",
    "formula_memory_error",
    "condition_missing",
    "multi-step_logic_error",
    "concept_confusion",
    "format_output_error",
    "tool_violation",
    "irrelevant_output",
    "prompt_echo",
    "parse_failed",
    "unknown_error",
)

SAMPLE_GRADES = (
    "train_eligible",
    "hard_unsuitable",
    "easy_sample",
)

ACCURACY_BUCKETS = (
    "below_0.1",
    "0.1-0.4",
    "above_0.4",
)

SUB_CATEGORY_TAXONOMY = {
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

ALL_SUB_CATEGORIES = tuple(
    sub_category
    for category in MATH_CATEGORIES
    for sub_category in SUB_CATEGORY_TAXONOMY[category]
)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "correct"}


def _as_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _contains_cjk(value):
    return bool(re.search(r"[\u3400-\u9fff]", str(value)))


def _english_only(value, fallback="", preserve_newlines=False):
    cleaned = re.sub(r"[\u3400-\u9fff]+", " ", str(value or ""))
    if preserve_newlines:
        cleaned = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in cleaned.splitlines()
        ).strip()
    else:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or fallback


# [ADDED] Write readable UTF-8 JSON atomically with two-space indentation.
def dump_standard_json(data, output_file):
    output_file = os.fspath(output_file)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    temporary_output = f"{output_file}.tmp"
    with open(temporary_output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary_output, output_file)


def read_json_records(path):
    """Read a standard JSON array/object or migrate a legacy JSONL cache."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        records = []
        for line in raw.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records
    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], list):
            return data[0]
        return data
    return [data] if isinstance(data, dict) else []


def normalize_math_category(category, sub_category="", question=""):
    text = f"{category} {sub_category} {question}".lower()
    category_aliases = (
        ("Composite Comprehensive", ("composite", "cross-domain", "comprehensive")),
        ("Linear Algebra", ("linear algebra", "matrix", "vector space", "eigen")),
        ("Calculus", ("calculus", "derivative", "integral", "limit", "continuity")),
        ("Number Theory", ("number theory", "prime", "divisib", "modular", "congruence")),
        ("Probability & Statistics", ("probability", "statistics", "random", "variance", "mean")),
        ("Geometry & Trigonometry", ("geometry", "trigonometry", "triangle", "circle", "volume")),
        ("Word Problems", ("word problem", "rate", "mixture", "distance", "interest", "work")),
        ("Algebra", ("algebra", "equation", "polynomial", "inequality", "variable")),
        ("Arithmetic", ("arithmetic", "fraction", "decimal", "addition", "subtraction")),
    )
    for normalized, aliases in category_aliases:
        if any(alias in text for alias in aliases):
            return normalized
    return "Arithmetic"


def normalize_sub_category(sub_category, category, question=""):
    value = re.sub(r"\s+", " ", str(sub_category or "")).strip()
    category_lookup = {
        item.lower(): item for item in SUB_CATEGORY_TAXONOMY[category]
    }
    if value.lower() in category_lookup:
        return category_lookup[value.lower()]
    all_lookup = {item.lower(): item for item in ALL_SUB_CATEGORIES}
    if value.lower() in all_lookup:
        value = ""
    if value and not _contains_cjk(value):
        return value

    text = f"{value} {question}".lower()
    keyword_map = (
        ("Solid Geometry", ("volume", "surface area", "sphere", "cylinder", "cone")),
        ("Trigonometric Reasoning", ("sine", "cosine", "tangent", "trigon")),
        ("Basic Probability", ("probability", "random", "dice", "coin")),
        ("Combinatorics", ("combination", "permutation", "arrangement")),
        ("Matrix Operations", ("matrix", "determinant")),
        ("Linear Systems", ("linear system", "simultaneous equation")),
        ("Differentiation", ("derivative", "differentiat")),
        ("Integration", ("integral", "integrat")),
        ("Limits and Continuity", ("limit", "continuity")),
        ("Modular Arithmetic", ("modular", "congruence", "remainder")),
        ("Prime Factorization", ("prime factor", "prime number")),
        ("Rate and Distance", ("speed", "distance", "travel", "rate")),
        ("Work and Mixture", ("mixture", "work together", "combined work")),
        ("Financial Applications", ("interest", "profit", "discount", "investment")),
        ("Linear Equations", ("linear equation", "solve for")),
        ("Fraction and Decimal Operations", ("fraction", "decimal")),
        ("Ratio and Percentage", ("ratio", "percent", "percentage")),
    )
    for normalized, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            return normalized
    return SUB_CATEGORY_TAXONOMY[category][0]


def build_unique_key(category, sub_category, question):
    normalized = "|".join(
        re.sub(r"\s+", " ", str(value)).strip().lower()
        for value in (category, sub_category, question)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def classify_error_tags(record):
    """Assign one or more tags from the fixed English error-tag enumeration."""
    category = normalize_math_category(
        record.get("category"),
        record.get("sub_category", record.get("subcat", "")),
        record.get("question", ""),
    )
    text = " ".join(
        str(record.get(key, ""))
        for key in (
            "question",
            "reasons",
            "error_reason",
            "gold_answer",
            "test_taker_response",
            "test_taker_answer",
        )
    ).lower()
    tags = []
    if category == "Arithmetic" or any(
        token in text for token in ("calculation", "arithmetic", "computed", "numeric")
    ):
        tags.append("calculation_error")
    if category in {"Geometry & Trigonometry", "Calculus", "Linear Algebra"} or any(
        token in text for token in ("formula", "identity", "theorem")
    ):
        tags.append("formula_memory_error")
    if any(
        token in text
        for token in ("condition", "constraint", "domain", "unit", "assumption", "missing")
    ):
        tags.append("condition_missing")
    if category in {"Word Problems", "Composite Comprehensive"} or len(
        re.findall(r"\w+", str(record.get("question", "")))
    ) >= 35:
        tags.append("multi-step_logic_error")
    if not tags or any(
        token in text for token in ("concept", "confus", "misunderstood", "incorrect method")
    ):
        tags.append("concept_confusion")
    return [tag for tag in ERROR_TAGS if tag in tags]


def canonicalize_math_record(record, index=0):
    """Convert new or legacy question/inference data to the required schema."""
    question = _english_only(
        record.get("question", ""), "Legacy non-English question"
    )
    category = normalize_math_category(
        record.get("category", ""),
        record.get("sub_category", record.get("subcat", "")),
        question,
    )
    sub_category = normalize_sub_category(
        record.get(
            "sub_category",
            record.get("subcat", record.get("subcategory_description", "")),
        ),
        category,
        question,
    )
    gold_answer = _english_only(
        record.get("gold_answer", record.get("answer", "")),
        "N/A",
    )
    response = _english_only(
        record.get("test_taker_response", record.get("test_taker_answer", ""))
    )
    prompt = _english_only(
        record.get("prompt", ""), preserve_newlines=True
    ) or (
        "Output just with the final answer to the question.\n"
        f"Question:{question}\nAnswer:"
    )
    unique_key = build_unique_key(category, sub_category, question)
    existing_tags = record.get("error_tags")
    error_tags = (
        [tag for tag in existing_tags if tag in ERROR_TAGS]
        if isinstance(existing_tags, list)
        else []
    )
    canonical = {
        "id": _as_int(record.get("id"), index + 1),
        "question_id": str(
            record.get(
                "question_id",
                record.get("id", index + 1),
            )
        ),
        "category": category,
        "sub_category": sub_category,
        "difficulty": _as_int(record.get("difficulty"), 5),
        "question": question,
        "gold_answer": gold_answer,
        "answer_type": str(record.get("answer_type", "text")),
        "canonical_answer": record.get(
            "canonical_answer",
            record.get("gold_answer", record.get("answer", "")),
        ),
        "display_answer": str(
            record.get(
                "display_answer",
                record.get("gold_answer", record.get("answer", "")),
            )
        ),
        "unit": record.get("unit"),
        "tolerance": record.get("tolerance"),
        "order_sensitive": bool(record.get("order_sensitive", False)),
        "test_taker_response": response,
        "prompt": prompt,
        "is_correct": _as_bool(record.get("is_correct")),
        "error_tags": error_tags,
        "unique_key": unique_key,
    }
    preserved_fields = (
        "generation_source",
        "reference_hard_sample_ids",
        "target_error_type",
        "generation_strategy",
        # [ADDED] Preserve the evaluator's independent gold-answer audit
        # through inference, evaluation, hard-pool, and research exports.
        "gold_answer_validation",
        "truth_validation_details",
        "test_taker_truth_validation",
        "failure_type",
        "raw_response",
        "parsed_response",
        "parse_status",
        "repair_attempts",
        "contains_prompt_echo",
        "contains_irrelevant_content",
        "extraneous_content_discarded",
        "discarded_prefix_chars",
        "discarded_suffix_chars",
        "reasoning_steps_truncated",
        "original_reasoning_step_count",
        "reasoning_steps_dropped",
        "reasoning_step_chars_truncated",
        "parser_version",
        "tool_violation",
        "test_taker_tool_call_count",
        "evaluation_status",
        "normalized_gold_answer",
        "normalized_test_taker_answer",
        "deterministic_checks",
        "primary_error_tag",
        "secondary_error_tags",
        "evidence",
        "attribution_confidence",
        "needs_review",
        "evaluator_tool_calls",
        "question_parse_success",
        "answer_validation_success",
        "evaluator_confidence",
        "semantic_judge",
        "semantic_judge_reason",
        "judge_deterministic_agreement",
        "ambiguous",
        "format_only_error",
        "fixed_test",
        "run_id",
        "cycle_id",
        "iteration_id",
        "iteration_uid",
        "config_hash",
        "prompt_hash",
        "cache_status",
    )
    for field in preserved_fields:
        if field in record:
            canonical[field] = copy.deepcopy(record[field])
    return canonical


def load_math_inference(inference_file):
    return [
        canonicalize_math_record(record, index)
        for index, record in enumerate(read_json_records(inference_file))
        if isinstance(record, dict) and record.get("question")
    ]


# [ADDED] Standard math inference cache; Wiki and Multilingual stay unchanged.
def generate_math_inference(
    question_inputs,
    test_model_info,
    outfile,
    bsz=1,
    temperature=0.01,
    max_length=50,
    research_config=None,
    progress_manager=None,
    cycle=None,
    iteration=None,
):
    if len(question_inputs) == 1 and isinstance(question_inputs[0], list):
        question_inputs = question_inputs[0]
    canonical_records = [
        canonicalize_math_record(record, index)
        for index, record in enumerate(question_inputs)
    ]
    existing_records = load_math_inference(outfile)
    existing_by_key = {record["unique_key"]: record for record in existing_records}
    for record in canonical_records:
        existing = existing_by_key.get(record["unique_key"])
        if existing:
            record.update(existing)

    if research_config:
        for record in canonical_records:
            record["prompt"] = test_taker_prompt(record, research_config)
            raw_response = str(record.get("raw_response", "") or "")
            if not raw_response:
                continue
            reparsed = parse_test_taker_output(
                raw_response,
                record["prompt"],
                record["answer_type"],
                research_config,
            )
            record.update(reparsed)
            record["parser_version"] = "structured_v2"
            if reparsed["parse_status"] == "success":
                record["test_taker_response"] = reparsed[
                    "parsed_response"
                ]["final_answer"]
            else:
                # [MODIFIED] Invalid historical responses are retryable. A
                # non-empty status marker must never make a failed record look
                # like a completed inference cache entry.
                record["test_taker_response"] = ""

    # Persist the full question set before the first request so this file alone
    # is sufficient to resume an interrupted inference run.
    dump_standard_json(canonical_records, outfile)
    if len(test_model_info) == 3:
        model_choice, tokenizer_choice, client_choice = test_model_info
        auth = None
        use_helm = False
    elif len(test_model_info) == 4:
        model_choice, tokenizer_choice, client_choice, auth = test_model_info
        use_helm = True
    else:
        raise ValueError("Unexpected test model configuration")

    pending = [
        index
        for index, record in enumerate(canonical_records)
        if not record["test_taker_response"]
    ]
    if not pending:
        print(f"FOUND completed inference cache {outfile}")
        return canonical_records
    print(
        f"writing to {outfile} "
        f"(resuming with {len(pending)}/{len(canonical_records)} unanswered records)"
    )
    if research_config:
        temperature = float(
            research_config["models"]["test_taker"]["temperature"]
        )
        max_length = int(
            research_config["models"]["test_taker"]["max_new_tokens"]
        )
    progress_context = (
        progress_manager.stage(
            "Infer",
            total=len(pending),
            cycle=cycle,
            iteration=iteration,
        )
        if progress_manager
        else contextlib.nullcontext(
            tqdm.tqdm(
                total=len(pending),
                leave=False,
                dynamic_ncols=True,
            )
        )
    )
    with progress_context as progress:
        for offset in range(0, len(pending), bsz):
            batch_indices = pending[offset:offset + bsz]
            prompts = [canonical_records[index]["prompt"] for index in batch_indices]
            request_result = gen_from_prompt(
                model=model_choice,
                tokenizer=tokenizer_choice,
                prompt=prompts,
                echo_prompt=False,
                temperature=temperature,
                max_tokens=max_length,
                service=client_choice,
                terminate_by_linebreak="no",
                stop_sequences=(
                    research_config["test_taker_prompt"]["stop_sequences"]
                    if research_config
                    else None
                ),
                use_helm=use_helm,
                auth=auth,
                verbose=False,
                structured_schema=(
                    TestTakerOutput
                    if research_config
                    and research_config["structured_output"]["enabled"]
                    and research_config["structured_output"][
                        "use_for_test_taker"
                    ]
                    else None
                ),
                structured_backend=(
                    research_config["structured_output"]["local_backend"]
                    if research_config
                    else "none"
                ),
                structured_fallback_backend=(
                    research_config["structured_output"]["fallback_backend"]
                    if research_config
                    else "none"
                ),
                structured_required=(
                    bool(research_config["structured_output"]["required"])
                    if research_config
                    else False
                ),
            )
            for index, completion in zip(batch_indices, request_result.completions):
                record = canonical_records[index]
                raw_response = str(completion.text or "")
                if research_config:
                    parsed = parse_test_taker_output(
                        raw_response,
                        record["prompt"],
                        record["answer_type"],
                        research_config,
                    )
                    record.update(parsed)
                    record["parser_version"] = "structured_v2"
                    record["test_taker_response"] = (
                        parsed["parsed_response"].get("final_answer", "")
                        if parsed["parse_status"] == "success"
                        else parsed["parse_status"].upper()
                    )
                    record["test_taker_tool_call_count"] = int(
                        parsed["tool_violation"]
                    )
                else:
                    record["test_taker_response"] = _english_only(
                        raw_response,
                        "NON_ENGLISH_RESPONSE",
                    )
                progress.update(1)
            dump_standard_json(canonical_records, outfile)
    return canonical_records


# [ADDED] Grade wrong answers using their measured sub-category accuracy.
def _sample_grade_for_accuracy(accuracy):
    if accuracy < 0.1:
        return "hard_unsuitable", "below_0.1"
    if accuracy <= 0.4:
        return "train_eligible", "0.1-0.4"
    return "easy_sample", "above_0.4"


# [MODIFIED] Deduplicate, grade, and retain wrong answers in the global pool.
def manage_hard_pool(
    inference_file,
    hard_pool_file,
    source_iter,
    source_cycle=1,
):
    inference_records = load_math_inference(inference_file)
    grouped = defaultdict(list)
    for record in inference_records:
        grouped[(record["category"], record["sub_category"])].append(record)
    accuracy_by_group = {
        key: sum(_as_bool(item.get("is_correct")) for item in records) / len(records)
        for key, records in grouped.items()
        if records
    }
    existing_raw = read_json_records(hard_pool_file)
    if (
        len(existing_raw) == 1
        and isinstance(existing_raw[0], dict)
        and isinstance(existing_raw[0].get("groups"), dict)
    ):
        existing_raw = [
            sample
            for group in existing_raw[0]["groups"].values()
            if isinstance(group, list)
            for sample in group
        ]
    existing = {}
    for index, sample in enumerate(existing_raw):
        if not isinstance(sample, dict) or not sample.get("question"):
            continue
        canonical = canonicalize_math_record(sample, index)
        grade = sample.get("sample_grade", "hard_unsuitable")
        if grade not in SAMPLE_GRADES:
            grade = "hard_unsuitable"
        bucket = sample.get("accuracy_bucket", "below_0.1")
        if bucket not in ACCURACY_BUCKETS:
            bucket = "below_0.1"
        hard_sample = {
            "source_iter": _as_int(sample.get("source_iter"), source_iter),
            "source_cycle": _as_int(sample.get("source_cycle"), source_cycle),
            "last_seen_iter": _as_int(
                sample.get("last_seen_iter", sample.get("source_iter")),
                source_iter,
            ),
            "last_seen_cycle": _as_int(
                sample.get("last_seen_cycle", sample.get("source_cycle")),
                source_cycle,
            ),
            "category": canonical["category"],
            "sub_category": canonical["sub_category"],
            "question": canonical["question"],
            "gold_answer": canonical["gold_answer"],
            "test_taker_response": canonical["test_taker_response"],
            "error_tags": canonical["error_tags"] or classify_error_tags(sample),
            "difficulty": canonical["difficulty"],
            "unique_key": canonical["unique_key"],
            "sample_grade": grade,
            "accuracy_bucket": bucket,
            "sub_category_accuracy": float(
                sample.get("sub_category_accuracy", 0.0)
            ),
            "occurrences": max(1, _as_int(sample.get("occurrences"), 1)),
        }
        existing[hard_sample["unique_key"]] = hard_sample

    before = len(existing)
    for record in inference_records:
        if record["is_correct"] or not record["test_taker_response"]:
            continue
        group_key = (record["category"], record["sub_category"])
        sub_category_accuracy = accuracy_by_group.get(group_key, 0.0)
        sample_grade, accuracy_bucket = _sample_grade_for_accuracy(
            sub_category_accuracy
        )
        sample = {
            "source_iter": int(source_iter),
            "source_cycle": int(source_cycle),
            "last_seen_iter": int(source_iter),
            "last_seen_cycle": int(source_cycle),
            "category": record["category"],
            "sub_category": record["sub_category"],
            "question": record["question"],
            "gold_answer": record["gold_answer"],
            "test_taker_response": record["test_taker_response"],
            "error_tags": record["error_tags"] or classify_error_tags(record),
            "difficulty": record["difficulty"],
            "unique_key": record["unique_key"],
            "sample_grade": sample_grade,
            "accuracy_bucket": accuracy_bucket,
            "sub_category_accuracy": sub_category_accuracy,
            "occurrences": 1,
        }
        previous = existing.get(sample["unique_key"])
        if previous:
            # [MODIFIED] A resumed iteration must not inflate occurrence counts.
            already_seen_in_iteration = (
                int(previous.get("last_seen_iter", -1)) == int(source_iter)
                and int(previous.get("last_seen_cycle", -1)) == int(source_cycle)
            )
            previous.update(
                {
                    "last_seen_iter": int(source_iter),
                    "last_seen_cycle": int(source_cycle),
                    "test_taker_response": sample["test_taker_response"],
                    "error_tags": sample["error_tags"],
                    "sample_grade": sample_grade,
                    "accuracy_bucket": accuracy_bucket,
                    "sub_category_accuracy": sub_category_accuracy,
                    "occurrences": (
                        previous.get("occurrences", 1)
                        if already_seen_in_iteration
                        else previous.get("occurrences", 1) + 1
                    ),
                }
            )
        else:
            existing[sample["unique_key"]] = sample
    samples = sorted(
        existing.values(),
        key=lambda item: (
            item["source_cycle"],
            item["source_iter"],
            item["unique_key"],
        ),
    )
    dump_standard_json(samples, hard_pool_file)
    return len(existing) - before, len(samples)


# [ADDED] Export eligible hard samples as Alpaca JSONL.
def export_training_dataset(hard_pool_file, output_file):
    records = [
        record
        for record in read_json_records(hard_pool_file)
        if isinstance(record, dict)
        and record.get("sample_grade") == "train_eligible"
        and record.get("question")
        and record.get("gold_answer")
    ]
    deduplicated = {
        record.get("unique_key")
        or build_unique_key(
            record.get("category", ""),
            record.get("sub_category", ""),
            record["question"],
        ): record
        for record in records
    }
    output_file = os.path.abspath(os.fspath(output_file))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    temporary_output = f"{output_file}.tmp"
    with open(temporary_output, "w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(deduplicated):
            record = deduplicated[key]
            alpaca_record = {
                "instruction": "Solve the math problem. Return only the final answer.",
                "input": _english_only(record["question"]),
                "output": _english_only(record["gold_answer"]),
            }
            handle.write(json.dumps(alpaca_record, ensure_ascii=False))
            handle.write("\n")
    os.replace(temporary_output, output_file)
    return len(deduplicated)


# [ADDED] Report free disk space before exporting or training.
def check_disk_space(path, warning_threshold_gb=10):
    target = Path(path).resolve()
    existing_target = target
    while not existing_target.exists() and existing_target != existing_target.parent:
        existing_target = existing_target.parent
    usage = shutil.disk_usage(existing_target)
    free_gb = usage.free / (1024 ** 3)
    status = {
        "ok": free_gb >= float(warning_threshold_gb),
        "path": target.as_posix(),
        "free_gb": round(free_gb, 3),
        "warning_threshold_gb": int(warning_threshold_gb),
    }
    level = "INFO" if status["ok"] else "WARNING"
    print(
        f"[DiskCheck] level={level} free_gb={status['free_gb']:.3f} "
        f"threshold_gb={status['warning_threshold_gb']} path={status['path']}"
    )
    return status


# [ADDED] Invoke the repository-local fine-tuning entry point.
def call_local_finetune(
    original_model,
    dataset_path,
    gpu,
    epoch,
    batch,
    lora_rank,
    output_path,
    metrics_path=None,
    run_id=None,
    config_hash=None,
    max_seq_length=None,
    learning_rate=None,
    wandb_enabled=False,
    wandb_mode=None,
    wandb_project=None,
    wandb_entity=None,
    wandb_group=None,
    wandb_tags=None,
    wandb_log_model=False,
):
    script_path = Path(__file__).resolve().with_name("train_llm.py")
    if not script_path.is_file():
        return {
            "success": False,
            "returncode": 2,
            "error": f"Built-in fine-tuning script not found: {script_path}",
        }
    command = [
        sys.executable,
        os.fspath(script_path),
        "--model_name_or_path",
        os.fspath(original_model),
        "--dataset_path",
        os.fspath(dataset_path),
        "--gpu",
        str(gpu),
        "--epoch",
        str(epoch),
        "--batch",
        str(batch),
        "--lora_rank",
        str(lora_rank),
        "--output_path",
        os.fspath(output_path),
    ]
    _optional_arguments = (
        ("--metrics_path", metrics_path),
        ("--run_id", run_id),
        ("--config_hash", config_hash),
        ("--max_seq_length", max_seq_length),
        ("--learning_rate", learning_rate),
        ("--wandb_mode", wandb_mode),
        ("--wandb_project", wandb_project),
        ("--wandb_entity", wandb_entity),
        ("--wandb_group", wandb_group),
        (
            "--wandb_tags",
            (
                ",".join(str(tag) for tag in wandb_tags)
                if isinstance(wandb_tags, (list, tuple))
                else wandb_tags
            ),
        ),
    )
    for option, value in _optional_arguments:
        if value is not None:
            command.extend([option, str(value)])
    if wandb_enabled:
        command.append("--wandb_enabled")
    if wandb_log_model:
        command.append("--wandb_log_model")
    print("[FineTune] command=" + subprocess.list2cmdline(command))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            bufsize=1,
        )
    except OSError as exc:
        return {
            "success": False,
            "returncode": 2,
            "error": str(exc),
        }
    output_lines = []
    if process.stdout is not None:
        for line in process.stdout:
            cleaned = line.rstrip()
            if cleaned:
                print(cleaned)
                output_lines.append(cleaned)
                if len(output_lines) > 200:
                    output_lines.pop(0)
    returncode = process.wait()
    error_output = "\n".join(output_lines)
    return {
        "success": returncode == 0,
        "returncode": returncode,
        "error": "" if returncode == 0 else error_output[-4000:],
    }


# [MODIFIED] Maintain global configuration and cycle-aware iteration indexes.
def update_meta_summary(
    meta_summary_file,
    run_config,
    sub_categories,
    iteration_entry,
    topic_salience_data=None,
):
    existing = read_json_records(meta_summary_file)
    meta = existing[0] if existing and isinstance(existing[0], dict) else {}
    indexes = [
        item
        for item in meta.get("iteration_index", [])
        if isinstance(item, dict)
        and (
            item.get("cycle_num", 1),
            item.get("iter_num"),
        )
        != (
            iteration_entry.get("cycle_num", 1),
            iteration_entry.get("iter_num"),
        )
    ]
    indexes.append(iteration_entry)
    indexes.sort(
        key=lambda item: (
            item.get("cycle_num", 1),
            item.get("iter_num", 0),
        )
    )
    all_sub_categories = sorted(
        {
            str(item)
            for item in meta.get("all_sub_categories", [])
            if not _contains_cjk(item)
        }
        | {
            str(item)
            for item in sub_categories
            if not _contains_cjk(item)
        }
    )
    payload = {
        "run_config": run_config,
        "all_sub_categories": all_sub_categories,
        "topic_salience_data": topic_salience_data or {},
        "iteration_index": indexes,
    }
    dump_standard_json(payload, meta_summary_file)
    return payload


# [ADDED] Remove redundant fragments, attempts, and invalid JSON files.
def clean_redundant_files(
    iteration_dir,
    preserve_json_paths=None,
    strict_json_allowlist=False,
):
    if not os.path.isdir(iteration_dir):
        return []
    preserved = {
        os.path.normcase(os.path.abspath(os.fspath(path)))
        for path in (preserve_json_paths or [])
    }
    redundant_patterns = (
        "**/*all_questions.json",
        "**/*subcat*.questions.json",
        "**/*subcat*questions_final.json",
        "**/*.attempt*.txt",
        "**/*.compare_answers.jsonl",
        "**/temp_log/**/*",
    )
    removed = []
    import glob as glob_module

    for pattern in redundant_patterns:
        for path in glob_module.glob(
            os.path.join(iteration_dir, pattern), recursive=True
        ):
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)
    for path in glob_module.glob(
        os.path.join(iteration_dir, "**", "*.json"), recursive=True
    ):
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) == 0:
                raise ValueError("empty JSON")
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            os.remove(path)
            removed.append(path)
            continue
        if (
            strict_json_allowlist
            and os.path.normcase(os.path.abspath(path)) not in preserved
        ):
            os.remove(path)
            removed.append(path)
    return removed


class HardSamplePool:
    """Read-only prompt and metric view over the canonical hard_pool.json."""

    def __init__(self, pool_path):
        self.pool_path = os.fspath(pool_path)
        self.samples = [
            sample
            for sample in read_json_records(self.pool_path)
            if isinstance(sample, dict) and sample.get("question")
        ]

    @property
    def total_count(self):
        return len(self.samples)

    @staticmethod
    def accuracy(records):
        records = [record for record in records if isinstance(record, dict)]
        if not records:
            return 0.0
        return sum(_as_bool(record.get("is_correct")) for record in records) / len(records)

    @staticmethod
    def coverage(records):
        taxonomy_lookup = {item.lower(): item for item in ALL_SUB_CATEGORIES}
        covered = {
            taxonomy_lookup[str(record.get("sub_category", "")).strip().lower()]
            for record in records
            if isinstance(record, dict)
            and str(record.get("sub_category", "")).strip().lower() in taxonomy_lookup
        }
        missing = [item for item in ALL_SUB_CATEGORIES if item not in covered]
        return len(covered) / len(ALL_SUB_CATEGORIES), sorted(covered), missing

    def get_history_context(self, max_samples=None):
        eligible = [
            sample
            for sample in self.samples
            if sample.get("sample_grade") == "train_eligible"
        ]
        samples = eligible or self.samples
        samples = samples if max_samples is None else samples[-max_samples:]
        grouped = defaultdict(list)
        for sample in samples:
            grouped[sample.get("sub_category", "Unknown")].append(sample)
        lines = ["Historical hard samples grouped by sub_category:"]
        for sub_category in sorted(grouped):
            lines.append(f"\n[{sub_category}]")
            for sample in grouped[sub_category]:
                tags = ",".join(sample.get("error_tags", [])) or "concept_confusion"
                lines.append(
                    f"- hard_sample_id: {sample.get('unique_key', '')[:16]} | "
                    f"difficulty: {sample.get('difficulty', 5)} | "
                    f"structural_pattern: {sub_category} problem | "
                    f"observed_failure: {tags} | "
                    "variation_requirements: change values, wording, and structure"
                )
        return "\n".join(lines)

    def get_variant_context(self, category, sub_category, max_samples=12):
        samples = [
            sample
            for sample in self.samples
            if sample.get("sample_grade") == "train_eligible"
            and sample.get("category") == category
            and sample.get("sub_category") == sub_category
        ]
        if not samples:
            samples = [
                sample
                for sample in self.samples
                if sample.get("sample_grade") == "train_eligible"
                and sample.get("category") == category
            ]
        lines = []
        for sample in samples[-max_samples:]:
            lines.append(
                f"- hard_sample_id: {sample.get('unique_key', '')[:16]} | "
                f"subcategory: {sample.get('sub_category', '')} | "
                f"difficulty: {sample.get('difficulty', 5)} | "
                f"structural_pattern: {sample.get('sub_category', '')} problem | "
                "variation_requirements: change all values and wording; "
                "preserve the reasoning trap; do not copy the source"
            )
        return "\n".join(lines)


def extract_code(text):
    """Return (language, code) pairs from Markdown fenced code blocks."""
    blocks = re.findall(
        r"```([\w.+-]*)[ \t]*\r?\n?(.*?)```", text, flags=re.DOTALL
    )
    return [(language or "", code.strip()) for language, code in blocks]


def execute_code(code, lang="python", timeout=120):
    """Execute generated Python with the active virtual-environment interpreter."""
    if not lang.lower().startswith("python"):
        return 1, f"Unsupported language: {lang}", None
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=os.getcwd(),
    )
    return result.returncode, result.stdout + result.stderr, None



# [ADDED] Extract quote-aware balanced JSON from surrounding prose.
def _balanced_json_fragments(text):
    fragments = []
    for start in (match.start() for match in re.finditer(r"[\[{]", text)):
        stack = []
        quote_char = None
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quote_char:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote_char:
                    quote_char = None
                continue
            if char in {'"', "'"}:
                quote_char = char
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                if not stack:
                    break
                opener = stack.pop()
                if (opener, char) not in {("[", "]"), ("{", "}")}:
                    break
                if not stack:
                    fragments.append(text[start:index + 1])
                    break
    return fragments


# [ADDED] Repair common formatting noise without changing field semantics.
def _repair_json_candidate(candidate):
    repaired = candidate.strip().lstrip("\ufeff")
    repaired = re.sub(r"^\s*json\s*", "", repaired, flags=re.IGNORECASE)
    repaired = repaired.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)",
        r'\1"\2"\3',
        repaired,
    )
    return repaired


# [MODIFIED] Parse fenced, raw, tagged, and balanced JSON model output.
def extract_json_v2(json_text, outfilename):
    if not isinstance(json_text, str):
        raise TypeError("Model response must be text")
    response = json_text.replace("TERMINATE", "").strip().lstrip("\ufeff")
    fenced_blocks = extract_code(response)
    candidates = [
        code for language, code in fenced_blocks if language.lower() == "json"
    ]
    candidates.extend(
        code
        for language, code in fenced_blocks
        if language.lower() != "json" and code.startswith(("[", "{"))
    )
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"<json[^>]*>(.*?)</json>", response, flags=re.DOTALL | re.IGNORECASE
        )
    )

    # Prefer explicit fenced/<json> blocks. Only scan the surrounding prose
    # when the model did not provide an explicit structured block.
    if not candidates:
        if response.startswith(("[", "{")):
            candidates.append(response)
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", response):
            try:
                _, end = decoder.raw_decode(response[match.start():])
            except json.JSONDecodeError:
                continue
            candidates.append(response[match.start():match.start() + end])
            break
        balanced_fragments = _balanced_json_fragments(response)
        if balanced_fragments:
            candidates.append(balanced_fragments[0])

    # Keep the original order while avoiding duplicate parsing of fenced JSON.
    candidates = list(dict.fromkeys(candidate.strip() for candidate in candidates))
    if not candidates:
        raise ValueError("Model response did not contain a JSON block")

    parsed_blocks = []
    errors = []
    for candidate in candidates:
        parsed = None
        parse_errors = []
        for candidate_variant in (candidate, _repair_json_candidate(candidate)):
            try:
                parsed = json.loads(candidate_variant)
                break
            except json.JSONDecodeError as json_exc:
                parse_errors.append(str(json_exc))
            try:
                parsed = ast.literal_eval(candidate_variant)
                break
            except (ValueError, SyntaxError) as ast_exc:
                parse_errors.append(str(ast_exc))
        if parsed is None:
            errors.append("; ".join(parse_errors))
            continue
        if parsed not in (None, [], {}):
            parsed_blocks.append(parsed)

    if not parsed_blocks:
        detail = errors[-1] if errors else "the JSON value was empty"
        raise ValueError(f"Model returned no usable JSON: {detail}")
    if outfilename is not None:
        dump_standard_json(parsed_blocks, outfilename)
    return parsed_blocks

def _questions_match(expected, cached):
    """Return whether a cached inference belongs to the expected question."""
    required_keys = ("id", "question")
    optional_keys = ("answer", "category", "subcat")
    if any(cached.get(key) != expected.get(key) for key in required_keys):
        return False
    return all(
        key not in expected or cached.get(key) == expected.get(key)
        for key in optional_keys
    )


def _load_valid_inference_cache(outfile, problem_json):
    """Load the valid cache prefix and discard a corrupt or stale suffix."""
    if not os.path.exists(outfile):
        return []

    valid_results = []
    cache_needs_rewrite = False
    with open(outfile, "r", encoding="utf-8") as in_handle:
        for index, raw_line in enumerate(in_handle):
            if index >= len(problem_json):
                cache_needs_rewrite = True
                break
            try:
                cached_line = json.loads(raw_line)
            except json.JSONDecodeError:
                cache_needs_rewrite = True
                break
            if not _questions_match(problem_json[index], cached_line):
                cache_needs_rewrite = True
                break
            valid_results.append(cached_line)

    # Canonicalize a partial cache before appending so even a valid final JSONL
    # record without a trailing newline cannot be concatenated with the next one.
    if cache_needs_rewrite or (
        valid_results and len(valid_results) < len(problem_json)
    ):
        temporary_outfile = f"{outfile}.tmp"
        with open(temporary_outfile, "w", encoding="utf-8") as out_handle:
            for line in valid_results:
                print(json.dumps(line, ensure_ascii=False), file=out_handle)
        os.replace(temporary_outfile, outfile)
    if cache_needs_rewrite:
        print(
            f"Discarded an invalid inference-cache suffix; "
            f"kept {len(valid_results)} verified records."
        )
    return valid_results


def test_taker_inference(
    test_model_info,
    problem_json,
    outfile,
    bsz=1,
    temperature=0.01,
    max_length=50,
    existing_results=None,
):
    if len(test_model_info) == 3:
        model_choice, tokenizer_choice, client_choice = test_model_info
        auth = None
        use_helm = False
    elif len(test_model_info) == 4:
        model_choice, tokenizer_choice, client_choice, auth = test_model_info
        use_helm = True

    existing_results = list(existing_results or [])
    if len(existing_results) > len(problem_json):
        raise ValueError("Inference cache is longer than the question list")
    if len(existing_results) == len(problem_json):
        return existing_results

    file_mode = "a" if existing_results else "w"
    print(
        f"writing to {outfile} "
        f"(resuming at {len(existing_results) + 1}/{len(problem_json)})"
    )
    full_result_lst = existing_results
    batch_lst, line_lst = [], []
    with open(outfile, file_mode, encoding="utf-8") as out_handle:
        for source_line in tqdm.tqdm(problem_json[len(existing_results):]):
            line = copy.deepcopy(source_line)
            line['prompt'] = "Output just with the final answer to the question.\nQuestion:" + line[
                'question'] + "\n" + "Answer:"
            line_lst.append(line)
            batch_lst.append(line['prompt'])
            if len(batch_lst) < bsz:
                continue  # batch not full yet
            request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                             echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                             service=client_choice,
                                             terminate_by_linebreak='no', use_helm=use_helm, auth=auth,
                                             verbose=False)

            for line, xx in zip(line_lst, request_result.completions):
                line['test_taker_response'] = xx.text
                print(
                    json.dumps(line, ensure_ascii=False),
                    file=out_handle,
                    flush=True,
                )
                full_result_lst.append(line)
            batch_lst, line_lst = [], []
        if len(batch_lst) > 0:
            request_result = gen_from_prompt(model=model_choice, tokenizer=tokenizer_choice, prompt=batch_lst,
                                             echo_prompt=False, temperature=temperature, max_tokens=max_length,
                                             service=client_choice, terminate_by_linebreak='no', use_helm=use_helm,
                                             auth=auth, verbose=False)
            for line, xx in zip(line_lst, request_result.completions):
                line['test_taker_response'] = xx.text
                print(
                    json.dumps(line, ensure_ascii=False),
                    file=out_handle,
                    flush=True,
                )
                full_result_lst.append(line)
    return full_result_lst


def _generate_lm_answers(question_inputs, test_model_info, agent_model_info, outfile_prefix='att1'):
    # test_taker_lm, test_taker_tokenizer, test_taker_client = test_model_info
    if isinstance(question_inputs, list) or isinstance(question_inputs, dict):
        question_inputs_str = json.dumps(question_inputs, indent=2)
    else:
        assert False

    if not isinstance(question_inputs, list) or not question_inputs:
        raise RuntimeError(
            "No benchmark questions were generated. Check the upstream dataset source and retry."
        )
    if isinstance(question_inputs[0], list):
        json_dict = question_inputs[0]
    else:
        json_dict = question_inputs

    inference_file = f"{outfile_prefix}.test_taker_inference.json"
    cached_results = _load_valid_inference_cache(inference_file, json_dict)
    if len(cached_results) == len(json_dict):
        print(f"FOUND completed inference cache {inference_file}")
        return cached_results
    if cached_results:
        print(
            f"FOUND partial inference cache with {len(cached_results)}/"
            f"{len(json_dict)} records; resuming."
        )

    full_result_lst = test_taker_inference(test_model_info, json_dict,
                                           outfile=inference_file,
                                           existing_results=cached_results)

    return full_result_lst



def search_related_pages(search_query):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "list": "search",
        "srsearch": search_query,
        "srlimit": 50,
        "srnamespace": 0,
    }
    try:
        response = requests.get(
            WIKIPEDIA_API_URL, params=params, headers=WIKIMEDIA_HEADERS, timeout=30
        )
        response.raise_for_status()
        return [item["title"] for item in response.json().get("query", {}).get("search", [])]
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Wikipedia search failed for {search_query!r}: {exc}")
        return []


def _resolve_wikipedia_title(search_query):
    results = search_related_pages(search_query)
    return results[0] if results else None


def get_pageviews(page_title, start_date="2020040100", end_date="2023040700"):
    resolved_title = _resolve_wikipedia_title(page_title) or page_title.replace("_", " ")
    encoded_title = quote(resolved_title.replace(" ", "_"), safe="")
    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia/all-access/all-agents/{encoded_title}/daily/{start_date}/{end_date}"
    )
    try:
        response = requests.get(url, headers=WIKIMEDIA_HEADERS, timeout=30)
        response.raise_for_status()
        views = sum(item["views"] for item in response.json().get("items", []))
        print("retrieved pageviews for", resolved_title)
        return views
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Pageview lookup failed for {resolved_title!r}: {exc}")
        return 0


def _legacy_get_pageviews(page_title, start_date="2020040100", end_date="2023040700"):
    access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIwMDFkMTFmNmQ2MzVmMGY4YmI3MDlkNWViN2ZhNDRlYiIsImp0aSI6IjMzOGQ0Mzc0YzNmZjE5NjBlZDkzNjIwNTdiYjMwYjExOWYzZTY2MzVkZjM3NmY3NDcyZjczMDcyMjNiYzU4ODFjODBkOTliOTZmMjAzZGNkIiwiaWF0IjoxNzEyNjEwMTg0LjY4OTIyNywibmJmIjoxNzEyNjEwMTg0LjY4OTIzLCJleHAiOjMzMjY5NTE4OTg0LjY4NzY1Mywic3ViIjoiNzUzODczODIiLCJpc3MiOiJodHRwczovL21ldGEud2lraW1lZGlhLm9yZyIsInJhdGVsaW1pdCI6eyJyZXF1ZXN0c19wZXJfdW5pdCI6NTAwMCwidW5pdCI6IkhPVVIifSwic2NvcGVzIjpbImJhc2ljIl19.YN0ZvSzsBuYe3Mg-r0C63cWxDXPU3GOCyspUqg4mMv27Qw1FJq9F9H6JKJAUMrqQxB-xyWZqpu8mekvMoxb3Ha5S2fpPbuM4gMB0JketqG2obaDd4QqgtJjg8KDYKwR8ieKoPRLDSHv3Tv4NcvIL-EvzjkRybqrukzQwttwuBUwxmlY8vhC1BZed7URt_-KhMYPsnNfJLSBeWivYJOmrqF2S04AOS0Egjul8Pz_yXAQ7q7aqpIwg6X2jod0ZN5h1gnmAvZmoLB7mKSAxrHEUL2zaQ8BVERWostWVA9ek556cuUJe5NusQ0XW7pcsYIi0YpFjKOBuq-tXzuOlbxFhlbwrp6xkhE_grQGNs1IxyT-w_sjQc2gI48FDe0ldDrTg6ZmgLELsjJM8xOxBy1ng1fY73p-QnaDdxX4hqRw2ZBDlZ1E2j84lvVrv62x_SHPiBNAeywEPcOqDRV_XbU6ArOyJ7QTZXRu9UOT0XDQ-Fx3maCRGb35W4aOtLSWL-SSXYLI8ZuOQ2BwKQQYYbEDMp0W7NjHWzh8YPv6Y2wDaMzsAqaxk2c36pNvTToiTc_P6_a56lydQwoT8ACx1kzzw5lTNPKPEPxPGNiMgtsL3VqtxJWMR7Lgq-ZKwI7cwQ5FTp2YriQDBYuvoaDQeG_eVh8BlNlyg26OYojtYbNos3os"
    client_id = "001d11f6d635f0f8bb709d5eb7fa44eb"
    client_secret = "630b434daa4c8f6cce03b1c294b59574c1ce9431"  # Example client secret
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': 'wikipagerank',
    }

    # Construct the API URL with the appropriate parameters
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{page_title}/daily/{start_date}/{end_date}"
    # Make the HTTP GET request to the API
    response = requests.get(url, headers=headers)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        # Extract the pageview data
        print('retrieved for ', page_title)
        views = sum(item['views'] for item in data['items'])
        return views
    else:
        print(f"Failed to retrieve pageviews data for {page_title}. Status code: {response.status_code}")
        return 0

def clean_str(p):
  try:
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")
  except:
    return ''

def filter_paragraph(paragraph_lst):
    return [p for p in paragraph_lst if len(p.split(" ")) > 2 and len(p.split(".")) > 1]


def get_page_obs(page):
    # find all paragraphs
    paragraphs = page.split("\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def search_step(entity, output_more=False):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": entity,
        "gsrnamespace": 0,
        "gsrlimit": 1,
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "exlimit": 1,
        "redirects": 1,
    }
    try:
        response = requests.get(
            WIKIPEDIA_API_URL, params=params, headers=WIKIMEDIA_HEADERS, timeout=30
        )
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", [])
        if not pages:
            print(f"Wikipedia page not found for {entity!r}")
            return [], entity
        page = pages[0]
        resolved_entity = page.get("title", entity)
        paragraphs = filter_paragraph(get_page_obs(page.get("extract", "")))
        if not output_more:
            paragraphs = paragraphs[:10]
        print("found entity", resolved_entity)
        return paragraphs, resolved_entity
    except (requests.RequestException, ValueError, KeyError) as exc:
        print(f"Wikipedia content lookup failed for {entity!r}: {exc}")
        return [], entity
