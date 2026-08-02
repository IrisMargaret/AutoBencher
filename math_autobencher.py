# Runtime path configuration must be established before project imports.
# ruff: noqa: E402

import argparse
import contextlib
import copy
import glob
import gc
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import tqdm

# Runtime artifacts and caches are configured after YAML resolution. Avoid
# writing repository-local bytecode during project imports.
sys.dont_write_bytecode = True

from autobencher.config import (
    ConfigurationError,
    cli_config_overrides,
    load_project_config,
    str2bool,
    thaw_config,
)
from autobencher.budget_ledger import (
    BudgetExhausted,
    active_ledger,
    estimate_tokens,
)
from autobencher.attribution_eval import export_review_sample
from autobencher.coverage import coverage_metrics, generation_schedule
from autobencher.dataset import (
    build_training_dataset,
    normalize_question_text,
    write_alpaca_jsonl,
)
from autobencher.difficulty import (
    assess_difficulty,
    target_difficulty_profile,
)
from autobencher.experiment import (
    ResearchRun,
    atomic_json,
    utc_now,
)
from autobencher.evaluator import (
    judge_answer_semantics,
    solve_with_privileged_python,
)
from autobencher.fixed_benchmark import (
    bootstrap_development_fixed_test_set,
    fixed_benchmark_summary,
    load_fixed_test_set,
)
from autobencher.generation_guidance import (
    load_generation_guidance_context,
)
from autobencher.policies import policy_runtime_descriptor
from autobencher.similarity import build_similarity_batch
from autobencher.structured import (
    answers_equivalent,
    attribute_error,
    fuse_equivalence_with_semantic_judge,
    normalize_generated_gold_contract,
    normalize_answer_type,
    validate_generated_question,
)
from autobencher.storage import configure_runtime_storage
from autobencher.truth_solver import FailureType, TruthSolver
from autobencher.training_protocol import (
    enforce_train_correct_incorrect_ratio,
    split_by_template_cluster,
    write_precomputed_training_splits,
)
from autobencher.evaluation_sets import load_evaluation_registry
from autobencher.evaluation_audit import load_json_questions
from autobencher.fingerprints import artifact_fingerprint
from util import gen_from_prompt, helm_process_args, process_args_for_models
from tool_util import (
    DEFAULT_SYSTEM_MESSAGE,
    HardSamplePool,
    MATH_CATEGORIES,
    SUB_CATEGORY_TAXONOMY,
    canonicalize_math_record,
    classify_error_tags,
    check_disk_space,
    clean_redundant_files,
    call_local_finetune,
    dump_standard_json,
    execute_code,
    export_training_dataset,
    extract_code,
    extract_json_v2,
    generate_math_inference,
    load_math_inference,
    manage_hard_pool,
    update_hard_pool_lifecycle,
    normalize_math_category,
    normalize_sub_category,
    read_json_records,
    update_meta_summary,
)
from run_scripts import log_math_iteration_metrics

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""


def _request_control_kwargs(research_config, model_role="evaluator"):
    """Route configured API timeouts/retries into shared model calls."""
    if not research_config:
        return {}
    model_config = research_config["models"][model_role]
    return {
        "request_timeout_seconds": float(
            model_config["request_timeout_seconds"]
        ),
        "max_num_retries": int(model_config["max_retries"]),
        "retry_delay_seconds": float(
            model_config.get("retry_delay_seconds", 5)
        ),
    }


def _generate_python_answers(problem_json, agent_lm, agent_tokenizer, agent_client, outfile_prefix='att1'):
    context = """Your goal is to generate the python code that solves the provided math questions. 
The provided math problems are in the format of json. You should write the python code for all the problems in python coding blocks. You should NOT need to write a large solve function that can solve all the problems and handle all cases, just provide problem specific solution for each question. 
Do not use string matching to figure out which problem to solve. Just do as follows, write the expression for each question inline, then print the answer. 
```python
ans1 = 1+1
print("1. ANSWER:", ans1)
ans2 = np.sin(1)
print("2. ANSWER:", ans2)
``` 
Also, make sure to answer questions **sequentially**, in the same order as question id. If there is a question that's not answerable, set the answer to be "N/A".
*You should not omit any questions, and DO NOT use ellipsis*
For questions with a numeric answer, make sure to simplify the final result and report the final answer as a decimal number. For example, for a sympy expression, you should simplify it and report the final answer as a decimal number via evalf(). But note that 'float' object has no attribute 'evalf'. 
If you are solving a trigonometric equation using sympy, make sure to use sympy.sin(x) for variable x. Do not use math.sin(x) or np.sin(x), which cause errors. Also, remember that sympy.sin takes in radians, not degrees. So you want to judge the unit and then convert degrees into radians. 
Implement error handling for each function, if there is an error (in input or in execution), output "N/A" for that question.
The python code should print the answer of each problem in the following format: 
1. ANSWER: <answer1>
2. ANSWER: <answer2>
3. ANSWER: <answer3>
...
"""
    if os.path.exists(f"{outfile_prefix}.python_answers.json"):
        print(f"{outfile_prefix}.python_answers.json exists. Skipping.")
        with open(f"{outfile_prefix}.python_answers.json", "r") as f:
            raw_logs = f.read()
        try:
            logs = json.loads(raw_logs)
        except json.JSONDecodeError:
            logs = raw_logs
        return f'Processed Already in {outfile_prefix}.python_code.json', logs

    elif os.path.exists(f"{outfile_prefix}.python_code.json"):
        with open(f"{outfile_prefix}.python_code.json", "r") as f:
            raw_code = f.read()
        try:
            extracted_python_code = json.loads(raw_code)
        except json.JSONDecodeError:
            extracted_python_code = raw_code
        exit_code, logs, image = execute_code(extracted_python_code,
                                              lang="python")
        print(logs)
        assert exit_code == 0
        return extracted_python_code, logs

    if isinstance(problem_json, list) or isinstance(problem_json, dict):
        problem_json_str = json.dumps(problem_json, indent=2)
    else:
        problem_json_str = problem_json
    context += problem_json_str + "Please write the python code for all the problems in a python coding block."
    context = DEFAULT_SYSTEM_MESSAGE + context

    request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context],
                                     echo_prompt=False, temperature=0.0, max_tokens=2000,
                                     process_func=None, service=agent_client,
                                     terminate_by_linebreak='no', )
    response = request_result.completions[0].text
    # parse the json file
    response = response.replace('TERMINATE', '')
    extracted_python_code = extract_code(response)
    # print(extracted_python_code, 'extracted_python_code')
    extracted_python_code = extracted_python_code[0][1]
    dump_standard_json(
        extracted_python_code, f"{outfile_prefix}.python_code.json"
    )
    exit_code, logs, image = execute_code(extracted_python_code, lang="python3.9")
    print(logs)
    if exit_code != 0:
        # error
        silver_string = "\n".join(
            [f"{idx + 1}. {line['answer']}" for idx, line in enumerate(problem_json)])
        return None, silver_string

    else:
        dump_standard_json(logs, f"{outfile_prefix}.python_answers.json")

        return extracted_python_code, logs

def solve_with_python(question_json, outfile_prefix, agent_info):
    # solve for 20 questions. at a time.
    agent_lm, agent_tokenizer, agent_client = agent_info
    question_json_copy = copy.deepcopy(question_json)

    python_log_lst = []
    for idx in range(0, len(question_json), 20):
        question_json = question_json_copy[idx:idx+20]
        question_json_2 = []
        for line in question_json:
            line2 = {}
            line2['question'] = line['question']
            line2['id'] = line['id']
            question_json_2.append(line2)

        python_output, python_logs = _generate_python_answers(question_json_2,
                                                              agent_lm, agent_tokenizer, agent_client,
                                                              outfile_prefix = outfile_prefix + f".{idx}")
        print('python_output', python_output)
        print('python_logs', python_logs)
        # process python_logs into answers.
        ans_lst = python_logs.strip().split('\n')
        assert len(ans_lst) == len(question_json)
        for idx, ans in enumerate(ans_lst):
            answer = ans.split("ANSWER:")[1].strip()
            python_log_lst.append(answer)
    print(python_log_lst)
    assert len(python_log_lst) == len(question_json_copy)
    for line, ans in zip(question_json_copy, python_log_lst):
        line['python_answer'] = ans

    dump_standard_json(
        question_json_copy, outfile_prefix + '.full_python_answers.json'
    )
    return python_log_lst, question_json_copy

def get_summary_of_results(json_dict, gold_key="answer", verbose=False):
    # a summary of the results.
    # summarize by each category.
    category2correct_count = defaultdict(list)
    category2question = defaultdict(list)
    str_summary = 'In the following, we summarize the evaluation results by each category in this agent iteration. \n We will report the accuracy for each category, and list the questions that are answered correctly and incorrectly. \n'
    for line in json_dict:
        sub_category = line.get("sub_category", line.get("subcat"))
        line['category2'] = (
            f"{line['category']} || {sub_category}"
            if sub_category else line['category']
        )
        category2correct_count[line['category2']].append(line['is_correct'])
        category2question[(line['category2'], line['is_correct'])].append(line)
    for category in category2correct_count:
        acc_temp = sum(
            1 if (x is True or str(x).lower() == 'true') else 0
            for x in category2correct_count[category]
        ) / len(category2correct_count[category])
        str_summary += f"category: {category}, accuracy: {round(acc_temp, 3)} " \
                       f"|| {sum([1 if (x is True or str(x).lower() == 'true') else 0 for x in category2correct_count[category]])} out of {len(category2correct_count[category])}" + "\n"
        if verbose:
            str_summary += "# Questions answered correctly:\n"
            correct_questions = [
                question
                for (group, correctness), questions in category2question.items()
                if group == category
                and (correctness is True or str(correctness).lower() == "true")
                for question in questions
            ]
            for qq in correct_questions:
                gold = qq.get(gold_key, qq.get("gold_answer", qq.get("answer", "")))
                prediction = qq.get(
                    "test_taker_answer", qq.get("test_taker_response", "")
                )
                str_summary += f"{qq['question']} || gold: {gold} || pred: {prediction}\n"
            str_summary += "# Questions answered incorrectly:\n"
            incorrect_questions = [
                question
                for (group, correctness), questions in category2question.items()
                if group == category
                and not (
                    correctness is True or str(correctness).lower() == "true"
                )
                for question in questions
            ]
            for qq in incorrect_questions:
                gold = qq.get(gold_key, qq.get("gold_answer", qq.get("answer", "")))
                prediction = qq.get(
                    "test_taker_answer", qq.get("test_taker_response", "")
                )
                str_summary += f"{qq['question']} || gold: {gold} || pred: {prediction}\n"
            str_summary += "\n + ------------------------------------ + \n"
    # print(str_summary)
    return str_summary

def summarize_over_history(history_json_dict, gold_key="python_answer", verbose=True):
    '''
    :param history: a list of dictionaries. Each dictionary corresponds to a run.
    :return: a summary of the results.
    '''
    # augment each line of the dictionary with the iteration number.
    for idx, json_dict in enumerate(history_json_dict):
        for line in json_dict:
            line['iteration'] = idx
    # concatenate the dictionaries.
    json_dict = [line for json_dict in history_json_dict for line in json_dict]
    # a summary of the results.
    str_summary = get_summary_of_results(json_dict, gold_key=gold_key, verbose=verbose)
    # print(str_summary)
    return str_summary


# [ADDED] Compute directed-generation triggers from existing result caches.
def _get_math_flywheel_state(history_json_dict):
    all_records = [
        record
        for iteration_records in history_json_dict
        for record in iteration_records
    ]
    coverage, covered_labels, missing_labels = HardSamplePool.coverage(all_records)
    if len(history_json_dict) < 2:
        return False, [], coverage, covered_labels, missing_labels

    recent_accuracies = [
        HardSamplePool.accuracy(records)
        for records in history_json_dict[-2:]
    ]
    triggers = []
    if all(accuracy >= 0.7 for accuracy in recent_accuracies):
        triggers.append("two_round_accuracy_gte_0.7")
    if coverage < 0.6:
        triggers.append("coverage_below_60pct")
    return bool(triggers), triggers, coverage, covered_labels, missing_labels


MATH_PLAN_SIZE = len(MATH_CATEGORIES)


def _write_json_atomic(data, output_file):
    # [MODIFIED] Route every math JSON write through the standard writer.
    dump_standard_json(data, output_file)


# [ADDED] Keep raw retry output in the disposable iteration log.
def _write_attempt_log(outfile_prefix, stage, attempt, response):
    iteration_dir = os.path.dirname(os.path.abspath(outfile_prefix))
    temp_log_dir = os.path.join(iteration_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)
    filename = (
        f"{os.path.basename(outfile_prefix)}.{stage}.attempt{attempt}.txt"
    )
    with open(os.path.join(temp_log_dir, filename), "w", encoding="utf-8") as f:
        f.write(re.sub(r"[\u3400-\u9fff]+", " ", str(response)))


def _normalize_math_plan(extracted_json):
    if not isinstance(extracted_json, list) or not extracted_json:
        raise ValueError("Expected one JSON array for the math plan")
    if len(extracted_json) == 1 and isinstance(extracted_json[0], list):
        plan = extracted_json[0]
    elif all(isinstance(item, dict) for item in extracted_json):
        plan = extracted_json
    else:
        raise ValueError("Expected one JSON array for the math plan")
    is_legacy_plan = all(
        isinstance(item, dict)
        and "sub_category" not in item
        and "subcategory_description" in item
        for item in plan
    )
    required_size = 5 if is_legacy_plan else MATH_PLAN_SIZE
    if len(plan) < required_size:
        raise ValueError(
            f"Expected at least {required_size} math-plan items, got {len(plan)}"
        )
    if any(not isinstance(item, dict) or "category" not in item for item in plan):
        raise ValueError(
            "Each plan item must contain category and sub_category"
        )
    normalized_plan = []
    for index, item in enumerate(plan[:required_size]):
        raw_sub_category = item.get(
            "sub_category", item.get("subcategory_description", "")
        )
        category = normalize_math_category(
            item.get("category", ""), raw_sub_category
        )
        normalized_plan.append(
            {
                "id": int(index + 1),
                "category": category,
                "sub_category": normalize_sub_category(
                    raw_sub_category, category
                ),
                "difficulty": max(1, min(10, int(float(item.get("difficulty", 5))))),
            }
        )
    if not is_legacy_plan:
        generated_categories = {item["category"] for item in normalized_plan}
        missing_categories = [
            category
            for category in MATH_CATEGORIES
            if category not in generated_categories
        ]
        if missing_categories:
            raise ValueError(
                "Math plan must cover every top-level category; missing: "
                + ", ".join(missing_categories)
            )
    if len(plan) > required_size:
        print(
            f"Model returned {len(plan)} math-plan items; "
            f"using the requested first {required_size}."
        )
    return [normalized_plan]


def _generate_cat_with_aim(
    aim,
    agent_lm,
    agent_tokenizer,
    agent_client,
    history,
    iters,
    outfile_prefix='att1',
    enable_hard_sample_guidance=False,
    coverage_summary="",
    hard_pool_file=None,
    research_config=None,
):
    # [MODIFIED] Enforce the fixed two-level, nine-category taxonomy.
    taxonomy_json = json.dumps(
        SUB_CATEGORY_TAXONOMY, ensure_ascii=True, indent=2
    )
    context = f"""
Your goal is to create a hierarchical math benchmark plan that converges to
the target accuracy range AIM.

The plan must contain exactly 9 items, one for each top-level category below,
in this exact category vocabulary:
{json.dumps(list(MATH_CATEGORIES), ensure_ascii=True)}

Choose each sub_category from this fixed taxonomy:
{taxonomy_json}

Planning rules:
1. Cover all 9 top-level categories in every iteration.
2. Track performance independently for each sub_category. Never use a broad
   category average to hide a weak or strong sub_category.
3. Prefer low-accuracy and missing sub_categories in later iterations.
4. Avoid trivial single-digit arithmetic and easy one-step equations unless
   their measured sub_category accuracy is within AIM.
5. Use only English strings.
6. Return only one JSON array. Do not add explanations, Markdown, comments,
   placeholders, or ellipses.

Each item must use exactly these keys:
[
  {{
    "id": 1,
    "category": "Arithmetic",
    "sub_category": "Fraction and Decimal Operations",
    "difficulty": 5
  }}
]

The full response must contain 9 objects and cover every top-level category
exactly once. You do not need to solve or generate questions in this step.
"""
    context = context.replace("AIM", str(aim))
    # [ADDED] Load the global pool and inject eligible weakness evidence.
    hard_pool_file = hard_pool_file or os.path.join(
        os.path.dirname(os.path.abspath(outfile_prefix)), "hard_pool.json"
    )
    hard_pool = HardSamplePool(hard_pool_file)
    if enable_hard_sample_guidance:
        hard_sample_context = hard_pool.get_history_context(max_samples=100)
        context += f"""

# Hard-sample-directed generation rules (mandatory)
The baseline stage has finished. Use the complete historical hard-sample pool
below as direct evidence of the test taker's real weaknesses.

1. Prioritize variant problems that preserve the traps, dependency structure,
   and numerical structure of historical wrong answers, while changing wording
   and values. Never copy a historical question verbatim.
2. Actively generate missing or very-low-coverage fine-grained math types.
3. Greatly reduce the weight of single-digit addition/subtraction, trivial
   arithmetic, and simple one-step linear equations that typically have 100%
   accuracy.
4. Return exactly one structured JSON array with 9 plan items and no natural
   language outside JSON.

Current coverage:
{coverage_summary}

Train-eligible historical hard-sample context:
{hard_sample_context}
"""
    else:
        context += (
            "\nThis is a baseline-collection iteration. Preserve broad category "
            "exploration and return only the required structured JSON plan. "
            f"The hard pool currently contains {hard_pool.total_count} samples."
        )
    if iters is None:
        iters = len(history) + 1
    if len(history) == 0:
        context += "Please start with iteration 1."
    else:
        context += "\n".join(history) + "\nPlease start with iteration {}. Remember your goal is to find the subcategory with accuracy {} for each category.".format(iters, aim)
    context = DEFAULT_JSON_MESSAGE + context
    output_file = f"{outfile_prefix}.question_plan_with_aim.json"
    last_error = None
    for attempt in range(1, 4):
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                "\nYour previous response could not be parsed. Return only one "
                "complete JSON array with all 9 required categories. Do not "
                "include Markdown, analysis, comments, or placeholders."
            )
        request_result = gen_from_prompt(
            model=agent_lm,
            tokenizer=agent_tokenizer,
            prompt=[context + retry_instruction],
            echo_prompt=False,
            temperature=0.0,
            max_tokens=4096,
            process_func=None,
            service=agent_client,
            terminate_by_linebreak='no',
            **_request_control_kwargs(research_config),
        )
        response = request_result.completions[0].text
        try:
            extracted_json = extract_json_v2(response, None)
            extracted_json = _normalize_math_plan(extracted_json)
            _write_json_atomic(extracted_json[0], output_file)
            return extracted_json
        except (ValueError, IndexError, TypeError) as exc:
            last_error = exc
            print(f"Math plan generation attempt {attempt}/3 failed: {exc}")
            _write_attempt_log(
                outfile_prefix, "question_plan_with_aim", attempt, response
            )
    raise RuntimeError("Failed to generate a valid math plan after 3 attempts") from last_error
# [ADDED] Bound deterministic SymPy work on Unix workers. Windows lacks
# SIGALRM, so the same parser limits inputs and performs an elapsed-time check.
@contextlib.contextmanager
def _sympy_validation_timeout(seconds):
    seconds = float(seconds)
    can_interrupt = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_interrupt:
        yield
        return

    def _raise_timeout(signum, frame):
        del signum, frame
        raise TimeoutError("SymPy gold-answer validation timed out")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _split_top_level_equations(text):
    """Split comma/semicolon/newline-delimited equations outside brackets."""
    parts = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and character in ",;\n":
            part = text[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    final = text[start:].strip()
    if final:
        parts.append(final)
    return parts


def _extract_ordered_tuple_system(question_text):
    """Parse variables and equations from the exact original question text."""
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    original_question = str(question_text)
    if not original_question.strip():
        raise ValueError("The original question is empty")
    if len(original_question) > 16000:
        raise ValueError("The original question exceeds the parser limit")

    equation_text = original_question
    if ":" in original_question:
        possible_tail = original_question.rsplit(":", 1)[1]
        if "=" in possible_tail:
            equation_text = possible_tail
    equation_text = re.sub(
        r"\s+\band\b\s+(?=[A-Za-z]\w*\s*[+\-*/^=])",
        ", ",
        equation_text,
        flags=re.IGNORECASE,
    )
    raw_equations = [
        part.rstrip().rstrip(".").strip()
        for part in _split_top_level_equations(equation_text)
        if "=" in part
    ]
    if not raw_equations:
        raise ValueError("No equations were found in the original question")
    if len(raw_equations) > 20:
        raise ValueError("The equation count exceeds the parser limit")

    variable_match = re.search(
        r"(?:for|variables?)\s*\(([^()]*)\)",
        original_question,
        flags=re.IGNORECASE,
    )
    if variable_match:
        variable_names = re.findall(
            r"[A-Za-z]\w*",
            variable_match.group(1),
        )
    else:
        identifiers = []
        for raw_equation in raw_equations:
            identifiers.extend(re.findall(r"[A-Za-z]\w*", raw_equation))
        excluded = {
            "sin",
            "cos",
            "tan",
            "sqrt",
            "exp",
            "log",
            "pi",
            "e",
        }
        variable_names = sorted(
            {
                identifier
                for identifier in identifiers
                if identifier.lower() not in excluded
            }
        )
    variable_names = list(dict.fromkeys(variable_names))
    if not variable_names or len(variable_names) > 10:
        raise ValueError("Unable to determine a bounded variable list")

    symbols = sympy.symbols(" ".join(variable_names))
    if len(variable_names) == 1:
        symbols = (symbols,)
    else:
        symbols = tuple(symbols)
    local_dict = dict(zip(variable_names, symbols))
    local_dict.update(
        {
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "sqrt": sympy.sqrt,
            "exp": sympy.exp,
            "log": sympy.log,
            "pi": sympy.pi,
            "e": sympy.E,
        }
    )
    allowed_names = set(local_dict)
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    equations = []
    equation_sources = []
    for raw_equation in raw_equations:
        normalized = re.sub(
            r"^\s*(?:eq(?:uation)?\s*\d+|[\[(]?\d+[\])]?)[.:]\s*",
            "",
            raw_equation,
            flags=re.IGNORECASE,
        )
        if normalized.count("=") != 1:
            raise ValueError(
                f"Expected one equality operator: {raw_equation}"
            )
        left_text, right_text = (
            piece.strip() for piece in normalized.split("=", 1)
        )
        if not left_text or not right_text:
            raise ValueError(f"Incomplete equation: {raw_equation}")
        for expression_text in (left_text, right_text):
            if "__" in expression_text or not re.fullmatch(
                r"[A-Za-z0-9_+\-*/^().\s]+",
                expression_text,
            ):
                raise ValueError(
                    f"Unsupported equation syntax: {raw_equation}"
                )
            identifiers = set(
                re.findall(r"[A-Za-z]\w*", expression_text)
            )
            unknown = identifiers - allowed_names
            if unknown:
                raise ValueError(
                    "Unknown identifiers in equation: "
                    + ", ".join(sorted(unknown))
                )
        left = parse_expr(
            left_text,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
        right = parse_expr(
            right_text,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
        equations.append(sympy.Eq(left, right, evaluate=False))
        equation_sources.append(f"{left_text} = {right_text}")
    return {
        "original_question": original_question,
        "variables": symbols,
        "variable_names": variable_names,
        "equations": equations,
        "equation_sources": equation_sources,
        "local_dict": local_dict,
    }


def _parse_ordered_tuple_candidate(candidate, system):
    """Parse a proposed tuple without using it in the independent solve."""
    import sympy

    text = str(candidate).strip()
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.strip("$").strip()
    if not (
        len(text) >= 2
        and text[0] in "(["
        and text[-1] in ")]"
    ):
        raise ValueError("The proposed answer is not an ordered tuple")
    components = _split_top_level_equations(text[1:-1])
    if len(components) != len(system["variables"]):
        raise ValueError(
            "The tuple length does not match the system variable count"
        )
    values = []
    for component in components:
        value_text = re.sub(
            r"^[A-Za-z]\w*\s*=\s*",
            "",
            component.strip(),
        )
        if "__" in value_text or not re.fullmatch(
            r"[A-Za-z0-9_+\-*/^().\s]+",
            value_text,
        ):
            raise ValueError("Unsupported tuple component syntax")
        value = sympy.sympify(
            value_text.replace("^", "**"),
            locals=system["local_dict"],
        )
        if value.free_symbols:
            raise ValueError("Tuple components must be fully specified")
        values.append(sympy.simplify(value))
    return tuple(values)


def _sympy_values_equal(left, right, tolerance):
    import sympy

    difference = sympy.simplify(left - right)
    if difference == 0 or difference.is_zero is True:
        return True
    if difference.free_symbols:
        return False
    try:
        return abs(float(sympy.N(difference))) <= float(tolerance)
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_ordered_tuple_system(
    question_text,
    proposed_gold_answer,
    research_config,
):
    """Independently solve and then substitute into the same parsed system."""
    # [MODIFIED] All authoritative SymPy work is centralized in TruthSolver.
    # This adapter preserves the existing gold-validation JSON field names.
    solver = TruthSolver.from_config(research_config)
    truth = solver.solve(question_text)
    candidate = solver.validate_candidate(
        question_text,
        proposed_gold_answer,
    )
    truth_details = dict(truth.truth_validation_details)
    substitution_details = list(
        candidate.get("substitution_details")
        or truth_details.get("substitution_details")
        or []
    )
    verification_passed = bool(
        truth.success
        and candidate.get("verification_passed")
    )
    source_hash = truth_details.get("source_question_sha256")
    return {
        "status": "passed" if verification_passed else "failed",
        "recomputed_answer": truth.canonical_answer or "",
        "answer_type": truth.answer_type or "ordered_tuple",
        "answer_type_consistent": truth.answer_type == "ordered_tuple",
        "verification_passed": verification_passed,
        "substitution_passed": bool(
            candidate.get("substitution_passed")
        ),
        "verification_method": (
            "TruthSolver independent solve and per-equation substitution"
        ),
        "failure_reason": (
            None
            if verification_passed
            else (
                truth.failure_summary
                or candidate.get("failure_type")
                or "The candidate failed deterministic truth validation."
            )
        ),
        "answer_equivalent": verification_passed,
        "question_source_consistent": bool(source_hash),
        "source_question_sha256": source_hash,
        "solve_question_sha256": source_hash,
        "substitution_question_sha256": source_hash,
        "solver_status": (
            truth_details.get("solver_branch")
            if truth.success
            else truth.failure_type
        ),
        "substitution_details": substitution_details,
        "failure_type": candidate.get("failure_type"),
        "truth_validation_details": truth_details,
    }

    # Legacy implementation retained below only until downstream migrations
    # finish; it is unreachable and no longer participates in truth solving.
    # [ADDED] Avoid an optional binary gmpy2 backend from making validation
    # unavailable when the host package is ABI-incompatible.
    os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
    os.environ.setdefault("MPMATH_NOGMPY", "1")
    import sympy
    from sympy.solvers.solveset import NonlinearError

    original_question = str(question_text)
    solve_question = original_question
    substitution_question = original_question
    if not (
        solve_question == substitution_question == original_question
    ):
        raise AssertionError(
            "Solve and substitution question sources must be identical"
        )
    timeout_seconds = float(
        research_config["generation"][
            "gold_validation_timeout_seconds"
        ]
    )
    tolerance = float(
        research_config["answer_normalization"][
            "absolute_tolerance"
        ]
    )
    started = time.monotonic()
    question_hash = hashlib.sha256(
        original_question.encode("utf-8")
    ).hexdigest()
    base_audit = {
        "status": "failed",
        "recomputed_answer": "",
        "answer_type": "ordered_tuple",
        "answer_type_consistent": True,
        "verification_passed": False,
        "substitution_passed": False,
        "verification_method": (
            "SymPy independent solve and per-equation substitution"
        ),
        "failure_reason": None,
        "answer_equivalent": False,
        "question_source_consistent": True,
        "source_question_sha256": question_hash,
        "solve_question_sha256": question_hash,
        "substitution_question_sha256": question_hash,
        "solver_status": "not_started",
        "substitution_details": [],
    }
    try:
        with _sympy_validation_timeout(timeout_seconds):
            system = _extract_ordered_tuple_system(original_question)
            expressions = [
                sympy.simplify(equation.lhs - equation.rhs)
                for equation in system["equations"]
            ]
            try:
                matrix_a, matrix_b = sympy.linear_eq_to_matrix(
                    expressions,
                    system["variables"],
                )
                solution_set = sympy.linsolve(
                    (matrix_a, matrix_b),
                    system["variables"],
                )
                solver_name = "linsolve"
            except NonlinearError:
                solution_set = sympy.nonlinsolve(
                    expressions,
                    system["variables"],
                )
                solver_name = "nonlinsolve"

            if solution_set is sympy.EmptySet or solution_set == sympy.EmptySet:
                base_audit["solver_status"] = "no_solution"
                base_audit["failure_reason"] = (
                    "The original system has no solution."
                )
                return base_audit
            solutions = list(solution_set)
            if len(solutions) != 1:
                base_audit["solver_status"] = (
                    "no_solution" if not solutions else "multiple_solutions"
                )
                base_audit["failure_reason"] = (
                    "The original system does not have exactly one solution."
                )
                return base_audit
            independent_solution = tuple(
                sympy.simplify(value) for value in solutions[0]
            )
            if any(
                value.free_symbols for value in independent_solution
            ):
                base_audit["solver_status"] = "infinite_solutions"
                base_audit["failure_reason"] = (
                    "The original system has infinitely many solutions."
                )
                return base_audit
            base_audit["solver_status"] = f"unique_{solver_name}"
            base_audit["recomputed_answer"] = (
                "("
                + ", ".join(
                    sympy.sstr(value)
                    for value in independent_solution
                )
                + ")"
            )

            candidate = _parse_ordered_tuple_candidate(
                proposed_gold_answer,
                system,
            )
            solution_match = all(
                _sympy_values_equal(candidate_value, solution_value, tolerance)
                for candidate_value, solution_value in zip(
                    candidate,
                    independent_solution,
                )
            )
            substitutions = dict(zip(system["variables"], candidate))
            substitution_details = []
            for index, (source, equation) in enumerate(
                zip(
                    system["equation_sources"],
                    system["equations"],
                ),
                start=1,
            ):
                left_value = sympy.simplify(
                    equation.lhs.subs(substitutions)
                )
                right_value = sympy.simplify(
                    equation.rhs.subs(substitutions)
                )
                difference = sympy.simplify(left_value - right_value)
                passed = _sympy_values_equal(
                    left_value,
                    right_value,
                    tolerance,
                )
                detail = {
                    "equation_index": index,
                    "original_equation": source,
                    "substituted_left": sympy.sstr(left_value),
                    "substituted_right": sympy.sstr(right_value),
                    "difference": sympy.sstr(difference),
                    "passed": passed,
                }
                substitution_details.append(detail)
                print(
                    "[GoldValidation] "
                    f"equation={index} source={source!r} "
                    f"left={detail['substituted_left']} "
                    f"right={detail['substituted_right']} "
                    f"difference={detail['difference']} "
                    f"passed={str(passed).lower()}"
                )
            substitution_passed = bool(substitution_details) and all(
                detail["passed"] for detail in substitution_details
            )
            verification_passed = (
                solution_match and substitution_passed
            )
            base_audit.update(
                {
                    "status": (
                        "passed" if verification_passed else "failed"
                    ),
                    "verification_passed": verification_passed,
                    "substitution_passed": substitution_passed,
                    "answer_equivalent": solution_match,
                    "failure_reason": (
                        None
                        if verification_passed
                        else (
                            "The proposed answer differs from the "
                            "independent solution."
                            if not solution_match
                            else (
                                "The proposed answer fails at least one "
                                "original equation."
                            )
                        )
                    ),
                    "substitution_details": substitution_details,
                }
            )
    except TimeoutError as exc:
        base_audit["solver_status"] = "timeout"
        base_audit["failure_reason"] = str(exc)
    except Exception as exc:
        # [ADDED] Fail closed for every parser/solver exception. Process-level
        # interrupts still propagate because they do not inherit Exception.
        base_audit["solver_status"] = "parse_or_solve_error"
        base_audit["failure_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
    elapsed = time.monotonic() - started
    base_audit["elapsed_seconds"] = round(elapsed, 6)
    if elapsed > timeout_seconds and base_audit["status"] == "passed":
        base_audit.update(
            {
                "status": "failed",
                "verification_passed": False,
                "substitution_passed": False,
                "solver_status": "timeout",
                "failure_reason": (
                    "SymPy gold-answer validation exceeded the timeout."
                ),
            }
        )
    return base_audit


# [ADDED] A separate evaluator pass must verify every proposed gold answer
# before the question can enter test-taker inference.
def _validate_generated_gold_answers(
    questions,
    agent_lm,
    agent_tokenizer,
    agent_client,
    research_config,
    outfile_prefix,
):
    """Independently recompute and substitute proposed gold answers."""
    if not research_config or not research_config["generation"][
        "require_gold_answer_validation"
    ]:
        return list(questions)
    validation_input = [
        {
            "validation_id": index,
            "question": question["question"],
            "answer_type": question["answer_type"],
            "proposed_gold_answer": question["canonical_answer"],
            "unit": question.get("unit"),
            "tolerance": question.get("tolerance"),
        }
        for index, question in enumerate(questions)
    ]
    prompt = f"""You are the privileged math gold-answer validator.
Independently solve every problem below. Do not trust the proposed answer.
For equations, inequalities, constraints, geometry, and word problems,
substitute the proposed answer back into every original condition. For direct
calculations, recompute the result using an independent derivation.

Return only one JSON array with exactly one object per validation_id:
[
  {{
    "validation_id": 0,
    "recomputed_answer": "standalone answer",
    "answer_type": "integer",
    "verification_passed": true,
    "substitution_passed": true,
    "verification_method": "independent recomputation and substitution",
    "failure_reason": null
  }}
]

Rules:
- verification_passed is true only when the proposed answer is mathematically
  correct, unique under the stated conditions, and has the requested type.
- substitution_passed is true only when the answer satisfies every applicable
  equation, domain, sign, unit, and problem constraint.
- Never copy the proposed answer without independently recomputing it.
- Use English strings and output no Markdown or additional text.

Problems:
{json.dumps(validation_input, ensure_ascii=False, indent=2)}
"""
    validation_config = research_config["generation"]
    last_error = None
    attempts = int(validation_config["gold_validation_attempts"])
    for attempt in range(1, attempts + 1):
        response = ""
        try:
            request_result = gen_from_prompt(
                model=agent_lm,
                tokenizer=agent_tokenizer,
                prompt=[prompt],
                echo_prompt=False,
                temperature=float(
                    validation_config["gold_validation_temperature"]
                ),
                max_tokens=int(
                    validation_config["gold_validation_max_tokens"]
                ),
                process_func=None,
                service=agent_client,
                terminate_by_linebreak="no",
                **_request_control_kwargs(research_config),
            )
            response = request_result.completions[0].text
            extracted = extract_json_v2(response, None)
            validations = extracted[0]
            if (
                not isinstance(validations, list)
                or len(validations) != len(questions)
            ):
                raise ValueError(
                    "Gold validation must return one result per question"
                )
            by_id = {}
            for item in validations:
                if not isinstance(item, dict):
                    raise ValueError(
                        "Every gold validation result must be an object"
                    )
                validation_id = int(item.get("validation_id", -1))
                if validation_id in by_id:
                    raise ValueError("Duplicate gold validation_id")
                by_id[validation_id] = item
            expected_ids = set(range(len(questions)))
            if set(by_id) != expected_ids:
                raise ValueError(
                    "Gold validation_id values must exactly match the input"
                )
            accepted = []
            audit_records = []
            for index, question in enumerate(questions):
                validation = by_id[index]
                if (
                    question["answer_type"] == "ordered_tuple"
                    and str(question["question"]).count("=") >= 1
                ):
                    # [MODIFIED] Never trust evaluator self-reported booleans
                    # for tuple-valued systems. Both layers use one parsed copy
                    # of the exact original question string.
                    audit = _validate_ordered_tuple_system(
                        question["question"],
                        question["canonical_answer"],
                        research_config,
                    )
                else:
                    recomputed_answer = validation.get(
                        "recomputed_answer",
                        "",
                    )
                    verification_passed = validation.get(
                        "verification_passed"
                    )
                    substitution_passed = validation.get(
                        "substitution_passed"
                    )
                    if not isinstance(verification_passed, bool):
                        raise ValueError(
                            "verification_passed must be a JSON boolean"
                        )
                    if not isinstance(substitution_passed, bool):
                        raise ValueError(
                            "substitution_passed must be a JSON boolean"
                        )
                    equivalence = answers_equivalent(
                        question["canonical_answer"],
                        recomputed_answer,
                        question["answer_type"],
                        research_config,
                    )
                    validator_answer_type = normalize_answer_type(
                        validation.get(
                            "answer_type",
                            question["answer_type"],
                        ),
                        recomputed_answer,
                    )
                    answer_type_consistent = (
                        validator_answer_type == question["answer_type"]
                    )
                    audit = {
                        "status": (
                            "passed"
                            if verification_passed
                            and substitution_passed
                            and equivalence["equivalent"]
                            and answer_type_consistent
                            else "failed"
                        ),
                        "recomputed_answer": recomputed_answer,
                        "answer_type": validator_answer_type,
                        "answer_type_consistent": answer_type_consistent,
                        "verification_passed": verification_passed,
                        "substitution_passed": substitution_passed,
                        "verification_method": str(
                            validation.get(
                                "verification_method",
                                "",
                            )
                        ).strip(),
                        "failure_reason": validation.get("failure_reason"),
                        "answer_equivalent": bool(
                            equivalence["equivalent"]
                        ),
                    }
                question["gold_answer_validation"] = audit
                audit_records.append(
                    {
                        "question_id": question.get("question_id"),
                        "question": question["question"],
                        "proposed_gold_answer": question[
                            "canonical_answer"
                        ],
                        "validation": audit,
                    }
                )
                if audit["status"] == "passed":
                    accepted.append(question)
            # [ADDED] Keep both accepted and rejected gold-answer checks for
            # reproducibility. Failed questions never enter test-taker inference.
            audit_file = f"{outfile_prefix}.gold_answer_validation.json"
            existing_audits = read_json_records(audit_file)
            dump_standard_json(
                existing_audits + audit_records,
                audit_file,
            )
            return accepted
        except (ValueError, TypeError, IndexError) as exc:
            last_error = exc
            _write_attempt_log(
                outfile_prefix,
                "gold_answer_validation",
                attempt,
                response,
            )
    raise RuntimeError(
        "Failed to validate generated gold answers"
    ) from last_error


def _generate_question_from_description(
    description_json,
    agent_lm,
    agent_tokenizer,
    agent_client,
    outfile_prefix='att1',
    questions_old=None,
    hard_sample_context="",
    question_count=50,
    research_config=None,
):
    question_count = int(question_count)
    generator_prompt_path = (
        Path(__file__).resolve().parent
        / "prompts"
        / "generator_question.txt"
    )
    context = generator_prompt_path.read_text(encoding="utf-8").format(
        question_count=question_count,
    )

    sub_category = description_json.get(
        "sub_category", description_json.get("subcategory_description", "")
    )
    context = (
        # [MODIFIED] Do not prepend the legacy TERMINATE instruction because
        # it conflicts with the strict JSON-only generation contract.
        context
        + f"\nTop-level category: {description_json['category']}"
        + f"\nSub_category: {sub_category}"
        + f"\nRequired generation_source: {description_json.get('generation_source', 'coverage_deficit')}"
        + f"\nRequired generation_strategy: {description_json.get('generation_strategy', 'quota_repair')}"
        + f"\nTarget difficulty: {description_json.get('difficulty', 5)}"
        + "\nUse English text only."
    )
    guidance_ids = []
    if research_config:
        guidance_context, guidance_ids = load_generation_guidance_context(
            research_config,
            category=str(description_json["category"]),
            subcategory=str(sub_category),
            target_difficulty=int(description_json.get("difficulty", 5)),
            selection_key=(
                f"{description_json.get('generation_source', '')}:"
                f"{description_json.get('generation_strategy', '')}:"
                f"{description_json.get('difficulty', 5)}"
            ),
        )
        if guidance_context:
            context += f"""

{guidance_context}

Use these records only as abstract structural guidance. Create a genuinely new
problem: change wording, constants, entities, representation, and at least one
reasoning dependency. Do not infer or reproduce a withheld source question or
answer.
"""
    if hard_sample_context:
        context += f"""

Use the train-eligible wrong-answer examples below as weakness evidence.
Generate variants in the same category and sub_category with accuracy expected
between 0.1 and 0.4. Preserve the mathematical trap and reasoning depth, but
change wording, values, and surface form. Never copy a question verbatim.

Train-eligible examples:
{hard_sample_context}
"""
    if questions_old:
        old_q_string = ''
        for q in questions_old:
            old_q_string += str(q) + '\n'
        remaining = max(1, question_count - len(questions_old))
        context += (
            f"\nQuestions already generated: {old_q_string}"
            f"\nGenerate exactly {remaining} new, non-repetitive questions."
        )
    extracted_json = None
    last_error = None
    valid_json = False
    for attempt in range(1, 4):
        retry_instruction = ""
        if attempt > 1:
            retry_instruction = (
                "\nYour previous response could not be parsed. Return only one "
                "complete JSON array. Do not include Markdown, analysis, "
                "comments, placeholders, or text outside the array."
            )
        request_result = gen_from_prompt(model=agent_lm, tokenizer=agent_tokenizer, prompt=[context + retry_instruction],
                                         echo_prompt=False, temperature=0.0, max_tokens=8192,
                                         process_func=None, service=agent_client,
                                         terminate_by_linebreak='no',
                                         **_request_control_kwargs(research_config), )
        response = request_result.completions[0].text
        try:
            extracted_json = extract_json_v2(response, None)
            questions = extracted_json[0]
            if not isinstance(questions, list) or not questions:
                raise ValueError("Expected a non-empty list of math questions")
            required_keys = {
                "question",
                "answer_type",
                "canonical_answer",
                "display_answer",
            }
            if any(not isinstance(item, dict) for item in questions):
                raise ValueError("Every generated question must be a JSON object")
            for item in questions:
                if "answer" in item and "canonical_answer" not in item:
                    item["canonical_answer"] = item["answer"]
                if "canonical_answer" in item and "display_answer" not in item:
                    item["display_answer"] = str(item["canonical_answer"])
                if "answer_type" not in item:
                    item["answer_type"] = "text"
                if not required_keys.issubset(item):
                    raise ValueError(
                        "Each generated question must contain question, answer_type, "
                        "canonical_answer, and display_answer"
                    )
            if any(
                re.search(
                    r"[\u3400-\u9fff\ufffd]",
                    (
                        f"{item.get('question', '')} "
                        f"{item.get('canonical_answer', '')} "
                        f"{item.get('display_answer', '')}"
                    ),
                )
                for item in questions
            ):
                raise ValueError(
                    "Generated questions and answers must use English text "
                    "only and must not contain corrupted Unicode"
                )
            valid_json = True
            break
        except (ValueError, IndexError, TypeError) as exc:
            last_error = exc
            print(f"Math question generation attempt {attempt}/3 failed: {exc}")
            _write_attempt_log(outfile_prefix, "questions", attempt, response)
    if not valid_json:
        raise RuntimeError("Failed to generate math questions after 3 attempts") from last_error
    for line in extracted_json[0]:
        line["id"] = line.get("id", line.get("question_id"))
        line["question_id"] = str(
            line.get("question_id", line.get("id", ""))
        )
        line["category"] = description_json["category"]
        line["subcategory"] = sub_category
        line["sub_category"] = sub_category
        raw_answer_type = str(line.get("answer_type", "text"))
        line["answer_type"] = normalize_answer_type(
            raw_answer_type,
            line.get("canonical_answer"),
        )
        if line["answer_type"] != raw_answer_type:
            line["raw_answer_type"] = raw_answer_type
        line["answer"] = line["canonical_answer"]
        line["gold_answer"] = line["canonical_answer"]
        line["difficulty"] = max(
            1,
            min(10, int(line.get("difficulty", description_json.get("difficulty", 5)))),
        )
        line["unit"] = line.get("unit")
        line["tolerance"] = line.get("tolerance")
        line["order_sensitive"] = bool(line.get("order_sensitive", False))
        line["generation_source"] = description_json.get(
            "generation_source",
            "coverage_deficit",
        )
        line["reference_hard_sample_ids"] = list(
            description_json.get("reference_hard_sample_ids", [])
        )
        line["generation_guidance_ids"] = list(guidance_ids)
        line["target_error_type"] = description_json.get("target_error_type")
        line["generation_strategy"] = description_json.get(
            "generation_strategy",
            "quota_repair",
        )
    extracted_json[0] = _validate_generated_gold_answers(
        extracted_json[0],
        agent_lm,
        agent_tokenizer,
        agent_client,
        research_config,
        outfile_prefix,
    )
    return extracted_json


_SYMPY_SUBCATEGORY_CONTRACTS = {
    "Integer Operations": "End with `Compute <integer expression>.`",
    "Fraction and Decimal Operations": (
        "End with `Compute <exact fraction/decimal expression>.`"
    ),
    "Ratio and Percentage": (
        "State the ratio or percentage story, then end with one equivalent "
        "`Compute <exact expression>.` clause."
    ),
    "Linear Equations": "Use `Solve for x: <left> = <right>.`",
    "Systems of Equations": (
        "Use `Solve the system for (x, y): <eq1>, <eq2>.` with a unique tuple."
    ),
    "Polynomials and Inequalities": (
        "Use either `Solve for x: <polynomial> = 0.` or "
        "`Solve the inequality for x: <relation>.`"
    ),
    "Plane Geometry": (
        "State the geometry facts, then end with `Compute <formula with values>.`"
    ),
    "Solid Geometry": (
        "State the solid dimensions, then end with `Compute <exact volume or "
        "surface-area expression>.`"
    ),
    "Trigonometric Reasoning": (
        "End with `Compute <exact expression using sin, cos, tan, sqrt, pi>.`"
    ),
    "Basic Probability": (
        "State the event, then end with `Compute <favorable/total expression>.`"
    ),
    "Combinatorics": (
        "End with `Compute binomial(n,k).` or a factorial expression."
    ),
    "Descriptive Statistics": (
        "End with `Compute mean(v1,...,vn).` or `Compute "
        "variance(v1,...,vn).`"
    ),
    "Rate and Distance": (
        "State the word problem, then end with `Compute <distance/rate/time "
        "expression>.`"
    ),
    "Work and Mixture": (
        "State the word problem, then end with `Compute <exact rational expression>.`"
    ),
    "Financial Applications": (
        "State the financial problem, then end with `Compute <exact expression>.`"
    ),
    "Divisibility and Factors": (
        "End with `Compute gcd(a,b).`, `Compute lcm(a,b).`, or "
        "`Compute divisor_count(n).`"
    ),
    "Prime Factorization": (
        "Ask a prime property with `Compute isprime(n).` or use "
        "`Compute divisor_count(n).`"
    ),
    "Modular Arithmetic": "End with `Compute Mod(a,m).`",
    "Limits and Continuity": (
        "Use `Find the limit of <expression> as x approaches <point>.`"
    ),
    "Differentiation": (
        "Use `Differentiate <expression> with respect to x.` optionally "
        "followed by `at x = <point>`."
    ),
    "Integration": (
        "Use `Integrate <expression> with respect to x from <a> to <b>.`"
    ),
    "Matrix Operations": (
        "End with `Compute det(Matrix([[...],[...]])).`"
    ),
    "Linear Systems": (
        "Use `Solve the system for (x, y): <eq1>, <eq2>.` with a unique tuple."
    ),
    "Vectors and Vector Spaces": (
        "End with `Compute dot(Matrix([...]),Matrix([...])).`"
    ),
    "Cross-Domain Multi-Step Problems": (
        "Give a short multi-step story and end with one exact `Compute "
        "<combined expression>.` clause."
    ),
    "Proof and Mathematical Reasoning": (
        "Ask for a decidable property and end with `Compute isprime(n).` or "
        "an exact identity-difference expression."
    ),
    "Constraint Synthesis": (
        "Use a complete equation system with a unique finite solution."
    ),
}


def _sympy_question_contract(sub_category):
    return _SYMPY_SUBCATEGORY_CONTRACTS.get(
        str(sub_category),
        "End with one exact `Compute <expression>.` clause.",
    )


# The generator emits only questions. The default gold source is the local
# deterministic SymPy solver; the legacy LLM-authored Python chain is opt-in.
def _generate_question_text_with_truth(
    description_json,
    agent_lm,
    agent_tokenizer,
    agent_client,
    outfile_prefix="att1",
    questions_old=None,
    hard_sample_context="",
    question_count=50,
    research_config=None,
    repair_feedback=None,
):
    question_count = int(question_count)
    sub_category = description_json.get(
        "sub_category",
        description_json.get("subcategory_description", ""),
    )
    generation_config = research_config["generation"]
    target_profile = description_json.get(
        "target_difficulty_profile",
    ) or target_difficulty_profile(
        int(description_json.get("difficulty", 5)),
        research_config,
    )
    max_retry = int(generation_config["generator_max_retry"])
    truth_solver = TruthSolver.from_config(research_config)
    context = f"""Generate exactly {question_count} English math questions for:
Category: {description_json["category"]}
Subcategory: {sub_category}
Difficulty: {description_json.get("difficulty", 5)}
Objective difficulty profile:
{json.dumps(target_profile, ensure_ascii=False, sort_keys=True)}

Return only one JSON array in this exact question-only format:
[
  {{"question": "What is 5 + 3?"}}
]

Mandatory rules:
1. Every object must contain exactly one key: question.
2. Never output an answer, candidate answer, gold answer, solution, answer type,
   explanation, reasoning, hint, metadata, or Markdown.
3. Every question must be self-contained, objectively solvable, and have a
   finite closed-form answer that SymPy can independently derive.
4. Use explicit solver-friendly wording:
   - equations: "Solve for x: ... = ...."
   - systems: "Solve the system for (x, y): eq1, eq2."
   - integrals: "Integrate EXPR with respect to x from A to B."
   - limits: "Find the limit of EXPR as x approaches A."
   - calculations: "Compute EXPR."
   - derivatives: "Differentiate EXPR with respect to x."
   - inequalities: "Solve the inequality for x: LEFT <= RIGHT."
5. Write powers as ^ or ** and use explicit equality signs.
6. Use English only and never copy an earlier question verbatim.
7. Keep every problem within difficulty
   {int(generation_config["minimum_difficulty"])} to
   {int(generation_config["maximum_difficulty"])} on a 1-10 scale and solvable
   in at most {int(generation_config["maximum_reasoning_steps"])} concise
   reasoning steps.
   Difficulty means the required mathematical work in the objective profile:
   reasoning steps, operations, constraints, symbolic depth, and
   representation load. Do not simulate difficulty with large numbers,
   verbose stories, obscure names, or unnecessary arithmetic.
8. Do not generate olympiad, contest-final, research-level, trick, or
   intentionally pathological problems. Prefer clear school or early
   undergraduate exercises with modest arithmetic.
9. The final solver clause must encode exactly the same computation described
   by any preceding story. Do not require unstated assumptions or mental
   arithmetic outside that clause.
10. Allowed exact functions are sin, cos, tan, sqrt, exp, log, Abs, factorial,
    binomial, gcd, lcm, Mod, floor, ceiling, Matrix, det, dot, mean, variance,
    totient, divisor_count, and isprime. Do not use prose number words inside
    the solver clause.

Subcategory-specific SymPy contract:
{_sympy_question_contract(sub_category)}
"""
    if "system" in sub_category.lower():
        context += """
System-specific constraint:
- Include the ordered variable tuple and every complete equation.
- Construct a consistent system with one unique solution.
- A valid solution must satisfy every equation simultaneously.
- Do not emit any numerical solution or candidate tuple.
"""
    if hard_sample_context:
        context += f"""
Use these weakness patterns only as structural inspiration. Do not copy their
answers or question text:
{hard_sample_context}
"""
    if questions_old:
        previous_questions = [
            str(item.get("question", "")).strip()
            for item in questions_old
            if isinstance(item, dict)
        ]
        context += (
            "\nPreviously accepted questions that must not be repeated:\n"
            + json.dumps(
                previous_questions,
                ensure_ascii=False,
                indent=2,
            )
        )
    feedback = str(repair_feedback or "").strip()
    failures = []
    accepted_questions = []
    raw_generator_output_count = 0
    for attempt in range(1, max_retry + 1):
        feedback_block = ""
        if feedback:
            feedback_block = f"""
Previous attempt failure summary:
{feedback}
Correct every listed failure. Do not repeat the same invalid output pattern.
"""
        response = ""
        try:
            ledger = active_ledger()
            if ledger is not None:
                ledger.assert_generation_available()
            generation_call_started = time.monotonic()
            request_result = gen_from_prompt(
                model=agent_lm,
                tokenizer=agent_tokenizer,
                prompt=[context + feedback_block],
                echo_prompt=False,
                temperature=float(generation_config["temperature"]),
                top_p=float(generation_config["top_p"]),
                max_tokens=8192,
                process_func=None,
                service=agent_client,
                terminate_by_linebreak="no",
                budget_role="generation",
                **_request_control_kwargs(research_config),
            )
            response = request_result.completions[0].text
            if ledger is not None and agent_client is None:
                ledger.record_generation_call(
                    input_tokens=estimate_tokens(context + feedback_block),
                    output_tokens=estimate_tokens(response),
                    wall_time_seconds=(
                        time.monotonic() - generation_call_started
                    ),
                    retry=attempt > 1,
                    api_calls=0,
                    exact_tokens=False,
                )
            extracted = extract_json_v2(response, None)
            items = extracted[0]
            if not isinstance(items, list) or not items:
                raise ValueError(
                    "Expected a non-empty question-only JSON array"
                )
            raw_generator_output_count += len(items)
            format_errors = []
            normalized_items = []
            for item_index, item in enumerate(items):
                if not isinstance(item, dict):
                    format_errors.append(
                        f"item {item_index} is not a JSON object"
                    )
                    continue
                keys = set(item)
                if keys != {"question"}:
                    forbidden = sorted(keys - {"question"})
                    failure_type = FailureType.GENERATOR_FORMAT_ERROR.value
                    candidate = next(
                        (
                            item.get(key)
                            for key in (
                                "canonical_answer",
                                "gold_answer",
                                "candidate_gold_answer",
                                "answer",
                            )
                            if item.get(key) is not None
                        ),
                        None,
                    )
                    if candidate is not None and item.get("question"):
                        candidate_check = truth_solver.validate_candidate(
                            item["question"],
                            candidate,
                        )
                        if candidate_check.get("failure_type") == (
                            FailureType.PARTIAL_SOLUTION.value
                        ):
                            failure_type = (
                                FailureType.PARTIAL_SOLUTION.value
                            )
                            format_errors.append(
                                "item "
                                f"{item_index} leaked a partial solution "
                                f"that passed "
                                f"{candidate_check['equations_passed']}/"
                                f"{candidate_check['equations_total']} "
                                "equations"
                            )
                    failures.append(
                        {
                            "stage": "generator",
                            "failure_type": failure_type,
                            "attempt": attempt,
                            "item_index": item_index,
                            "question": item.get("question"),
                            "unexpected_fields": forbidden,
                        }
                    )
                    format_errors.append(
                        f"item {item_index} contains forbidden fields: "
                        + ", ".join(forbidden)
                    )
                    continue
                question_text = str(item["question"]).strip()
                if not question_text:
                    format_errors.append(
                        f"item {item_index} has an empty question"
                    )
                    continue
                if re.search(r"[\u3400-\u9fff\ufffd]", question_text):
                    format_errors.append(
                        f"item {item_index} is not valid English UTF-8 text"
                    )
                    continue
                normalized_items.append({"question": question_text})
            if format_errors:
                feedback = "; ".join(format_errors[:12])
                failures.append(
                    {
                        "stage": "generator",
                        "failure_type": (
                            FailureType.GENERATOR_FORMAT_ERROR.value
                        ),
                        "attempt": attempt,
                        "failure_summary": feedback,
                    }
                )
                _write_attempt_log(
                    outfile_prefix,
                    "question_only_generator",
                    attempt,
                    response,
                )
                # Keep valid siblings from a mixed batch. Invalid candidates
                # are skipped individually; the outer quota-repair loop may
                # request replacements without terminating the iteration.
                if normalized_items:
                    accepted_questions = normalized_items
                    break
                continue
            accepted_questions = normalized_items
            break
        except (ValueError, TypeError, IndexError) as exc:
            feedback = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "stage": "generator",
                    "failure_type": (
                        FailureType.GENERATOR_FORMAT_ERROR.value
                    ),
                    "attempt": attempt,
                    "failure_summary": feedback,
                }
            )
            _write_attempt_log(
                outfile_prefix,
                "question_only_generator",
                attempt,
                response,
            )

    if not accepted_questions:
        failures.append(
            {
                "stage": "generator",
                "failure_type": FailureType.REPAIR_EXHAUSTED.value,
                "attempts": max_retry,
                "failure_summary": (
                    "Question-only generator repair attempts were exhausted."
                ),
            }
        )

    gold_solver_backend = str(
        generation_config["gold_solver_backend"]
    )
    evaluator_enabled = bool(
        gold_solver_backend == "llm_python"
        and research_config["evaluator_pipeline"]["enabled"]
    )
    evaluator_truths = [None] * len(accepted_questions)
    evaluator_cache_path = (
        f"{outfile_prefix}.evaluator_python_solutions.json"
    )
    configured_workers = int(
        research_config["evaluator_pipeline"]["max_parallel_questions"]
    )
    is_thread_safe_api_client = bool(
        agent_client is not None
        and hasattr(agent_client, "chat")
        and hasattr(agent_client.chat, "completions")
    )
    evaluator_workers = (
        min(configured_workers, len(accepted_questions))
        if evaluator_enabled and is_thread_safe_api_client
        else 1
    )
    if evaluator_enabled and accepted_questions:
        print(
            "[Generate] gold_verification_start "
            f"questions={len(accepted_questions)} "
            f"parallel_workers={evaluator_workers}",
            flush=True,
        )

        def solve_candidate(candidate_index):
            try:
                return solve_with_privileged_python(
                    accepted_questions[candidate_index]["question"],
                    (agent_lm, agent_tokenizer, agent_client),
                    research_config,
                    cache_path=evaluator_cache_path,
                )
            except Exception as exc:
                # One provider/client failure rejects only this candidate.
                # The remaining quota continues through the normal repair loop.
                return {
                    "status": "failed",
                    "failure_reason": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }

        if evaluator_workers > 1:
            with ThreadPoolExecutor(
                max_workers=evaluator_workers,
                thread_name_prefix="gold-evaluator",
            ) as executor:
                futures = {
                    executor.submit(solve_candidate, index): index
                    for index in range(len(accepted_questions))
                }
                completed = 0
                for future in as_completed(futures):
                    candidate_index = futures[future]
                    evaluator_truths[candidate_index] = future.result()
                    completed += 1
                    print(
                        "[Generate] gold_verification_progress "
                        f"completed={completed}/{len(accepted_questions)}",
                        flush=True,
                    )
        else:
            for index in range(len(accepted_questions)):
                evaluator_truths[index] = solve_candidate(index)
                print(
                    "[Generate] gold_verification_progress "
                    f"completed={index + 1}/{len(accepted_questions)}",
                    flush=True,
                )

    solved_questions = []
    for index, item in enumerate(accepted_questions):
        question_text = item["question"]
        evaluator_truth = evaluator_truths[index]
        deterministic_truth = truth_solver.solve(question_text)
        if evaluator_truth is None:
            if not deterministic_truth.success:
                failures.append(
                    {
                        "stage": "truth_solver",
                        "failure_type": deterministic_truth.failure_type,
                        "item_index": index,
                        "question": question_text,
                        "failure_summary": deterministic_truth.failure_summary,
                        "truth_validation_details": (
                            deterministic_truth.truth_validation_details
                        ),
                    }
                )
                continue
            evaluator_truth = {
                "status": "passed",
                "canonical_answer": deterministic_truth.canonical_answer,
                "answer_type": deterministic_truth.answer_type,
                "verification_passed": True,
                "substitution_passed": bool(
                    deterministic_truth.truth_validation_details.get(
                        "substitution_passed",
                        True,
                    )
                ),
                "source_question_sha256": (
                    deterministic_truth.truth_validation_details.get(
                        "source_question_sha256"
                    )
                ),
                "solver_question_sha256": (
                    deterministic_truth.truth_validation_details.get(
                        "solver_question_sha256"
                    )
                ),
                "solver_prompt_version": "sympy_gold_executor_v1",
                "python_code_sha256": None,
                "verification_details": [],
                "postcheck": {},
                "training_reasoning_summary": (
                    truth_solver.training_reasoning(
                        deterministic_truth
                    )
                ),
                "estimated_difficulty": int(
                    description_json.get("difficulty", 5)
                ),
                "difficulty_acceptable": True,
            }
        if evaluator_truth.get("status") != "passed":
            failure_reason = str(
                evaluator_truth.get(
                    "failure_reason",
                    "privileged evaluator rejected the solution",
                )
            )
            failure_type = (
                FailureType.DIFFICULTY_REJECTED.value
                if "difficulty" in failure_reason.lower()
                else FailureType.EVALUATOR_CODE_FAILURE.value
            )
            failures.append(
                {
                    "stage": "privileged_evaluator",
                    "failure_type": failure_type,
                    "item_index": index,
                    "question": question_text,
                    "failure_summary": failure_reason,
                    "truth_validation_details": evaluator_truth,
                }
            )
            continue
        canonical_answer = str(evaluator_truth["canonical_answer"])
        answer_type = normalize_answer_type(
            evaluator_truth["answer_type"],
            canonical_answer,
        )
        try:
            answer_contract = normalize_generated_gold_contract(
                question_text,
                canonical_answer,
                answer_type,
                item.get("tolerance"),
                decimal_tolerance=float(
                    research_config["answer_normalization"][
                        "decimal_gold_tolerance"
                    ]
                ),
            )
        except (ValueError, TypeError, OverflowError) as exc:
            # A malformed solver/evaluator answer is a rejected candidate, not
            # a fatal iteration error. The outer quota loop can request a
            # replacement while retaining valid siblings from the same batch.
            failures.append(
                {
                    "stage": "gold_contract",
                    "failure_type": FailureType.TRUTH_PARSE_FAIL.value,
                    "item_index": index,
                    "question": question_text,
                    "failure_summary": f"{type(exc).__name__}: {exc}",
                    "solver_answer": canonical_answer,
                    "solver_answer_type": answer_type,
                }
            )
            continue
        canonical_answer = answer_contract["canonical_answer"]
        answer_type = answer_contract["answer_type"]
        answer_tolerance = answer_contract["tolerance"]
        exact_canonical_answer = answer_contract[
            "exact_canonical_answer"
        ]
        deterministic_agreement = None
        if evaluator_enabled and deterministic_truth.success:
            try:
                deterministic_contract = normalize_generated_gold_contract(
                    question_text,
                    deterministic_truth.canonical_answer,
                    deterministic_truth.answer_type,
                    decimal_tolerance=float(
                        research_config["answer_normalization"][
                            "decimal_gold_tolerance"
                        ]
                    ),
                )
            except (ValueError, TypeError, OverflowError) as exc:
                failures.append(
                    {
                        "stage": "truth_agreement",
                        "failure_type": FailureType.TRUTH_PARSE_FAIL.value,
                        "item_index": index,
                        "question": question_text,
                        "failure_summary": f"{type(exc).__name__}: {exc}",
                        "truth_solver_answer": (
                            deterministic_truth.canonical_answer
                        ),
                        "truth_solver_answer_type": (
                            deterministic_truth.answer_type
                        ),
                    }
                )
                continue
            deterministic_agreement = answers_equivalent(
                canonical_answer,
                deterministic_contract["canonical_answer"],
                answer_type,
                research_config,
                tolerance=answer_tolerance,
            )
            if not deterministic_agreement["equivalent"]:
                failures.append(
                    {
                        "stage": "truth_agreement",
                        "failure_type": FailureType.TRUTH_DISAGREEMENT.value,
                        "item_index": index,
                        "question": question_text,
                        "failure_summary": (
                            "Privileged Python answer disagrees with "
                            "deterministic TruthSolver"
                        ),
                        "evaluator_answer": canonical_answer,
                        "truth_solver_answer": (
                            deterministic_truth.canonical_answer
                        ),
                    }
                )
                continue
        elif deterministic_truth.success:
            # With the SymPy backend, evaluator_truth is constructed directly
            # from this exact TruthSolver result. Sending set/interval/symbolic
            # answers through the generic cross-backend normalizer can only
            # introduce a false disagreement and pointless repair rounds.
            deterministic_agreement = {
                "equivalent": True,
                "method": "same_sympy_result",
            }
        if evaluator_enabled:
            evaluator_audit_summary = {
                key: value
                for key, value in evaluator_truth.items()
                if key
                    not in {
                        "python_code",
                        "analysis_summary",
                        "solver_analysis_summary",
                        "independent_python_code",
                        "independent_analysis_summary",
                    }
            }
            truth_details = {
                "source_question": question_text,
                "source_question_sha256": evaluator_truth.get(
                    "source_question_sha256"
                ),
                "solver_question_sha256": evaluator_truth.get(
                    "solver_question_sha256"
                ),
                "solver_branch": "isolated_evaluator_python",
                "solver_prompt_version": evaluator_truth.get(
                    "solver_prompt_version"
                ),
                "python_code_sha256": evaluator_truth.get(
                    "python_code_sha256"
                ),
                "substitution_passed": bool(
                    evaluator_truth.get("substitution_passed")
                ),
                "runtime_verification_passed": bool(
                    evaluator_truth.get("verification_passed")
                ),
                "postcheck": evaluator_truth.get("postcheck", {}),
                "deterministic_truth_solver": deterministic_truth.to_dict(),
                "deterministic_agreement": deterministic_agreement,
                "evaluator_audit": evaluator_audit_summary,
            }
        else:
            truth_details = dict(
                deterministic_truth.truth_validation_details
            )
            truth_details["solver_backend"] = "sympy"
        requested_difficulty = max(
            int(research_config["generation"]["minimum_difficulty"]),
            min(
                int(research_config["generation"]["maximum_difficulty"]),
                int(
                    evaluator_truth.get(
                        "estimated_difficulty",
                        description_json.get("difficulty", 5),
                    )
                ),
            ),
        )
        legacy_evaluator_difficulty = requested_difficulty
        requested_difficulty = max(
            int(research_config["generation"]["minimum_difficulty"]),
            min(
                int(research_config["generation"]["maximum_difficulty"]),
                int(description_json.get("difficulty", 5)),
            ),
        )
        gold_reasoning_summary = list(
            evaluator_truth.get(
                "training_reasoning_summary",
                evaluator_truth.get("analysis_summary", []),
            )
        )
        difficulty_profile = assess_difficulty(
            question_text,
            answer_type,
            truth_details,
            gold_reasoning_summary,
            requested_difficulty,
            research_config,
        )
        difficulty_profile["legacy_evaluator_estimated_score"] = (
            legacy_evaluator_difficulty
        )
        difficulty_config = research_config["difficulty"]
        difficulty_rejection_reason = None
        if (
            difficulty_profile["difficulty_module_enabled"]
            and
            difficulty_profile["profile_trusted"]
            and bool(
                difficulty_config[
                    "reject_outside_generation_bounds"
                ]
            )
            and not difficulty_profile["within_generation_bounds"]
        ):
            difficulty_rejection_reason = (
                "objective observed difficulty "
                f"{difficulty_profile['score']} is outside generation bounds "
                f"{research_config['generation']['minimum_difficulty']}.."
                f"{research_config['generation']['maximum_difficulty']}"
            )
        elif (
            difficulty_profile["difficulty_module_enabled"]
            and
            difficulty_profile["profile_trusted"]
            and str(difficulty_config["mismatch_action"]) == "reject"
            and not difficulty_profile["within_target_tolerance"]
        ):
            difficulty_rejection_reason = (
                "objective observed difficulty "
                f"{difficulty_profile['score']} differs from requested "
                f"{requested_difficulty} by more than tolerance "
                f"{difficulty_profile['target_tolerance']}"
            )
        if difficulty_rejection_reason:
            failures.append(
                {
                    "stage": "difficulty_profile",
                    "failure_type": FailureType.DIFFICULTY_REJECTED.value,
                    "item_index": index,
                    "question": question_text,
                    "failure_summary": difficulty_rejection_reason,
                    "difficulty_profile": difficulty_profile,
                }
            )
            continue
        effective_difficulty = int(
            difficulty_profile["effective_score"]
        )
        truth_details["difficulty_profile"] = difficulty_profile
        solved_questions.append(
            {
                "id": f"q_{index + 1}",
                "question_id": f"q_{index + 1}",
                "category": description_json["category"],
                "subcategory": sub_category,
                "sub_category": sub_category,
                "difficulty": effective_difficulty,
                "target_difficulty": requested_difficulty,
                "requested_difficulty": requested_difficulty,
                "observed_difficulty": int(
                    difficulty_profile["score"]
                ),
                "effective_difficulty": effective_difficulty,
                "difficulty_profile": difficulty_profile,
                "question": question_text,
                "answer_type": answer_type,
                "canonical_answer": canonical_answer,
                "display_answer": canonical_answer,
                "answer": canonical_answer,
                "gold_answer": canonical_answer,
                "gold_reasoning_summary": gold_reasoning_summary,
                "unit": None,
                "tolerance": answer_tolerance,
                "exact_canonical_answer": exact_canonical_answer,
                "order_sensitive": answer_type == "ordered_tuple",
                "generation_source": description_json.get(
                    "generation_source",
                    "coverage_deficit",
                ),
                "reference_hard_sample_ids": list(
                    description_json.get(
                        "reference_hard_sample_ids",
                        [],
                    )
                ),
                "target_error_type": description_json.get(
                    "target_error_type"
                ),
                "generation_strategy": description_json.get(
                    "generation_strategy",
                    "quota_repair",
                ),
                "truth_validation_details": truth_details,
                "failure_type": None,
                "gold_answer_validation": {
                    "status": "passed",
                    "recomputed_answer": canonical_answer,
                    "answer_type": answer_type,
                    "answer_type_consistent": True,
                    "verification_passed": True,
                    "substitution_passed": bool(
                        truth_details.get("substitution_passed", True)
                    ),
                    "verification_method": (
                        (
                            "isolated evaluator Python execution with "
                            "runtime verification and SymPy agreement"
                        )
                        if evaluator_enabled
                        else (
                            "deterministic SymPy execution with exact "
                            "solution and substitution verification"
                        )
                    ),
                    "failure_reason": None,
                    "answer_equivalent": True,
                    "failure_type": None,
                    "truth_validation_details": truth_details,
                },
            }
        )

    failure_file = f"{outfile_prefix}.generation_failures.json"
    existing_failures = read_json_records(failure_file)
    dump_standard_json(
        existing_failures + failures,
        failure_file,
    )
    truth_audits = [
        {
            "question_id": item["question_id"],
            "question": item["question"],
            "proposed_gold_answer": None,
            "validation": item["gold_answer_validation"],
        }
        for item in solved_questions
    ]
    audit_file = f"{outfile_prefix}.gold_answer_validation.json"
    dump_standard_json(
        read_json_records(audit_file) + truth_audits,
        audit_file,
    )
    failure_counts = defaultdict(int)
    for failure in failures:
        failure_counts[str(failure.get("failure_type", "unknown"))] += 1
    ledger = active_ledger()
    if ledger is not None:
        difficulty_rejections = int(
            failure_counts.get(
                FailureType.DIFFICULTY_REJECTED.value,
                0,
            )
        )
        ledger.record_generation_batch(
            requested=question_count,
            output=raw_generator_output_count,
            accepted=len(solved_questions),
            failed=max(
                max(
                    0,
                    raw_generator_output_count - len(solved_questions),
                ),
                sum(int(value) for value in failure_counts.values()),
            ),
            difficulty_rejections=difficulty_rejections,
            sympy_validations=len(accepted_questions),
            validation_failures=max(
                0,
                len(accepted_questions) - len(solved_questions),
            ),
        )
    dump_standard_json(
        {
            "gold_solver_backend": gold_solver_backend,
            "llm_gold_solver_calls": (
                len(accepted_questions) if evaluator_enabled else 0
            ),
            "requested_questions": question_count,
            "generator_output_questions": raw_generator_output_count,
            "valid_truth_solved_questions": len(solved_questions),
            "discarded_questions": max(
                0,
                len(accepted_questions) - len(solved_questions),
            ),
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        f"{outfile_prefix}.generation_batch_summary.json",
    )
    return [solved_questions]


def _ask_question_v3(
    agent_info,
    history,
    iters,
    outfile_prefix,
    aim_acc=None,
    enable_hard_sample_guidance=False,
    coverage_summary="",
    hard_pool_file=None,
    generation_plan=None,
    progress_manager=None,
    cycle_number=None,
    research_config=None,
):
    agent_lm, agent_tokenizer, agent_client = agent_info
    plan_outfile = f"{outfile_prefix}.question_plan_with_aim.json"
    if generation_plan is not None:
        plan_json = list(generation_plan["allocations"])
        dump_standard_json(plan_json, plan_outfile)
    elif not os.path.exists(plan_outfile):
        # [MODIFIED] Pass hard-pool evidence only in directed-generation rounds.
        plan_json = _generate_cat_with_aim(
            aim_acc,
            agent_lm,
            agent_tokenizer,
            agent_client,
            history,
            iters=iters,
            outfile_prefix=outfile_prefix,
            enable_hard_sample_guidance=enable_hard_sample_guidance,
            coverage_summary=coverage_summary,
            hard_pool_file=hard_pool_file,
            research_config=research_config,
        )

    else:
        print('FOUND THE PLAN FILE', plan_outfile)
        with open(plan_outfile, 'r', encoding="utf-8") as f:
            plan_json = json.load(f)
        normalized_plan = _normalize_math_plan(plan_json)
        if normalized_plan[0] != plan_json:
            _write_json_atomic(normalized_plan[0], plan_outfile)
        plan_json = normalized_plan
    if generation_plan is None:
        plan_json = plan_json[0]

    # [MODIFIED] Keep question fragments in memory; inference is canonical.
    question_json_full = []
    normalized_question_texts = set()
    hard_pool = HardSamplePool(hard_pool_file) if hard_pool_file else None
    repair_limit = (
        int(
            research_config["generation"][
                "max_quota_repair_rounds"
            ]
        )
        if research_config
        else 3
    )
    allow_partial_budget = (
        bool(
            research_config["generation"][
                "allow_partial_question_budget"
            ]
        )
        if research_config
        else True
    )
    minimum_verified_questions = (
        int(
            research_config["generation"][
                "minimum_verified_questions"
            ]
        )
        if research_config
        else 1
    )
    subcategory_shortfalls = []
    subcategory_statistics = []
    generation_health_file = os.path.join(
        os.path.dirname(os.path.abspath(hard_pool_file))
        if hard_pool_file
        else os.path.dirname(os.path.abspath(outfile_prefix)),
        "generation_health.json",
    )
    health_records = read_json_records(generation_health_file)
    generation_health = (
        dict(health_records[0])
        if health_records and isinstance(health_records[0], dict)
        else {}
    )
    current_global_iteration = int(
        generation_plan.get("global_iteration", iters)
        if generation_plan
        else iters
    )
    cooldown_threshold = (
        int(
            research_config["generation"][
                "subcategory_failure_cooldown_threshold"
            ]
        )
        if research_config
        else 3
    )
    cooldown_iterations = (
        int(
            research_config["generation"][
                "subcategory_cooldown_iterations"
            ]
        )
        if research_config
        else 2
    )
    generation_total = (
        int(generation_plan["question_budget"])
        if generation_plan is not None
        else sum(int(item.get("question_count", 50)) for item in plan_json)
    )
    progress_context = (
        progress_manager.stage(
            "Generate",
            total=generation_total,
            cycle=cycle_number,
            iteration=iters,
        )
        if progress_manager
        else contextlib.nullcontext(None)
    )
    with progress_context as progress:
        for idx, plan_line in enumerate(plan_json):
            outfile_prefix2 = outfile_prefix + '.subcat{}'.format(idx)
            health_key = (
                f"{plan_line['category']}|||{plan_line['sub_category']}"
            )
            health_state = dict(generation_health.get(health_key, {}))
            cooldown_until = int(
                health_state.get("cooldown_until_iteration", 0)
            )
            target_count = int(plan_line.get("question_count", 50))
            subcategory_started_at = time.monotonic()
            print(
                "[Generate] subcategory_start "
                f"index={idx + 1}/{len(plan_json)} "
                f"category={plan_line['category']!r} "
                f"sub_category={plan_line['sub_category']!r} "
                f"requested={target_count}",
                flush=True,
            )
            if current_global_iteration <= cooldown_until:
                cooldown_failure = {
                    "category": plan_line["category"],
                    "sub_category": plan_line["sub_category"],
                    "requested": target_count,
                    "verified": 0,
                    "shortfall": target_count,
                    "repair_rounds": 0,
                    "reason": "subcategory_cooldown",
                    "cooldown_until_iteration": cooldown_until,
                }
                subcategory_shortfalls.append(cooldown_failure)
                subcategory_statistics.append(
                    {
                        "category": plan_line["category"],
                        "sub_category": plan_line["sub_category"],
                        "generated_total": 0,
                        "valid_samples": 0,
                        "failure_counts": {
                            FailureType.SUBCATEGORY_COOLDOWN.value: (
                                target_count
                            )
                        },
                        "coverage_gap": target_count,
                    }
                )
                print(
                    "[Generate] subcategory_cooldown "
                    f"sub_category={plan_line['sub_category']!r} "
                    f"until_iteration={cooldown_until}",
                    flush=True,
                )
                continue
            is_hard_variant = (
                plan_line.get("generation_source") == "hard_pool_variant"
            )
            error_type_targeting_enabled = bool(
                (generation_plan or {})
                .get("component_state", {})
                .get("error_type_targeting", True)
            )
            variant_context = (
                hard_pool.get_variant_context(
                    plan_line["category"],
                    plan_line["sub_category"],
                    confidence_threshold=float(
                        research_config["error_attribution"][
                            "confidence_threshold"
                        ]
                    ),
                    include_error_targeting=(
                        error_type_targeting_enabled
                    ),
                )
                if enable_hard_sample_guidance and is_hard_variant and hard_pool
                else ""
            )
            if is_hard_variant and hard_pool:
                matching_samples = [
                    sample
                    for sample in hard_pool.samples
                    if sample.get("sample_grade") == "train_eligible"
                    and sample.get("category") == plan_line["category"]
                    and sample.get("sub_category") == plan_line["sub_category"]
                ]
                references = [
                    sample.get("unique_key", "")[:16]
                    for sample in matching_samples
                ][:12]
                plan_line["reference_hard_sample_ids"] = references
                error_counts = Counter(
                    tag
                    for sample in matching_samples
                    if (
                        error_type_targeting_enabled
                        and sample.get("verification_tier")
                        == "deterministic"
                        and float(
                            sample.get(
                                "attribution_confidence",
                                0.0,
                            )
                            or 0.0
                        )
                        >= float(
                            research_config["error_attribution"][
                                "confidence_threshold"
                            ]
                        )
                        and bool(sample.get("evidence"))
                    )
                    for tag in sample.get("error_tags", [])
                    if tag and tag != "unknown_error"
                )
                plan_line["target_error_type"] = (
                    error_counts.most_common(1)[0][0]
                    if error_counts
                    else None
                )
            try:
                question_json = _generate_question_text_with_truth(
                    plan_line,
                    agent_lm,
                    agent_tokenizer,
                    agent_client,
                    outfile_prefix2,
                    hard_sample_context=variant_context,
                    question_count=target_count,
                    research_config=research_config,
                )
            except BudgetExhausted as exc:
                subcategory_shortfalls.append(
                    {
                        "category": plan_line["category"],
                        "sub_category": plan_line["sub_category"],
                        "requested": target_count,
                        "verified": 0,
                        "shortfall": target_count,
                        "repair_rounds": 0,
                        "reason": "cost_budget_exhausted",
                        "detail": str(exc),
                    }
                )
                print(f"[Budget] generation_stopped reason={exc}", flush=True)
                break
            if len(question_json) == 1:
                question_json = question_json[0]
            question_json = [
                item
                for item in question_json
                if normalize_question_text(item.get("question"))
                and normalize_question_text(item.get("question"))
                not in normalized_question_texts
            ]
            normalized_question_texts.update(
                normalize_question_text(item.get("question"))
                for item in question_json
            )
            batch_summary_records = read_json_records(
                f"{outfile_prefix2}.generation_batch_summary.json"
            )
            batch_summary = (
                dict(batch_summary_records[0])
                if batch_summary_records
                and isinstance(batch_summary_records[0], dict)
                else {}
            )
            aggregate_generated = int(
                batch_summary.get("generator_output_questions", 0)
            )
            aggregate_failures = defaultdict(int)
            for key, value in batch_summary.get(
                "failure_counts",
                {},
            ).items():
                aggregate_failures[str(key)] += int(value)
            repair_round = 0
            while (
                len(question_json) < target_count
                and repair_round < repair_limit
            ):
                repair_round += 1
                missing_count = target_count - len(question_json)
                try:
                    question_json_new = _generate_question_text_with_truth(
                        plan_line,
                        agent_lm,
                        agent_tokenizer,
                        agent_client,
                        outfile_prefix2,
                        questions_old=question_json,
                        hard_sample_context=variant_context,
                        question_count=missing_count,
                        research_config=research_config,
                        repair_feedback=(
                            f"The previous batch still has {missing_count} "
                            "unfilled positions because some candidates were "
                            "duplicate, malformed, ambiguous, or failed gold "
                            "verification. Skip those failed candidates and "
                            "generate new structurally different questions while "
                            "preserving the assigned subcategory."
                        ),
                    )
                except BudgetExhausted as exc:
                    print(
                        f"[Budget] quota_repair_stopped reason={exc}",
                        flush=True,
                    )
                    break
                question_json_new = question_json_new[0]
                repair_summary_records = read_json_records(
                    f"{outfile_prefix2}.generation_batch_summary.json"
                )
                repair_summary = (
                    dict(repair_summary_records[0])
                    if repair_summary_records
                    and isinstance(repair_summary_records[0], dict)
                    else {}
                )
                aggregate_generated += int(
                    repair_summary.get(
                        "generator_output_questions",
                        0,
                    )
                )
                for key, value in repair_summary.get(
                    "failure_counts",
                    {},
                ).items():
                    aggregate_failures[str(key)] += int(value)
                unique_new = []
                for item in question_json_new:
                    signature = normalize_question_text(item.get("question"))
                    if not signature or signature in normalized_question_texts:
                        continue
                    normalized_question_texts.add(signature)
                    unique_new.append(item)
                question_json.extend(unique_new)
            accepted = question_json[:target_count]
            question_json_full.extend(accepted)
            verified_count = len(accepted)
            shortfall = max(0, target_count - verified_count)
            print(
                "[Generate] subcategory_done "
                f"index={idx + 1}/{len(plan_json)} "
                f"sub_category={plan_line['sub_category']!r} "
                f"verified={verified_count}/{target_count} "
                f"repair_rounds={repair_round} "
                f"elapsed={time.monotonic() - subcategory_started_at:.1f}s",
                flush=True,
            )
            plan_line["verified_question_count"] = verified_count
            plan_line["question_shortfall"] = shortfall
            plan_line["quota_repair_rounds_used"] = repair_round
            if verified_count:
                health_state.update(
                    {
                        "consecutive_failures": 0,
                        "cooldown_until_iteration": 0,
                        "last_success_iteration": (
                            current_global_iteration
                        ),
                    }
                )
            else:
                consecutive_failures = int(
                    health_state.get("consecutive_failures", 0)
                ) + 1
                health_state["consecutive_failures"] = (
                    consecutive_failures
                )
                health_state["last_failure_iteration"] = (
                    current_global_iteration
                )
                if consecutive_failures >= cooldown_threshold:
                    health_state["cooldown_until_iteration"] = (
                        current_global_iteration + cooldown_iterations
                    )
            health_state["last_failure_counts"] = dict(
                sorted(aggregate_failures.items())
            )
            generation_health[health_key] = health_state
            subcategory_statistics.append(
                {
                    "category": plan_line["category"],
                    "sub_category": plan_line["sub_category"],
                    "generated_total": aggregate_generated,
                    "valid_samples": verified_count,
                    "failure_counts": dict(
                        sorted(aggregate_failures.items())
                    ),
                    "coverage_gap": shortfall,
                }
            )
            if shortfall:
                subcategory_shortfalls.append(
                    {
                        "category": plan_line["category"],
                        "sub_category": plan_line["sub_category"],
                        "requested": target_count,
                        "verified": verified_count,
                        "shortfall": shortfall,
                        "repair_rounds": repair_round,
                        "reason": (
                            "insufficient_unique_gold_verified_questions"
                        ),
                    }
                )
                print(
                    "[Generate] quota_repair_exhausted "
                    f"sub_category={plan_line['sub_category']!r} "
                    f"verified={verified_count}/{target_count} "
                    f"repair_rounds={repair_round}"
                )
            if progress is not None:
                progress.update(len(accepted))
                progress.set_postfix(
                    verified=len(question_json_full),
                    shortfall=sum(
                        item["shortfall"]
                        for item in subcategory_shortfalls
                    ),
                )
    if generation_plan is not None:
        expected = int(generation_plan["question_budget"])
        verified_total = len(question_json_full)
        hard_variant_lines = [
            line
            for line in plan_json
            if line.get("generation_source") == "hard_pool_variant"
        ]
        targeted_lines = [
            line
            for line in hard_variant_lines
            if line.get("target_error_type")
        ]
        difficulty_rejection_count = sum(
            int(
                item.get("failure_counts", {}).get(
                    FailureType.DIFFICULTY_REJECTED.value,
                    0,
                )
            )
            for item in subcategory_statistics
        )
        realized_component_state = generation_plan.get(
            "component_state",
            {},
        )
        component_execution_evidence = {
            "hard_pool_variant_allocation_count": len(hard_variant_lines),
            "error_targeted_allocation_count": len(targeted_lines),
            "hard_pool_disabled_verified": (
                realized_component_state.get("hard_pool_variants", True)
                or not hard_variant_lines
            ),
            "error_type_targeting_disabled_verified": (
                realized_component_state.get("error_type_targeting", True)
                or not targeted_lines
            ),
            "difficulty_rejection_count": difficulty_rejection_count,
            "difficulty_module_disabled_verified": (
                realized_component_state.get("difficulty_module", True)
                or difficulty_rejection_count == 0
            ),
        }
        if not all(
            (
                component_execution_evidence["hard_pool_disabled_verified"],
                component_execution_evidence[
                    "error_type_targeting_disabled_verified"
                ],
                component_execution_evidence[
                    "difficulty_module_disabled_verified"
                ],
            )
        ):
            raise RuntimeError(
                "A disabled Hard Pool/error-targeting component affected "
                "the realized generation plan."
            )
        generation_result = {
            "requested_questions": expected,
            "verified_questions": verified_total,
            "question_shortfall": max(0, expected - verified_total),
            "partial_iteration": verified_total != expected,
            "below_minimum_verified": (
                verified_total < minimum_verified_questions
            ),
            "partial_budget_allowed": allow_partial_budget,
            "subcategory_shortfalls": subcategory_shortfalls,
            "subcategory_statistics": subcategory_statistics,
            "component_execution_evidence": component_execution_evidence,
        }
        dump_standard_json(generation_health, generation_health_file)
        generation_plan["generation_result"] = generation_result
        dump_standard_json(generation_result, (
            f"{outfile_prefix}.generation_quota_summary.json"
        ))
        if verified_total != expected:
            print(
                "[Generate] partial_verified_iteration "
                f"verified={verified_total}/{expected} "
                "coverage_target=cumulative "
                "action=continue_without_cycle_failure"
            )
        # [MODIFIED] Persist the realized verified counts alongside the
        # original plan without requiring all subcategories in one iteration.
        dump_standard_json(plan_json, plan_outfile)
        for index, question in enumerate(question_json_full, start=1):
            question["id"] = index
            question["question_id"] = (
                f"c{generation_plan.get('cycle', 0)}_"
                f"i{generation_plan['global_iteration']}_q{index:05d}"
            )
            # [ADDED] Fail before evaluation if a generated record violates the
            # checked-in, versioned question contract.
            validate_generated_question(question)
    return question_json_full, plan_json



def _build_compare_summary(iter_number, inference_records, research_config=None):
    grouped = defaultdict(list)
    difficulty_grouped = defaultdict(list)
    difficulty_gaps = []
    for record in inference_records:
        grouped[(record["category"], record["sub_category"])].append(record)
        difficulty_grouped[int(record.get("difficulty", 5))].append(record)
        profile = record.get("difficulty_profile")
        if isinstance(profile, dict):
            try:
                difficulty_gaps.append(
                    abs(
                        float(profile["score"])
                        - float(
                            profile.get(
                                "requested_score",
                                record.get("target_difficulty"),
                            )
                        )
                    )
                )
            except (KeyError, TypeError, ValueError):
                pass
    category_statistics = []
    for (category, sub_category), records in sorted(grouped.items()):
        correct_count = sum(record["is_correct"] for record in records)
        category_statistics.append(
            {
                "category": category,
                "sub_category": sub_category,
                "total_count": len(records),
                "correct_count": correct_count,
                "accuracy": correct_count / len(records),
            }
        )
    total_questions = len(inference_records)
    total_correct = sum(record["is_correct"] for record in inference_records)
    summary = {
        "iter_number": int(iter_number),
        "total_questions": total_questions,
        "global_accuracy": (
            total_correct / total_questions if total_questions else 0.0
        ),
        "category_statistics": category_statistics,
        "difficulty_statistics": [
            {
                "difficulty": difficulty,
                "total_count": len(records),
                "correct_count": sum(
                    bool(record.get("is_correct"))
                    for record in records
                ),
                "accuracy": (
                    sum(
                        bool(record.get("is_correct"))
                        for record in records
                    )
                    / len(records)
                ),
            }
            for difficulty, records in sorted(
                difficulty_grouped.items()
            )
            if records
        ],
        "difficulty_profiled_count": sum(
            isinstance(record.get("difficulty_profile"), dict)
            for record in inference_records
        ),
        "difficulty_mean_absolute_target_gap": (
            sum(difficulty_gaps) / len(difficulty_gaps)
            if difficulty_gaps
            else None
        ),
        "parse_failed_count": sum(
            record.get("parse_status") == "parse_failed"
            for record in inference_records
        ),
        "format_error_count": sum(
            record.get("evaluation_status") == "format_only_error"
            for record in inference_records
        ),
        "tool_violation_count": sum(
            bool(record.get("tool_violation"))
            for record in inference_records
        ),
        "irrelevant_output_count": sum(
            bool(record.get("contains_irrelevant_content"))
            for record in inference_records
        ),
        "prompt_echo_count": sum(
            bool(record.get("contains_prompt_echo"))
            for record in inference_records
        ),
        "test_taker_tool_call_count": sum(
            int(record.get("test_taker_tool_call_count", 0))
            for record in inference_records
        ),
    }
    if research_config:
        summary.update(coverage_metrics(inference_records, research_config))
    return summary


def _lowest_sub_categories(compare_summary, limit=10):
    return sorted(
        compare_summary.get("category_statistics", []),
        key=lambda item: (item["accuracy"], -item["total_count"], item["sub_category"]),
    )[:limit]


def _semantic_judge_cache_record_matches(cached, gold, predicted):
    return bool(
        isinstance(cached, dict)
        and str(cached.get("question", "")).strip()
        == str(gold.get("question", "")).strip()
        and str(cached.get("test_taker_answer", "")).strip()
        == str(predicted.get("test_taker_response", "")).strip()
        and str(cached.get("gold_answer", "")).strip()
        == str(gold.get("answer", "")).strip()
    )


def _failed_semantic_judgment(
    *,
    gold,
    predicted,
    research_config,
    error,
):
    """Convert a per-question judge outage into an auditable failed result."""
    deterministic = answers_equivalent(
        gold["answer"],
        predicted["test_taker_response"],
        predicted.get("answer_type", "text"),
        research_config,
    )
    detail = _sanitize_error(error)
    return {
        "semantically_equivalent": False,
        "confidence": 0.0,
        "reason": f"semantic judge unavailable: {detail}",
        "format_only_difference": False,
        "status": "failed",
        "failure_type": "provider_error",
        "attempt": int(
            research_config["evaluator_pipeline"]["semantic_judge_attempts"]
        ),
        "prompt_version": "semantic_answer_judge_unavailable",
        "prompt_sha256": None,
        "deterministic_equivalent": bool(deterministic["equivalent"]),
    }


def _evaluate_semantic_judgments(
    gold_records,
    test_taker_output,
    *,
    tool_info,
    research_config,
    judge_cache_path,
    evaluator_progress=None,
):
    """Evaluate independently, checkpointing each item and retrying failures.

    A provider failure is data about one judge call, not a reason to discard an
    otherwise valid iteration. Successful cached items are reused on resume;
    failed cached items are retried.
    """
    cached = read_json_records(judge_cache_path)
    cache_compatible = (
        len(cached) <= len(test_taker_output)
        and all(
            _semantic_judge_cache_record_matches(item, gold, predicted)
            for item, gold, predicted in zip(
                cached,
                gold_records,
                test_taker_output,
            )
        )
    )
    if not cache_compatible:
        cached = []
        for stale_path in (
            judge_cache_path,
            f"{os.path.splitext(judge_cache_path)[0]}.jsonl",
        ):
            if os.path.isfile(stale_path):
                os.remove(stale_path)

    judgments = list(cached)
    total = len(test_taker_output)
    pending_indices = []
    for index, (gold, predicted) in enumerate(
        zip(gold_records, test_taker_output)
    ):
        reusable = (
            index < len(judgments)
            and isinstance(judgments[index], dict)
            and isinstance(
                judgments[index].get("semantic_judge"),
                dict,
            )
            and judgments[index]["semantic_judge"].get("status")
            == "success"
        )
        if reusable:
            if evaluator_progress is not None:
                evaluator_progress.update(1)
            continue
        pending_indices.append(index)
        placeholder = {
            "question": gold["question"],
            "gold_answer": gold["answer"],
            "test_taker_answer": predicted["test_taker_response"],
            "is_correct": False,
            "confidence": 0.0,
            "reasons": "semantic judgment pending",
            "semantic_judge": {"status": "pending"},
        }
        if index < len(judgments):
            judgments[index] = placeholder
        else:
            judgments.append(placeholder)

    if not pending_indices:
        print(
            "[Evaluate] reused semantic judge cache "
            f"questions={total}",
            flush=True,
        )
        return judgments

    # Persist positional placeholders before parallel calls. If the process is
    # interrupted, completed indices remain reusable and unfinished indices
    # are retried without invalidating the whole cache.
    dump_standard_json(judgments, judge_cache_path)

    def evaluate_index(index):
        gold = gold_records[index]
        predicted = test_taker_output[index]
        try:
            semantic = judge_answer_semantics(
                question=gold["question"],
                gold_answer=gold["answer"],
                predicted_answer=predicted["test_taker_response"],
                answer_type=predicted.get("answer_type", "text"),
                evaluator_info=tool_info,
                config=research_config,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            semantic = _failed_semantic_judgment(
                gold=gold,
                predicted=predicted,
                research_config=research_config,
                error=exc,
            )
        return {
            "question": gold["question"],
            "gold_answer": gold["answer"],
            "test_taker_answer": predicted["test_taker_response"],
            "is_correct": semantic["semantically_equivalent"],
            "confidence": semantic["confidence"],
            "reasons": semantic["reason"],
            "semantic_judge": semantic,
        }

    client = tool_info[2] if tool_info and len(tool_info) > 2 else None
    thread_safe_client = bool(
        client is not None
        and hasattr(client, "chat")
        and hasattr(client.chat, "completions")
    )
    configured_workers = int(
        research_config["evaluator_pipeline"]["max_parallel_questions"]
    )
    worker_count = (
        min(configured_workers, len(pending_indices))
        if thread_safe_client
        else 1
    )
    print(
        "[Evaluate] semantic_judge_start "
        f"questions={len(pending_indices)} parallel_workers={worker_count}",
        flush=True,
    )

    def store(index, judgment):
        judgments[index] = judgment
        # Only the parent thread writes the ordered cache, so concurrent API
        # completion cannot corrupt or reorder the checkpoint file.
        dump_standard_json(judgments, judge_cache_path)
        semantic = judgment["semantic_judge"]
        if semantic.get("status") != "success":
            print(
                "[Evaluate] semantic_judge_failed "
                f"index={index + 1}/{total} "
                f"error={semantic['reason']} action=continue",
                flush=True,
            )
        if evaluator_progress is not None:
            evaluator_progress.update(1)

    if worker_count > 1:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="semantic-judge",
        ) as executor:
            futures = {
                executor.submit(evaluate_index, index): index
                for index in pending_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                store(index, future.result())
    else:
        for index in pending_indices:
            store(index, evaluate_index(index))
    return judgments


def _finite_confidence(value, default=0.0):
    """Coerce optional evaluator confidence without aborting an experiment."""
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return confidence if math.isfinite(confidence) else float(default)


# [MODIFIED] Persist only canonical inference details and comparison statistics.
def test_and_eval(
    question_json,
    outfile_prefix,
    test_taker_info,
    agent_info,
    tool_info,
    gold_ans_key='answer',
    iter_number=0,
    temp_log_dir=None,
    research_config=None,
    progress_manager=None,
    cycle_number=None,
    event_logger=None,
):
    inference_file = f"{outfile_prefix}.test_taker_inference.json"
    compare_file = f"{outfile_prefix}.compare_answers.json"
    cached_inference = load_math_inference(inference_file)
    cached_compare = read_json_records(compare_file)
    if (
        cached_inference
        and all(record["test_taker_response"] for record in cached_inference)
        and (
            not research_config
            or all(
                record.get("parse_status") == "success"
                and record.get("parser_version") == "structured_v2"
                and record.get("semantic_judge", {}).get("status")
                == "success"
                for record in cached_inference
            )
        )
        and len(cached_compare) == 1
        and "category_statistics" in cached_compare[0]
        and cached_compare[0].get("total_questions") == len(cached_inference)
    ):
        print("FOUND completed standardized iteration cache", compare_file)
        return cached_inference

    print(len(question_json), "number of questions.")
    if event_logger:
        event_logger.event(
            "INFO",
            "Infer",
            "stage_started",
            f"questions={len(question_json)}",
            cycle=cycle_number,
            iteration=iter_number,
        )
    test_taker_output = generate_math_inference(
        question_json,
        test_taker_info,
        inference_file,
        research_config=research_config,
        progress_manager=progress_manager,
        cycle=cycle_number,
        iteration=iter_number,
    )
    if event_logger:
        event_logger.event(
            "INFO",
            "Infer",
            "stage_completed",
            f"completed={len(test_taker_output)}",
            cycle=cycle_number,
            iteration=iter_number,
        )
    if len(question_json) != len(test_taker_output):
        raise RuntimeError(
            "Inference cache is incomplete after generation: "
            f"{len(test_taker_output)}/{len(question_json)} records. "
            "Rerun the same command to resume."
        )

    if gold_ans_key == 'python_answer':
        # Keep the original privileged Python/SymPy answer path unchanged.
        _, solved_questions = solve_with_python(
            question_json, outfile_prefix, agent_info
        )
        gold_records = [
            dict(record, answer=record.get("python_answer", record.get("answer", "")))
            for record in solved_questions
        ]
    else:
        gold_records = [
            {
                "id": record["id"],
                "question": record["question"],
                "answer": record["gold_answer"],
                "category": record["category"],
                "sub_category": record["sub_category"],
                "difficulty": record["difficulty"],
            }
            for record in test_taker_output
        ]

    temp_log_dir = temp_log_dir or os.path.join(
        os.path.dirname(os.path.abspath(outfile_prefix)), "temp_log"
    )
    os.makedirs(temp_log_dir, exist_ok=True)
    judge_prefix = os.path.join(temp_log_dir, "judge")
    judge_cache_path = f"{judge_prefix}.compare_answers.json"
    if event_logger:
        event_logger.event(
            "INFO",
            "Evaluate",
            "stage_started",
            f"questions={len(test_taker_output)}",
            cycle=cycle_number,
            iteration=iter_number,
        )
    evaluator_progress_context = (
        progress_manager.stage(
            "Evaluate",
            total=len(test_taker_output),
            cycle=cycle_number,
            iteration=iter_number,
        )
        if progress_manager
        else contextlib.nullcontext(None)
    )
    with evaluator_progress_context as evaluator_progress:
        original_tqdm = tqdm.tqdm

        def _tracked_evaluator_iterator(iterable, *args, **kwargs):
            del args, kwargs
            for item in iterable:
                yield item
                if evaluator_progress is not None:
                    evaluator_progress.update(1)

        if progress_manager:
            tqdm.tqdm = _tracked_evaluator_iterator
        try:
            evaluator_cache_exists = os.path.exists(judge_cache_path)
            if research_config:
                judgments = _evaluate_semantic_judgments(
                    gold_records,
                    test_taker_output,
                    tool_info=tool_info,
                    research_config=research_config,
                    judge_cache_path=judge_cache_path,
                    evaluator_progress=evaluator_progress,
                )
            else:
                judgments = [
                    {
                        "question": gold["question"],
                        "gold_answer": gold["answer"],
                        "test_taker_answer": predicted[
                            "test_taker_response"
                        ],
                        "is_correct": (
                            str(gold["answer"]).strip()
                            == str(
                                predicted["test_taker_response"]
                            ).strip()
                        ),
                        "confidence": 1.0,
                        "reasons": "legacy exact comparison",
                    }
                    for gold, predicted in zip(
                        gold_records,
                        test_taker_output,
                    )
                ]
                dump_standard_json(judgments, judge_cache_path)
                if evaluator_cache_exists and evaluator_progress is not None:
                    evaluator_progress.update(len(test_taker_output))
        finally:
            tqdm.tqdm = original_tqdm
    if event_logger:
        event_logger.event(
            "INFO",
            "Evaluate",
            "stage_completed",
            f"completed={len(judgments)}",
            cycle=cycle_number,
            iteration=iter_number,
        )
    if len(judgments) != len(test_taker_output):
        raise RuntimeError("Judgement count does not match inference count")

    standardized_records = []
    truth_solver = (
        TruthSolver.from_config(research_config)
        if research_config
        else None
    )
    for index, (record, judgment) in enumerate(
        zip(test_taker_output, judgments)
    ):
        standardized = canonicalize_math_record(record, index)
        # Parse failures do not enter semantic-judge fusion, so they cannot
        # have a deterministic-vs-judge conflict. Keep the per-record default
        # explicit to avoid carrying or reading an unassigned branch value.
        judge_conflict = False
        evaluator_confidence = _finite_confidence(
            judgment.get("confidence"),
            default=0.0,
        )
        evaluator_is_correct = str(
            judgment.get("is_correct", "")
        ).strip().lower() == "true"
        if research_config:
            parse_result = {
                "parse_status": standardized.get("parse_status", "parse_failed"),
                "parsed_response": standardized.get("parsed_response", {}),
                "contains_prompt_echo": standardized.get(
                    "contains_prompt_echo",
                    False,
                ),
                "contains_irrelevant_content": standardized.get(
                    "contains_irrelevant_content",
                    False,
                ),
                "tool_violation": standardized.get("tool_violation", False),
            }
            equivalence = answers_equivalent(
                standardized.get(
                    "canonical_answer",
                    standardized["gold_answer"],
                ),
                standardized["test_taker_response"],
                standardized.get("answer_type", "text"),
                research_config,
                tolerance=standardized.get("tolerance"),
            )
            candidate_truth_check = None
            if (
                truth_solver is not None
                and standardized.get("answer_type") == "ordered_tuple"
                and "=" in standardized.get("question", "")
                and standardized.get("test_taker_response")
            ):
                candidate_truth_check = truth_solver.validate_candidate(
                    standardized["question"],
                    standardized["test_taker_response"],
                )
                standardized["test_taker_truth_validation"] = (
                    candidate_truth_check
                )
                if candidate_truth_check.get("failure_type") == (
                    FailureType.PARTIAL_SOLUTION.value
                ):
                    standardized["failure_type"] = (
                        FailureType.PARTIAL_SOLUTION.value
                    )
            status = parse_result["parse_status"]
            if status != "success":
                evaluation_status = status
                standardized["is_correct"] = False
            else:
                semantic_judge = judgment.get("semantic_judge", {})
                semantic_judge_valid = (
                    semantic_judge.get("status") == "success"
                )
                semantic_threshold = float(
                    research_config["evaluator_pipeline"][
                        "semantic_judge_confidence_threshold"
                    ]
                )
                fusion = fuse_equivalence_with_semantic_judge(
                    equivalence,
                    judge_is_correct=evaluator_is_correct,
                    judge_valid=semantic_judge_valid,
                    judge_confidence=evaluator_confidence,
                    confidence_threshold=semantic_threshold,
                    require_semantic_judge=bool(
                        research_config["evaluator_pipeline"][
                            "require_semantic_judge"
                        ]
                    ),
                )
                standardized["is_correct"] = bool(fusion["is_correct"])
                evaluation_status = str(fusion["status"])
                judge_conflict = bool(fusion["judge_conflict"])
                if standardized["is_correct"] and bool(
                    semantic_judge.get("format_only_difference")
                ):
                    standardized["format_only_error"] = True
            attribution_equivalence = dict(equivalence)
            attribution_equivalence["equivalent"] = bool(
                standardized["is_correct"]
            )
            attribution = attribute_error(
                standardized,
                parse_result,
                attribution_equivalence,
                research_config,
            )
            evaluator_tool_calls = [
                {
                    "tool_name": "language_model",
                    "purpose": "isolated_semantic_answer_equivalence",
                    "privileged_side": "evaluator",
                }
            ]
            if standardized.get("answer_type") in {
                "symbolic_expression",
                "equation",
                "inequality",
            }:
                evaluator_tool_calls.append(
                    {
                        "tool_name": "sympy",
                        "purpose": "deterministic_symbolic_equivalence",
                        "privileged_side": "evaluator",
                    }
                )
            standardized.update(
                {
                    "evaluation_status": evaluation_status,
                    "normalized_gold_answer": equivalence["gold_normalized"],
                    "normalized_test_taker_answer": equivalence[
                        "predicted_normalized"
                    ],
                    "deterministic_checks": equivalence[
                        "deterministic_checks"
                    ],
                    "evaluator_tool_calls": evaluator_tool_calls,
                    "evaluator_confidence": evaluator_confidence,
                    "semantic_judge": judgment.get(
                        "semantic_judge",
                        {},
                    ),
                    "semantic_judge_reason": judgment.get("reasons", ""),
                    "judge_deterministic_agreement": (
                        bool(equivalence["equivalent"])
                        == evaluator_is_correct
                    ),
                    "equivalence_status": equivalence["status"],
                    "equivalence_needs_review": bool(
                        equivalence.get("needs_review", False)
                        or judge_conflict
                    ),
                    "equivalence_backend_disagreement": bool(
                        equivalence.get("disagreement", False)
                    ),
                    "equivalence_backend_results": equivalence.get(
                        "backend_results", {}
                    ),
                    "authoritative_equivalence_method": equivalence.get(
                        "authoritative_method"
                    ),
                    "answer_validation_success": bool(
                        equivalence["gold_normalized"]["success"]
                    ),
                    "question_parse_success": True,
                    "test_taker_truth_validation": candidate_truth_check,
                    **attribution,
                }
            )
            primary = attribution.get("primary_error_tag")
            standardized["error_tags"] = (
                [primary] if primary else []
            ) + list(attribution.get("secondary_error_tags", []))
        else:
            standardized["is_correct"] = evaluator_is_correct
            standardized["error_tags"] = (
                []
                if standardized["is_correct"]
                else classify_error_tags({**standardized, **judgment})
            )
        standardized_records.append(standardized)
    dump_standard_json(standardized_records, inference_file)

    compare_summary = _build_compare_summary(
        iter_number,
        standardized_records,
        research_config,
    )
    dump_standard_json(compare_summary, compare_file)
    return standardized_records


# [MODIFIED] Map the original prefix to eval or cycle-aware output layers.
def _build_iteration_paths(outfile_prefix1, iter_number, cycle_number=None):
    raw_prefix = os.fspath(outfile_prefix1)
    output_root = os.path.abspath(os.path.dirname(raw_prefix) or ".")
    raw_base_name = os.path.basename(raw_prefix)
    base_name = raw_base_name.rstrip(".") or "math"
    cycle_dir = (
        os.path.join(output_root, "cycle", f"cycle_{cycle_number}")
        if cycle_number is not None
        else output_root
    )
    iteration_dir = os.path.join(cycle_dir, f"iter_{iter_number}")
    os.makedirs(os.path.join(iteration_dir, "temp_log"), exist_ok=True)
    filename_suffix = (
        f"cycle{cycle_number}.iter{iter_number}"
        if cycle_number is not None
        else str(iter_number)
    )
    iteration_prefix = os.path.join(iteration_dir, f"{base_name}.{filename_suffix}")
    legacy_prefix = os.path.join(
        output_root, f"{raw_base_name}{iter_number}"
    )
    return {
        "output_root": output_root,
        "cycle_dir": cycle_dir,
        "iteration_dir": iteration_dir,
        "temp_log_dir": os.path.join(iteration_dir, "temp_log"),
        "iteration_prefix": iteration_prefix,
        "legacy_prefix": legacy_prefix,
        "plan_file": f"{iteration_prefix}.question_plan_with_aim.json",
        "generation_plan_file": os.path.join(
            iteration_dir,
            "generation_plan.json",
        ),
        "inference_file": f"{iteration_prefix}.test_taker_inference.json",
        "compare_file": f"{iteration_prefix}.compare_answers.json",
        "hard_pool_file": os.path.join(output_root, "hard_pool.json"),
        "meta_summary_file": os.path.join(output_root, "meta_summary.json"),
    }


def _records_from_legacy_cache(paths, iter_number):
    """Migrate flat legacy caches without repeating completed API work."""
    plan_file = paths["plan_file"]
    legacy_plan = f"{paths['legacy_prefix']}.question_plan_with_aim.json"
    if not os.path.exists(plan_file) and os.path.exists(legacy_plan):
        legacy_data = read_json_records(legacy_plan)
        try:
            normalized = _normalize_math_plan(
                [legacy_data] if legacy_data and isinstance(legacy_data[0], dict) else legacy_data
            )
            dump_standard_json(normalized[0], plan_file)
        except (ValueError, TypeError, IndexError):
            pass

    inference_file = paths["inference_file"]
    legacy_inference = f"{paths['legacy_prefix']}.test_taker_inference.json"
    legacy_compare = f"{paths['legacy_prefix']}.compare_answers.json"
    legacy_questions = f"{paths['legacy_prefix']}.all_questions.json"
    records = load_math_inference(inference_file)
    if not records and os.path.exists(legacy_inference):
        records = load_math_inference(legacy_inference)
    if not records and os.path.exists(legacy_questions):
        records = [
            canonicalize_math_record(record, index)
            for index, record in enumerate(read_json_records(legacy_questions))
            if isinstance(record, dict) and record.get("question")
        ]

    legacy_judgments = read_json_records(legacy_compare)
    if records and legacy_judgments and not (
        len(legacy_judgments) == 1
        and "category_statistics" in legacy_judgments[0]
    ):
        judgments_by_question = {
            str(item.get("question", "")).strip(): item
            for item in legacy_judgments
            if isinstance(item, dict)
        }
        for record in records:
            judgment = judgments_by_question.get(record["question"])
            if not judgment:
                continue
            record["test_taker_response"] = str(
                judgment.get(
                    "test_taker_answer",
                    judgment.get("test_taker_response", record["test_taker_response"]),
                )
            ).strip()
            record["is_correct"] = str(
                judgment.get("is_correct", "")
            ).strip().lower() == "true"
            record["error_tags"] = (
                []
                if record["is_correct"]
                else classify_error_tags({**record, **judgment})
            )
    if records:
        dump_standard_json(records, inference_file)

    compare_file = paths["compare_file"]
    if not os.path.exists(compare_file):
        if (
            legacy_judgments
            and len(legacy_judgments) == 1
            and "category_statistics" in legacy_judgments[0]
        ):
            dump_standard_json(legacy_judgments[0], compare_file)
        elif records and legacy_judgments:
            dump_standard_json(
                _build_compare_summary(iter_number, records), compare_file
            )
    return records


def _clean_legacy_redundant_files(paths):
    patterns = (
        f"{paths['legacy_prefix']}.all_questions.json",
        f"{paths['legacy_prefix']}.subcat*.questions.json",
        f"{paths['legacy_prefix']}.subcat*questions_final.json",
        f"{paths['legacy_prefix']}*.attempt*.txt",
        f"{paths['legacy_prefix']}.compare_answers.jsonl",
    )
    removed = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                os.remove(path)
                removed.append(path)
    for path in glob.glob(f"{paths['legacy_prefix']}*.json"):
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
    return removed


# [ADDED] Cleanup is idempotent and is invoked from an iteration finally block,
# so generation, inference, or evaluation failures cannot leave attempt caches.
def _cleanup_iteration_cache(args, cycle_number, iter_number):
    if not bool(getattr(args, "clean_cycle_cache", True)):
        return []
    cycle_layer = (
        cycle_number
        if getattr(args, "mode", "eval") == "data_flywheel"
        else None
    )
    paths = _build_iteration_paths(
        args.outfile_prefix1,
        iter_number,
        cycle_number=cycle_layer,
    )
    try:
        removed = clean_redundant_files(
            paths["iteration_dir"],
            preserve_json_paths=(
                paths["plan_file"],
                paths["generation_plan_file"],
                paths["compare_file"],
                paths["inference_file"],
            ),
            strict_json_allowlist=True,
        )
        if getattr(args, "mode", "eval") == "eval":
            removed.extend(_clean_legacy_redundant_files(paths))
    except OSError as exc:
        research_run = getattr(args, "research_run", None)
        if research_run:
            research_run.logger.event(
                "WARNING",
                "Cleanup",
                "iteration_cache_cleanup_failed",
                f"{type(exc).__name__}: {exc}",
                cycle=cycle_number,
                iteration=iter_number,
            )
        else:
            print(
                "[Cleanup] iteration_cache_cleanup_failed "
                f"error={type(exc).__name__}: {exc}"
            )
        return []
    research_run = getattr(args, "research_run", None)
    if research_run:
        research_run.logger.event(
            "INFO",
            "Cleanup",
            "iteration_cache_cleaned",
            f"removed={len(removed)}",
            cycle=cycle_number,
            iteration=iter_number,
        )
    elif removed:
        print(f"[Cleanup] removed_redundant_files={len(removed)}")
    return removed


def _relative_json_path(path, output_root):
    return os.path.relpath(path, output_root).replace("\\", "/")


def _performance_context(history_json_dict):
    records = [
        record
        for iteration_records in history_json_dict
        for record in iteration_records
    ]
    if not records:
        return "No measured sub_category accuracy is available yet."
    summary = _build_compare_summary(0, records)
    lowest = sorted(
        summary["category_statistics"],
        key=lambda item: (item["accuracy"], item["sub_category"]),
    )[:10]
    return "\n".join(
        f"- {item['category']} / {item['sub_category']}: "
        f"accuracy={item['accuracy']:.3f}, total={item['total_count']}"
        for item in lowest
    )


# [ADDED] Parse Boolean values accepted by the direct math entry point.
def _parse_bool(value):
    try:
        return str2bool(value)
    except ConfigurationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _utc_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _output_root(outfile_prefix):
    return os.path.abspath(os.path.dirname(os.fspath(outfile_prefix)) or ".")


def _bound_outfile_prefix(run_dir, configured_prefix):
    """Keep all legacy cycle state inside the run-id-owned directory."""
    configured = Path(str(configured_prefix).rstrip("."))
    prefix_name = configured.name or "output"
    return str(Path(run_dir) / f"{prefix_name}.")


def _load_test_taker_info(model_name, use_helm):
    if use_helm == "yes":
        print("[Model] loaded HELM test taker")
        return helm_process_args(model_name)
    # [ADDED] A tagged local identifier can use an OpenAI-compatible vLLM
    # endpoint when VLLM_BASE_URL is configured; otherwise util.py uses Ollama.
    drive, _ = os.path.splitdrive(str(model_name))
    if ":" in str(model_name) and not drive and os.getenv("VLLM_BASE_URL"):
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            base_url=os.environ["VLLM_BASE_URL"].rstrip("/"),
        )
        print(
            f"[Model] local_backend=vllm model={model_name} "
            f"base_url={os.environ['VLLM_BASE_URL']}"
        )
        return model_name, None, client
    model, tokenizer, _, client = process_args_for_models(model_name)
    return model, tokenizer, client


def _load_agent_and_evaluator(args):
    agent_lm, agent_tokenizer, agent_name, agent_client = (
        process_args_for_models(args.agent_modelname)
    )
    agent_info = (agent_lm, agent_tokenizer, agent_client)
    if args.tool_modelname is None:
        return agent_info, agent_info
    tool_lm, tool_tokenizer, _, tool_client = process_args_for_models(
        args.tool_modelname
    )
    return agent_info, (tool_lm, tool_tokenizer, tool_client)


# [ADDED] Release local GPU allocations before starting QLoRA.
def _release_model_info(model_info):
    if not model_info:
        return
    model = model_info[0] if len(model_info) > 0 else None
    service = model_info[2] if len(model_info) > 2 else None
    if service is not None and hasattr(service, "loaded_models"):
        for model_name in list(service.loaded_models):
            try:
                service.session.post(
                    f"{service.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "keep_alive": 0,
                        "stream": False,
                    },
                    timeout=30,
                )
            except requests.RequestException as exc:
                print(f"[Model] Ollama unload warning: {exc}")
        service.loaded_models.clear()
    if model is not None and hasattr(model, "cpu"):
        try:
            model.cpu()
        except (RuntimeError, AttributeError):
            pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_model_component(model_name):
    base_name = os.path.basename(os.path.normpath(str(model_name))) or "model"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base_name).strip("._") or "model"


def _complete_merged_model(path):
    model_path = Path(str(path)).expanduser()
    return (
        model_path.is_dir()
        and (model_path / "config.json").is_file()
        and bool(
            list(model_path.glob("*.safetensors"))
            or list(model_path.glob("*.bin"))
        )
    )


def _load_resumable_cycle_checkpoint(path, research_run, base_model):
    """Load a completed merged checkpoint only when its identity still matches."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        return None
    checkpoint = _load_cycle_record(str(checkpoint_path))
    merged_model_path = checkpoint.get("merged_model_path", "")
    if (
        checkpoint.get("run_id") != research_run.run_id
        or checkpoint.get("config_hash") != research_run.config_hash
        or checkpoint.get("base_model") != base_model
        or not _complete_merged_model(merged_model_path)
        or checkpoint.get("merged_model_sha256")
        != artifact_fingerprint(
            merged_model_path,
            allow_missing=False,
        )["sha256"]
    ):
        raise RuntimeError(
            "Existing checkpoint cannot be resumed because its identity or "
            "model directory is invalid"
        )
    return checkpoint


def _cycle_record_file(args):
    return os.path.join(_output_root(args.outfile_prefix1), "cycle_record.json")


def _load_cycle_record(path):
    records = read_json_records(path)
    if records and isinstance(records[0], dict):
        return records[0]
    return {}


def _save_cycle_record(path, record):
    record["updated_at"] = _utc_timestamp()
    dump_standard_json(record, path)


def _failure_type(stage, exc):
    if isinstance(exc, KeyboardInterrupt):
        return "interrupted"
    detail = str(exc).lower()
    if "out of memory" in detail or "cuda oom" in detail:
        return "finetune_oom"
    if "disk" in detail or "free space" in detail:
        return "disk_space"
    if "timeout" in detail or "timed out" in detail:
        return "api_timeout"
    if "model" in stage:
        return "model_loading"
    if "json" in detail:
        return "json_parsing"
    return "runtime_error"


def _sanitize_error(exc):
    message = str(exc).strip()
    if not message:
        message = repr(exc).strip() or type(exc).__name__
    message = re.sub(r"[\u3400-\u9fff]+", " ", message)
    return re.sub(r"\s+", " ", message).strip()[:4000]


def _sanitize_traceback(exc):
    """Preserve an actionable traceback even when an exception has no message."""
    formatted = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    formatted = re.sub(r"[\u3400-\u9fff]+", " ", formatted)
    return formatted.strip()[-12000:]


def _study_runtime_metadata(config, question_budget=None):
    """Return the effective strategy state used by manifests and logs."""
    metadata = dict(policy_runtime_descriptor(config))
    metadata["component_state"] = dict(metadata["component_state"])
    metadata["seed"] = int(config["experiment"]["seed"])
    metadata["question_budget"] = int(
        config["experiment"]["questions_per_iteration"]
        if question_budget is None
        else question_budget
    )
    metadata["budget_protocol"] = str(config["budget"]["protocol"])
    return metadata


def _save_research_cycle_manifest(args, cycle_entry):
    research_run = getattr(args, "research_run", None)
    if research_run is None:
        return
    atomic_json(
        {
            **research_run.metadata(),
            "cycle_id": cycle_entry["cycle"],
            **cycle_entry,
        },
        research_run.cycle_root
        / f"cycle_{cycle_entry['cycle']}"
        / "cycle_manifest.json",
    )


def _load_completed_cycle_history(output_root, cycles):
    history = []
    root = Path(output_root)
    for cycle in sorted(cycles, key=lambda item: int(item.get("cycle", 0))):
        # Completed iterations in an interrupted/failed cycle are durable and
        # must be restored before resuming that same model version.
        cycle_number = int(cycle.get("cycle", 0))
        for iteration_number in range(
            1,
            int(cycle.get("iterations_completed", 0)) + 1,
        ):
            iteration_dir = (
                root
                / "cycle"
                / f"cycle_{cycle_number}"
                / f"iter_{iteration_number}"
            )
            candidates = sorted(
                iteration_dir.glob("*.test_taker_inference.json")
            )
            if candidates:
                records = load_math_inference(candidates[0])
                if records:
                    history.append(records)
    return history


def _upsert_history_iteration(
    history,
    records,
    cycle_number,
    iteration_number,
):
    for index, existing in enumerate(history):
        if not existing:
            continue
        first = existing[0]
        if (
            int(first.get("cycle_id", first.get("source_cycle", -1)))
            == int(cycle_number)
            and int(first.get("iteration_id", first.get("source_iter", -1)))
            == int(iteration_number)
        ):
            history[index] = records
            return
    history.append(records)


# [ADDED] Run one adaptive iteration and update all global governance files.
def _run_math_iteration(
    args,
    current_test_taker_model,
    test_taker_info,
    agent_info,
    evaluator_info,
    history_dict,
    cycle_number,
    iter_number,
):
    started_at = utc_now()
    research_run = getattr(args, "research_run", None)
    research_config = (
        research_run.config if research_run is not None else None
    )
    global_iter_number = (cycle_number - 1) * args.num_iters + iter_number
    cycle_layer = cycle_number if args.mode == "data_flywheel" else None
    paths = _build_iteration_paths(
        args.outfile_prefix1,
        iter_number,
        cycle_number=cycle_layer,
    )
    args.outfile_prefix = paths["iteration_prefix"]
    migrated_records = (
        _records_from_legacy_cache(paths, iter_number)
        if args.mode == "eval"
        else load_math_inference(paths["inference_file"])
    )
    # [ADDED] Never reuse a legacy cache whose gold did not originate from
    # TruthSolver. This prevents pre-migration LLM-authored gold from entering
    # evaluation, the hard pool, or training exports.
    untrusted_truth_cache = [
        record.get("question_id", record.get("id", "unknown"))
        for record in migrated_records
        if not record.get("truth_validation_details", {}).get(
            "source_question_sha256"
        )
    ]
    if untrusted_truth_cache:
        if research_run:
            research_run.logger.event(
                "WARNING",
                "TruthPipeline",
                "legacy_gold_cache_invalidated",
                (
                    "failure_type=truth_parse_fail "
                    f"questions={len(untrusted_truth_cache)}"
                ),
                cycle=cycle_number,
                iteration=iter_number,
                metrics={
                    "failure_type": (
                        FailureType.TRUTH_PARSE_FAIL.value
                    ),
                    "question_ids": untrusted_truth_cache[:20],
                },
            )
        migrated_records = []
    # [MODIFIED] Never resume an iteration whose cached question text contains
    # non-English or encoding-corrupted symbols. Regeneration is safer than
    # evaluating or exporting an ambiguous mathematical expression.
    corrupted_cached_questions = [
        record.get("question_id", record.get("id", "unknown"))
        for record in migrated_records
        if re.search(
            r"[\u3400-\u9fff\ufffd]",
            str(record.get("question", "")),
        )
    ]
    if corrupted_cached_questions:
        if research_run:
            research_run.logger.event(
                "WARNING",
                "Generate",
                "corrupted_question_cache_invalidated",
                f"questions={len(corrupted_cached_questions)}",
                cycle=cycle_number,
                iteration=iter_number,
                metrics={
                    "question_ids": corrupted_cached_questions[:20],
                },
            )
        migrated_records = []
    if research_run and migrated_records:
        cached_hashes = {
            str(record.get("config_hash"))
            for record in migrated_records
            if record.get("config_hash")
        }
        if cached_hashes and cached_hashes != {research_run.config_hash}:
            raise ConfigurationError(
                "cache.config_hash",
                "cached iteration was produced by a different resolved config",
                sorted(cached_hashes),
            )
    (
        triggered,
        flywheel_triggers,
        prior_coverage,
        covered_sub_categories,
        missing_sub_categories,
    ) = _get_math_flywheel_state(history_dict)
    hard_pool = HardSamplePool(paths["hard_pool_file"])
    has_train_eligible = any(
        sample.get("sample_grade") == "train_eligible"
        for sample in hard_pool.samples
    )
    generation_plan = None
    if research_config:
        prior_records = [
            record
            for iteration_records in history_dict
            for record in iteration_records
        ]
        generation_plan = generation_schedule(
            prior_records,
            research_config,
            global_iter_number,
            sum(
                sample.get("sample_grade") == "train_eligible"
                for sample in hard_pool.samples
            ),
            hard_pool_records=hard_pool.samples,
            previous_round_records=(
                history_dict[-1]
                if iter_number > 1 and history_dict
                else []
            ),
            cycle=cycle_number,
            seed=int(research_config["experiment"]["seed"]),
            question_budget=int(
                research_config["experiment"]["questions_per_iteration"]
            ),
        )
        should_direct_generation = bool(
            generation_plan["hard_pool_injection_enabled"]
        )
        atomic_json(
            {
                **research_run.metadata(),
                **generation_plan,
            },
            paths["generation_plan_file"],
        )
        research_run.logger.event(
            "INFO",
            "Generate",
            "generation_plan_created",
            (
                f"questions={generation_plan['question_budget']} "
                f"quota_status={generation_plan['cumulative_quota_status']} "
                "coverage_target=cumulative"
            ),
            cycle=cycle_number,
            iteration=iter_number,
            metrics={
                "source_budget": generation_plan["source_budget"],
                "quota_feasible": generation_plan["quota_feasible"],
                "cumulative_quota_status": generation_plan[
                    "cumulative_quota_status"
                ],
                "remaining_questions_for_full_quota": generation_plan[
                    "remaining_questions_for_full_quota_before_iteration"
                ],
                "previous_round_accuracy_state": generation_plan[
                    "previous_round_accuracy_state"
                ],
                "policy_name": generation_plan["policy_name"],
                "policy_version": generation_plan["policy_version"],
                "variant": generation_plan["variant"],
                "component_state": generation_plan["component_state"],
                "seed": generation_plan["seed"],
                "question_budget": generation_plan["question_budget"],
            },
        )
        if generation_plan.get("diagnostics", {}).get(
            "hard_pool_disabled_by_ablation"
        ):
            research_run.logger.event(
                "INFO",
                "Generate",
                "hard_pool_disabled_by_ablation",
                (
                    "hard_pool_disabled_by_ablation "
                    f"variant={generation_plan['variant']}"
                ),
                cycle=cycle_number,
                iteration=iter_number,
                metrics={
                    "source_budget": generation_plan["source_budget"],
                    "component_state": generation_plan["component_state"],
                },
            )
    else:
        should_direct_generation = (
            (iter_number >= 3 and triggered)
            or (args.mode == "data_flywheel" and has_train_eligible)
        )
    if has_train_eligible and "train_eligible_hard_pool" not in flywheel_triggers:
        flywheel_triggers.append("train_eligible_hard_pool")
    coverage_summary = (
        f"taxonomy_coverage={prior_coverage:.1%}\n"
        f"covered_sub_categories={covered_sub_categories}\n"
        f"missing_sub_categories={missing_sub_categories}\n"
        "lowest_accuracy_sub_categories:\n"
        f"{_performance_context(history_dict)}"
    )
    cached_compare = read_json_records(paths["compare_file"])
    has_standard_compare = (
        len(cached_compare) == 1
        and isinstance(cached_compare[0], dict)
        and "category_statistics" in cached_compare[0]
        and cached_compare[0].get("total_questions") == len(migrated_records)
    )
    if (
        migrated_records
        and all(record["test_taker_response"] for record in migrated_records)
        and (
            not research_config
            or all(
                record.get("parse_status") == "success"
                and record.get("parser_version") == "structured_v2"
                and record.get("semantic_judge", {}).get("status")
                == "success"
                for record in migrated_records
            )
        )
        and has_standard_compare
        and os.path.exists(paths["inference_file"])
    ):
        print(f"[Cache] completed iteration: {paths['compare_file']}")
        json_dict = load_math_inference(paths["inference_file"])
        compare_summary = _build_compare_summary(
            iter_number,
            json_dict,
            research_config,
        )
        dump_standard_json(compare_summary, paths["compare_file"])
    else:
        if migrated_records:
            json_category = migrated_records
        else:
            summarized_content = summarize_over_history(
                history_dict,
                gold_key="gold_answer",
                verbose=False,
            )
            history = [
                summarized_content,
                _performance_context(history_dict),
            ]
            json_category, _ = _ask_question_v3(
                agent_info,
                history,
                iter_number,
                outfile_prefix=args.outfile_prefix,
                aim_acc=args.acc_target,
                enable_hard_sample_guidance=should_direct_generation,
                coverage_summary=coverage_summary,
                hard_pool_file=paths["hard_pool_file"],
                generation_plan=generation_plan,
                progress_manager=(
                    research_run.progress if research_run else None
                ),
                cycle_number=cycle_number,
                research_config=research_config,
            )
            if research_run:
                atomic_json(
                    {
                        **research_run.metadata(),
                        **generation_plan,
                    },
                    paths["generation_plan_file"],
                )
                generation_result = (
                    generation_plan.get("generation_result", {})
                    if generation_plan
                    else {}
                )
                for subcategory_stat in generation_result.get(
                    "subcategory_statistics",
                    [],
                ):
                    for failure_type, failure_count in (
                        subcategory_stat.get(
                            "failure_counts",
                            {},
                        ).items()
                    ):
                        research_run.logger.event(
                            "WARNING",
                            "TruthPipeline",
                            "sample_generation_failure",
                            (
                                f"failure_type={failure_type} "
                                f"count={failure_count} "
                                "sub_category="
                                f"{subcategory_stat.get('sub_category')!r}"
                            ),
                            cycle=cycle_number,
                            iteration=iter_number,
                            metrics={
                                "failure_type": failure_type,
                                "failure_count": failure_count,
                                "category": subcategory_stat.get(
                                    "category"
                                ),
                                "sub_category": subcategory_stat.get(
                                    "sub_category"
                                ),
                            },
                        )
                if generation_result.get("partial_iteration"):
                    research_run.logger.event(
                        "WARNING",
                        "Generate",
                        "verified_question_shortfall",
                        (
                            f"verified={len(json_category)}/"
                            f"{generation_result.get('requested_questions')}"
                        ),
                        cycle=cycle_number,
                        iteration=iter_number,
                        metrics=generation_result,
                    )
                research_run.logger.event(
                    "INFO",
                    "Generate",
                    "stage_completed",
                    (
                        f"completed={len(json_category)} "
                        "gold_verified=true"
                    ),
                    cycle=cycle_number,
                    iteration=iter_number,
                )
        if json_category:
            json_dict = test_and_eval(
                copy.deepcopy(json_category),
                args.outfile_prefix,
                test_taker_info,
                agent_info,
                evaluator_info,
                gold_ans_key="answer",
                iter_number=iter_number,
                temp_log_dir=paths["temp_log_dir"],
                research_config=research_config,
                progress_manager=(
                    research_run.progress if research_run else None
                ),
                cycle_number=cycle_number,
                event_logger=(
                    research_run.logger if research_run else None
                ),
            )
        else:
            # [ADDED] All single-sample failures are isolated. An iteration
            # with zero surviving truths is recorded, not raised as a Cycle
            # runtime error.
            json_dict = []
            dump_standard_json([], paths["inference_file"])
            if research_run:
                research_run.logger.event(
                    "WARNING",
                    "Generate",
                    "no_verified_questions",
                    "completed=0 action=continue_cycle",
                    cycle=cycle_number,
                    iteration=iter_number,
                    metrics=(
                        generation_plan.get("generation_result", {})
                        if generation_plan
                        else {}
                    ),
                )
        compare_summary = _build_compare_summary(
            iter_number,
            json_dict,
            research_config,
        )
        dump_standard_json(compare_summary, paths["compare_file"])

    if research_run:
        iteration_uid = (
            f"{research_run.run_id}:cycle_{cycle_number}:iter_{iter_number}"
        )
        for record in json_dict:
            record.update(
                {
                    "run_id": research_run.run_id,
                    "cycle_id": cycle_number,
                    "iteration_id": iter_number,
                    "iteration_uid": iteration_uid,
                    "config_hash": research_run.config_hash,
                    "prompt_hash": research_run.prompt_bundle[
                        "combined_sha256"
                    ],
                    "cache_status": (
                        "hit" if migrated_records else "generated"
                    ),
                }
            )
        dump_standard_json(json_dict, paths["inference_file"])
    _upsert_history_iteration(
        history_dict,
        json_dict,
        cycle_number,
        iter_number,
    )
    new_hard_count, hard_total = manage_hard_pool(
        paths["inference_file"],
        paths["hard_pool_file"],
        iter_number,
        source_cycle=cycle_number,
    )
    run_config = {
        "mode": args.mode,
        "agent_model": args.agent_modelname,
        "test_taker_model": current_test_taker_model,
        "tool_model": args.tool_modelname or args.agent_modelname,
        "target_accuracy_range": args.acc_target,
        "iterations_per_cycle": args.num_iters,
        "max_cycle": args.max_cycle if args.mode == "data_flywheel" else 1,
        "output_root": paths["output_root"].replace("\\", "/"),
    }
    update_meta_summary(
        paths["meta_summary_file"],
        run_config,
        [record["sub_category"] for record in json_dict],
        {
            "cycle_num": cycle_number,
            "iter_num": iter_number,
            "global_iter_num": global_iter_number,
            "plan_file_path": _relative_json_path(
                paths["plan_file"], paths["output_root"]
            ),
            "infer_file_path": _relative_json_path(
                paths["inference_file"], paths["output_root"]
            ),
            "stat_file_path": _relative_json_path(
                paths["compare_file"], paths["output_root"]
            ),
            "new_hard_sample_count": new_hard_count,
        },
    )
    if research_run:
        hard_pool_snapshot = read_json_records(paths["hard_pool_file"])
        coverage_payload = _build_compare_summary(
            iter_number,
            json_dict,
            research_config,
        )
        normalized_answers = [
            {
                "question_id": record.get("question_id"),
                "gold": record.get("normalized_gold_answer"),
                "predicted": record.get("normalized_test_taker_answer"),
                "status": record.get("evaluation_status"),
            }
            for record in json_dict
        ]
        evaluation_results = [
            {
                "question_id": record.get("question_id"),
                "status": record.get("evaluation_status"),
                "is_correct": record.get("is_correct"),
                "deterministic_checks": record.get(
                    "deterministic_checks",
                    {},
                ),
                "evaluator_tool_calls": record.get(
                    "evaluator_tool_calls",
                    [],
                ),
                "failure_type": record.get("failure_type"),
                "truth_validation_details": record.get(
                    "truth_validation_details",
                    {},
                ),
            }
            for record in json_dict
        ]
        error_attributions = [
            {
                "question_id": record.get("question_id"),
                "is_correct": record.get("is_correct"),
                "primary_error_tag": record.get("primary_error_tag"),
                "secondary_error_tags": record.get(
                    "secondary_error_tags",
                    [],
                ),
                "evidence": record.get("evidence", []),
                "attribution_confidence": record.get(
                    "attribution_confidence",
                    1.0,
                ),
                "needs_review": record.get("needs_review", False),
                "attribution_method": record.get("attribution_method"),
                "verification_tier": record.get("verification_tier"),
                "first_error_step": record.get("first_error_step"),
                "taxonomy_version": record.get("taxonomy_version"),
                "deterministic_checks": record.get(
                    "deterministic_checks",
                    {},
                ),
            }
            for record in json_dict
        ]
        iteration_summary = {
            "iteration_uid": (
                f"{research_run.run_id}:cycle_{cycle_number}:iter_{iter_number}"
            ),
            "started_at": started_at,
            "completed_at": utc_now(),
            "question_count": len(json_dict),
            "global_accuracy": compare_summary["global_accuracy"],
            "hard_pool_size_before": (
                hard_total - new_hard_count
            ),
            "hard_pool_size_after": hard_total,
            "hard_pool_injection_enabled": bool(
                generation_plan
                and generation_plan["hard_pool_injection_enabled"]
            ),
            "hard_pool_reference_count": (
                generation_plan["hard_pool_reference_count"]
                if generation_plan
                else 0
            ),
            "directed_generation_question_count": (
                generation_plan["directed_generation_question_count"]
                if generation_plan
                else 0
            ),
            "coverage_repair_question_count": (
                generation_plan["coverage_repair_question_count"]
                if generation_plan
                else 0
            ),
            "retention_question_count": (
                generation_plan["retention_question_count"]
                if generation_plan
                else 0
            ),
            "cache_status": "hit" if migrated_records else "generated",
            **(
                {
                    key: generation_plan[key]
                    for key in (
                        "policy_name",
                        "policy_version",
                        "variant",
                        "component_state",
                        "seed",
                        "question_budget",
                    )
                }
                if generation_plan
                else _study_runtime_metadata(research_config)
            ),
        }
        attribution_config = research_config["error_attribution"]
        if bool(attribution_config.get("export_review_csv", False)):
            review_path = (
                research_run.iteration_dir(cycle_number, iter_number)
                / "error_attribution_review.csv"
            )
            review_count = export_review_sample(
                json_dict,
                review_path,
                sample_size=int(
                    attribution_config.get("review_sample_size", 100)
                ),
                seed=(
                    int(research_config["experiment"].get("seed", 42))
                    + global_iter_number
                ),
            )
            iteration_summary.update(
                {
                    "attribution_review_sample_count": review_count,
                    "attribution_review_path": _relative_json_path(
                        str(review_path),
                        paths["output_root"],
                    ),
                }
            )
        if not bool(args.clean_cycle_cache):
            research_run.export_iteration(
                cycle_number,
                iter_number,
                {
                "generation_plan": generation_plan or {},
                "generated_questions": [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {
                            "raw_response",
                            "parsed_response",
                            "test_taker_response",
                        }
                    }
                    for record in json_dict
                ],
                "test_taker_outputs": [
                    {
                        "question_id": record.get("question_id"),
                        "raw_response": record.get("raw_response"),
                        "parsed_response": record.get("parsed_response"),
                        "parse_status": record.get("parse_status"),
                        "tool_violation": record.get("tool_violation", False),
                    }
                    for record in json_dict
                ],
                "normalized_answers": normalized_answers,
                "evaluation_results": evaluation_results,
                "error_attributions": error_attributions,
                "coverage_metrics": coverage_payload,
                "adaptive_sampler_state": (
                    generation_plan["adaptive_sampler_state"]
                    if generation_plan
                    else []
                ),
                "hard_pool_snapshot": hard_pool_snapshot,
                "iteration_summary": iteration_summary,
                },
            )
        research_run.logger.event(
            "INFO",
            "Iteration",
            "iteration_completed",
            f"accuracy={compare_summary['global_accuracy']:.4f}",
            cycle=cycle_number,
            iteration=iter_number,
            metrics=compare_summary,
        )
    cumulative_records = [
        record
        for iteration_records in history_dict
        for record in iteration_records
    ]
    current_coverage, _, _ = HardSamplePool.coverage(cumulative_records)
    log_math_iteration_metrics(
        global_iter_number,
        compare_summary["global_accuracy"],
        current_coverage,
        hard_total,
        should_direct_generation,
        flywheel_triggers,
        _lowest_sub_categories(compare_summary),
    )
    if json_dict:
        print(
            get_summary_of_results(
                json_dict,
                gold_key="gold_answer",
                verbose=False,
            )
        )
    else:
        print("[MathFlywheel] verified_questions=0 action=continue_cycle")
    return {
        "paths": paths,
        "global_iter_number": global_iter_number,
        "global_accuracy": compare_summary["global_accuracy"],
        "hard_pool_total": hard_total,
        "generation_result": (
            generation_plan.get("generation_result", {})
            if generation_plan
            else {}
        ),
    }


# [ADDED] Aggregate generator and TruthSolver health at Cycle granularity.
def _aggregate_cycle_generation_statistics(iteration_results):
    grouped = {}
    for result in iteration_results:
        for item in result.get(
            "subcategory_statistics",
            [],
        ):
            key = (
                str(item.get("category", "")),
                str(item.get("sub_category", "")),
            )
            aggregate = grouped.setdefault(
                key,
                {
                    "category": key[0],
                    "sub_category": key[1],
                    "generated_total": 0,
                    "valid_samples": 0,
                    "failure_counts": {},
                    "coverage_gap": 0,
                },
            )
            aggregate["generated_total"] += int(
                item.get("generated_total", 0)
            )
            aggregate["valid_samples"] += int(
                item.get("valid_samples", 0)
            )
            aggregate["coverage_gap"] += int(
                item.get("coverage_gap", 0)
            )
            for failure_type, count in item.get(
                "failure_counts",
                {},
            ).items():
                aggregate["failure_counts"][failure_type] = (
                    aggregate["failure_counts"].get(failure_type, 0)
                    + int(count)
                )
    categories = sorted(
        grouped.values(),
        key=lambda item: (
            item["category"],
            item["sub_category"],
        ),
    )
    total_generated = sum(
        item["generated_total"] for item in categories
    )
    total_valid = sum(item["valid_samples"] for item in categories)
    total_failures = defaultdict(int)
    for item in categories:
        for failure_type, count in item["failure_counts"].items():
            total_failures[failure_type] += count
    return {
        "generated_total": total_generated,
        "valid_samples": total_valid,
        "valid_rate": (
            total_valid / total_generated if total_generated else 0.0
        ),
        "failure_counts": dict(sorted(total_failures.items())),
        "coverage_gap": sum(
            item["coverage_gap"] for item in categories
        ),
        "subcategory_statistics": categories,
    }


def _load_fixed_test_benchmark_cache(
    research_run,
    *,
    model_name,
    fixed_questions,
    fixed_metadata,
    stage_name,
    cycle_number=None,
    suite_name="fixed_test",
):
    stage_dir = research_run.run_dir / str(suite_name) / stage_name
    inference_path = stage_dir / "fixed_math.test_taker_inference.json"
    comparison_path = stage_dir / "fixed_math.compare_answers.json"
    summary_path = stage_dir / "summary.json"
    if not (
        inference_path.is_file()
        and comparison_path.is_file()
    ):
        return None
    # Read the durable standardized records directly.  `load_math_inference`
    # intentionally drops unanswered rows, which is useful during generation
    # but would make a completed parse-failure row look like a missing cache.
    inference_records = read_json_records(inference_path)
    normalized_model_name = str(model_name).replace("\\", "/")
    cached_comparison = read_json_records(comparison_path)
    comparison_summary = (
        cached_comparison[0]
        if len(cached_comparison) == 1
        and isinstance(cached_comparison[0], dict)
        else {}
    )
    if not (
        len(inference_records) == len(fixed_questions)
        and comparison_summary.get("total_questions") == len(fixed_questions)
        and "category_statistics" in comparison_summary
    ):
        return None
    expected_ids = [
        str(item.get("question_id", item.get("id", "")))
        for item in fixed_questions
    ]
    observed_ids = [
        str(item.get("question_id", item.get("id", "")))
        for item in inference_records
    ]
    if all(expected_ids) and expected_ids != observed_ids:
        return None

    cached_summary = (
        _load_cycle_record(str(summary_path))
        if summary_path.is_file()
        else {}
    )
    cached_outcomes = cached_summary.get("item_outcomes", [])
    summary_complete = bool(
        cached_summary.get("dataset_sha256") == fixed_metadata.get("sha256")
        and cached_summary.get("model_name") == normalized_model_name
        and int(cached_summary.get("total_questions", 0))
        == len(fixed_questions)
        and isinstance(cached_outcomes, list)
        and len(cached_outcomes) == len(fixed_questions)
        and _optional_accuracy_delta(
            0.0,
            cached_summary.get("accuracy"),
        )
        is not None
    )
    if not summary_complete:
        cached_summary = fixed_benchmark_summary(
            inference_records,
            stage=stage_name,
            model_name=model_name,
            dataset_sha256=fixed_metadata["sha256"],
        )
        cached_summary["answer_artifacts"] = {
            "test_taker_inference": _relative_json_path(
                inference_path,
                research_run.run_dir,
            ),
            "answer_comparison": _relative_json_path(
                comparison_path,
                research_run.run_dir,
            ),
            "contains_raw_response": True,
            "contains_parsed_reasoning_summary": True,
            "contains_normalized_answers": True,
            "contains_semantic_judgment": True,
        }
        atomic_json(
            {**research_run.metadata(), **cached_summary},
            summary_path,
        )
        research_run.logger.event(
            "INFO",
            "FixedTest",
            "stage_summary_rebuilt",
            f"stage={stage_name} questions={len(fixed_questions)}",
            cycle=cycle_number,
        )
    research_run.logger.event(
        "INFO",
        "FixedTest",
        "stage_resume_hit",
        f"stage={stage_name} questions={len(fixed_questions)}",
        cycle=cycle_number,
    )
    return cached_summary


def _run_fixed_test_benchmark(
    args,
    *,
    model_name,
    test_taker_info,
    agent_info,
    evaluator_info,
    fixed_questions,
    fixed_metadata,
    stage_name,
    cycle_number=None,
    suite_name="fixed_test",
):
    """Evaluate one model against the immutable holdout without training it."""
    research_run = getattr(args, "research_run", None)
    if research_run is None:
        raise RuntimeError("Fixed benchmark requires a ResearchRun")
    stage_dir = research_run.run_dir / str(suite_name) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    inference_path = stage_dir / "fixed_math.test_taker_inference.json"
    comparison_path = stage_dir / "fixed_math.compare_answers.json"
    summary_path = stage_dir / "summary.json"

    # A cycle can fail after inference/evaluation has completed but before the
    # caller records the derived baseline delta.  On resume, treat the three
    # complete fixed-test artifacts as an atomic cache: reusing them avoids a
    # second model pass and, more importantly, a second semantic-judge pass.
    cached_summary = _load_fixed_test_benchmark_cache(
        research_run,
        model_name=model_name,
        fixed_questions=fixed_questions,
        fixed_metadata=fixed_metadata,
        stage_name=stage_name,
        cycle_number=cycle_number,
        suite_name=suite_name,
    )
    if cached_summary is not None:
        return cached_summary
    records = test_and_eval(
        copy.deepcopy(fixed_questions),
        str(stage_dir / "fixed_math"),
        test_taker_info,
        agent_info,
        evaluator_info,
        gold_ans_key="answer",
        iter_number=0 if cycle_number is None else int(cycle_number),
        temp_log_dir=str(stage_dir / "temp_log"),
        research_config=research_run.config,
        progress_manager=research_run.progress,
        cycle_number=cycle_number,
        event_logger=research_run.logger,
    )
    summary = fixed_benchmark_summary(
        records,
        stage=stage_name,
        model_name=model_name,
        dataset_sha256=fixed_metadata["sha256"],
    )
    summary["answer_artifacts"] = {
        "test_taker_inference": _relative_json_path(
            inference_path,
            research_run.run_dir,
        ),
        "answer_comparison": _relative_json_path(
            comparison_path,
            research_run.run_dir,
        ),
        "contains_raw_response": True,
        "contains_parsed_reasoning_summary": True,
        "contains_normalized_answers": True,
        "contains_semantic_judgment": True,
    }
    atomic_json(
        {**research_run.metadata(), **summary},
        summary_path,
    )
    clean_redundant_files(
        str(stage_dir),
        preserve_json_paths=(
            inference_path,
            comparison_path,
            summary_path,
        ),
        strict_json_allowlist=True,
    )
    return summary


def _optional_accuracy_delta(baseline, current):
    """Return an accuracy delta only when both endpoints are numeric."""
    if baseline is None or current is None:
        return None
    try:
        baseline_value = float(baseline)
        current_value = float(current)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(baseline_value) or not math.isfinite(current_value):
        return None
    return current_value - baseline_value


def _retention_delta(baseline, current):
    def wilson_interval(successes, total, z=1.959963984540054):
        if total <= 0:
            return {"lower": None, "upper": None, "method": "wilson_95"}
        proportion = successes / total
        denominator = 1 + z * z / total
        center = (proportion + z * z / (2 * total)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1 - proportion) / total
                + z * z / (4 * total * total)
            )
            / denominator
        )
        return {
            "lower": max(0.0, center - radius),
            "upper": min(1.0, center + radius),
            "method": "wilson_95",
        }
    baseline_items = {
        item["question_id"]: bool(item["is_correct"])
        for item in (baseline or {}).get("item_outcomes", [])
    }
    current_items = {
        item["question_id"]: item
        for item in (current or {}).get("item_outcomes", [])
    }
    baseline_correct = {
        identifier
        for identifier, is_correct in baseline_items.items()
        if is_correct
    }
    forgotten = sorted(
        identifier
        for identifier in baseline_correct
        if identifier in current_items
        and not bool(current_items[identifier]["is_correct"])
    )
    by_dimension = defaultdict(lambda: {"baseline_correct": 0, "forgotten": 0})
    for identifier in baseline_correct:
        item = current_items.get(identifier, {})
        dimension = str(item.get("retention_dimension") or "unknown")
        by_dimension[dimension]["baseline_correct"] += 1
        by_dimension[dimension]["forgotten"] += int(identifier in forgotten)
    return {
        "baseline_accuracy": (baseline or {}).get("accuracy"),
        "current_accuracy": (current or {}).get("accuracy"),
        "accuracy_delta": _optional_accuracy_delta(
            (baseline or {}).get("accuracy"),
            (current or {}).get("accuracy"),
        ),
        "baseline_correct_count": len(baseline_correct),
        "forgotten_count": len(forgotten),
        "forgetting_rate": (
            len(forgotten) / len(baseline_correct)
            if baseline_correct
            else 0.0
        ),
        "forgetting_rate_ci95": wilson_interval(
            len(forgotten), len(baseline_correct)
        ),
        "forgotten_question_ids": forgotten,
        "dimension_statistics": {
            name: {
                **counts,
                "forgetting_rate": (
                    counts["forgotten"] / counts["baseline_correct"]
                    if counts["baseline_correct"]
                    else 0.0
                ),
                "forgetting_rate_ci95": wilson_interval(
                    counts["forgotten"], counts["baseline_correct"]
                ),
            }
            for name, counts in sorted(by_dimension.items())
        },
    }


def _evaluation_leakage_holdouts(config, project_root, fixed, retention):
    """Collect every locally visible non-blind evaluation question."""
    records = [*fixed, *retention]
    sources = {
        "active_fixed": len(fixed),
        "retention": len(retention),
    }
    evaluation = config.get("evaluation_sets", {})
    registry_path = evaluation.get("registry_path")
    if registry_path:
        registry = load_evaluation_registry(registry_path, project_root)
        for identifier, spec in registry["sets"].items():
            if spec.get("role") == "blind_final" or not spec.get("dataset_path"):
                continue
            path = Path(str(spec["dataset_path"])).expanduser()
            if not path.is_absolute():
                path = Path(project_root) / path
            if path.is_file():
                loaded = load_json_questions(path)
                records.extend(loaded)
                sources[f"evaluation_set:{identifier}"] = len(loaded)
    deduplicated = {}
    for record in records:
        key = str(record.get("question_id", record.get("id", ""))) or str(
            record.get("question", record.get("input", ""))
        )
        deduplicated[key] = dict(record)
    return list(deduplicated.values()), dict(sorted(sources.items()))


# [ADDED] Execute eval or the complete evaluation-training flywheel.
def _run_autobencher(args, agent_info, evaluator_info):
    output_root = _output_root(args.outfile_prefix1)
    os.makedirs(output_root, exist_ok=True)
    cycle_record_path = _cycle_record_file(args)
    study_runtime = _study_runtime_metadata(args.research_run.config)
    run_config = {
        "mode": args.mode,
        "agent_model": args.agent_modelname,
        "initial_test_taker_model": args.test_taker_modelname,
        "num_iters": args.num_iters,
        "max_cycle": args.max_cycle if args.mode == "data_flywheel" else 1,
        "export_interval": args.export_interval,
        "finetune_gpu": args.finetune_gpu,
        "finetune_epoch": args.finetune_epoch,
        "finetune_batch": args.finetune_batch,
        "lora_rank": args.lora_rank,
        "config_hash": (
            args.research_run.config_hash
            if getattr(args, "research_run", None)
            else None
        ),
        **study_runtime,
    }
    cycle_record = _load_cycle_record(cycle_record_path)
    if cycle_record.get("run_config") != run_config:
        cycle_record = {
            "schema_version": 1,
            "status": "running",
            "created_at": _utc_timestamp(),
            "run_config": run_config,
            "active_test_taker_model": args.test_taker_modelname,
            "cycles": [],
        }
    else:
        cycle_record["status"] = "running"
    _save_cycle_record(cycle_record_path, cycle_record)

    fixed_questions = []
    fixed_metadata = {}
    baseline_fixed_summary = None
    retention_questions = []
    retention_metadata = {}
    baseline_retention_summary = None
    if args.research_run.config["fixed_test"]["enabled"]:
        try:
            fixed_questions, fixed_metadata = load_fixed_test_set(
                args.research_run.config,
                args.research_run.project_root,
            )
            fixed_root = args.research_run.run_dir / "fixed_test"
            fixed_root.mkdir(parents=True, exist_ok=True)
            atomic_json(
                {
                    **args.research_run.metadata(),
                    **fixed_metadata,
                    "questions": fixed_questions,
                },
                fixed_root / "dataset_snapshot.json",
            )
            if args.research_run.config["fixed_test"][
                "evaluate_baseline"
            ]:
                prior_fixed = cycle_record.get("fixed_test", {})
                prior_dataset = prior_fixed.get("dataset", {})
                prior_baseline = prior_fixed.get("baseline", {})
                if (
                    args.research_run.resumed
                    and prior_dataset.get("sha256") == fixed_metadata.get("sha256")
                    and int(prior_baseline.get("total_questions", 0))
                    == len(fixed_questions)
                ):
                    baseline_fixed_summary = prior_baseline
                    args.research_run.logger.event(
                        "INFO", "FixedTest", "baseline_resume_hit",
                        f"questions={len(fixed_questions)}",
                    )
                else:
                    baseline_info = _load_test_taker_info(
                        args.test_taker_modelname,
                        args.use_helm,
                    )
                    try:
                        baseline_fixed_summary = _run_fixed_test_benchmark(
                            args,
                            model_name=args.test_taker_modelname,
                            test_taker_info=baseline_info,
                            agent_info=agent_info,
                            evaluator_info=evaluator_info,
                            fixed_questions=fixed_questions,
                            fixed_metadata=fixed_metadata,
                            stage_name="baseline",
                        )
                    finally:
                        _release_model_info(baseline_info)
                cycle_record["fixed_test"] = {
                    "dataset": fixed_metadata,
                    "baseline": baseline_fixed_summary,
                }
                _save_cycle_record(cycle_record_path, cycle_record)
        except Exception as exc:
            cycle_record["status"] = "failed"
            cycle_record["failed_stage"] = "fixed_test_baseline"
            cycle_record["error"] = _sanitize_error(exc)
            _save_cycle_record(cycle_record_path, cycle_record)
            args.research_run.finalize(
                "failed",
                {
                    "stage": "fixed_test_baseline",
                    "error": _sanitize_error(exc),
                },
            )
            print(
                "[FixedTest] baseline failed "
                f"error={_sanitize_error(exc)}"
            )
            return 1

    if args.research_run.config["retention_test"]["enabled"]:
        try:
            retention_config = {
                "fixed_test": {
                    "dataset_path": args.research_run.config[
                        "retention_test"
                    ]["dataset_path"],
                    "require_all_subcategories": False,
                }
            }
            retention_questions, retention_metadata = load_fixed_test_set(
                retention_config,
                args.research_run.project_root,
            )
            minimum_retention_questions = int(
                args.research_run.config["retention_test"][
                    "minimum_question_count"
                ]
            )
            if len(retention_questions) < minimum_retention_questions:
                raise ValueError(
                    "Retention set is too small: "
                    f"{len(retention_questions)} < {minimum_retention_questions}"
                )
            if args.research_run.config["retention_test"][
                "evaluate_baseline"
            ]:
                retention_info = _load_test_taker_info(
                    args.test_taker_modelname,
                    args.use_helm,
                )
                try:
                    baseline_retention_summary = _run_fixed_test_benchmark(
                        args,
                        model_name=args.test_taker_modelname,
                        test_taker_info=retention_info,
                        agent_info=agent_info,
                        evaluator_info=evaluator_info,
                        fixed_questions=retention_questions,
                        fixed_metadata=retention_metadata,
                        stage_name="baseline",
                        suite_name="retention_test",
                    )
                finally:
                    _release_model_info(retention_info)
                cycle_record["retention_test"] = {
                    "dataset": retention_metadata,
                    "baseline": baseline_retention_summary,
                }
                _save_cycle_record(cycle_record_path, cycle_record)
        except Exception as exc:
            cycle_record["status"] = "failed"
            cycle_record["failed_stage"] = "retention_test_baseline"
            cycle_record["error"] = _sanitize_error(exc)
            _save_cycle_record(cycle_record_path, cycle_record)
            args.research_run.finalize(
                "failed",
                {
                    "stage": "retention_test_baseline",
                    "error": _sanitize_error(exc),
                },
            )
            return 1

    if study_runtime["policy_name"] == "base":
        evaluation_provenance = thaw_config(
            args.research_run.config["evaluation_provenance"]
        )
        cycle_record["status"] = "completed"
        cycle_record["active_test_taker_model"] = args.test_taker_modelname
        cycle_record["evaluation_only"] = True
        cycle_record.update(study_runtime)
        _save_cycle_record(cycle_record_path, cycle_record)
        baseline_accuracy = (
            baseline_fixed_summary.get("accuracy")
            if baseline_fixed_summary
            else None
        )
        experiment_summary = {
            **args.research_run.metadata(),
            **study_runtime,
            "status": "completed",
            "evaluation_only": True,
            "cycle_count": 0,
            "iteration_count": 0,
            "generated_question_count": 0,
            "total_questions": (
                int(baseline_fixed_summary.get("total_questions", 0))
                if baseline_fixed_summary
                else 0
            ),
            "hard_pool_size": 0,
            "training_sample_count": 0,
            "baseline_accuracy": baseline_accuracy,
            "final_accuracy": baseline_accuracy,
            "accuracy_delta": 0.0 if baseline_accuracy is not None else None,
            "active_test_taker_model": args.test_taker_modelname,
            "fixed_test": cycle_record.get("fixed_test", {}),
            "execution_policy": (
                evaluation_provenance.get("execution_policy") or "base"
            ),
            "evaluated_method": (
                evaluation_provenance.get("evaluated_method") or "base"
            ),
            "evaluation_provenance": evaluation_provenance,
        }
        atomic_json(
            experiment_summary,
            args.research_run.run_dir / "experiment_summary.json",
        )
        if hasattr(args.research_run, "budget_ledger"):
            args.research_run.budget_ledger.record_accuracy(
                baseline_accuracy,
                baseline_accuracy,
            )
        args.research_run.finalize(
            "completed",
            {
                **study_runtime,
                "evaluation_only": True,
                "cycle_count": 0,
                "iteration_count": 0,
                "training_sample_count": 0,
                "baseline_accuracy": baseline_accuracy,
                "execution_policy": experiment_summary["execution_policy"],
                "evaluated_method": experiment_summary["evaluated_method"],
            },
        )
        args.research_run.logger.event(
            "INFO",
            "Study",
            "base_evaluation_completed",
            (
                "policy=base evaluation_only=true "
                "generation=false training_export=false finetune=false"
            ),
            metrics=experiment_summary,
        )
        return 0

    current_test_taker_model = cycle_record.get(
        "active_test_taker_model",
        args.test_taker_modelname,
    )
    cycle_limit = args.max_cycle if args.mode == "data_flywheel" else 1
    cycles_by_number = {
        item.get("cycle"): item
        for item in cycle_record.get("cycles", [])
        if isinstance(item, dict)
    }
    run_history = _load_completed_cycle_history(
        output_root,
        list(cycles_by_number.values()),
    )
    for cycle_number in range(1, cycle_limit + 1):
        data_matched_protocol = bool(
            getattr(args, "research_run", None)
            and args.research_run.config["budget"]["protocol"]
            == "data_matched"
        )
        prior_cycle = cycles_by_number.get(cycle_number)
        if prior_cycle and prior_cycle.get("status") in {
            "completed",
            "completed_without_training",
            "collecting_data_budget",
        }:
            current_test_taker_model = prior_cycle.get(
                "next_test_taker_model",
                current_test_taker_model,
            )
            print(f"[Cycle] resume_skip={cycle_number} status={prior_cycle['status']}")
            continue
        if prior_cycle:
            current_test_taker_model = prior_cycle.get(
                "test_taker_model",
                current_test_taker_model,
            )

        if prior_cycle:
            cycle_entry = {
                **prior_cycle,
                "status": "running",
                "resumed_at": _utc_timestamp(),
                "resume_count": int(prior_cycle.get("resume_count", 0)) + 1,
                **study_runtime,
            }
            cycle_entry.pop("failed_stage", None)
            cycle_entry.pop("failure_type", None)
            cycle_entry.pop("error", None)
            cycle_entry.pop("traceback", None)
        else:
            cycle_entry = {
                "cycle": cycle_number,
                "status": "running",
                "started_at": _utc_timestamp(),
                "test_taker_model": current_test_taker_model,
                "iterations_completed": 0,
                "training_export": None,
                "finetune_output": None,
                "finetune_status": "not_started",
                "next_test_taker_model": current_test_taker_model,
                **study_runtime,
            }
        if getattr(args, "research_run", None):
            _save_research_cycle_manifest(args, cycle_entry)
        cycle_record["cycles"] = [
            item
            for item in cycle_record.get("cycles", [])
            if item.get("cycle") != cycle_number
        ]
        cycle_record["cycles"].append(cycle_entry)
        cycle_record["cycles"].sort(key=lambda item: item["cycle"])
        _save_cycle_record(cycle_record_path, cycle_record)

        # Fast-path a crash that happened after the merged checkpoint and
        # fixed-test artifacts were fully published.  This path deliberately
        # does not reload the base model, rebuild the training dataset, rerun
        # QLoRA, regenerate answers, or call the semantic judge.
        post_training_resume = bool(
            prior_cycle
            and prior_cycle.get("failed_stage")
            in {"fixed_test_evaluation", "retention_test_evaluation"}
            and int(cycle_entry.get("iterations_completed", 0))
            >= int(args.num_iters)
            and getattr(args, "research_run", None)
        )
        if post_training_resume:
            training_dir = (
                args.research_run.cycle_root
                / f"cycle_{cycle_number}"
                / "training"
            )
            recovered_checkpoint = _load_resumable_cycle_checkpoint(
                training_dir / "checkpoint_manifest.json",
                args.research_run,
                cycle_entry["test_taker_model"],
            )
            recovered_model = (
                recovered_checkpoint.get("merged_model_path")
                if recovered_checkpoint
                else None
            )
            cached_fixed = None
            fixed_cache_required = bool(
                fixed_questions
                and args.research_run.config["fixed_test"][
                    "evaluate_after_each_training_cycle"
                ]
            )
            if recovered_model and fixed_cache_required:
                cached_fixed = _load_fixed_test_benchmark_cache(
                    args.research_run,
                    model_name=recovered_model,
                    fixed_questions=fixed_questions,
                    fixed_metadata=fixed_metadata,
                    stage_name=f"cycle_{cycle_number}",
                    cycle_number=cycle_number,
                )
            cached_retention = None
            retention_cache_required = bool(
                retention_questions
                and args.research_run.config["retention_test"][
                    "evaluate_after_each_training_cycle"
                ]
            )
            if recovered_model and retention_cache_required:
                cached_retention = _load_fixed_test_benchmark_cache(
                    args.research_run,
                    model_name=recovered_model,
                    fixed_questions=retention_questions,
                    fixed_metadata=retention_metadata,
                    stage_name=f"cycle_{cycle_number}",
                    cycle_number=cycle_number,
                    suite_name="retention_test",
                )
            complete_post_training_cache = bool(
                recovered_checkpoint
                and (not fixed_cache_required or cached_fixed is not None)
                and (
                    not retention_cache_required
                    or cached_retention is not None
                )
            )
            if complete_post_training_cache:
                current_test_taker_model = str(recovered_model)
                if cached_fixed is not None:
                    baseline_accuracy = (baseline_fixed_summary or {}).get(
                        "accuracy"
                    )
                    cached_fixed["baseline_accuracy"] = baseline_accuracy
                    cached_fixed["accuracy_delta"] = _optional_accuracy_delta(
                        baseline_accuracy,
                        cached_fixed.get("accuracy"),
                    )
                    cycle_entry["fixed_test"] = cached_fixed
                    atomic_json(
                        {**args.research_run.metadata(), **cached_fixed},
                        (
                            args.research_run.run_dir
                            / "fixed_test"
                            / f"cycle_{cycle_number}"
                            / "summary.json"
                        ),
                    )
                if cached_retention is not None:
                    cached_retention["forgetting"] = _retention_delta(
                        baseline_retention_summary,
                        cached_retention,
                    )
                    cycle_entry["retention_test"] = cached_retention
                training_cost = recovered_checkpoint.get("training_cost", {})
                if hasattr(args.research_run, "budget_ledger"):
                    args.research_run.budget_ledger.record_training(
                        training_cost,
                        event_id=f"cycle_{cycle_number}",
                    )
                export_path = Path(output_root) / str(
                    cycle_entry.get("training_export", "")
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "status": "completed",
                        "returncode": 0,
                        "base_model": cycle_entry["test_taker_model"],
                        "dataset_path": str(export_path).replace("\\", "/"),
                        "output_path": current_test_taker_model.replace(
                            "\\", "/"
                        ),
                        "seed": int(
                            args.research_run.config["experiment"]["seed"]
                        ),
                        "training_cost": training_cost,
                        "resumed_checkpoint": True,
                    },
                    training_dir / "finetune_summary.json",
                )
                cycle_entry["status"] = "completed"
                cycle_entry["finetune_status"] = "completed"
                cycle_entry["next_test_taker_model"] = (
                    current_test_taker_model.replace("\\", "/")
                )
                cycle_entry["completed_at"] = _utc_timestamp()
                cycle_entry["post_training_resume"] = {
                    "checkpoint_reused": True,
                    "fixed_test_reused": cached_fixed is not None,
                    "retention_test_reused": cached_retention is not None,
                }
                cycle_record["active_test_taker_model"] = (
                    current_test_taker_model
                )
                _save_cycle_record(cycle_record_path, cycle_record)
                _save_research_cycle_manifest(args, cycle_entry)
                args.research_run.logger.event(
                    "INFO",
                    "Cycle",
                    "post_training_resume_hit",
                    (
                        f"cycle={cycle_number} checkpoint=true "
                        f"fixed_test={cached_fixed is not None} "
                        f"retention_test={cached_retention is not None}"
                    ),
                    cycle=cycle_number,
                )
                continue
        test_taker_info = None
        stage = "test_taker_model_loading"
        try:
            test_taker_info = _load_test_taker_info(
                current_test_taker_model,
                args.use_helm,
            )
            if cycle_number > 1:
                hard_pool_path = Path(output_root) / "hard_pool.json"
                hard_pool = HardSamplePool(hard_pool_path)
                retest_questions = hard_pool.select_retest(
                    current_cycle=cycle_number,
                    max_samples=args.research_run.config["hard_pool"][
                        "retest_samples_per_model_version"
                    ],
                    decay_lambda=args.research_run.config[
                        "adaptive_sampling"
                    ]["decay_lambda"],
                )
                if retest_questions:
                    retest_dir = (
                        args.research_run.run_dir
                        / "hard_pool_retest"
                        / f"cycle_{cycle_number}"
                    )
                    retest_dir.mkdir(parents=True, exist_ok=True)
                    for item in retest_questions:
                        item["answer"] = item.get("gold_answer", "")
                    retest_records = test_and_eval(
                        copy.deepcopy(retest_questions),
                        str(retest_dir / "hard_pool"),
                        test_taker_info,
                        agent_info,
                        evaluator_info,
                        gold_ans_key="answer",
                        iter_number=0,
                        temp_log_dir=str(retest_dir / "temp_log"),
                        research_config=args.research_run.config,
                        progress_manager=args.research_run.progress,
                        cycle_number=cycle_number,
                        event_logger=args.research_run.logger,
                    )
                    lifecycle = update_hard_pool_lifecycle(
                        hard_pool_path,
                        retest_records,
                        model_version=current_test_taker_model,
                        current_cycle=cycle_number,
                        mastered_correct_streak=args.research_run.config[
                            "hard_pool"
                        ]["mastered_correct_streak"],
                        stale_after_cycles=args.research_run.config["hard_pool"][
                            "stale_after_cycles"
                        ],
                        retire_after_cycles=args.research_run.config["hard_pool"][
                            "retire_after_cycles"
                        ],
                    )
                    cycle_entry["hard_pool_retest"] = {
                        "question_count": len(retest_records),
                        **lifecycle,
                    }
                    _save_cycle_record(cycle_record_path, cycle_record)
            history_dict = run_history
            stage = "evaluation"
            last_iteration = None
            cycle_generation_results = []
            first_iteration = int(cycle_entry.get("iterations_completed", 0)) + 1
            for iter_number in range(first_iteration, args.num_iters + 1):
                try:
                    last_iteration = _run_math_iteration(
                        args,
                        current_test_taker_model,
                        test_taker_info,
                        agent_info,
                        evaluator_info,
                        history_dict,
                        cycle_number,
                        iter_number,
                    )
                finally:
                    # [ADDED] Always clean the just-finished iteration,
                    # including partial generation and exception paths.
                    _cleanup_iteration_cache(
                        args,
                        cycle_number,
                        iter_number,
                    )
                cycle_entry["iterations_completed"] = iter_number
                cycle_entry["last_global_accuracy"] = last_iteration[
                    "global_accuracy"
                ]
                cycle_generation_results.append(
                    last_iteration.get("generation_result", {})
                )
                _save_cycle_record(cycle_record_path, cycle_record)

            cycle_generation_statistics = (
                _aggregate_cycle_generation_statistics(
                    cycle_generation_results
                )
            )
            cycle_entry["generation_statistics"] = (
                cycle_generation_statistics
            )
            if getattr(args, "research_run", None):
                args.research_run.save_cycle_artifact(
                    cycle_number,
                    "metrics",
                    "generation_statistics",
                    cycle_generation_statistics,
                )
            if args.mode == "eval":
                cycle_entry["status"] = "completed"
                cycle_entry["finetune_status"] = "disabled"
                cycle_entry["completed_at"] = _utc_timestamp()
                cycle_record["active_test_taker_model"] = current_test_taker_model
                _save_cycle_record(cycle_record_path, cycle_record)
                _save_research_cycle_manifest(args, cycle_entry)
                continue

            completed_before = (cycle_number - 1) * args.num_iters
            completed_now = cycle_number * args.num_iters
            crossed_export_boundary = (
                completed_now // args.export_interval
                > completed_before // args.export_interval
            )
            if not crossed_export_boundary:
                cycle_entry["status"] = "completed_without_training"
                cycle_entry["finetune_status"] = "waiting_for_export_interval"
                cycle_entry["completed_at"] = _utc_timestamp()
                _save_cycle_record(cycle_record_path, cycle_record)
                _save_research_cycle_manifest(args, cycle_entry)
                continue

            stage = "disk_check"
            disk_status = check_disk_space(
                output_root,
                args.disk_warning_threshold,
            )
            cycle_entry["disk_check"] = disk_status
            if not disk_status["ok"]:
                raise RuntimeError(
                    f"Insufficient free disk space: {disk_status['free_gb']:.3f} GB"
                )

            stage = "training_export"
            if getattr(args, "research_run", None):
                training_dir = (
                    args.research_run.cycle_root
                    / f"cycle_{cycle_number}"
                    / "training"
                )
                training_dir.mkdir(parents=True, exist_ok=True)
                data_matched_protocol = (
                    args.research_run.config["budget"]["protocol"]
                    == "data_matched"
                )
                candidates = [
                    record
                    for iteration_records in history_dict
                    for record in iteration_records
                    if data_matched_protocol
                    or int(record.get("cycle_id", -1))
                    == int(cycle_number)
                ]
                leakage_holdouts, leakage_holdout_sources = (
                    _evaluation_leakage_holdouts(
                        args.research_run.config,
                        args.research_run.project_root,
                        fixed_questions,
                        retention_questions,
                    )
                )
                selected, dataset_manifest, rejected = build_training_dataset(
                    candidates,
                    args.research_run.config,
                    seed=int(args.research_run.config["experiment"]["seed"]),
                    holdout_records=leakage_holdouts,
                )
                dataset_manifest["leakage_holdout_sources"] = (
                    leakage_holdout_sources
                )
                dataset_manifest["leakage_holdout_count"] = len(
                    leakage_holdouts
                )
                if data_matched_protocol:
                    target_samples = int(
                        args.research_run.config["budget"][
                            "data_matched_target_samples"
                        ]
                    )
                    dataset_manifest.update(
                        {
                            "budget_protocol": "data_matched",
                            "data_matched_target_samples": target_samples,
                            "matching_unit": "post_split_train_samples",
                        }
                    )
                if (
                    selected
                    and bool(
                        args.research_run.config["training_mix"][
                            "strict_correct_incorrect_ratio"
                        ]
                    )
                    and not dataset_manifest["strict_ratio_satisfied"]
                ):
                    raise RuntimeError(
                        "Training dataset violated the configured "
                        "25% correct / 75% wrong ratio"
                    )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "data": candidates,
                    },
                    training_dir / "dataset_candidates.json",
                )
                export_path = str(training_dir / "dataset_selected.jsonl")
                exported_count = write_alpaca_jsonl(selected, export_path)
                split_manifest = None
                validation_export_path = None
                internal_test_export_path = None
                if data_matched_protocol and not selected:
                    if cycle_number < cycle_limit:
                        cycle_entry["status"] = "collecting_data_budget"
                        cycle_entry["finetune_status"] = (
                            "waiting_for_post_split_train_target"
                        )
                        cycle_entry["training_sample_count"] = 0
                        cycle_entry["completed_at"] = _utc_timestamp()
                        _save_cycle_record(cycle_record_path, cycle_record)
                        _save_research_cycle_manifest(args, cycle_entry)
                        continue
                    raise RuntimeError(
                        "Data-matched run has no eligible post-filter samples"
                    )
                if (
                    selected
                    and bool(
                        args.research_run.config["finetune"]["enabled"]
                    )
                ):
                    split_config = args.research_run.config["finetune"][
                        "training_split"
                    ]
                    try:
                        splits, split_manifest = split_by_template_cluster(
                            selected,
                            split_config,
                            seed=int(
                                args.research_run.config["experiment"]["seed"]
                            ),
                        )
                    except ValueError as split_error:
                        if data_matched_protocol and cycle_number < cycle_limit:
                            cycle_entry["status"] = "collecting_data_budget"
                            cycle_entry["finetune_status"] = (
                                "waiting_for_template_clusters"
                            )
                            cycle_entry["training_sample_count"] = 0
                            cycle_entry["completed_at"] = _utc_timestamp()
                            cycle_entry["split_error"] = str(split_error)
                            _save_cycle_record(cycle_record_path, cycle_record)
                            _save_research_cycle_manifest(args, cycle_entry)
                            continue
                        raise
                    if data_matched_protocol:
                        target_samples = int(
                            args.research_run.config["budget"][
                                "data_matched_target_samples"
                            ]
                        )
                        correct_target = target_samples // 4
                        wrong_target = target_samples - correct_target
                        train_correct = [
                            item for item in splits["train"]
                            if item.get("_metadata", {}).get("training_source")
                            == "correct_retention_samples"
                        ]
                        train_wrong = [
                            item for item in splits["train"]
                            if item.get("_metadata", {}).get("training_source")
                            != "correct_retention_samples"
                        ]
                        shortfall = max(0, correct_target - len(train_correct)) + max(
                            0, wrong_target - len(train_wrong)
                        )
                        dataset_manifest["data_matched_train_shortfall"] = shortfall
                        dataset_manifest["post_split_train_pool_count"] = len(
                            splits["train"]
                        )
                        if target_samples % 4 or shortfall:
                            if cycle_number < cycle_limit:
                                cycle_entry["status"] = "collecting_data_budget"
                                cycle_entry["finetune_status"] = (
                                    "waiting_for_post_split_train_target"
                                )
                                cycle_entry["training_sample_count"] = 0
                                cycle_entry["completed_at"] = _utc_timestamp()
                                _save_cycle_record(cycle_record_path, cycle_record)
                                _save_research_cycle_manifest(args, cycle_entry)
                                args.research_run.logger.event(
                                    "INFO", "BuildDataset", "data_budget_shortfall",
                                    (
                                        f"post_split_train={len(splits['train'])} "
                                        f"target={target_samples} shortfall={shortfall}"
                                    ),
                                    cycle=cycle_number,
                                    metrics=dataset_manifest,
                                )
                                continue
                            raise RuntimeError(
                                "Data-matched post-split train target was not "
                                f"reached: train={len(splits['train'])}, "
                                f"target={target_samples}, ratio_shortfall={shortfall}"
                            )
                        splits["train"] = [
                            *train_correct[:correct_target],
                            *train_wrong[:wrong_target],
                        ]
                        split_manifest["split_record_counts"]["train"] = target_samples
                        split_manifest["data_matched_train_cap"] = {
                            "target": target_samples,
                            "correct": correct_target,
                            "incorrect": wrong_target,
                            "actual": len(splits["train"]),
                        }
                    elif bool(
                        args.research_run.config["training_mix"][
                            "strict_correct_incorrect_ratio"
                        ]
                    ):
                        splits, ratio_audit = (
                            enforce_train_correct_incorrect_ratio(
                                splits,
                                correct_fraction=float(
                                    args.research_run.config["training_mix"][
                                        "correct_retention_samples"
                                    ]
                                ),
                            )
                        )
                        split_manifest["strict_train_ratio_cap"] = ratio_audit
                    split_manifest = write_precomputed_training_splits(
                        splits,
                        training_dir,
                        split_manifest,
                    )
                    export_path = split_manifest["paths"]["train"]
                    validation_export_path = split_manifest["paths"][
                        "validation"
                    ]
                    internal_test_export_path = split_manifest["paths"][
                        "internal_test"
                    ]
                    exported_count = int(
                        split_manifest["split_record_counts"]["train"]
                    )
                if hasattr(args.research_run, "budget_ledger"):
                    args.research_run.budget_ledger.record_dataset(
                        dataset_manifest,
                        selected,
                        split_manifest=split_manifest,
                        event_id=f"cycle_{cycle_number}",
                    )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        **dataset_manifest,
                    },
                    training_dir / "dataset_manifest.json",
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "rejected": rejected,
                        "rejection_reasons": dataset_manifest[
                            "rejection_reasons"
                        ],
                    },
                    training_dir / "dedup_report.json",
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "template_cluster_count": dataset_manifest[
                            "template_cluster_count"
                        ],
                        "selected_mix_counts": dataset_manifest[
                            "selected_mix_counts"
                        ],
                    },
                    training_dir / "diversity_report.json",
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "base_model": current_test_taker_model,
                        "gpu": args.finetune_gpu,
                        "epochs": args.finetune_epoch,
                        "batch_size": args.finetune_batch,
                        "lora_rank": args.lora_rank,
                        "max_seq_length": args.research_run.config[
                            "finetune"
                        ]["max_seq_length"],
                        "learning_rate": args.research_run.config[
                            "finetune"
                        ]["learning_rate"],
                        "training_split": split_manifest,
                        "max_training_tokens": args.research_run.config[
                            "finetune"
                        ]["max_training_tokens"],
                        "max_optimizer_steps": args.research_run.config[
                            "finetune"
                        ]["max_optimizer_steps"],
                        "evaluation_strategy": args.research_run.config[
                            "finetune"
                        ]["evaluation_strategy"],
                        "eval_steps": args.research_run.config["finetune"][
                            "eval_steps"
                        ],
                        "save_steps": args.research_run.config["finetune"][
                            "save_steps"
                        ],
                        "load_best_model_at_end": args.research_run.config[
                            "finetune"
                        ]["load_best_model_at_end"],
                        "metric_for_best_model": args.research_run.config[
                            "finetune"
                        ]["metric_for_best_model"],
                        "early_stopping_patience": args.research_run.config[
                            "finetune"
                        ]["early_stopping_patience"],
                        "checkpoint_selection_rule": args.research_run.config[
                            "finetune"
                        ]["checkpoint_selection_rule"],
                        "evaluation_set_used_for_model_selection": False,
                        "seed": int(
                            args.research_run.config["experiment"]["seed"]
                        ),
                    },
                    training_dir / "finetune_config.json",
                )
                rejection_reason_counts = dataset_manifest[
                    "rejection_reasons"
                ]
                args.research_run.logger.event(
                    "INFO",
                    "BuildDataset",
                    "stage_completed",
                    (
                        f"selected={exported_count} "
                        f"rejected={len(rejected)} "
                        "rejection_reasons="
                        f"{json.dumps(rejection_reason_counts, sort_keys=True)}"
                    ),
                    cycle=cycle_number,
                    metrics=dataset_manifest,
                )
            else:
                export_path = os.path.join(
                    output_root,
                    "training_export",
                    f"cycle_{cycle_number}_train.jsonl",
                )
                exported_count = export_training_dataset(
                    os.path.join(output_root, "hard_pool.json"),
                    export_path,
                )
                validation_export_path = None
                internal_test_export_path = None
            cycle_entry["training_export"] = _relative_json_path(
                export_path,
                output_root,
            )
            cycle_entry["training_sample_count"] = exported_count
            _save_cycle_record(cycle_record_path, cycle_record)
            if (
                getattr(args, "research_run", None)
                and bool(args.research_run.config["finetune"]["enabled"])
                and not dataset_manifest["minimum_sample_requirement_met"]
            ):
                raise RuntimeError(
                    "Training dataset has fewer eligible samples than "
                    "training_mix.minimum_samples: "
                    f"selected={exported_count}, "
                    "required="
                    f"{args.research_run.config['training_mix']['minimum_samples']}, "
                    "rejection_reasons="
                    f"{json.dumps(rejection_reason_counts, sort_keys=True)}"
                )
            if exported_count == 0:
                cycle_entry["status"] = "completed_without_training"
                cycle_entry["finetune_status"] = "no_train_eligible_samples"
                cycle_entry["completed_at"] = _utc_timestamp()
                _save_cycle_record(cycle_record_path, cycle_record)
                _save_research_cycle_manifest(args, cycle_entry)
                continue

            stage = "finetune"
            _release_model_info(test_taker_info)
            test_taker_info = None
            model_dir_name = (
                f"{_safe_model_component(current_test_taker_model)}_"
                f"{_safe_model_component(args.new_local_model_suffix)}_"
                f"cycle_{cycle_number}"
            )
            finetune_output = os.path.join(
                output_root,
                "models",
                model_dir_name,
            )
            cycle_entry["finetune_output"] = _relative_json_path(
                finetune_output,
                output_root,
            )
            cycle_entry["finetune_status"] = "running"
            _save_cycle_record(cycle_record_path, cycle_record)
            if hasattr(args.research_run, "budget_ledger"):
                args.research_run.budget_ledger.assert_training_available()
            checkpoint_resume_path = (
                training_dir / "checkpoint_manifest.json"
                if getattr(args, "research_run", None)
                else None
            )
            recovered_checkpoint = None
            if checkpoint_resume_path:
                recovered_checkpoint = _load_resumable_cycle_checkpoint(
                    checkpoint_resume_path,
                    args.research_run,
                    current_test_taker_model,
                )
            if recovered_checkpoint:
                args.research_run.logger.event(
                    "INFO",
                    "FineTune",
                    "checkpoint_resume_hit",
                    (
                        f"cycle={cycle_number} model="
                        f"{recovered_checkpoint['merged_model_path']}"
                    ),
                    cycle=cycle_number,
                )
            result = (
                {
                    "success": True,
                    "returncode": 0,
                    "training_summary": _load_cycle_record(
                        str(training_dir / "training_cost_summary.json")
                    ),
                    "resumed_checkpoint": True,
                }
                if recovered_checkpoint
                else call_local_finetune(
                current_test_taker_model,
                export_path,
                args.finetune_gpu,
                args.finetune_epoch,
                args.finetune_batch,
                args.lora_rank,
                finetune_output,
                metrics_path=(
                    str(training_dir / "finetune_metrics.jsonl")
                    if getattr(args, "research_run", None)
                    else None
                ),
                summary_path=(
                    str(training_dir / "training_cost_summary.json")
                    if getattr(args, "research_run", None)
                    else None
                ),
                run_id=(
                    args.research_run.run_id
                    if getattr(args, "research_run", None)
                    else None
                ),
                config_hash=(
                    args.research_run.config_hash
                    if getattr(args, "research_run", None)
                    else None
                ),
                max_seq_length=(
                    args.research_run.config["finetune"]["max_seq_length"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                learning_rate=(
                    args.research_run.config["finetune"]["learning_rate"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_enabled=(
                    bool(
                        args.research_run.config["tracking"]["wandb"][
                            "enabled"
                        ]
                    )
                    if getattr(args, "research_run", None)
                    else False
                ),
                wandb_mode=(
                    args.research_run.config["tracking"]["wandb"]["mode"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_project=(
                    args.research_run.config["tracking"]["wandb"]["project"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_entity=(
                    args.research_run.config["tracking"]["wandb"]["entity"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_group=(
                    args.research_run.config["tracking"]["wandb"]["group"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_tags=(
                    args.research_run.config["tracking"]["wandb"]["tags"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                wandb_log_model=(
                    bool(
                        args.research_run.config["tracking"]["wandb"][
                            "log_model"
                        ]
                    )
                    if getattr(args, "research_run", None)
                    else False
                ),
                seed=(
                    int(args.research_run.config["experiment"]["seed"])
                    if getattr(args, "research_run", None)
                    else 42
                ),
                eval_dataset_path=validation_export_path,
                internal_test_dataset_path=internal_test_export_path,
                max_training_tokens=(
                    args.research_run.config["finetune"][
                        "max_training_tokens"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                max_optimizer_steps=(
                    args.research_run.config["finetune"][
                        "max_optimizer_steps"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                evaluation_strategy=(
                    args.research_run.config["finetune"][
                        "evaluation_strategy"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                eval_steps=(
                    args.research_run.config["finetune"]["eval_steps"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                save_steps=(
                    args.research_run.config["finetune"]["save_steps"]
                    if getattr(args, "research_run", None)
                    else None
                ),
                load_best_model_at_end=(
                    args.research_run.config["finetune"][
                        "load_best_model_at_end"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                metric_for_best_model=(
                    args.research_run.config["finetune"][
                        "metric_for_best_model"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                greater_is_better=(
                    args.research_run.config["finetune"][
                        "greater_is_better"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                early_stopping_patience=(
                    args.research_run.config["finetune"][
                        "early_stopping_patience"
                    ]
                    if getattr(args, "research_run", None)
                    else None
                ),
                )
            )
            training_cost_summary = result.get("training_summary") or {}
            if not result["success"]:
                raise RuntimeError(
                    result.get("error")
                    or f"Fine-tuning exited with code {result['returncode']}"
                )
            if not recovered_checkpoint and getattr(args, "research_run", None):
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "status": "completed",
                        "base_model": current_test_taker_model,
                        "merged_model_path": os.path.abspath(finetune_output).replace(
                            "\\", "/"
                        ),
                        "merged_model_sha256": artifact_fingerprint(
                            finetune_output,
                            allow_missing=False,
                        )["sha256"],
                        "training_cost": training_cost_summary,
                    },
                    training_dir / "checkpoint_manifest.json",
                )
            if (
                getattr(args, "research_run", None)
                and hasattr(args.research_run, "budget_ledger")
            ):
                # The checkpoint is published first. A crash after publication
                # is recovered without retraining, and this idempotent event
                # then fills any missing ledger entry exactly once.
                args.research_run.budget_ledger.record_training(
                    training_cost_summary,
                    event_id=f"cycle_{cycle_number}",
                )
            current_test_taker_model = (
                str(recovered_checkpoint["merged_model_path"])
                if recovered_checkpoint
                else os.path.abspath(finetune_output)
            )
            if (
                fixed_questions
                and args.research_run.config["fixed_test"][
                    "evaluate_after_each_training_cycle"
                ]
            ):
                stage = "fixed_test_evaluation"
                trained_test_taker_info = _load_test_taker_info(
                    current_test_taker_model,
                    args.use_helm,
                )
                try:
                    fixed_cycle_summary = _run_fixed_test_benchmark(
                        args,
                        model_name=current_test_taker_model,
                        test_taker_info=trained_test_taker_info,
                        agent_info=agent_info,
                        evaluator_info=evaluator_info,
                        fixed_questions=fixed_questions,
                        fixed_metadata=fixed_metadata,
                        stage_name=f"cycle_{cycle_number}",
                        cycle_number=cycle_number,
                    )
                finally:
                    _release_model_info(trained_test_taker_info)
                baseline_accuracy = (baseline_fixed_summary or {}).get(
                    "accuracy"
                )
                fixed_cycle_summary["baseline_accuracy"] = baseline_accuracy
                fixed_cycle_summary["accuracy_delta"] = _optional_accuracy_delta(
                    baseline_accuracy,
                    fixed_cycle_summary.get("accuracy"),
                )
                cycle_entry["fixed_test"] = fixed_cycle_summary
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        **fixed_cycle_summary,
                    },
                    (
                        args.research_run.run_dir
                        / "fixed_test"
                        / f"cycle_{cycle_number}"
                        / "summary.json"
                    ),
                )
            if (
                retention_questions
                and args.research_run.config["retention_test"][
                    "evaluate_after_each_training_cycle"
                ]
            ):
                stage = "retention_test_evaluation"
                retention_model_info = _load_test_taker_info(
                    current_test_taker_model,
                    args.use_helm,
                )
                try:
                    retention_cycle_summary = _run_fixed_test_benchmark(
                        args,
                        model_name=current_test_taker_model,
                        test_taker_info=retention_model_info,
                        agent_info=agent_info,
                        evaluator_info=evaluator_info,
                        fixed_questions=retention_questions,
                        fixed_metadata=retention_metadata,
                        stage_name=f"cycle_{cycle_number}",
                        cycle_number=cycle_number,
                        suite_name="retention_test",
                    )
                finally:
                    _release_model_info(retention_model_info)
                retention_cycle_summary["forgetting"] = _retention_delta(
                    baseline_retention_summary,
                    retention_cycle_summary,
                )
                cycle_entry["retention_test"] = retention_cycle_summary
            if getattr(args, "research_run", None):
                training_dir = (
                    args.research_run.cycle_root
                    / f"cycle_{cycle_number}"
                    / "training"
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "status": "completed",
                        "returncode": result["returncode"],
                        "base_model": cycle_entry["test_taker_model"],
                        "dataset_path": str(export_path).replace("\\", "/"),
                        "output_path": current_test_taker_model.replace("\\", "/"),
                        "seed": int(
                            args.research_run.config["experiment"]["seed"]
                        ),
                        "training_cost": training_cost_summary,
                    },
                    training_dir / "finetune_summary.json",
                )
                # checkpoint_manifest.json was already published atomically
                # immediately after model merge with the required model SHA
                # and training-cost fields. Do not overwrite that recoverable
                # manifest with a reduced summary after evaluation.
            cycle_entry["status"] = "completed"
            cycle_entry["finetune_status"] = "completed"
            cycle_entry["next_test_taker_model"] = current_test_taker_model.replace(
                "\\",
                "/",
            )
            cycle_entry["completed_at"] = _utc_timestamp()
            cycle_record["active_test_taker_model"] = current_test_taker_model
            _save_cycle_record(cycle_record_path, cycle_record)
            _save_research_cycle_manifest(args, cycle_entry)
            if data_matched_protocol:
                break
        except (Exception, KeyboardInterrupt) as exc:
            cycle_entry["status"] = "failed"
            cycle_entry["failed_stage"] = stage
            cycle_entry["failure_type"] = _failure_type(stage, exc)
            cycle_entry["error"] = _sanitize_error(exc)
            cycle_entry["exception_type"] = type(exc).__name__
            cycle_entry["traceback"] = _sanitize_traceback(exc)
            cycle_entry["completed_at"] = _utc_timestamp()
            cycle_record["status"] = "failed"
            cycle_record["active_test_taker_model"] = current_test_taker_model
            _save_cycle_record(cycle_record_path, cycle_record)
            if getattr(args, "research_run", None):
                failure_payload = {
                    "status": "failed",
                    "cycle_id": cycle_number,
                    "stage": stage,
                    "failure_type": cycle_entry["failure_type"],
                    "error": cycle_entry["error"],
                    "exception_type": cycle_entry["exception_type"],
                    "traceback": cycle_entry["traceback"],
                    "base_model": current_test_taker_model,
                    "completed_at": cycle_entry["completed_at"],
                }
                _save_research_cycle_manifest(args, cycle_entry)
                args.research_run.save_cycle_artifact(
                    cycle_number,
                    "training" if stage in {"training_export", "finetune"} else "failure",
                    "finetune_summary" if stage == "finetune" else "failure_summary",
                    failure_payload,
                )
                args.research_run.finalize(
                    "failed",
                    {
                        "cycle": cycle_number,
                        "stage": stage,
                        "failure_type": cycle_entry["failure_type"],
                    },
                )
            print(
                f"[Cycle] failed cycle={cycle_number} stage={stage} "
                f"type={cycle_entry['failure_type']} error={cycle_entry['error']}"
            )
            return 1
        finally:
            _release_model_info(test_taker_info)

    cycle_record["status"] = "completed"
    cycle_record["active_test_taker_model"] = current_test_taker_model
    _save_cycle_record(cycle_record_path, cycle_record)
    if getattr(args, "research_run", None):
        iteration_question_counts = []
        for path in args.research_run.cycle_root.glob(
            "cycle_*/iter_*/*.test_taker_inference.json"
        ):
            iteration_question_counts.append(
                len(read_json_records(path))
            )
        baseline_summary = cycle_record.get("fixed_test", {}).get(
            "baseline",
            {},
        )
        trained_cycle_summaries = [
            item
            for item in cycle_record.get("cycles", [])
            if item.get("finetune_status") == "completed"
            and isinstance(item.get("fixed_test"), dict)
        ]
        final_fixed_summary = (
            trained_cycle_summaries[-1]["fixed_test"]
            if trained_cycle_summaries
            else {}
        )
        trained_retention_summaries = [
            item["retention_test"]
            for item in cycle_record.get("cycles", [])
            if isinstance(item.get("retention_test"), dict)
        ]
        final_retention_summary = (
            trained_retention_summaries[-1]
            if trained_retention_summaries
            else {}
        )
        training_sample_count = sum(
            int(item.get("training_sample_count", 0) or 0)
            for item in cycle_record.get("cycles", [])
        )
        experiment_summary = {
            **args.research_run.metadata(),
            **study_runtime,
            "status": "completed",
            "cycle_count": cycle_limit,
            "iteration_count": len(iteration_question_counts),
            "total_questions": sum(iteration_question_counts),
            "hard_pool_size": len(
                read_json_records(os.path.join(output_root, "hard_pool.json"))
            ),
            "training_sample_count": training_sample_count,
            "baseline_accuracy": baseline_summary.get("accuracy"),
            "final_accuracy": final_fixed_summary.get("accuracy"),
            "accuracy_delta": final_fixed_summary.get("accuracy_delta"),
            "fixed_test_question_count": final_fixed_summary.get(
                "total_questions"
            ),
            "active_test_taker_model": current_test_taker_model,
            "fixed_test": cycle_record.get("fixed_test", {}),
            "execution_policy": (
                args.research_run.config["evaluation_provenance"].get(
                    "execution_policy"
                )
                or "train_and_evaluate"
            ),
            "evaluated_method": (
                args.research_run.config["evaluation_provenance"].get(
                    "evaluated_method"
                )
                or study_runtime["policy_name"]
            ),
            "evaluation_provenance": thaw_config(
                args.research_run.config["evaluation_provenance"]
            ),
            "retention_test": {
                **cycle_record.get("retention_test", {}),
                "final": final_retention_summary,
            },
        }
        if hasattr(args.research_run, "budget_ledger"):
            args.research_run.budget_ledger.record_accuracy(
                experiment_summary["baseline_accuracy"],
                experiment_summary["final_accuracy"],
            )
        atomic_json(
            experiment_summary,
            args.research_run.run_dir / "experiment_summary.json",
        )
        args.research_run.finalize(
            "completed",
            {
                **study_runtime,
                "cycle_count": experiment_summary["cycle_count"],
                "iteration_count": experiment_summary["iteration_count"],
                "active_test_taker_model": current_test_taker_model,
            },
        )
        print(
            "[MathFlywheel] run_completed "
            + json.dumps(
                {
                    "status": "completed",
                    "run_dir": str(args.research_run.run_dir).replace(
                        "\\",
                        "/",
                    ),
                    "training_sample_count": training_sample_count,
                    "baseline_accuracy": experiment_summary[
                        "baseline_accuracy"
                    ],
                    "final_accuracy": experiment_summary["final_accuracy"],
                    "accuracy_delta": experiment_summary["accuracy_delta"],
                    "active_test_taker_model": current_test_taker_model,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


def _run_python_solve(args, agent_info):
    datafile = args.outfile_prefix1
    full_list = []
    try:
        with open(datafile, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                record["test_taker_response"] = "PLACEHOLDER"
                full_list.append(record)
    except (json.JSONDecodeError, TypeError):
        with open(datafile, "r", encoding="utf-8") as handle:
            for record in json.load(handle):
                record["test_taker_response"] = "PLACEHOLDER"
                full_list.append(record)
    solve_with_python(full_list, args.outfile_prefix1, agent_info)
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="math_autobencher.py",
        description="Run adaptive math evaluation or the local LoRA data flywheel.",
    )
    parser.add_argument("--test_taker_modelname", default="gpt-3.5-turbo")
    parser.add_argument("--test_taker_modelname2", default=None)
    parser.add_argument("--agent_modelname", default="gpt-4-turbo-preview")
    parser.add_argument("--tool_modelname", default=None)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--pairwise", type=str, default="no")
    parser.add_argument("--exp_mode", type=str, default="autobencher")
    parser.add_argument("--theme", type=str, default="history")
    parser.add_argument("--use_helm", type=str, default="no")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--acc_target", type=str, default="0.1--0.3")
    parser.add_argument("--num_iters", type=int, default=8)
    parser.add_argument("--outfile_prefix1", type=str, default=None)
    parser.add_argument(
        "--config",
        "--experiment",
        type=str,
        default=None,
        help="YAML configuration profile; existing CLI options override it.",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Optional environment YAML merged below the experiment profile.",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate the complete flywheel environment without starting a run",
    )
    parser.add_argument("--resume", type=_parse_bool, default=None)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="PATH=VALUE",
        help="Temporary dotted-path YAML overrides.",
    )
    # [ADDED] Flywheel and built-in QLoRA parameters.
    parser.add_argument(
        "--mode",
        choices=["eval", "data_flywheel"],
        default="eval",
    )
    parser.add_argument("--export_interval", type=int, default=1)
    parser.add_argument("--max_cycle", type=int, default=1)
    parser.add_argument("--finetune_gpu", type=str, default="0")
    parser.add_argument("--finetune_epoch", type=int, default=3)
    parser.add_argument("--finetune_batch", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument(
        "--new_local_model_suffix",
        type=str,
        default="finetuned",
    )
    parser.add_argument("--disk_warning_threshold", type=int, default=10)
    parser.add_argument(
        "--clean_cycle_cache",
        type=_parse_bool,
        default=True,
    )
    return parser


def _run_preflight(config, storage_paths):
    import sympy

    from train_llm import validate_dependencies

    solver = TruthSolver.from_config(config)
    self_test = solver.solve("Solve for x: 2*x + 3 = 11.")
    if not self_test.success or self_test.canonical_answer != "4":
        raise RuntimeError(
            "SymPy gold solver self-test failed: "
            f"{self_test.to_dict()}"
        )
    difficulty_self_test = assess_difficulty(
        "Solve for x: 2*x + 3 = 11.",
        self_test.answer_type or "equation",
        self_test.truth_validation_details,
        solver.training_reasoning(self_test),
        int(config["adaptive_sampling"]["initial_difficulty"]),
        config,
    )
    similarity_probe_config = dict(config["dataset"])
    # Probe MinHash independently. Loading the embedding model here would make
    # a nominal preflight download a large optional model.
    similarity_probe_config["sentence_transformers_enabled"] = False
    similarity_probe_config["sentence_transformers_required"] = False
    similarity_probe = build_similarity_batch(
        ["solve x + 2 = 5", "solve x + 2 = 6"],
        similarity_probe_config,
    )
    fixed_questions = []
    fixed_metadata = {}
    if config["fixed_test"]["enabled"]:
        fixed_questions, fixed_metadata = load_fixed_test_set(
            config,
            Path(__file__).resolve().parent,
        )
    cuda_available = None
    if config["finetune"]["enabled"]:
        validate_dependencies(
            bool(config["tracking"]["wandb"]["enabled"])
        )
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            raise RuntimeError(
                "CUDA is unavailable but finetune.enabled is true."
            )
    evaluator_name = str(config["models"]["evaluator"]["model_name"])
    if (
        "deepseek" in evaluator_name.lower()
        and not os.environ.get("DEEPSEEK_API_KEY")
    ):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is required for the configured evaluator."
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "gold_solver_backend": config["generation"][
                    "gold_solver_backend"
                ],
                "sympy_version": sympy.__version__,
                "sympy_self_test_answer": self_test.canonical_answer,
                "difficulty_rubric_version": config["difficulty"][
                    "rubric_version"
                ],
                "difficulty_self_test": difficulty_self_test,
                "difficulty_generation_bounds": [
                    int(config["generation"]["minimum_difficulty"]),
                    int(config["generation"]["maximum_difficulty"]),
                ],
                "minhash_backend": similarity_probe.minhash_backend,
                "minhash_backend_error": similarity_probe.minhash_error,
                "sentence_transformers_enabled": bool(
                    config["dataset"]["sentence_transformers_enabled"]
                ),
                "sentence_transformers_required": bool(
                    config["dataset"]["sentence_transformers_required"]
                ),
                "output_root": storage_paths["output_root"],
                "temp_dir": storage_paths["temp_dir"],
                "fixed_test_question_count": len(fixed_questions),
                "fixed_test_sha256": fixed_metadata.get("sha256"),
                "finetune_enabled": bool(config["finetune"]["enabled"]),
                "cuda_available": cuda_available,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cli_option_present(*names):
    return any(
        token == name or token.startswith(name + "=")
        for token in sys.argv[1:]
        for name in names
    )


def _set_nested(mapping, path, value):
    cursor = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _configuration_cli_overrides(args, config_explicit):
    explicit_options = {
        token.split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    return cli_config_overrides(
        args,
        explicit_options=explicit_options,
        include_implicit_defaults=not config_explicit,
    )


def _apply_resolved_configuration(args, config):
    args.agent_modelname = str(config["models"]["evaluator"]["model_name"])
    args.test_taker_modelname = str(config["models"]["test_taker"]["model_path"])
    args.exp_mode = str(config["experiment"]["exp_mode"])
    args.num_iters = int(config["experiment"]["num_iterations"])
    args.mode = str(config["experiment"]["mode"])
    args.export_interval = int(config["experiment"]["export_interval"])
    args.max_cycle = int(config["experiment"]["max_cycles"])
    args.clean_cycle_cache = bool(config["experiment"]["clean_cycle_cache"])
    args.finetune_gpu = str(config["finetune"]["gpu"])
    args.finetune_epoch = int(config["finetune"]["epochs"])
    args.finetune_batch = int(config["finetune"]["batch_size"])
    args.lora_rank = int(config["finetune"]["lora_rank"])
    args.new_local_model_suffix = str(
        config["finetune"]["new_local_model_suffix"]
    )
    args.disk_warning_threshold = float(
        config["experiment"]["disk_warning_threshold_gb"]
    )
    args.temperature = float(config["models"]["evaluator"]["temperature"])
    args.top_p = float(config["models"]["evaluator"]["top_p"])
    args.tool_modelname = config["models"]["judge"]["model_name"]
    args.use_helm = (
        "yes" if config["compatibility"]["use_helm"] else "no"
    )
    args.acc_target = (
        f"{config['adaptive_sampling']['target_accuracy_low']},"
        f"{config['adaptive_sampling']['target_accuracy_high']}"
    )
    if not _cli_option_present(
        "--outfile_prefix1",
        "--outfile-prefix1",
    ):
        output_root = Path(str(config["paths"]["output_root"])).expanduser()
        prefix = str(config["paths"]["outfile_prefix"]).rstrip(".")
        args.outfile_prefix1 = str(output_root / f"{prefix}.")


def main():
    parser = _build_parser()
    args = parser.parse_args()
    config_explicit = args.config is not None
    config_path = args.config or str(
        Path(__file__).resolve().parent / "configs" / "math_flywheel.yaml"
    )
    try:
        cli_overrides = _configuration_cli_overrides(args, config_explicit)
        bootstrap_config, _ = load_project_config(
            config_path,
            environment_path=args.environment,
            cli_overrides=cli_overrides,
            temporary_overrides=args.override,
            validate_paths=False,
        )
        bootstrap = bootstrap_development_fixed_test_set(
            bootstrap_config,
            project_root=Path(__file__).resolve().parent,
        )
        if bootstrap["status"] in {"installed", "already_installed"}:
            print(
                "[Bootstrap] development_fixed_test="
                f"{bootstrap['status']} path={bootstrap['output_path']}"
            )
        resolved_config, provenance = load_project_config(
            config_path,
            environment_path=args.environment,
            cli_overrides=cli_overrides,
            temporary_overrides=args.override,
            validate_paths=True,
        )
        storage_paths = configure_runtime_storage(resolved_config)
        _apply_resolved_configuration(
            args,
            resolved_config,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "[Storage] output_root="
        f"{storage_paths['output_root']} temp_dir={storage_paths['temp_dir']}"
    )
    for name in (
        "num_iters",
        "export_interval",
        "max_cycle",
        "finetune_epoch",
        "finetune_batch",
        "lora_rank",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1")
    if args.disk_warning_threshold < 0:
        parser.error("--disk_warning_threshold cannot be negative")
    if args.mode == "data_flywheel" and args.use_helm == "yes":
        parser.error("data_flywheel requires a local non-HELM test taker")
    if args.preflight_only:
        try:
            return _run_preflight(resolved_config, storage_paths)
        except Exception as exc:
            print(
                "[Preflight] status=failed error="
                f"{type(exc).__name__}: {exc}"
            )
            return 2

    output_root = _output_root(args.outfile_prefix1)
    os.makedirs(output_root, exist_ok=True)
    resume_enabled = bool(resolved_config["experiment"]["resume"])
    run_id = args.run_id or (
        f"{resolved_config['experiment']['name']}-"
        f"{provenance['config_hash'][:8]}"
    )
    if not resume_enabled and args.run_id is None:
        run_id += "-" + time.strftime("%Y%m%dT%H%M%S", time.localtime())
    research_run = ResearchRun(
        resolved_config,
        provenance,
        run_id,
        project_root=Path(__file__).resolve().parent,
        legacy_output_root=(None if config_explicit else output_root),
        resume=resume_enabled,
    )
    # run_id deterministically owns one directory. A resumed invocation reuses
    # its cycle record, hard pool, ledger, checkpoints, and cached iterations.
    # Study Runner supplies an isolated absolute legacy prefix. Joining an
    # absolute path would discard run_dir and split resumable state across two
    # directories, so only its file name is used inside the bound run.
    args.outfile_prefix1 = _bound_outfile_prefix(
        research_run.run_dir,
        resolved_config["paths"]["outfile_prefix"],
    )
    output_root = _output_root(args.outfile_prefix1)
    args.research_run = research_run
    serializable_args = {
        key: value
        for key, value in vars(args).items()
        if key != "research_run"
    }
    research_run.initialize(serializable_args)
    research_run.logger.event(
        "INFO",
        "Startup",
        "run_initialized",
        f"config_hash={research_run.config_hash}",
    )
    cycle_record_path = _cycle_record_file(args)
    try:
        agent_info, evaluator_info = _load_agent_and_evaluator(args)
    except Exception as exc:
        failure_record = {
            "schema_version": 1,
            "status": "failed",
            "created_at": _utc_timestamp(),
            "updated_at": _utc_timestamp(),
            "active_test_taker_model": args.test_taker_modelname,
            "failure_type": _failure_type("agent_model_loading", exc),
            "failed_stage": "agent_model_loading",
            "error": _sanitize_error(exc),
            "cycles": [],
        }
        dump_standard_json(failure_record, cycle_record_path)
        research_run.finalize(
            "failed",
            {
                "stage": "agent_model_loading",
                "failure_type": failure_record["failure_type"],
            },
        )
        print(f"[Startup] model loading failed: {failure_record['error']}")
        return 1

    if args.exp_mode == "autobencher":
        return _run_autobencher(args, agent_info, evaluator_info)
    if args.exp_mode == "python_solve":
        return _run_python_solve(args, agent_info)
    parser.error(f"Unsupported --exp_mode: {args.exp_mode}")


if __name__ == "__main__":
    raise SystemExit(main())
