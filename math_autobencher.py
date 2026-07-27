import glob
import gc
import random
import sys
import contextlib
from pathlib import Path

import requests
import copy
import re, time
import os, argparse, ast, json, tqdm
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np

from autobencher.config import (
    ConfigurationError,
    load_resolved_config,
    str2bool,
)
from autobencher.coverage import coverage_metrics, generation_schedule
from autobencher.dataset import (
    build_training_dataset,
    normalize_question_text,
    write_alpaca_jsonl,
)
from autobencher.experiment import (
    ResearchRun,
    atomic_json,
    utc_now,
)
from autobencher.structured import (
    answers_equivalent,
    attribute_error,
    normalize_answer_type,
    validate_generated_question,
)
from util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
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
    normalize_math_category,
    normalize_sub_category,
    read_json_records,
    search_related_pages,
    search_step,
    get_pageviews,
    update_meta_summary,
)
from run_scripts import log_math_iteration_metrics
from wiki_autobencher import fast_compare_answers

DEFAULT_JSON_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your reasoning and language skills.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
Reply "TERMINATE" in the end when everything is done.
"""


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



def _generate_question_from_description(
    description_json,
    agent_lm,
    agent_tokenizer,
    agent_client,
    outfile_prefix='att1',
    questions_old=None,
    hard_sample_context="",
    question_count=50,
):
    question_count = int(question_count)
    context = f"""Your goal is to generate exactly {question_count} math questions
that match one assigned category and subcategory.
In each iteration, you receive a sub_category that describes the exact type of question to ask.
Return one JSON array. Every object must satisfy the generated-question schema:
[
  {{
    "question_id": "q_1",
    "category": "Arithmetic",
    "subcategory": "Integer Operations",
    "difficulty": 1,
    "question": "What is 5 + 3?",
    "answer_type": "integer",
    "canonical_answer": "8",
    "display_answer": "8",
    "unit": null,
    "tolerance": null,
    "order_sensitive": false,
    "generation_source": "coverage_deficit",
    "reference_hard_sample_ids": [],
    "target_error_type": null,
    "generation_strategy": "quota_repair"
  }}
]

Mandatory quality and output rules:
1. Output only the JSON array. Do not output Markdown, prompts, system text,
   explanations, role prefixes, placeholders, or ellipses.
2. Every problem must match the assigned category and subcategory.
3. Include all necessary conditions and ensure the answer is unique unless the
   answer schema explicitly represents multiple solutions.
4. Do not generate subjective or unverifiable questions.
5. Do not copy historical questions. Variants must change values, wording, or
   mathematical structure.
6. Ensure canonical_answer can be independently verified.
7. Use only English text.
"""

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
                                         terminate_by_linebreak='no', )
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
                    r"[\u3400-\u9fff]",
                    f"{item.get('question', '')} {item.get('answer', '')}",
                )
                for item in questions
            ):
                raise ValueError(
                    "Generated questions and answers must use English text only"
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
        line["answer_type"] = normalize_answer_type(raw_answer_type)
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
        line["target_error_type"] = description_json.get("target_error_type")
        line["generation_strategy"] = description_json.get(
            "generation_strategy",
            "quota_repair",
        )
    return extracted_json

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
            is_hard_variant = (
                plan_line.get("generation_source") == "hard_pool_variant"
            )
            variant_context = (
                hard_pool.get_variant_context(
                    plan_line["category"],
                    plan_line["sub_category"],
                )
                if enable_hard_sample_guidance and is_hard_variant and hard_pool
                else ""
            )
            if is_hard_variant and hard_pool:
                references = [
                    sample.get("unique_key", "")[:16]
                    for sample in hard_pool.samples
                    if sample.get("sample_grade") == "train_eligible"
                    and sample.get("category") == plan_line["category"]
                    and sample.get("sub_category") == plan_line["sub_category"]
                ][:12]
                plan_line["reference_hard_sample_ids"] = references
            target_count = int(plan_line.get("question_count", 50))
            question_json = _generate_question_from_description(
                plan_line,
                agent_lm,
                agent_tokenizer,
                agent_client,
                outfile_prefix2,
                hard_sample_context=variant_context,
                question_count=target_count,
            )
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

            repair_round = 0
            while len(question_json) < target_count:
                repair_round += 1
                if repair_round > 3:
                    raise RuntimeError(
                        f"Generation quota repair failed for {plan_line['sub_category']}"
                    )
                question_json_new = _generate_question_from_description(
                    plan_line,
                    agent_lm,
                    agent_tokenizer,
                    agent_client,
                    outfile_prefix2,
                    questions_old=question_json,
                    hard_sample_context=variant_context,
                    question_count=target_count,
                )
                question_json_new = question_json_new[0]
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
            if progress is not None:
                progress.update(len(accepted))
    if generation_plan is not None:
        expected = int(generation_plan["question_budget"])
        if len(question_json_full) != expected:
            raise RuntimeError(
                f"Generated question count mismatch: {len(question_json_full)}/{expected}"
            )
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
    for record in inference_records:
        grouped[(record["category"], record["sub_category"])].append(record)
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
            evaluator_cache_exists = os.path.exists(
                f"{judge_prefix}.compare_answers.json"
            )
            _, judgments = fast_compare_answers(
                gold_records,
                test_taker_output,
                tool_info,
                outfile_prefix=judge_prefix,
                gold_ans_key="answer",
            )
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
    for index, (record, judgment) in enumerate(
        zip(test_taker_output, judgments)
    ):
        standardized = canonicalize_math_record(record, index)
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
            )
            status = parse_result["parse_status"]
            if status != "success":
                evaluation_status = status
                standardized["is_correct"] = False
            else:
                standardized["is_correct"] = bool(equivalence["equivalent"])
                evaluation_status = equivalence["status"]
                if (
                    standardized["is_correct"]
                    and not evaluator_is_correct
                ):
                    evaluation_status = "format_only_error"
                    standardized["format_only_error"] = True
            attribution = attribute_error(
                standardized,
                parse_result,
                equivalence,
                research_config,
            )
            evaluator_tool_calls = []
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
                    "evaluator_confidence": float(
                        judgment.get("confidence", 1.0)
                    ),
                    "answer_validation_success": bool(
                        equivalence["gold_normalized"]["success"]
                    ),
                    "question_parse_success": True,
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
        os.path.join(output_root, f"cycle_{cycle_number}")
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
    message = re.sub(r"[\u3400-\u9fff]+", " ", str(exc))
    return re.sub(r"\s+", " ", message).strip()[:4000]


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
        research_run.run_dir
        / f"cycle_{cycle_entry['cycle']}"
        / "cycle_manifest.json",
    )


def _load_completed_cycle_history(output_root, cycles):
    history = []
    root = Path(output_root)
    for cycle in sorted(cycles, key=lambda item: int(item.get("cycle", 0))):
        if cycle.get("status") not in {
            "completed",
            "completed_without_training",
        }:
            continue
        cycle_number = int(cycle.get("cycle", 0))
        for iteration_number in range(
            1,
            int(cycle.get("iterations_completed", 0)) + 1,
        ):
            iteration_dir = (
                root / f"cycle_{cycle_number}" / f"iter_{iteration_number}"
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
        )
        generation_plan["cycle"] = cycle_number
        should_direct_generation = bool(
            generation_plan["hard_pool_injection_enabled"]
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
            )
            if research_run:
                research_run.logger.event(
                    "INFO",
                    "Generate",
                    "stage_completed",
                    f"completed={len(json_category)}",
                    cycle=cycle_number,
                    iteration=iter_number,
                )
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
                    "prompt_hash": research_run.provenance["config_hash"],
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
        }
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
    if args.clean_cycle_cache:
        removed = clean_redundant_files(paths["iteration_dir"])
        if args.mode == "eval":
            removed.extend(_clean_legacy_redundant_files(paths))
        if removed:
            print(f"[MathFlywheel] removed_redundant_files={len(removed)}")

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
    print(
        get_summary_of_results(
            json_dict,
            gold_key="gold_answer",
            verbose=False,
        )
    )
    return {
        "paths": paths,
        "global_iter_number": global_iter_number,
        "global_accuracy": compare_summary["global_accuracy"],
        "hard_pool_total": hard_total,
    }


# [ADDED] Execute eval or the complete evaluation-training flywheel.
def _run_autobencher(args, agent_info, evaluator_info):
    output_root = _output_root(args.outfile_prefix1)
    os.makedirs(output_root, exist_ok=True)
    cycle_record_path = _cycle_record_file(args)
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
        prior_cycle = cycles_by_number.get(cycle_number)
        if prior_cycle and prior_cycle.get("status") in {
            "completed",
            "completed_without_training",
        }:
            current_test_taker_model = prior_cycle.get(
                "next_test_taker_model",
                current_test_taker_model,
            )
            print(f"[Cycle] resume_skip={cycle_number} status={prior_cycle['status']}")
            continue

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
        test_taker_info = None
        stage = "test_taker_model_loading"
        try:
            test_taker_info = _load_test_taker_info(
                current_test_taker_model,
                args.use_helm,
            )
            history_dict = run_history
            stage = "evaluation"
            last_iteration = None
            for iter_number in range(1, args.num_iters + 1):
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
                cycle_entry["iterations_completed"] = iter_number
                cycle_entry["last_global_accuracy"] = last_iteration[
                    "global_accuracy"
                ]
                _save_cycle_record(cycle_record_path, cycle_record)

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
                    args.research_run.run_dir
                    / f"cycle_{cycle_number}"
                    / "training"
                )
                training_dir.mkdir(parents=True, exist_ok=True)
                candidates = [
                    record
                    for iteration_records in history_dict
                    for record in iteration_records
                ]
                selected, dataset_manifest, rejected = build_training_dataset(
                    candidates,
                    args.research_run.config,
                    seed=int(args.research_run.config["experiment"]["seed"])
                    + cycle_number,
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
                    },
                    training_dir / "finetune_config.json",
                )
                args.research_run.logger.event(
                    "INFO",
                    "BuildDataset",
                    "stage_completed",
                    f"selected={exported_count} rejected={len(rejected)}",
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
            cycle_entry["training_export"] = _relative_json_path(
                export_path,
                output_root,
            )
            cycle_entry["training_sample_count"] = exported_count
            _save_cycle_record(cycle_record_path, cycle_record)
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
            result = call_local_finetune(
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
            )
            if not result["success"]:
                raise RuntimeError(
                    result.get("error")
                    or f"Fine-tuning exited with code {result['returncode']}"
                )
            current_test_taker_model = os.path.abspath(finetune_output)
            if getattr(args, "research_run", None):
                training_dir = (
                    args.research_run.run_dir
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
                    },
                    training_dir / "finetune_summary.json",
                )
                atomic_json(
                    {
                        **args.research_run.metadata(),
                        "cycle_id": cycle_number,
                        "status": "completed",
                        "merged_model_path": current_test_taker_model.replace(
                            "\\",
                            "/",
                        ),
                    },
                    training_dir / "checkpoint_manifest.json",
                )
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
        except (Exception, KeyboardInterrupt) as exc:
            cycle_entry["status"] = "failed"
            cycle_entry["failed_stage"] = stage
            cycle_entry["failure_type"] = _failure_type(stage, exc)
            cycle_entry["error"] = _sanitize_error(exc)
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
        all_iteration_summaries = []
        for path in args.research_run.run_dir.glob(
            "cycle_*/iter_*/iteration_summary.json"
        ):
            records = read_json_records(path)
            if records:
                all_iteration_summaries.append(records[0].get("data", {}))
        atomic_json(
            {
                **args.research_run.metadata(),
                "status": "completed",
                "cycle_count": cycle_limit,
                "iteration_count": len(all_iteration_summaries),
                "total_questions": sum(
                    int(item.get("question_count", 0))
                    for item in all_iteration_summaries
                ),
                "hard_pool_size": len(
                    read_json_records(os.path.join(output_root, "hard_pool.json"))
                ),
                "active_test_taker_model": current_test_taker_model,
            },
            args.research_run.run_dir / "experiment_summary.json",
        )
        args.research_run.finalize(
            "completed",
            {
                "cycle_count": cycle_limit,
                "iteration_count": len(all_iteration_summaries),
                "active_test_taker_model": current_test_taker_model,
            },
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
    parser.add_argument("--outfile_prefix1", type=str, default="att1")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML configuration profile; existing CLI options override it.",
    )
    parser.add_argument("--run_id", type=str, default=None)
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
    definitions = (
        (("agent_modelname",), "agent_modelname", "models.evaluator.model_name"),
        (
            ("test_taker_modelname",),
            "test_taker_modelname",
            "models.test_taker.model_path",
        ),
        (("exp_mode",), "exp_mode", "experiment.exp_mode"),
        (("num_iters",), "num_iters", "experiment.num_iterations"),
        (("mode",), "mode", "experiment.mode"),
        (("export_interval",), "export_interval", "experiment.export_interval"),
        (("max_cycle",), "max_cycle", "experiment.max_cycles"),
        (
            ("clean_cycle_cache",),
            "clean_cycle_cache",
            "experiment.clean_cycle_cache",
        ),
        (("finetune_gpu",), "finetune_gpu", "finetune.gpu"),
        (("finetune_epoch",), "finetune_epoch", "finetune.epochs"),
        (("finetune_batch",), "finetune_batch", "finetune.batch_size"),
        (("lora_rank",), "lora_rank", "finetune.lora_rank"),
        (
            ("new_local_model_suffix",),
            "new_local_model_suffix",
            "finetune.new_local_model_suffix",
        ),
    )
    overlay = {}
    for option_names, attribute, path in definitions:
        explicit = any(
            _cli_option_present(
                f"--{name}",
                f"--{name.replace('_', '-')}",
            )
            for name in option_names
        )
        if explicit or not config_explicit:
            _set_nested(overlay, path, getattr(args, attribute))
    if _cli_option_present("--acc_target", "--acc-target") or not config_explicit:
        pieces = [
            piece
            for piece in re.split(r"\s*(?:--|,)\s*", args.acc_target)
            if piece
        ]
        if len(pieces) != 2:
            raise ConfigurationError(
                "acc_target",
                "expected low,high or low--high",
                args.acc_target,
            )
        low, high = map(float, pieces)
        _set_nested(
            overlay,
            "adaptive_sampling.target_accuracy_low",
            low,
        )
        _set_nested(
            overlay,
            "adaptive_sampling.target_accuracy_high",
            high,
        )
        _set_nested(
            overlay,
            "adaptive_sampling.target_accuracy_mid",
            (low + high) / 2,
        )
    if _cli_option_present("--outfile_prefix1", "--outfile-prefix1") or not config_explicit:
        raw_prefix = os.path.abspath(args.outfile_prefix1)
        _set_nested(
            overlay,
            "paths.output_root",
            os.path.dirname(raw_prefix) or os.getcwd(),
        )
        _set_nested(
            overlay,
            "paths.outfile_prefix",
            os.path.basename(raw_prefix).rstrip(".") or "math",
        )
    if args.resume is not None:
        _set_nested(overlay, "experiment.resume", args.resume)
    return overlay


def _apply_resolved_configuration(args, config, config_explicit):
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
    args.acc_target = (
        f"{config['adaptive_sampling']['target_accuracy_low']},"
        f"{config['adaptive_sampling']['target_accuracy_high']}"
    )
    if config_explicit and not _cli_option_present(
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
        resolved_config, provenance = load_resolved_config(
            config_path,
            cli_overrides=cli_overrides,
            temporary_overrides=args.override,
            validate_paths=True,
        )
        _apply_resolved_configuration(
            args,
            resolved_config,
            config_explicit,
        )
    except ConfigurationError as exc:
        parser.error(str(exc))
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
    )
    if config_explicit and not _cli_option_present(
        "--outfile_prefix1",
        "--outfile-prefix1",
    ):
        prefix = str(resolved_config["paths"]["outfile_prefix"]).rstrip(".")
        args.outfile_prefix1 = str(research_run.run_dir / f"{prefix}.")
        output_root = _output_root(args.outfile_prefix1)
        os.makedirs(output_root, exist_ok=True)
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
