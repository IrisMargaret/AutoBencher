import glob
import random

import requests
import copy
import re, time
import os, argparse, ast, json, tqdm
from pydantic import BaseModel, Extra, root_validator
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from time import sleep
from collections import defaultdict
import numpy as np

from util import gen_from_prompt, load_model, process_args_for_models, helm_process_args
from tool_util import (
    DEFAULT_SYSTEM_MESSAGE,
    HardSamplePool,
    MATH_CATEGORIES,
    SUB_CATEGORY_TAXONOMY,
    canonicalize_math_record,
    classify_error_tags,
    clean_redundant_files,
    dump_standard_json,
    execute_code,
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


# 【新增】基于既有判分缓存计算飞轮触发条件，不改变原始结果结构。
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
    # 【修改】所有数学 JSON 统一由标准导出函数写入。
    dump_standard_json(data, output_file)


# 【新增】重试原文只进入 temp_log，迭代成功后由清理函数移除。
def _write_attempt_log(outfile_prefix, stage, attempt, response):
    iteration_dir = os.path.dirname(os.path.abspath(outfile_prefix))
    temp_log_dir = os.path.join(iteration_dir, "temp_log")
    os.makedirs(temp_log_dir, exist_ok=True)
    filename = (
        f"{os.path.basename(outfile_prefix)}.{stage}.attempt{attempt}.txt"
    )
    with open(os.path.join(temp_log_dir, filename), "w", encoding="utf-8") as f:
        f.write(response)


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
    # 【修改】九大一级类 + 固定英文 sub_category 双层规划约束。
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
    # 【新增】每轮均加载全局困难样本池；第 3 轮起满足触发条件时注入错题上下文。
    hard_pool_file = hard_pool_file or os.path.join(
        os.path.dirname(os.path.abspath(outfile_prefix)), "hard_pool.json"
    )
    hard_pool = HardSamplePool(hard_pool_file)
    if enable_hard_sample_guidance:
        hard_sample_context = hard_pool.get_history_context(max_samples=None)
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

Complete historical hard-sample context:
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



def _generate_question_from_description(description_json, agent_lm, agent_tokenizer, agent_client, outfile_prefix='att1',
                                        questions_old=None):
    context = """Your goal is to come up with math questions that match the description. 
In each iteration, you receive a sub_category that describes the exact type of question to ask.
Return exactly 50 math questions matching that sub_category as one JSON array.
Each question must contain these keys: id, question, answer, difficulty.
You do not need to answer the questions. 

Note: do not come up with repetitive questions. If you have asked a question, do not ask it again! 
Come up with exactly 50 concrete questions, and write them in the following format.
Do not leave place holders or ellipsis!!!
It's helpful to first come up with a plan for this iteration, and then write the questions.
The response must follow this format:
[
  {"id": 1, "question": "What is 5 + 3?", "answer": "8", "difficulty": 1},
  {"id": 2, "question": "What is 5 - 3?", "answer": "2", "difficulty": 1}
]
Do not use a code block. Do not write any explanation before or after the JSON array.
Do not use placeholders or ellipses.
"""

    sub_category = description_json.get(
        "sub_category", description_json.get("subcategory_description", "")
    )
    context = (
        DEFAULT_JSON_MESSAGE
        + context
        + f"\nTop-level category: {description_json['category']}"
        + f"\nSub_category: {sub_category}"
        + "\nUse English text only."
    )
    if questions_old:
        old_q_string = ''
        for q in questions_old:
            old_q_string += str(q) + '\n'
        remaining = max(1, 50 - len(questions_old))
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
            required_keys = {"id", "question", "answer"}
            if any(
                not isinstance(item, dict) or not required_keys.issubset(item)
                for item in questions
            ):
                raise ValueError(
                    "Each generated question must contain id, question, and answer"
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
        line['category'] = description_json['category']
        line['sub_category'] = sub_category
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
):
    agent_lm, agent_tokenizer, agent_client = agent_info
    plan_outfile = f"{outfile_prefix}.question_plan_with_aim.json"
    if not os.path.exists(plan_outfile):
        # 【修改】仅在触发轮次向原规划函数传入困难样本与覆盖率约束。
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
    plan_json = plan_json[0]

    # 【修改】题目分片只在内存中存在，唯一逐题持久源由 inference 文件承担。
    question_json_full = []
    for idx, plan_line in enumerate(plan_json):
        outfile_prefix2 = outfile_prefix + '.subcat{}'.format(idx)
        question_json = _generate_question_from_description(
            plan_line,
            agent_lm,
            agent_tokenizer,
            agent_client,
            outfile_prefix2,
        )
        if len(question_json) == 1:
            question_json = question_json[0]

        while len(question_json) < 50:
            question_json_new = _generate_question_from_description(plan_line, agent_lm, agent_tokenizer,
                                                                    agent_client, outfile_prefix2,
                                                                    questions_old=question_json)
            question_json_new = question_json_new[0]
            question_json.extend(question_json_new)
        question_json_full.extend(question_json[:50])
    return question_json_full, plan_json



def _build_compare_summary(iter_number, inference_records):
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
    return {
        "iter_number": int(iter_number),
        "total_questions": total_questions,
        "global_accuracy": (
            total_correct / total_questions if total_questions else 0.0
        ),
        "category_statistics": category_statistics,
    }


def _lowest_sub_categories(compare_summary, limit=10):
    return sorted(
        compare_summary.get("category_statistics", []),
        key=lambda item: (item["accuracy"], -item["total_count"], item["sub_category"]),
    )[:limit]


# 【修改】保留原判分调用，但只持久化标准 inference 明细和 compare 统计对象。
def test_and_eval(
    question_json,
    outfile_prefix,
    test_taker_info,
    agent_info,
    tool_info,
    gold_ans_key='answer',
    iter_number=0,
    temp_log_dir=None,
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
    test_taker_output = generate_math_inference(
        question_json,
        test_taker_info,
        inference_file,
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
    _, judgments = fast_compare_answers(
        gold_records,
        test_taker_output,
        tool_info,
        outfile_prefix=judge_prefix,
        gold_ans_key="answer",
    )
    if len(judgments) != len(test_taker_output):
        raise RuntimeError("Judgement count does not match inference count")

    standardized_records = []
    for index, (record, judgment) in enumerate(
        zip(test_taker_output, judgments)
    ):
        standardized = canonicalize_math_record(record, index)
        standardized["is_correct"] = str(
            judgment.get("is_correct", "")
        ).strip().lower() == "true"
        standardized["error_tags"] = (
            []
            if standardized["is_correct"]
            else classify_error_tags({**standardized, **judgment})
        )
        standardized_records.append(standardized)
    dump_standard_json(standardized_records, inference_file)

    compare_summary = _build_compare_summary(iter_number, standardized_records)
    dump_standard_json(compare_summary, compare_file)
    return standardized_records


# 【新增】保持原 outfile_prefix1 入参，内部映射到标准 iter_N 目录。
def _build_iteration_paths(outfile_prefix1, iter_number):
    raw_prefix = os.fspath(outfile_prefix1)
    output_root = os.path.abspath(os.path.dirname(raw_prefix) or ".")
    raw_base_name = os.path.basename(raw_prefix)
    base_name = raw_base_name.rstrip(".") or "math"
    iteration_dir = os.path.join(output_root, f"iter_{iter_number}")
    os.makedirs(os.path.join(iteration_dir, "temp_log"), exist_ok=True)
    iteration_prefix = os.path.join(
        iteration_dir, f"{base_name}.{iter_number}"
    )
    legacy_prefix = os.path.join(
        output_root, f"{raw_base_name}{iter_number}"
    )
    return {
        "output_root": output_root,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')

    # parser.add_argument('--model', default='gpt-3.5-turbo')  # option that takes a value
    parser.add_argument('--test_taker_modelname', default='gpt-3.5-turbo')  # option that takes a value
    parser.add_argument('--test_taker_modelname2', default=None)  # option that takes a value
    parser.add_argument('--agent_modelname', default='gpt-4-turbo-preview')  # option that takes a value
    parser.add_argument('--tool_modelname', default=None)  # option that takes a value
    parser.add_argument('--temperature', type=float, default=0.001)  # option that takes a value
    parser.add_argument('--pairwise', type=str, default='no')  # option that takes a value
    parser.add_argument('--exp_mode', type=str, default='ki_wiki')  # option that takes a value
    parser.add_argument('--theme', type=str, default='history')  # option that takes a value
    parser.add_argument('--use_helm', type=str, default='yes')  # option that takes a value
    parser.add_argument('--top_p', type=float, default=0.9)  # option that takes a value
    parser.add_argument('--acc_target', type=str, default="0.3--0.5")  # option that takes a value
    parser.add_argument('--num_iters', type=int, default=8)

    parser.add_argument('--outfile_prefix1', type=str, default='att1')  # option that takes a value

    args2 = parser.parse_args()
    args = copy.deepcopy(args2)

    if args.use_helm == 'yes':
        test_taker_info = helm_process_args(args.test_taker_modelname)
        print('loaded helm models')
    else:
        # load the test taker model.
        test_taker_lm, test_taker_tokenizer, modelpath_name, test_taker_client = process_args_for_models(
            args.test_taker_modelname)
        test_taker_info = (test_taker_lm, test_taker_tokenizer, test_taker_client)

        if args.test_taker_modelname2 is not None:
            test_taker_lm2, test_taker_tokenizer2, modelpath_name2, test_taker_client2 = process_args_for_models(
                args.test_taker_modelname2)
            test_taker_info2 = (test_taker_lm2, test_taker_tokenizer2, test_taker_client2)


    agent_lm, agent_tokenizer, agent_name, agent_client = process_args_for_models(args.agent_modelname)

    if args.tool_modelname is None:
        tool_lm, tool_tokenizer, tool_name, tool_client = agent_lm, agent_tokenizer, agent_name, agent_client
    else:
        tool_lm, tool_tokenizer, tool_name, tool_client = process_args_for_models(args.tool_modelname)

    evaluator_info = (tool_lm, tool_tokenizer, tool_client)
    agent_info = (agent_lm, agent_tokenizer, agent_client) # agent model


    if args.exp_mode == 'autobencher':
        history_dict = []
        for iters in range(args.num_iters):
            iter_number = iters + 1
            paths = _build_iteration_paths(args.outfile_prefix1, iter_number)
            args.outfile_prefix = paths["iteration_prefix"]
            migrated_records = _records_from_legacy_cache(paths, iter_number)
            (
                should_direct_generation,
                flywheel_triggers,
                prior_coverage,
                covered_sub_categories,
                missing_sub_categories,
            ) = _get_math_flywheel_state(history_dict)
            # 【修改】前两轮强制采集基线，第 3 轮起执行双条件触发。
            should_direct_generation = (
                iters >= 2 and should_direct_generation
            )
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
                and cached_compare[0].get("total_questions")
                == len(migrated_records)
            )

            if (
                migrated_records
                and all(
                    record["test_taker_response"]
                    for record in migrated_records
                )
                and has_standard_compare
                and os.path.exists(paths["inference_file"])
            ):
                print("FOUND completed iteration cache", paths["compare_file"])
                json_dict = load_math_inference(paths["inference_file"])
                compare_summary = _build_compare_summary(iter_number, json_dict)
                dump_standard_json(compare_summary, paths["compare_file"])
            else:
                if migrated_records:
                    json_category = migrated_records
                else:
                    summarized_content = summarize_over_history(
                        history_dict, gold_key='gold_answer', verbose=False
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
                    )
                json_dict = test_and_eval(
                    copy.deepcopy(json_category),
                    args.outfile_prefix,
                    test_taker_info,
                    agent_info,
                    evaluator_info,
                    gold_ans_key='answer',
                    iter_number=iter_number,
                    temp_log_dir=paths["temp_log_dir"],
                )
                compare_summary = _build_compare_summary(iter_number, json_dict)
                dump_standard_json(compare_summary, paths["compare_file"])

            history_dict.append(json_dict)
            # 【新增】固定顺序：统计 → hard_pool → meta_summary → 清理。
            new_hard_count, hard_total = manage_hard_pool(
                paths["inference_file"],
                paths["hard_pool_file"],
                iter_number,
            )
            run_config = {
                "agent_model": args.agent_modelname,
                "test_taker_model": args.test_taker_modelname,
                "tool_model": args.tool_modelname or args.agent_modelname,
                "target_accuracy_range": args.acc_target,
                "max_iterations": args.num_iters,
                "output_root": paths["output_root"].replace("\\", "/"),
            }
            update_meta_summary(
                paths["meta_summary_file"],
                run_config,
                [record["sub_category"] for record in json_dict],
                {
                    "iter_num": iter_number,
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
            removed = clean_redundant_files(paths["iteration_dir"])
            removed.extend(_clean_legacy_redundant_files(paths))
            if removed:
                print(
                    f"[MathFlywheel] removed_redundant_files={len(removed)}"
                )

            cumulative_records = [
                record
                for iteration_records in history_dict
                for record in iteration_records
            ]
            current_coverage, _, _ = HardSamplePool.coverage(
                cumulative_records
            )
            weakest = _lowest_sub_categories(compare_summary)
            log_math_iteration_metrics(
                iter_number,
                compare_summary["global_accuracy"],
                current_coverage,
                hard_total,
                should_direct_generation,
                flywheel_triggers,
                weakest,
            )
            print(
                get_summary_of_results(
                    json_dict, gold_key="gold_answer", verbose=False
                )
            )


    elif args.exp_mode == 'python_solve':
        datafile = args.outfile_prefix1
        full_lst = []
        try:
            with open(datafile, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    line['test_taker_response'] = "PLACEHOLDER"
                    full_lst.append(line)
        except:
            with open(datafile, 'r') as f:
                line_lst = json.load(f)
                for line in line_lst:
                    line['test_taker_response'] = "PLACEHOLDER"
                    full_lst.append(line)
        solve_with_python(full_lst, args.outfile_prefix1, agent_info)
