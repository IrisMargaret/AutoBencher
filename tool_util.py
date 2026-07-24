import copy
import re
import requests
import os, argparse, ast, json, tqdm
import subprocess
import sys
from urllib.parse import quote
from time import sleep
from collections import defaultdict
import numpy as np
from util import gen_from_prompt


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



def extract_json_v2(json_text, outfilename):
    response = json_text.replace("TERMINATE", "").strip()
    fenced_blocks = extract_code(response)
    candidates = [
        code for language, code in fenced_blocks if language.lower() == "json"
    ]
    candidates.extend(
        code
        for language, code in fenced_blocks
        if language.lower() != "json" and code.startswith(("[", "{"))
    )
    if response.startswith(("[", "{")):
        candidates.append(response)

    # Some models add a short explanation even when asked for JSON only. Find
    # the first decodable array/object instead of failing the entire benchmark.
    if not candidates:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", response):
            try:
                _, end = decoder.raw_decode(response[match.start():])
            except json.JSONDecodeError:
                continue
            candidates.append(response[match.start():match.start() + end])
            break

    # Keep the original order while avoiding duplicate parsing of fenced JSON.
    candidates = list(dict.fromkeys(candidate.strip() for candidate in candidates))
    if not candidates:
        raise ValueError("Model response did not contain a JSON block")

    parsed_blocks = []
    errors = []
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as json_exc:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError) as ast_exc:
                errors.append(f"{json_exc}; {ast_exc}")
                continue
        if parsed not in (None, [], {}):
            parsed_blocks.append(parsed)

    if not parsed_blocks:
        detail = errors[-1] if errors else "the JSON value was empty"
        raise ValueError(f"Model returned no usable JSON: {detail}")
    if outfilename is not None:
        temporary_outfile = f"{outfilename}.tmp"
        with open(temporary_outfile, "w", encoding="utf-8") as f:
            json.dump(parsed_blocks, f, ensure_ascii=False)
        os.replace(temporary_outfile, outfilename)
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
