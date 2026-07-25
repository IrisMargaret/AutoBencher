# AutoBencher

[简体中文](README.zh-CN.md) | English

AutoBencher automatically builds and evaluates benchmark questions. This
checkout is configured for the OpenAI-compatible DeepSeek API and supports
Windows, macOS, and Linux with Python 3.10 or later.

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local activation scripts, run this once in the current
terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows Command Prompt

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, `python` and `pip` below refer to the virtual environment.

## 2. Install dependencies

For the tested, reproducible environment:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` contains exact tested versions. Use
`requirements.txt` only when you intentionally want compatible newer package
versions:

```shell
python -m pip install -r requirements.txt
```

## 3. Configure the API

Create a `.env` file in the project root:

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Do not commit `.env` or share your API key.

## 4. Verify the installation

The smoke test checks project imports, JSON parsing, API configuration, and one
small API request:

```shell
python smoke_test.py
```

Run the offline regression tests separately:

```shell
python -m unittest discover -s tests -v
```

## 5. Run a benchmark

With the virtual environment activated, the commands are:

```shell
python run_scripts.py math
python run_scripts.py wiki
python run_scripts.py multilingual
```

Start with one iteration to verify the complete workflow at lower cost:

```shell
python run_scripts.py math --num-iters 1
```

Select another model if the API account exposes it:

```shell
python run_scripts.py math --model deepseek-v4-pro --num-iters 1
```

The agent and test-taker can use different providers. Ollama model tags (model
names containing a tag such as `qwen2.5:7b-instruct`) are automatically sent
to Ollama's native local API:

```powershell
python run_scripts.py math `
  --agent_modelname deepseek-v4-pro `
  --test_taker_modelname qwen2.5:7b-instruct `
  --exp_mode autobencher `
  --use_helm no `
  --num_iters 2 `
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. `
  --acc_target 0.1--0.3
```

The default Ollama endpoint is `http://localhost:11434`. Before the first
question, AutoBencher preloads the selected model and keeps it loaded for 30
minutes. Transient model-runner and HTTP 502 failures are retried up to 10
times. This mode does not require PyTorch, Transformers, or Accelerate.

These optional `.env` settings customize Ollama behavior:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_RETRIES=10
OLLAMA_RETRY_DELAY_SECONDS=5
OLLAMA_REQUEST_TIMEOUT=300
```

Activation is optional if you call the environment's interpreter directly. On
Windows, for example:

```powershell
.\.venv\Scripts\python.exe run_scripts.py math
```

## Output, cache, and resuming

- Math results are written under `math_v5/` using one directory per iteration.
- Wiki results are written under `KI/`.
- Multilingual results are written under `multilingual/`.
- Completed stages are cached. Rerun the same command to resume an interrupted
  benchmark without repeating completed work.
- Before the first test-taker request, the math branch writes the complete
  question set to its inference file. Each completed batch is saved atomically,
  so the same inference file is sufficient to resume an interrupted run.
- Full runs generate hundreds of questions and may take a long time or consume
  substantial API tokens.

### Math hierarchical benchmark

The math branch uses nine fixed top-level categories:

```text
Arithmetic
Algebra
Geometry & Trigonometry
Probability & Statistics
Word Problems
Number Theory
Calculus
Linear Algebra
Composite Comprehensive
```

Every question also has an independently measured `sub_category`. The first
two iterations collect a broad baseline. Starting with iteration three,
hard-sample-directed generation is enabled when either of these conditions is
true:

- both previous global accuracies are at least `0.7`;
- fixed-taxonomy `sub_category` coverage is below `60%`.

Wrong answers are tagged only with this fixed vocabulary:

```text
calculation_error
formula_memory_error
condition_missing
multi-step_logic_error
concept_confusion
```

### Math output layout

For the default launcher prefix, the structure is:

```text
math_v5/
├── meta_summary.json
├── hard_pool.json
├── iter_1/
│   ├── temp_log/
│   ├── <prefix>.1.question_plan_with_aim.json
│   ├── <prefix>.1.test_taker_inference.json
│   └── <prefix>.1.compare_answers.json
├── iter_2/
└── iter_N/
```

All persistent math JSON uses UTF-8, English keys and values, and two-space
indentation. The three iteration files are retained permanently:

- `question_plan_with_aim.json` records the nine-category Meta Agent plan.
- `test_taker_inference.json` is the only question-level source and contains
  questions, gold answers, responses, correctness, error tags, and unique keys.
- `compare_answers.json` contains global accuracy plus independent statistics
  for every `(category, sub_category)` pair.

`hard_pool.json` incrementally deduplicates wrong answers by `unique_key`.
`meta_summary.json` records run configuration, all observed subcategories, and
the iteration file index.

Question fragments, `all_questions.json`, retry text, temporary judgement
files, empty JSON, and invalid JSON are removed at the end of each completed
iteration. The `temp_log/` directory remains but is emptied.

Legacy flat math caches and JSONL inference files are detected and migrated to
the iteration layout. Wiki and multilingual output formats are unchanged.

Model output is not always perfectly formatted. Math JSON parsing accepts
fenced JSON, raw JSON surrounded by explanation text, `<json>` blocks, smart
quotes, unquoted English keys, and trailing commas. Invalid generations retry
up to three times. Retry responses exist only under `temp_log/` while the
iteration is running and are removed after success.

## Troubleshooting

### `DEEPSEEK_API_KEY is not configured`

Confirm that `.env` is in the project root and that its variable name is
exactly `DEEPSEEK_API_KEY`. Then rerun `python smoke_test.py`.

### `Model response did not contain a JSON block`

Rerun the same benchmark command. Existing completed files will be reused, and
the failed JSON stage will retry automatically. During a failed active run,
inspect `iter_N/temp_log/`; successful iterations clear this directory.

### Gold-answer and inference counts differ

Rerun the same benchmark command. The inference cache is checked against the
current questions, any invalid suffix is discarded, and generation resumes
from the first missing record.

### Ollama returns HTTP 502

AutoBencher now preloads Ollama models through `/api/generate`, uses the native
`/api/chat` endpoint, keeps the model resident, and retries transient failures.
Rerun the exact same command after updating; generated questions and completed
inference records are reused.

Confirm that Ollama is running and the requested model is installed:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If all retries still fail, free enough RAM/VRAM for the selected model or use a
smaller Ollama model. The final error now includes Ollama's response detail and
the endpoint that was used.

### PowerShell cannot activate `.venv`

Use the process-scoped execution-policy command in section 1, or skip
activation and run `.\.venv\Scripts\python.exe` directly.

### Local Hugging Face models or HELM

The default install is intentionally API-focused. Local models additionally
need `torch`, `transformers`, and `accelerate`. HELM and Anthropic modes require
their optional SDKs and credentials.

## Stop the environment

```shell
deactivate
```
