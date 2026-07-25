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

- Math results are written under `math_v5/`.
- Wiki results are written under `KI/`.
- Multilingual results are written under `multilingual/`.
- Completed stages are cached. Rerun the same command to resume an interrupted
  benchmark without repeating completed work.
- Partially written inference caches are validated record by record. A rerun
  continues at the first missing question instead of treating the partial file
  as complete.
- Full runs generate hundreds of questions and may take a long time or consume
  substantial API tokens.

Model output is not always perfectly formatted. JSON-producing math stages
retry invalid output up to three times and only save a cache after validation.
Rejected responses are kept as `*.attemptN.txt` for diagnosis.
Math plans are limited to the five requested subcategories so an oversized
model response cannot unexpectedly multiply the benchmark size.

## Troubleshooting

### `DEEPSEEK_API_KEY is not configured`

Confirm that `.env` is in the project root and that its variable name is
exactly `DEEPSEEK_API_KEY`. Then rerun `python smoke_test.py`.

### `Model response did not contain a JSON block`

Rerun the same benchmark command. Existing completed files will be reused, and
the failed JSON stage will retry automatically. Inspect the related
`*.attemptN.txt` file if all retries fail.

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
