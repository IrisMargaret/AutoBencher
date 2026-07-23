# AutoBencher

This checkout is configured for the OpenAI-compatible DeepSeek API on Windows,
macOS, and Linux. Python 3.10+ is supported (the configured environment uses
Python 3.13).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` reproduces the tested environment exactly;
`requirements.txt` contains the looser direct dependencies for future upgrades.

Create `.env` (it is ignored by Git):

```dotenv
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Verify the API and all project imports with one low-cost request:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

## Run benchmarks

```powershell
.\.venv\Scripts\python.exe run_scripts.py wiki
.\.venv\Scripts\python.exe run_scripts.py multilingual
.\.venv\Scripts\python.exe run_scripts.py math
```

Use `--num-iters 1` for a shorter run. Full runs generate hundreds of questions
and can consume substantial API tokens. Output files are cached, so an
interrupted command can be rerun without repeating completed steps.

To use a different DeepSeek model:

```powershell
.\.venv\Scripts\python.exe run_scripts.py math --model deepseek-v4-pro --num-iters 1
```

Local Hugging Face models remain supported, but require optional GPU packages:
`torch`, `transformers`, and `accelerate`. HELM and Anthropic paths similarly
require their optional SDKs and API credentials.
