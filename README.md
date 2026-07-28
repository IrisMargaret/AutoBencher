<!-- [MODIFIED] Paper-grade math evaluation and local LoRA flywheel guide. -->

# AutoBencher

[Simplified Chinese](README.zh-CN.md) | English

AutoBencher automatically builds and evaluates benchmark questions. The math
pipeline now supports reproducible adaptive evaluation and an end-to-end local
QLoRA data flywheel. Wiki and Multilingual entry points remain compatible.

## Table of contents

- [Overview](#overview)
- [System architecture](#system-architecture)
- [Project layout](#project-layout)
- [Environment requirements](#environment-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick start](#quick-start)
- [Local and API model routing](#local-and-api-model-routing)
- [Volcengine remote GPU operation](#volcengine-remote-gpu-operation)
- [PyCharm remote development](#pycharm-remote-development)
- [Outputs and data contracts](#outputs-and-data-contracts)
- [Coverage and adaptive sampling](#coverage-and-adaptive-sampling)
- [Test-taker isolation and evaluation](#test-taker-isolation-and-evaluation)
- [Training dataset and LoRA](#training-dataset-and-lora)
- [Cache recovery, logs, and progress](#cache-recovery-logs-and-progress)
- [Experiment reproduction](#experiment-reproduction)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Citation, license, and contribution](#citation-license-and-contribution)

## Overview

Two math modes are available:

- `eval` generates and evaluates a benchmark without training.
- `data_flywheel` evaluates, updates the hard pool, builds a mixed training
  dataset, invokes the repository-local `train_llm.py`, merges the LoRA adapter,
  switches to the merged model, and starts the next cycle.

Core capabilities include:

- a fixed nine-category, 27-subcategory math taxonomy;
- exact per-iteration question-budget allocation;
- raw and quota-aware effective coverage metrics;
- Beta-Binomial adaptive sampling by subcategory and difficulty;
- a no-tool, strict JSON contract for the test taker;
- deterministic normalization for numeric and non-numeric answers;
- auditable evaluator-only symbolic checks and error attribution;
- hard-pool injection only from iteration 3 onward;
- a 45% hard-variant, 40% coverage-repair, 15% retention mix after warmup;
- mixed correct/incorrect training data with exact, near, semantic, and template
  deduplication;
- atomic artifacts, configuration hashes, manifests, structured logs, and
  resumable caches.

## System architecture

```text
YAML configuration + explicit CLI overrides
  -> taxonomy quota and adaptive-priority scheduler
  -> quality-constrained question generation
  -> no-tool test-taker structured inference
  -> JSON parsing and controlled format repair
  -> answer normalization and deterministic equivalence
  -> privileged evaluator verification and tool audit
  -> hierarchical error attribution
  -> hard-pool update and coverage state
  -> mixed dataset construction and deduplication
  -> built-in 4-bit QLoRA training
  -> adapter merge and next-cycle model switch
```

The evaluator and test taker are intentionally different roles. The test taker
receives no tool schema and must answer from model reasoning only. The evaluator
may use deterministic Python/SymPy-side checks; every such use is stored in
`evaluator_tool_calls`.

An iteration is one generate/infer/evaluate/update pass. A cycle contains
`num_iterations` iterations and, in `data_flywheel` mode, may end in one
fine-tuning job.

## Project layout

```text
AutoBencher/
├── autobencher/
│   ├── attribution_eval.py
│   ├── config.py
│   ├── coverage.py
│   ├── dataset.py
│   ├── experiment.py
│   └── structured.py
├── configs/
│   ├── environments/
│   │   ├── local.yaml
│   │   ├── server.yaml
│   │   └── volcengine.yaml
│   ├── experiments/
│   │   ├── math_flywheel.yaml
│   │   └── smoke_test.yaml
│   ├── math_flywheel.yaml
│   └── math_flywheel_smoke_test.yaml
├── schemas/
│   ├── error_attribution.schema.json
│   ├── evaluation_result.schema.json
│   ├── finetune_manifest.schema.json
│   ├── generated_question.schema.json
│   ├── iteration_manifest.schema.json
│   └── test_taker_output.schema.json
├── tests/
├── math_autobencher.py
├── multilingual_autobencher.py
├── run_scripts.py
├── tool_util.py
├── train_llm.py
├── util.py
├── wiki_autobencher.py
├── .env.example
├── requirements.txt
└── requirements-lock.txt
```

## Environment requirements

- Python 3.10 or later.
- A DeepSeek/OpenAI-compatible endpoint for the generation agent and evaluator.
- A local Transformers directory, vLLM endpoint, or Ollama model for the test
  taker.
- An NVIDIA CUDA GPU for `data_flywheel` training.
- Local Hugging Face-compatible Qwen weights for offline QLoRA.
- Sufficient persistent disk space for base weights, temporary adapters, merged
  weights, and run artifacts.

Evaluation can run without a training GPU when the test taker is served
separately. `train_llm.py` deliberately fails on CPU because bitsandbytes 4-bit
QLoRA requires CUDA.

## Installation

### Virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Windows Command Prompt:

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencies

For the pinned environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Use `requirements.txt` only when compatible newer versions are intentional:

```bash
python -m pip install -r requirements.txt
```

On a managed CUDA image, inspect the existing PyTorch build before installing
anything. Do not unconditionally replace a working CUDA-specific PyTorch build
with an incompatible wheel:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

### API environment

Copy `.env.example` to `.env`, then supply credentials locally:

```dotenv
DEEPSEEK_API_KEY=replace-me
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

Never commit `.env`, an API key, an SSH password, or a private key.

## Configuration

The math entry point loads configuration exactly once and passes one recursively
immutable `ProjectConfig` to runtime modules. New experiments should be created
by copying an experiment YAML; do not edit Python constants to tune an
experiment.

Configuration profiles:

| File | Purpose |
| --- | --- |
| `configs/math_flywheel.yaml` | Full default research flywheel. |
| `configs/math_flywheel_smoke_test.yaml` | One CPU/mock-sized 27-question scheduling profile. |
| `configs/experiments/math_flywheel.yaml` | Recommended full experiment entry point. |
| `configs/experiments/smoke_test.yaml` | Recommended smoke experiment entry point. |
| `configs/environments/local.yaml` | Local workstation paths and model route. |
| `configs/environments/server.yaml` | Generic persistent server paths. |
| `configs/environments/volcengine.yaml` | Volcengine model and persistent paths. |

Resolution precedence is:

```text
security invariants
  < schema defaults
  < inherited base YAML
  < environment YAML
  < experiment YAML
  < explicitly typed legacy CLI
  < --override
  < sensitive environment variables
```

Environment variables are used only for credentials and private service
addresses declared under `sensitive_environment`. Their values are never
written to snapshots. Launcher defaults are stripped when `run_scripts.py
--config` is used, so they do not silently replace YAML values.

Recommended launch:

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml
```

Use temporary dotted-path overrides without editing a profile:

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/local.yaml \
  --override experiment.seed=43 finetune.gpu=1
```

Boolean overrides accept `true/false`, `yes/no`, `on/off`, and `1/0`.
Configuration validation fails before model loading for invalid ratios, quotas,
thresholds, unknown fields, test-taker tool access, model paths, or output
permissions. Legacy YAML fields such as `finetune_epoch` are migrated with a
`DeprecationWarning`; unsupported schema versions are rejected.

To create an experiment, copy the nearest profile, keep `extends`, and override
only changed fields. Taxonomy quotas live under `taxonomy`, generation ratios
under `generation_mix`, dataset ratios under `training_mix`, and progress
settings under `logging`.

Every run writes these configuration audit files inside its allocated
`test_<N>` directory:

```text
resolved_config.yaml
resolved_config.json
config_sources.json
config_validation.json
config_hash.txt
```

`config_sources.json` records the winning source and value for each leaf field.
Sensitive values are redacted. Compare `resolved_config.json` or
`config_hash.txt` between two `test_<N>` directories to reproduce or audit
experiments. A result-affecting configuration change changes the hash and
invalidates incompatible caches.

Backward-compatible options remain supported and are mapped centrally:

| Legacy CLI | Configuration field |
| --- | --- |
| `--agent_modelname` | `models.evaluator.model_name` |
| `--test_taker_modelname` | `models.test_taker.model_path` |
| `--num_iters` | `experiment.num_iterations` |
| `--max_cycle` | `experiment.max_cycles` |
| `--acc_target` | `adaptive_sampling.target_accuracy_low/high/mid` |
| `--finetune_gpu` | `finetune.gpu` |
| `--finetune_epoch` | `finetune.epochs` |
| `--finetune_batch` | `finetune.batch_size` |
| `--lora_rank` | `finetune.lora_rank` |
| `--clean_cycle_cache` | `experiment.clean_cycle_cache` |

## Quick start

### Verify the checkout

```bash
python smoke_test.py
python math_autobencher.py --help
python run_scripts.py --help
python train_llm.py --help
```

`smoke_test.py` can make a small API request when credentials are present. The
pytest suite described below is the API-free default validation.

### Evaluation only

This compatibility command builds benchmarks and hard-pool evidence but never
starts training:

```bash
python run_scripts.py math \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname qwen2.5:7b-instruct \
  --exp_mode autobencher \
  --use_helm no \
  --num_iters 2 \
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. \
  --acc_target 0.1--0.3 \
  --mode eval
```

PowerShell equivalent:

```powershell
python run_scripts.py math `
  --agent_modelname deepseek-v4-pro `
  --test_taker_modelname qwen2.5:7b-instruct `
  --exp_mode autobencher `
  --use_helm no `
  --num_iters 2 `
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. `
  --acc_target 0.1--0.3 `
  --mode eval
```

### Configuration-driven flywheel

```bash
python math_autobencher.py \
  --config configs/math_flywheel_local.yaml
```

No external training-script path is accepted. The main process calls the root
`train_llm.py` automatically.

### Complete compatibility command

The following existing long-form command remains supported. Explicit values
override their YAML equivalents:

```bash
python math_autobencher.py \
  --exp_mode autobencher \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname /vepfs-mlp2/queue010/20262202597/Qwen2.5-7B-Instruct \
  --use_helm no \
  --num_iters 5 \
  --outfile_prefix1 /vepfs-mlp2/queue010/20262202597/math_flywheel/qwen7b_dsagent. \
  --acc_target 0.1,0.3 \
  --mode data_flywheel \
  --export_interval 1 \
  --max_cycle 3 \
  --finetune_gpu 0 \
  --finetune_epoch 3 \
  --finetune_batch 8 \
  --lora_rank 6 \
  --new_local_model_suffix math_lora \
  --clean_cycle_cache false
```

### Other benchmark modules

The original launcher workflows remain:

```bash
python run_scripts.py wiki
python run_scripts.py multilingual
python run_scripts.py math --num-iters 1
```

Wiki direct example:

```bash
python wiki_autobencher.py \
  --exp_mode autobencher \
  --test_taker_modelname deepseek-v4-pro \
  --use_helm no \
  --agent_modelname deepseek-v4-pro \
  --theme history \
  --outfile_prefix1 KI/history.
```

HELM and optional SDK modes require their own dependencies and credentials.

### Important CLI parameters

| Parameter | Default | Effect |
| --- | --- | --- |
| `--config` | none | YAML profile. |
| `--run_id` | name plus config hash | Stable run identity for resume. |
| `--resume` | YAML value | Reuse completed compatible caches. |
| `--override PATH=VALUE` | none | Highest-priority temporary settings. |
| `--mode` | `eval` in legacy CLI | `eval` or `data_flywheel`. |
| `--num_iters` | `8` in legacy CLI | Iterations per cycle. |
| `--max_cycle` | `1` | Maximum flywheel cycles. |
| `--export_interval` | `1` | Completed-cycle boundary for dataset export. |
| `--finetune_gpu` | `0` | CUDA device exposed to the trainer. |
| `--finetune_epoch` | `3` | QLoRA epochs. |
| `--finetune_batch` | `8` | Effective batch via gradient accumulation. |
| `--lora_rank` | `8` in legacy CLI | LoRA rank; YAML default is `6`. |
| `--new_local_model_suffix` | `finetuned` | Merged-model directory suffix. |
| `--disk_warning_threshold` | `10` | Minimum free GB before training. |
| `--clean_cycle_cache` | `true` | Remove redundant temporary fragments. |

## Local and API model routing

The common mixed deployment is:

```text
agent/evaluator: DeepSeek API
test taker: local vLLM, Ollama, or Transformers
trainer: local Transformers-compatible Qwen weights
```

### vLLM

```dotenv
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

When `VLLM_BASE_URL` is present, a tagged test-taker name such as
`qwen2.5:7b-instruct` is sent to that OpenAI-compatible endpoint.

### Ollama

Without `VLLM_BASE_URL`, tagged identifiers use the native Ollama service:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_RETRIES=10
OLLAMA_RETRY_DELAY_SECONDS=5
OLLAMA_REQUEST_TIMEOUT=300
```

### Transformers

Pass a local Hugging Face-compatible model directory. It must contain model
configuration, tokenizer files, and weights.

An Ollama/vLLM alias does not expose trainable source weights. Map it before a
flywheel:

```dotenv
AUTOBENCHER_MODEL_MAP={"qwen2.5:7b-instruct":"/models/Qwen2.5-7B-Instruct"}
```

Alternatives are `AUTOBENCHER_LOCAL_MODEL_PATH` and
`AUTOBENCHER_LOCAL_MODEL_ROOT`. Training uses `local_files_only=True`; it does
not download model weights.

## Volcengine remote GPU operation

Volcengine image availability, CUDA versions, and console labels can change.
Inspect the currently available preset Python images in the
[Volcengine ML Platform console](https://console.volcengine.com/ml-platform/region:ml-platform+cn-beijing/mirror/detail?Id=vemlp-cn-beijing.cr.volces.com/preset-images/python)
and choose an image compatible with the selected GPU and PyTorch build.

### Persistent layout

Use persistent storage rather than a temporary system root:

```text
/vepfs-mlp2/queue010/20262202597/
├── AutoBencher/
├── Qwen2.5-7B-Instruct/
├── .cache/
└── math_flywheel/
```

Do not keep the only checkpoint on ephemeral storage. Back up any non-persistent
path before rebuilding a development machine or changing images.

### Environment checks

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
df -h /vepfs-mlp2/queue010/20262202597
```

Run artifacts also save package, platform, CUDA, device, image, driver, and Git
metadata to `test_<N>/environment.json`.

### Install and run

```bash
cd /vepfs-mlp2/queue010/20262202597/AutoBencher
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python math_autobencher.py \
  --config configs/math_flywheel_volcengine.yaml
```

Set secrets through the platform environment or a protected `.env`; never put
real credentials in the configuration file.

### Interactive, tmux, nohup, and platform jobs

Interactive development:

```bash
python math_autobencher.py \
  --config configs/math_flywheel_volcengine.yaml
```

`tmux`:

```bash
tmux new -s math_flywheel
cd /vepfs-mlp2/queue010/20262202597/AutoBencher
source .venv/bin/activate
python math_autobencher.py \
  --config configs/math_flywheel_volcengine.yaml
```

Detach with `Ctrl+B`, then `D`; reconnect with:

```bash
tmux attach -t math_flywheel
```

`nohup`:

```bash
nohup python math_autobencher.py \
  --config configs/math_flywheel_volcengine.yaml \
  > /vepfs-mlp2/queue010/20262202597/math_flywheel/launcher.log \
  2>&1 &
tail -f /vepfs-mlp2/queue010/20262202597/math_flywheel/launcher.log
```

For a managed training or remote task: select a verified Python/CUDA image and
GPU, mount persistent storage, set the project working directory and protected
environment variables, use the configuration command above as the entry point,
and point every output path to persistent storage. Platform UI field names are
intentionally not assumed here.

## PyCharm remote development

1. Create and verify the remote Volcengine development machine.
2. Configure SSH access without placing passwords or keys in the repository.
3. In PyCharm, select Remote Development or configure an SSH interpreter.
4. Open the remote `AutoBencher` directory.
5. Select the remote `.venv` interpreter.
6. Mark the repository root as the working directory.
7. Configure API variables in a protected remote run configuration or `.env`.
8. Verify `nvidia-smi` and the PyTorch CUDA probe in the PyCharm terminal.
9. Run the smoke profile before a full experiment.
10. Use `tmux`, `nohup`, or a platform job for long runs; do not depend on the
    local PyCharm process remaining connected.

## Outputs and data contracts

Configuration-driven output:

```text
<output_root>/
├── test_1/
│   ├── resolved_config.yaml
│   ├── resolved_config.json
│   ├── config_sources.json
│   ├── config_validation.json
│   ├── config_hash.txt
│   ├── environment.json
│   ├── run_manifest.json
│   ├── taxonomy_snapshot.json
│   ├── experiment_summary.json
│   ├── hard_pool.json
│   ├── meta_summary.json
│   ├── cycle_record.json
│   ├── logs/
│   │   ├── run.log
│   │   └── events.jsonl
│   └── cycle/
│       └── cycle_1/
│           ├── cycle_manifest.json
│           ├── iter_1/
│           │   ├── generation_plan.json
│           │   ├── generated_questions.json
│           │   ├── test_taker_outputs.json
│           │   ├── normalized_answers.json
│           │   ├── evaluation_results.json
│           │   ├── error_attributions.json
│           │   ├── coverage_metrics.json
│           │   ├── adaptive_sampler_state.json
│           │   ├── hard_pool_snapshot.json
│           │   └── iteration_summary.json
│           └── training/
│               ├── dataset_candidates.json
│               ├── dataset_selected.jsonl
│               ├── dataset_manifest.json
│               ├── dedup_report.json
│               ├── diversity_report.json
│               ├── finetune_config.json
│               ├── finetune_metrics.jsonl
│               ├── finetune_summary.json
│               └── checkpoint_manifest.json
├── test_2/
└── test_3/
```

At startup, AutoBencher scans sibling directories matching `test_<number>` and
atomically creates `test_<maximum+1>`. Unrelated names are ignored and existing
runs are never overwritten. Every artifact produced by that invocation stays
inside the allocated test directory. All `cycle_<N>` directories are grouped
under its `cycle/` directory.

Global compatibility files remain available inside the active `test_<N>` root:
`hard_pool.json`, `meta_summary.json`, and `cycle_record.json`. Iteration
compatibility files retain `test_taker_inference.json`,
`compare_answers.json`, and `question_plan_with_aim.json`.

JSON artifacts are UTF-8, two-space indented, newline terminated, and written
through temporary files plus atomic replacement. JSONL is one complete UTF-8
object per line. Every research envelope includes schema version, `run_id`,
timestamp, Git commit, and `config_hash`.

Representative hard-pool sample:

```json
{
  "source_iter": 3,
  "source_cycle": 1,
  "last_seen_iter": 3,
  "last_seen_cycle": 1,
  "category": "Algebra",
  "sub_category": "Linear Equations",
  "question": "Solve 3x + 5 = 20.",
  "gold_answer": "5",
  "test_taker_response": "4",
  "error_tags": [
    "calculation_error"
  ],
  "difficulty": 4,
  "unique_key": "sha256-record-key",
  "sample_grade": "train_eligible",
  "accuracy_bucket": "0.1-0.4",
  "sub_category_accuracy": 0.24,
  "occurrences": 1
}
```

Representative evaluation and attribution fields:

```json
{
  "question_id": "c1_i3_q00001",
  "status": "incorrect",
  "is_correct": false,
  "deterministic_checks": {
    "answer_parse_success": true,
    "numeric_equivalence": false,
    "symbolic_equivalence": false,
    "unit_consistent": true,
    "format_valid": true
  },
  "evaluator_tool_calls": [],
  "primary_error_tag": "calculation_error",
  "secondary_error_tags": [],
  "attribution_confidence": 0.72,
  "needs_review": false
}
```

Temporary `subcat` fragments, `all_questions.json`, `*.attempt*.txt`, empty
JSON, and damaged JSON are removed when cleanup is enabled. Research manifests,
final iteration artifacts, hard-pool state, and training exports are retained.

## Coverage and adaptive sampling

The taxonomy contains nine categories and three subcategories per category.
Each subcategory defines `min_quota` and `base_weight`.

- `subcategory_coverage` is the backward-compatible raw coverage:
  subcategories with at least one sample divided by all subcategories.
- `effective_subcategory_coverage` counts only subcategories meeting their
  configured minimum quota.

Balance metrics include normalized Shannon entropy, Jensen-Shannon divergence
from uniform, count coefficient of variation, and max/min nonzero count ratio.

The scheduler uses largest-remainder allocation, so integer allocations always
sum exactly to `questions_per_iteration`. Minimum quotas are cumulative across
iterations; one iteration is not required to cover every subcategory. The
generation plan reports `multi_iteration_progress`, `scheduled_this_iteration`,
or `complete`, and records remaining subcategories without treating incomplete
single-iteration coverage as a runtime error. The compatibility field
`quota_feasible` means that the current integer budget was allocated correctly.

Per `(subcategory, difficulty)` accuracy is estimated with a Beta-Binomial
posterior. Boundary proximity, coverage deficit, uncertainty, persistent error,
and retention are combined into an auditable priority. Difficulty changes only
after the minimum observation count.

## Generated gold-answer verification

Question generation and gold-answer verification are separate evaluator calls.
After the evaluator proposes a question and `canonical_answer`, the system sends
the complete question back to the privileged evaluator with an explicit
instruction to solve it independently and substitute the proposed answer into
all applicable equations, domains, units, and constraints.

A question enters test-taker inference only when all of the following hold:

- independent recomputation succeeds;
- constraint/substitution verification succeeds;
- the recomputed answer is deterministically equivalent to the proposed gold;
- the validator answer type matches the generated-question answer type.

Rejected questions are removed before inference and the existing quota-repair
loop generates replacements. Every accepted or rejected check is recorded in
`*.subcat<N>.gold_answer_validation.json`; accepted question records also carry
the `gold_answer_validation` object. This behavior is controlled by
`generation.require_gold_answer_validation` and the related validation retry,
temperature, and token settings in YAML.

## Test-taker isolation and evaluation

The test taker receives no external tools and must return exactly:

```json
{
  "reasoning_summary": [
    "Compute the required intermediate value."
  ],
  "final_answer": "8",
  "answer_type": "integer",
  "confidence": 0.95
}
```

The parser records raw response, parsed response, repair count, parse status,
prompt echo, irrelevant content, and tool violation. Output statuses are
separate from mathematical correctness.

Local Transformers inference applies the model chat template, uses the
tokenizer EOS ID, forwards configured role stop sequences, and stops as soon as
the first complete structured-answer JSON object closes. If a model still adds
an explanation before the JSON or leaks another dialogue after it, the parser
accepts the unique valid structured object and records
`extraneous_content_discarded`, `discarded_prefix_chars`, and
`discarded_suffix_chars`. Multiple structured answers, prompt echo, and tool
calls remain hard failures.

An otherwise valid answer is not discarded only because the model emitted more
reasoning steps than requested. Excess steps and overlong step text are safely
truncated while `reasoning_steps_truncated`,
`original_reasoning_step_count`, `reasoning_steps_dropped`, and
`reasoning_step_chars_truncated` preserve the audit trail. Generated questions
containing CJK text or the Unicode replacement character are rejected before
inference so encoding-corrupted math notation cannot enter evaluation or
training data.

Supported answer types include integer, decimal, rational, percentage, Boolean,
symbolic expression, equation, inequality, set, interval, tuple, collection,
vector, matrix, unit value, multiple choice, and text. Numeric tolerance and
unit rules come from YAML. Common model aliases are canonicalized before Schema
validation, including `fraction` to `rational`, `percent` to `percentage`, and
`bool` to `boolean`. Mixed-number aliases such as `mixed_number` and
`mixed_fraction` also map to `rational`; unknown labels fall back to
deterministic inference from the canonical answer.

Before numeric type parsing, both the canonical gold answer and the extracted
test-taker answer are compared through a temporary cleaned copy. The cleanup
removes surrounding whitespace, a leading scalar assignment such as `x = `,
`y=`, or `z =`, and a trailing degree symbol. Raw responses and the persisted
`parsed_response.final_answer` are never rewritten. Equation, symbolic, unit,
and structured answer types retain their original syntax.

Error attribution follows output validity, answer equivalence, then
mathematical evidence. Fixed tags are:

```text
concept_confusion
formula_memory_error
calculation_error
multi_step_logic_error
condition_missing
format_output_error
tool_violation
irrelevant_output
prompt_echo
parse_failed
unknown_error
```

Low-confidence cases become `unknown_error` and set `needs_review=true`.
`autobencher/attribution_eval.py` can export a review CSV and compute precision,
recall, F1, confusion matrices, and Cohen's kappa from reviewed labels.

## Training dataset and LoRA

The mixed target distribution is configurable and defaults to:

- 55% incorrect boundary samples;
- 25% correct retention samples;
- 15% coverage repair samples;
- 5% format-instruction samples.

Noise filtering rejects ambiguous questions, invalid answers, low evaluator
confidence, tool violations, irrelevant output, prompt echo, and empty fields.
Deduplication applies exact signatures, token-shingle Jaccard, dependency-free
TF-IDF cosine similarity, and numeric/template clustering. Rejection reasons
and source counts are exported.

The selected file uses strict Alpaca JSONL:

```json
{
  "instruction": "Solve the math problem and return a JSON object with reasoning_summary, final_answer, answer_type, and confidence.",
  "input": "What is 5 + 3?",
  "output": "{\"reasoning_summary\":[\"Add the integers.\"],\"final_answer\":\"8\",\"answer_type\":\"integer\",\"confidence\":1.0}"
}
```

`train_llm.py` uses Transformers, PEFT, bitsandbytes, TRL, and Datasets to:

1. load offline local weights;
2. quantize the base model to NF4 4-bit;
3. prepare Qwen projection modules for LoRA;
4. train with gradient checkpointing and gradient accumulation;
5. save per-step metrics;
6. reload the base model and merge the adapter;
7. atomically publish a full local model directory.

Direct trainer help:

```bash
python train_llm.py --help
```

The main flywheel passes model, dataset, GPU, epoch, batch, rank, sequence
length, learning rate, run ID, config hash, metrics path, and output directory.

## Cache recovery, logs, and progress

Rerun the same `run_id` and compatible configuration to resume. Completed
inference records and iterations are reused, completed cycles are skipped, and
the active merged model is restored from `cycle_record.json`. A cache carrying a
different `config_hash` is rejected. The compound identity
`run_id + cycle_id + iteration_id + question_id` prevents repeated statistics,
and replaying one iteration does not increment hard-pool occurrences.

Each stage owns at most one dynamic progress bar. Nested bars raise an error.
Non-TTY profiles disable the bar, and Transformers/Datasets advisory bars are
disabled during evaluation. In an interactive terminal, Generate, Infer, and
Evaluate each reuse one in-place bar and close it before the next stage starts.
The trainer keeps its single Trainer progress bar. Redirected `nohup` output is
non-TTY, so it uses stage log messages instead of emitting repeated pseudo-bars.

Human-readable and machine-readable logs are:

```text
test_<N>/logs/run.log
test_<N>/logs/events.jsonl
```

Events include timestamp, level, run ID, cycle, iteration, stage, event,
message, and metrics. Failures such as model loading, fine-tuning OOM, disk
shortage, timeout, and JSON errors are stored before exit.

## Experiment reproduction

Archive the run directory together with:

- the Git commit in each manifest;
- resolved YAML and JSON plus config hash;
- taxonomy and prompt version/hash;
- model name/path and active merged checkpoint;
- environment and CUDA snapshot;
- seeds, run/cycle/iteration/question identities;
- generated questions and raw structured outputs;
- deterministic checks and error attributions;
- hard-pool snapshots and adaptive sampler state;
- selected/rejected dataset manifests;
- fine-tuning configuration and step metrics.

For an exact resume, keep the same `run_id`, configuration, model files, and
output root. Use `--resume false` to generate a timestamped new run ID when one
is not provided.

## Troubleshooting

### Local model loading

Confirm the directory contains `config.json`, tokenizer files, and weights.
Check read permissions, available VRAM, `VLLM_BASE_URL` including `/v1`, and the
served model alias. For training an alias, set `AUTOBENCHER_MODEL_MAP`.

### Fine-tuning OOM

Reduce `finetune.batch_size`, `finetune.lora_rank`, or
`finetune.max_seq_length`; use a smaller model; and ensure vLLM is not retaining
the selected training GPU. Evaluation data and manifests are preserved on
failure.

### Disk threshold

Free persistent space or adjust `--disk_warning_threshold` only after verifying
that both temporary adapter and merged model will fit.

### API timeout

Check endpoint, credentials, account access, network, and service status. Rerun
the same run ID to reuse completed work.

### Structured JSON failure

Inspect `raw_response`, `parse_status`, and iteration logs. The parser can
repair fenced JSON, smart quotes, unquoted English keys, and trailing commas,
but rejects multiple objects, role-prefixed text, prompt echo, and tool calls.

### Missing training packages

```bash
python -m pip install -r requirements-lock.txt
python -c "import torch, transformers, peft, bitsandbytes, trl, datasets; print('ready')"
```

## Tests

Default CPU and mock suite:

```bash
python -m pytest -q
```

GPU and real API tests are opt-in and skipped by default:

```bash
AUTOBENCHER_RUN_GPU_TESTS=1 python -m pytest -m gpu
AUTOBENCHER_RUN_API_TESTS=1 python -m pytest -m api
```

Windows PowerShell:

```powershell
$env:AUTOBENCHER_RUN_GPU_TESTS = "1"
python -m pytest -m gpu
```

The CPU mock covers configuration, quota scheduling, structured inference,
answer normalization, evaluation, dataset construction, atomic exports, and
research manifests without a real API or GPU. Real DeepSeek-v4-pro behavior,
full local Qwen inference, CUDA QLoRA, and Volcengine scheduling must be
validated in the target environment; offline unit success is not evidence that
those external systems ran.

## Citation, license, and contribution

This checkout does not currently include a `CITATION.cff` or a repository
license file. Do not infer a license from package availability. Add the
upstream project citation and license before redistributing modified code.

Contributions should include:

- a focused change with backward-compatible migration notes;
- updated YAML/schema/README documentation;
- CPU tests for deterministic behavior;
- `gpu` or `api` markers for external-system tests;
- no credentials, private model paths, caches, or generated checkpoints.
