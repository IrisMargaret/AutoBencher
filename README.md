<!-- [MODIFIED] Document adaptive evaluation and the built-in LoRA flywheel. -->

# AutoBencher

AutoBencher builds adaptive math benchmarks, evaluates a test-taker model, and
maintains a deduplicated hard-sample pool. It supports two math workflows:

- `eval` generates and evaluates benchmarks without training.
- `data_flywheel` evaluates, exports eligible mistakes, invokes the built-in
  `train_llm.py` QLoRA trainer, merges the LoRA adapter, loads the new local
  model, and repeats until `max_cycle`.

The Wiki and Multilingual modules remain unchanged.

## Features

- Nine fixed top-level math categories with a two-level taxonomy.
- Mixed deployment with a DeepSeek API agent and a local test taker.
- Local inference through vLLM, Ollama, or a Transformers model directory.
- Built-in 4-bit QLoRA training with Transformers, PEFT, bitsandbytes, and TRL.
- Alpaca JSONL export from `train_eligible` hard samples.
- Resumable iteration caches and cycle records.
- Atomic, readable UTF-8 JSON with two-space indentation.
- Automatic cleanup of redundant fragments, attempt logs, empty JSON, and
  damaged JSON.

## Environment setup

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
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

### 2. Install dependencies

The lock file includes the benchmark and local training stacks:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

For QLoRA, use a supported NVIDIA GPU, a working CUDA driver, and a CUDA-enabled
PyTorch build. Verify the runtime before starting a flywheel:

```shell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

The trainer imports its optional GPU packages only when training starts, so
API-only evaluation can still use a smaller environment.

### 3. Configure the DeepSeek agent

Create `.env` in the project root:

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

Do not commit `.env`.

### 4. Configure a local test taker

AutoBencher supports three local routes.

#### vLLM

Start an OpenAI-compatible vLLM server and set:

```dotenv
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

When `VLLM_BASE_URL` is present, a tagged identifier such as
`qwen2.5:7b-instruct` is sent to that endpoint.

#### Ollama

Without `VLLM_BASE_URL`, a tagged identifier is sent to the native Ollama API:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_RETRIES=10
OLLAMA_RETRY_DELAY_SECONDS=5
OLLAMA_REQUEST_TIMEOUT=300
```

#### Transformers

Pass a local Hugging Face-compatible model directory as
`--test_taker_modelname`. The directory must contain the model configuration,
tokenizer, and weight files.

### 5. Map an inference alias to trainable source weights

An Ollama or vLLM alias does not expose the original Hugging Face weights.
Before a flywheel run with an alias, map it to a local Qwen model directory:

```dotenv
AUTOBENCHER_MODEL_MAP={"qwen2.5:7b-instruct":"D:/models/Qwen2.5-7B-Instruct"}
```

Alternatives:

```dotenv
AUTOBENCHER_LOCAL_MODEL_PATH=D:/models/Qwen2.5-7B-Instruct
AUTOBENCHER_LOCAL_MODEL_ROOT=D:/models
```

`train_llm.py` always uses `local_files_only=True`; it does not download model
weights. A local path is therefore required unless the repository identifier is
already present in the Hugging Face cache.

## Run mode 1: evaluation only

The default mode is `eval`. The following command is directly compatible and
does not invoke training:

```bash
python run_scripts.py math \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname qwen2.5:7b-instruct \
  --exp_mode autobencher \
  --use_helm no \
  --num_iters 2 \
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. \
  --acc_target 0.1--0.3
```

Parameter behavior:

- `--agent_modelname` selects the benchmark-generation agent.
- `--test_taker_modelname` selects the evaluated model.
- `--exp_mode autobencher` runs adaptive benchmark generation and evaluation.
- `--use_helm no` uses direct API or local model routing.
- `--num_iters 2` runs two adaptive iterations.
- `--outfile_prefix1` sets the output root and retained filename prefix.
- `--acc_target 0.1--0.3` guides the agent toward the target accuracy range.

Adding `--mode eval` is optional because it is the default.

## Run mode 2: automatic data flywheel

This command evaluates, exports, trains, merges, switches models, and repeats
without an external training script:

```bash
python run_scripts.py math \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname qwen2.5:7b-instruct \
  --exp_mode autobencher \
  --use_helm no \
  --num_iters 2 \
  --outfile_prefix1 math_flywheel/qwen7b_dsagent. \
  --acc_target 0.1--0.3 \
  --mode data_flywheel \
  --export_interval 2 \
  --max_cycle 3 \
  --finetune_gpu 0 \
  --finetune_epoch 3 \
  --finetune_batch 8 \
  --lora_rank 8 \
  --new_local_model_suffix math_lora \
  --disk_warning_threshold 10 \
  --clean_cycle_cache true
```

`run_scripts.py` automatically invokes the root `train_llm.py`. No external
script path is accepted or required.

The effective training batch is implemented as a micro-batch of one plus
gradient accumulation. For example, `--finetune_batch 8` accumulates eight
steps.

If the initial test taker uses vLLM on the same GPU selected for training, the
vLLM process can retain VRAM. Use a separate inference GPU or stop the vLLM
worker before starting the fully automatic run. Ollama models loaded through
AutoBencher are explicitly unloaded before training.

## Command-line parameters

### Common launcher parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `math`, `wiki`, `multilingual` | positional | required | Benchmark module. |
| `--model` | string | `DEEPSEEK_MODEL` | Shorthand for both agent and test taker. |
| `--agent_modelname` | string | launcher model | Generation agent model. |
| `--test_taker_modelname` | string | launcher model | Evaluated API, tagged local, or local-path model. |
| `--test_taker_modelname2` | string | none | Preserved compatibility option. |
| `--tool_modelname` | string | agent model | Answer-comparison model. |
| `--exp_mode` | string | `autobencher` | Experiment workflow. |
| `--use_helm` | string | `no` | Use HELM when set to `yes`. |
| `--num_iters` | integer | `8` | Adaptive iterations per cycle. |
| `--outfile_prefix1` | string | module-specific | Output root and filename prefix. |
| `--acc_target` | string | math: `0.1--0.3` | Target benchmark accuracy interval. |
| `--temperature` | float | module default | Model sampling temperature. |
| `--top_p` | float | module default | Nucleus sampling parameter. |
| `--pairwise` | string | module default | Preserved comparison option. |
| `--theme` | string | `history` | Wiki theme; ignored by math. |

Hyphenated aliases such as `--agent-modelname` are also accepted by the
launcher.

### Flywheel parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `--mode` | choice | `eval` | `eval` or `data_flywheel`. |
| `--export_interval` | integer | `1` | Export when completed iterations cross each N-iteration boundary. |
| `--max_cycle` | integer | `1` | Maximum evaluation-training cycles. |
| `--finetune_gpu` | string | `0` | GPU identifier exposed to the trainer. |
| `--finetune_epoch` | integer | `3` | QLoRA training epochs. |
| `--finetune_batch` | integer | `8` | Effective training batch. |
| `--lora_rank` | integer | `8` | LoRA rank; alpha is twice this value. |
| `--new_local_model_suffix` | string | `finetuned` | Merged-model directory suffix. |
| `--disk_warning_threshold` | integer | `10` | Minimum free disk space in GB. |
| `--clean_cycle_cache` | Boolean | `true` | Remove redundant iteration files after success. |

### Direct `train_llm.py` parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `--model_name_or_path` | string | yes | Local model path, cached identifier, or mapped alias. |
| `--dataset_path` | path | yes | AutoBencher Alpaca JSONL export. |
| `--gpu` | string | no | CUDA device, default `0`. |
| `--epoch` | integer | no | Epoch count, default `3`. |
| `--batch` | integer | no | Effective batch, default `8`. |
| `--lora_rank` | integer | no | LoRA rank, default `8`. |
| `--output_path` | path | yes | Complete merged-model output directory. |
| `--max_seq_length` | integer | no | Token limit, default `2048`. |
| `--learning_rate` | float | no | Learning rate, default `2e-4`. |

The flywheel passes the required trainer parameters automatically.

## Math taxonomy

The fixed top-level category set is:

```json
[
  "Arithmetic",
  "Algebra",
  "Geometry & Trigonometry",
  "Probability & Statistics",
  "Word Problems",
  "Number Theory",
  "Calculus",
  "Linear Algebra",
  "Composite Comprehensive"
]
```

Every question also contains `sub_category`. Coverage and accuracy are tracked
independently for each `(category, sub_category)` pair.

The fixed error-tag enumeration is:

```json
[
  "calculation_error",
  "formula_memory_error",
  "condition_missing",
  "multi-step_logic_error",
  "concept_confusion"
]
```

## Hard-pool grading

Only wrong answers with a non-empty response enter `hard_pool.json`.
Sub-category accuracy determines the grade:

| Accuracy | `sample_grade` | `accuracy_bucket` | Exported |
| --- | --- | --- | --- |
| Below `0.1` | `hard_unsuitable` | `below_0.1` | no |
| `0.1` through `0.4` | `train_eligible` | `0.1-0.4` | yes |
| Above `0.4` | `easy_sample` | `above_0.4` | no |

Records are deduplicated by a SHA-256 `unique_key` derived from category,
sub-category, and normalized question text.

In directed flywheel rounds, the agent receives same-category
`train_eligible` examples and is instructed to create non-copying variants
expected to remain in the `0.1` to `0.4` accuracy range. Missing and weak
sub-categories are prioritized to balance coverage.

## Output layout

Evaluation-only layout:

```text
math_test/
|-- hard_pool.json
|-- meta_summary.json
|-- cycle_record.json
|-- iter_1/
|   |-- qwen7b_dsagent.0.3.1.question_plan_with_aim.json
|   |-- qwen7b_dsagent.0.3.1.test_taker_inference.json
|   `-- qwen7b_dsagent.0.3.1.compare_answers.json
`-- iter_N/
```

Flywheel layout:

```text
math_flywheel/
|-- hard_pool.json
|-- meta_summary.json
|-- cycle_record.json
|-- training_export/
|   |-- cycle_1_train.jsonl
|   `-- cycle_N_train.jsonl
|-- models/
|   |-- qwen2.5_7b-instruct_math_lora_cycle_1/
|   `-- qwen2.5_7b-instruct_math_lora_cycle_N/
|-- cycle_1/
|   |-- iter_1/
|   `-- iter_N/
`-- cycle_N/
    |-- iter_1/
    `-- iter_N/
```

## Persistent files

- `question_plan_with_aim.json` stores the nine-item plan for every iteration.
- `test_taker_inference.json` stores canonical questions, responses, and
  judgments for every iteration.
- `compare_answers.json` stores global and sub-category statistics.
- `hard_pool.json` stores deduplicated, graded wrong answers.
- `meta_summary.json` indexes all retained iteration files and run metadata.
- `cycle_record.json` checkpoints model transitions, exports, training status,
  disk checks, and failures.
- `training_export/*.jsonl` stores Alpaca training records.
- `train_llm.py` performs local 4-bit QLoRA and merged-model export.

## JSON standards

Persistent `.json` files use UTF-8, English keys and values, and two-space
indentation. A final newline is always written. Writes are atomic.

JSONL has one JSON object per physical line as required by the JSON Lines
format; it is UTF-8 and is not minified with custom separators.

### `question_plan_with_aim.json`

```json
[
  {
    "id": 1,
    "category": "Arithmetic",
    "sub_category": "Fraction and Decimal Operations",
    "difficulty": 6
  }
]
```

The real plan contains exactly nine objects and covers every top-level category.

### `test_taker_inference.json`

```json
[
  {
    "id": 1,
    "category": "Algebra",
    "sub_category": "Systems of Equations",
    "difficulty": 6,
    "question": "Solve 2x + y = 9 and x - y = 3.",
    "gold_answer": "x = 4, y = 1",
    "test_taker_response": "x = 3, y = 3",
    "prompt": "Output just with the final answer to the question.\nQuestion:Solve 2x + y = 9 and x - y = 3.\nAnswer:",
    "is_correct": false,
    "error_tags": [
      "calculation_error"
    ],
    "unique_key": "57c7c3fcf3d58a48e41db723feebcdb49fa61ea612126903258ea51ade680b6e"
  }
]
```

### `compare_answers.json`

```json
{
  "iter_number": 1,
  "total_questions": 50,
  "global_accuracy": 0.3,
  "category_statistics": [
    {
      "category": "Algebra",
      "sub_category": "Systems of Equations",
      "total_count": 10,
      "correct_count": 3,
      "accuracy": 0.3
    }
  ]
}
```

### `hard_pool.json`

```json
[
  {
    "source_iter": 1,
    "source_cycle": 1,
    "last_seen_iter": 1,
    "last_seen_cycle": 1,
    "category": "Algebra",
    "sub_category": "Systems of Equations",
    "question": "Solve 2x + y = 9 and x - y = 3.",
    "gold_answer": "x = 4, y = 1",
    "test_taker_response": "x = 3, y = 3",
    "error_tags": [
      "calculation_error"
    ],
    "difficulty": 6,
    "unique_key": "57c7c3fcf3d58a48e41db723feebcdb49fa61ea612126903258ea51ade680b6e",
    "sample_grade": "train_eligible",
    "accuracy_bucket": "0.1-0.4",
    "sub_category_accuracy": 0.3,
    "occurrences": 1
  }
]
```

### `meta_summary.json`

```json
{
  "run_config": {
    "mode": "data_flywheel",
    "agent_model": "deepseek-v4-pro",
    "test_taker_model": "qwen2.5:7b-instruct",
    "tool_model": "deepseek-v4-pro",
    "target_accuracy_range": "0.1--0.3",
    "iterations_per_cycle": 2,
    "max_cycle": 3,
    "output_root": "C:/project/math_flywheel"
  },
  "all_sub_categories": [
    "Systems of Equations"
  ],
  "topic_salience_data": {},
  "iteration_index": [
    {
      "cycle_num": 1,
      "iter_num": 1,
      "global_iter_num": 1,
      "plan_file_path": "cycle_1/iter_1/run.cycle1.iter1.question_plan_with_aim.json",
      "infer_file_path": "cycle_1/iter_1/run.cycle1.iter1.test_taker_inference.json",
      "stat_file_path": "cycle_1/iter_1/run.cycle1.iter1.compare_answers.json",
      "new_hard_sample_count": 12
    }
  ]
}
```

### `cycle_record.json`

```json
{
  "schema_version": 1,
  "status": "completed",
  "created_at": "2026-07-25T10:00:00Z",
  "run_config": {
    "mode": "data_flywheel",
    "agent_model": "deepseek-v4-pro",
    "initial_test_taker_model": "qwen2.5:7b-instruct",
    "num_iters": 2,
    "max_cycle": 1,
    "export_interval": 2,
    "finetune_gpu": "0",
    "finetune_epoch": 3,
    "finetune_batch": 8,
    "lora_rank": 8
  },
  "active_test_taker_model": "C:/project/math_flywheel/models/qwen_math_lora_cycle_1",
  "cycles": [
    {
      "cycle": 1,
      "status": "completed",
      "test_taker_model": "qwen2.5:7b-instruct",
      "iterations_completed": 2,
      "training_export": "training_export/cycle_1_train.jsonl",
      "training_sample_count": 120,
      "finetune_output": "models/qwen_math_lora_cycle_1",
      "finetune_status": "completed",
      "next_test_taker_model": "C:/project/math_flywheel/models/qwen_math_lora_cycle_1"
    }
  ],
  "updated_at": "2026-07-25T12:00:00Z"
}
```

### Training export

Each line of `cycle_N_train.jsonl` has exactly three fields:

```json
{"instruction": "Solve the math problem. Return only the final answer.", "input": "Solve 2x + y = 9 and x - y = 3.", "output": "x = 4, y = 1"}
```

## Automatic cleanup

When `--clean_cycle_cache true`, AutoBencher removes:

- `*subcat*.questions.json`;
- `*subcat*questions_final.json`;
- `*all_questions.json`;
- `*.attempt*.txt`;
- legacy `*.compare_answers.jsonl`;
- files under iteration `temp_log` directories;
- empty `.json` files;
- invalid or damaged `.json` files.

The three permanent iteration JSON files, global JSON files, training exports,
and merged models are retained.

## Workflows

### Evaluation

1. Load the API agent and local or remote test taker.
2. Generate a nine-category question plan.
3. Generate questions for each selected sub-category.
4. Run test-taker inference.
5. Compare answers and calculate sub-category accuracy.
6. Grade and deduplicate wrong answers in `hard_pool.json`.
7. Update metadata, clean redundant files, and stop after `num_iters`.

No dataset export or training occurs.

### Data flywheel

1. Load the DeepSeek agent and current local test taker.
2. Run `num_iters` adaptive evaluation iterations.
3. Store graded mistakes and checkpoint `cycle_record.json`.
4. When an export boundary is crossed, check free disk space.
5. Export all deduplicated `train_eligible` samples to Alpaca JSONL.
6. Release the locally loaded test-taker model.
7. Invoke the root `train_llm.py`.
8. Load the source model in 4-bit, train LoRA, and merge the adapter.
9. Switch `test_taker_modelname` to the merged local model directory.
10. Start the next cycle until `max_cycle` is reached.

If no eligible samples exist, the cycle is recorded as
`completed_without_training`; existing data is retained and the next cycle can
collect more samples.

## Resume behavior

Rerun the same command and output prefix after interruption:

- completed inference batches are reused;
- completed iteration JSON is reused;
- damaged cache suffixes are discarded;
- completed cycles are skipped;
- the next recorded local model is restored;
- failed cycles restart from retained evaluation data.

## Troubleshooting

### Local model loading fails

- Confirm the path contains `config.json`, tokenizer files, and weight files.
- Confirm the process has read permission.
- Confirm the model fits GPU memory for inference.
- For an alias, configure `AUTOBENCHER_MODEL_MAP`.
- For vLLM, confirm `VLLM_BASE_URL` includes `/v1` and the served model name
  matches `--test_taker_modelname`.

The failure is written to `cycle_record.json` as `model_loading`.

### An Ollama alias cannot be fine-tuned

Ollama stores inference-oriented weights, not the original trainable Hugging
Face model directory. Set `AUTOBENCHER_MODEL_MAP` or
`AUTOBENCHER_LOCAL_MODEL_PATH` to the offline Qwen source weights.

### Fine-tuning runs out of memory

- Reduce `--finetune_batch`.
- Reduce `--lora_rank`.
- Reduce `--max_seq_length` when calling `train_llm.py` directly.
- Use a smaller Qwen model.
- Ensure no vLLM worker retains the selected training GPU.

The trainer returns a nonzero code, and the main process records
`finetune_oom` while preserving the hard pool and training export.

### Disk space is below the threshold

Free space under the output volume or reduce `--disk_warning_threshold` only
after confirming the merged model and temporary adapter will fit. The run saves
`cycle_record.json` and exits before training.

### DeepSeek API timeout

Check `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, network access, and API service
status. Rerun the same command; completed iteration data is reused. Timeout
failures are recorded as `api_timeout`.

### JSON parsing fails

Model output parsing accepts fenced JSON, raw JSON surrounded by text, tagged
JSON, smart quotes, unquoted English keys, and trailing commas. Invalid output
is retried three times. If all retries fail, inspect the active `temp_log`
before cleanup and rerun with the same prefix.

### Training dependencies are missing

Install the lock file again inside the active environment:

```shell
python -m pip install -r requirements-lock.txt
```

Then verify:

```shell
python -c "import torch, transformers, peft, bitsandbytes, trl; print('ready')"
```

## Validation

Run offline regression tests:

```shell
python -m unittest discover -s tests -v
```

Check CLI wiring without starting a benchmark:

```shell
python run_scripts.py --help
python train_llm.py --help
```
