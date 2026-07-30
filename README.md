# AutoBencher Math Data Flywheel

[简体中文](README.zh-CN.md) | English

AutoBencher is an adaptive mathematics benchmark and local training-data
flywheel. It keeps the original AutoBencher loop—plan, generate, evaluate,
collect hard examples, train, and repeat—while making gold-answer generation,
answer judging, holdout evaluation, and cache retention explicit and auditable.

The maintained entry point is `math_autobencher.py`. The former Wiki and
Multilingual entry points have been removed from this math-only repository.

## What the pipeline does

- Covers a fixed taxonomy of 9 categories and 27 subcategories.
- Generates moderate questions within configurable difficulty bounds.
- Gives the test taker no tool schema or external-tool access; it answers with
  its own reasoning under a strict JSON contract.
- Uses local SymPy as the authoritative gold-answer source, including exact
  solving and equation/system substitution. LLM-authored solver code is an
  explicit compatibility mode, not the default.
- Normalizes numeric, rational, symbolic, set, interval, tuple, matrix, Boolean,
  and textual answers before comparison.
- Uses a separate LLM semantic judge to decide whether differently formatted
  answers mean the same thing. Deterministic checks and the semantic decision
  are both retained for audit.
- Builds every cycle's training set only from that cycle: exactly 25% correctly
  answered examples and 75% incorrectly answered examples. Every exported
  sample contains a verified gold answer and safe, non-empty solution steps.
- Evaluates the original model and every trained cycle on one immutable,
  all-subcategory holdout and reports the accuracy delta.
- Rejects training questions that are identical or highly similar to the fixed
  test set using exact/template checks, datasketch MinHash/LSH informed by
  Text-Dedup, and Sentence-Transformers semantic similarity.
- The 8-question functional profile may use the deterministic built-in MinHash
  with exhaustive pair comparison when optional datasketch is absent. The
  production profiles still require datasketch and Sentence-Transformers.
- Supports Outlines with Guidance fallback for token-constrained JSON from
  directly loaded local models, and W&B tracking in configurable offline,
  online, or disabled mode.
- Cleans each completed iteration to exactly three JSON caches: the target
  question plan, test-taker answers, and comparison results.

## End-to-end architecture

```text
YAML config + explicit CLI overrides
  -> original AutoBencher quota/adaptive scheduler
  -> moderate question-only generator
  -> controlled SymPy solver clause
  -> deterministic parse + exact local execution
  -> equation/system substitution and fail-closed verification
  -> answer-anchored training reasoning from solver evidence
  -> accepted canonical gold answer
  -> no-tool test-taker reasoning
  -> deterministic normalization / Math-Verify fallback
  -> isolated LLM semantic-equivalence judge
  -> error attribution + hard pool + coverage update
  -> current-cycle 25% correct / 75% wrong training set
  -> holdout leakage filtering
  -> local 4-bit QLoRA, adapter merge, model switch
  -> fixed-test re-evaluation and accuracy delta
```

The generator, semantic judge, and test taker have separate prompts and model
calls. Gold generation itself is local and deterministic.

## Project layout

```text
AutoBencher/
├── autobencher/
│   ├── config.py
│   ├── coverage.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── fixed_benchmark.py
│   ├── output_schemas.py
│   ├── reasoning.py
│   ├── similarity.py
│   ├── structured.py
│   └── truth_solver.py
├── benchmarks/
│   └── fixed_math_test_set.json
├── configs/
│   ├── math_flywheel.yaml
│   ├── environments/
│   └── experiments/
├── prompts/
│   ├── evaluator_python_solver.txt
│   ├── evaluator_independent_solver.txt
│   ├── evaluator_postcheck.txt
│   ├── semantic_answer_judge.txt
│   └── tora_evaluator_strategy.txt
├── tests/
├── math_autobencher.py
├── run_scripts.py
├── train_llm.py
├── tool_util.py
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
└── requirements-lock.txt
```

## Requirements

- Python 3.10 or later.
- A DeepSeek/OpenAI-compatible model for question generation and evaluation.
- A local Transformers model directory, vLLM endpoint, or Ollama model for the
  test taker.
- An NVIDIA CUDA GPU for `data_flywheel` training with bitsandbytes QLoRA.
- Enough persistent storage for base weights, adapters, merged weights, and run
  artifacts.

Evaluation may run without a training GPU when the test taker is served
separately. `train_llm.py` intentionally fails on CPU for 4-bit QLoRA.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source /root/.virtualenvs/AutoBencher/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` is the reproducible environment. Use
`requirements.txt` only when compatible newer versions are intentional.
Inspect an existing CUDA installation before changing its PyTorch build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

Create `.env` from `.env.example` or configure secrets in the job platform:

```dotenv
DEEPSEEK_API_KEY=replace-me
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

Do not commit `.env`, access keys, private endpoints, or SSH credentials.

## Configuration

All tunable behavior is declared in YAML. Python code contains validated safe
defaults, not experiment-specific paths or credentials.

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/local.yaml
```

Temporary overrides do not require source changes:

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=43 finetune.gpu=1
```

The precedence order is safe defaults, inherited base YAML, environment YAML,
experiment YAML, explicit compatibility CLI arguments, then `--override`.
Unknown fields and unsafe values fail validation before model loading.

Important sections in `configs/math_flywheel.yaml`:

| Section | Purpose |
| --- | --- |
| `generation` | Difficulty 2–6 by default, reasoning limit, retries, quota repair. |
| `evaluator_pipeline` | Prompt paths, Python timeout/size limits, retries, semantic-judge threshold. |
| `test_taker_prompt` | No-tool, strict-JSON, reasoning and output-injection limits. |
| `answer_normalization` | Numeric tolerances, symbolic rules, Math-Verify switch. |
| `adaptive_sampling` | Previous-round global bands plus per-subcategory Beta-Binomial sampling. |
| `dataset` | Deduplication and fixed-test similarity thresholds. |
| `training_mix` | Exact 25/75 ratio and `current_cycle` scope. |
| `fixed_test` | Dataset path, taxonomy coverage, baseline and post-cycle runs. |
| `finetune` | GPU, epochs, batch, LoRA rank, sequence length, learning rate. |
| `structured_output` | Outlines/Guidance local JSON backends and fail policy. |
| `tracking.wandb` | W&B mode, project, group, tags, and model logging. |

The evaluator and generation difficulty bounds must match. The validated
defaults prohibit competition-level questions.

### Adaptive difficulty and question allocation

The next round uses the immediately previous test-taker round—not the fixed
holdout score—as a coarse global difficulty guardrail:

| Previous-round overall accuracy | Default next-round bias |
| --- | --- |
| Below `0.35` | Decrease difficulty by 1 |
| `0.35` through `0.70` | Keep difficulty |
| Above `0.70` | Increase difficulty by 1 |

The global rule starts after
`adaptive_sampling.global_accuracy_min_observations: 10`. Configure the two
bounds with `global_accuracy_low` and `global_accuracy_high`, the step with
`global_difficulty_step`, and the safety cap with
`max_difficulty_change_per_iteration`. Difficulty is always clamped to
`generation.minimum_difficulty` and `generation.maximum_difficulty` (2–6 by
default).

This does not replace AutoBencher's adaptive framework. The global bias is
combined with the original per-subcategory and per-difficulty Beta-Binomial
posterior, coverage quota deficits, uncertainty, persistent errors, retention
probes, and eligible hard-pool variants. The local posterior target remains
separately configurable as `target_accuracy_low/mid/high` (0.10/0.20/0.30 by
default). A subcategory also resumes from its most recently sampled difficulty
instead of resetting every round.

## Running

For the complete low-cost 8-question path (baseline -> SymPy-verified
generation -> QLoRA -> fixed-set re-evaluation), see
[`docs/mini_flywheel_zh-CN.md`](docs/mini_flywheel_zh-CN.md).

Evaluation only:

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode eval \
  --num-iters 2
```

Complete 27-question quick flywheel (one iteration, one cycle, one epoch):

```bash
python run_scripts.py math \
  --config configs/experiments/quick_flywheel_27.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id quick-flywheel-27
```

This profile evaluates the original model on the 27-question fixed set,
generates 27 adaptive training candidates, performs one QLoRA cycle when a
complete 25/75 training block is available, and evaluates the merged model on
the fixed set again.

Complete data flywheel:

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode data_flywheel \
  --num-iters 5 \
  --max-cycle 3 \
  --finetune-gpu 0 \
  --finetune-epoch 3 \
  --finetune-batch 8 \
  --lora-rank 8
```

The legacy long-form arguments accepted by `math_autobencher.py` remain
available for existing math workflows. Explicit CLI values override YAML.

## Gold answer pipeline

The default `generation.gold_solver_backend: sympy` path never accepts an
LLM-produced gold answer and does not ask an LLM to author solver code.
DeepSeek emits question text only. The local deterministic solver parses the
controlled final solver clause, computes an exact result with SymPy, verifies
equation/system answers by substitution, and derives answer-anchored training
steps from that evidence. Unsupported or ambiguous questions fail closed and
are replaced by quota repair.

The older multi-call LLM-authored Python evaluator remains available only as
the explicit compatibility setting `gold_solver_backend: llm_python`.

In that non-default compatibility mode:

1. `evaluator_python_solver.txt` asks the primary evaluator to analyze the
   problem and return only `analysis_summary` and problem-specific
   `python_code`. Its
   configurable `tora_evaluator_strategy.txt` policy adapts ToRA's
   plan–program–output–answer pattern to a single fail-closed round.
2. `evaluator_independent_solver.txt` gives a second solver only the original
   question. It cannot see the first solver's reasoning, code, or answer.
3. Both programs are parsed with `ast`; imports, file/network access, dynamic
   execution, private attributes, definitions, and unsupported calls are
   rejected.
4. Each validated program runs with `python -I`, a temporary working directory, a
   minimal environment, restricted built-ins, output limits, and a hard
   timeout. It must return a canonical answer and pass its own verification and
   substitution checks. The two runtime answers must be equivalent.
5. `evaluator_postcheck.txt` runs in a fresh model call and sees only the
   question plus the two verified runtime results. It independently
   adjudicates the answer and rejects out-of-band difficulty.
6. Both runtime answers and the adjudicated answer must be deterministically
   equivalent and have compatible answer types.

The existing deterministic `TruthSolver` remains as an independent
cross-check where it supports the question. A disagreement rejects the
question rather than silently choosing one answer.

Generated Python and raw evaluator analysis exist only in disposable evaluator
cache fragments and never enter the test-taker prompt. A bounded, role-free
`analysis_summary` is retained separately as the verified gold solution steps
for training; code, prompt fragments, and the independent solver transcript are
excluded.

DeepSeek-compatible requests intentionally omit `max_tokens`; the evaluator
therefore is not truncated by an application output-token cap. Local model
generation remains bounded for process safety.

API calls in planning, question generation, gold solving, postchecking,
semantic judging, and API-backed test-taker inference all honor the configured
`request_timeout_seconds`, `max_retries`, and evaluator
`retry_delay_seconds`. Runtime logs emit `[API] request_start`,
`request_done`, and `request_retry`, while generation emits
`[Generate] subcategory_start` and `subcategory_done`. A provider stall
therefore times out and retries instead of leaving the process silently
blocked.

Only the non-default `llm_python` compatibility mode uses the three-stage
evaluator chain. In that mode,
`evaluator_pipeline.max_parallel_questions` controls the worker count.

## Test-taker isolation and grading

The test taker receives one question at a time and no external tools. Its
response must be one JSON object:

```json
{
  "reasoning_summary": ["short reasoning step", "self-check step"],
  "final_answer": "canonical answer text",
  "answer_type": "integer",
  "confidence": 0.9
}
```

Tool traces, prompt echo, irrelevant output, malformed JSON, or unapproved
fields fail closed. Formatting is normalized before grading so equivalent
fractions, decimals, symbolic forms, sets, intervals, tuples, and matrices are
not rejected merely because they are written differently.

Every otherwise valid answer also goes through the isolated semantic judge. It
compares only the question, canonical gold, answer type, normalized forms, and
test-taker answer. The final record retains both deterministic evidence and the
LLM decision, including confidence and disagreement flags.

## Fixed test and leakage protection

`benchmarks/fixed_math_test_set.json` is immutable input to the training loop and
contains one moderate question for every configured subcategory.

- At startup, the original test taker is evaluated and its baseline accuracy is
  stored.
- After every successful training cycle, the merged model is evaluated on the
  same dataset.
- Each cycle summary records baseline accuracy, current accuracy, and delta.
- Fixed-test questions never enter the hard-example training candidates.
- Training candidates are rejected on exact question match, normalized
  template match, token overlap, TF-IDF cosine similarity, datasketch
  MinHash/LSH, or Sentence-Transformers cosine similarity above configurable
  holdout thresholds.

The default semantic backend is
`sentence-transformers/all-MiniLM-L6-v2`. Its model name, revision, device,
cache directory, batch size, and `local_files_only` behavior are configurable
under `dataset`. In data-flywheel mode this guard is required and fails closed:
if the embedding model cannot load, training export stops instead of silently
allowing possible test leakage. The first online run may download the selected
model; provision its cache in advance and enable
`sentence_transformers_local_files_only` for an offline deployment.

Keep the test-set file under version control. Changing it changes its SHA-256
and creates a different benchmark; do not compare the resulting scores as if
they came from the same holdout.

## Training dataset contract

Only records from the current cycle are eligible. After quality filtering,
holdout filtering, and deduplication, selection uses complete four-example
blocks:

- 1 correct retention example (25%);
- 3 wrong examples from the cycle's wrong pool (75%).

The wrong portion is distributed among boundary errors, coverage repair, and
format/instruction errors according to `training_mix`. If a source is short,
other wrong records may fill its share, but correct records never fill a wrong
slot. Incomplete four-example blocks are not exported. The manifest records the
selected counts and asserts the exact ratio before fine-tuning.

Every eligible record must also contain a concrete
`gold_reasoning_summary` within the configured step and character limits. The
default path derives these steps from SymPy's exact result and substitution
evidence, including the verified final answer and an independent check. Plan-only
lists such as `["compute", "solve", "check"]`, missing steps, role injection,
tool requests, Markdown fences, ungrounded answers, and oversized steps are
rejected. Alpaca output always contains these postchecked steps,
`final_answer`, `answer_type`, and confidence—there is no generic placeholder
solution.

QLoRA is implemented with Hugging Face TRL's `SFTConfig`/`SFTTrainer`. When
`tracking.wandb.enabled` is true, Trainer reports metrics to W&B. The default
`offline` mode writes local run data without uploading; switching to `online`
is an explicit configuration choice.

## Outputs and cache retention

Each run stores resolved configuration, source provenance, validation results,
configuration hash, environment snapshot, manifests, logs, global hard pool,
training artifacts, model artifacts, and fixed-test results.

The configured output root contains one automatically allocated `test_<N>`
directory per invocation. Server configurations enforce
`/vepfs-mlp2/queue010/20262202597/math_flywheel`; writable output, caches,
temporary files, logs, datasets, checkpoints, and model artifacts outside that
root are rejected. The newest run can be located with
`ls -dt /vepfs-mlp2/queue010/20262202597/math_flywheel/test_* | head -1`.

With `experiment.clean_cycle_cache: true`, every iteration directory contains
exactly these JSON files after its `finally` cleanup:

```text
cycle/cycle_<N>/iter_<M>/
├── <prefix>.question_plan_with_aim.json
├── <prefix>.test_taker_inference.json
└── <prefix>.compare_answers.json
```

All `subcat*.json`, generator attempts, evaluator Python caches, temporary
judgments, malformed/empty JSON, and redundant iteration summaries are removed.
Cleanup is idempotent and runs after success or failure. Cycle-level training,
metrics, hard-pool, and fixed-test artifacts are retained because they are
results, not iteration caches.

Fixed-test outputs are kept separately:

```text
fixed_test/
├── dataset_snapshot.json
├── baseline/
│   ├── fixed_math.test_taker_inference.json
│   ├── fixed_math.compare_answers.json
│   └── summary.json
└── cycle_<N>/
    ├── fixed_math.test_taker_inference.json
    ├── fixed_math.compare_answers.json
    └── summary.json
```

The baseline runs immediately after the original test-taker is first loaded.
After every successful training and merge, the resulting model is loaded and
evaluated on the same immutable set. Each stage keeps the raw response, parsed
`reasoning_summary`, normalized answers, semantic judgment, per-answer-type
and per-subcategory statistics, confidence metrics, and accuracy—not only a
single aggregate score.

## Open-source foundations

The benchmark loop retains the planning, generation, evaluation, and feedback
topology of [XiangLi1999/AutoBencher](https://github.com/XiangLi1999/AutoBencher).
The evaluator strategy adapts the MIT-licensed
[Microsoft ToRA](https://github.com/microsoft/ToRA) tool-integrated reasoning
pattern, while replacing free-form tool execution with the restricted local
runtime described above.

The project also uses Hugging Face
[Math-Verify](https://github.com/huggingface/Math-Verify) as a maintained,
Apache-2.0-licensed fallback for parsing and comparing mathematical answer
forms. AutoBencher's own typed normalization and SymPy checks run first, and
the isolated LLM semantic judge remains mandatory by default.

Training/holdout leakage prevention uses an Apache-2.0-compatible, clean-room
MinHash adaptation informed by
[ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) and the
MIT-licensed [datasketch](https://github.com/ekzhu/datasketch) MinHash/LSH
implementation, followed by the Apache-2.0-licensed
[Hugging Face Sentence Transformers](https://github.com/huggingface/sentence-transformers).
Local constrained JSON uses
[Outlines](https://github.com/dottxt-ai/outlines) with
[Guidance](https://github.com/guidance-ai/guidance) fallback. Fine-tuning uses
[TRL](https://github.com/huggingface/trl), and configurable experiment tracking
uses [Weights & Biases](https://github.com/wandb/wandb).
See `THIRD_PARTY_NOTICES.md` for the precise reuse boundary and license notes.

## Verification

Run the CPU test suite:

```bash
python -m pytest -q
```

Syntax and CLI checks:

```bash
python -m py_compile math_autobencher.py run_scripts.py tool_util.py autobencher/*.py
python math_autobencher.py --help
python run_scripts.py --help
```

Real API inference, local model loading, CUDA QLoRA, and remote scheduling
require their respective credentials, model files, and hardware; unit tests do
not claim those external integrations passed.
