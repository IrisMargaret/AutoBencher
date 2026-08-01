# AutoBencher Math Data Flywheel

[简体中文](README.zh-CN.md) | English

AutoBencher is an adaptive mathematics benchmark and local training-data
flywheel. It keeps the original AutoBencher loop—plan, generate, evaluate,
collect hard examples, train, and repeat—while making gold-answer generation,
answer judging, holdout evaluation, and cache retention explicit and auditable.

The maintained entry point is `math_autobencher.py`. The former Wiki and
Multilingual entry points have been removed from this math-only repository.
See [Baselines and ablations](docs/baselines_and_ablations_zh-CN.md) for the
seven reproducible study methods, component matrix, commands, and manifest
checks.

## What the pipeline does

- Covers a fixed taxonomy of 9 categories and 27 subcategories.
- Separates requested difficulty from an objective, reproducible five-dimension
  profile. The observed score—not unsupported LLM self-rating—drives later
  adaptive sampling.
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
- Attributes errors only when a parser, type checker, SymPy equality,
  substitution check, or numeric signature supplies reproducible evidence;
  otherwise it abstains with `unknown_error`.
- Builds every cycle's training set only from that cycle: exactly 25% correctly
  answered examples and 75% incorrectly answered examples. Every exported
  sample contains a verified gold answer and safe, non-empty solution steps.
- Evaluates the original model and every trained cycle on one immutable,
  project-native holdout and reports overall, subcategory, answer type, and
  difficulty-stratum accuracy. No Hugging Face dataset supplies questions.
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
  -> observable five-dimension difficulty profile and effective score
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
|-- autobencher/
|   |-- config.py
|   |-- coverage.py
|   |-- dataset.py
|   |-- difficulty.py
|   |-- evaluator.py
|   |-- fixed_benchmark.py
|   |-- policies.py
|   |-- similarity.py
|   |-- structured.py
|   `-- truth_solver.py
|-- benchmarks/
|   `-- fixed_math_test_set.json
|-- configs/
|   |-- environments/
|   |-- experiments/
|   |-- studies/
|   `-- math_flywheel.yaml
|-- docs/
|-- prompts/
|-- tests/
|-- evaluate_error_attribution.py
|-- prepare_fixed_math_benchmark.py
|-- math_autobencher.py
|-- run_scripts.py
|-- train_llm.py
|-- tool_util.py
|-- THIRD_PARTY_NOTICES.md
|-- requirements.txt
`-- requirements-lock.txt
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
python -m pip check
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
```

`requirements-lock.txt` is the exact reproducible environment;
`requirements.txt` contains only supported direct-dependency ranges. Use a
fresh virtual environment when `pip check` reports a conflict—do not repair a
shared base environment by silently changing project pins.
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

## Authoritative run sequence

Run these offline checks after installation and after every code change. They
do not load a model, use a GPU, or call a paid API:

```bash
python verify_project.py
python smoke_test.py
python -m ruff check .
python -m pytest -q
```

On the Linux training server, route Python bytecode and every runtime artifact
to VEPFS, install the project-native benchmark, and run the fail-closed
preflight before any paid generation or GPU training:

```bash
export AUTOBENCHER_DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel
export PYTHONDONTWRITEBYTECODE=1

python -B prepare_fixed_math_benchmark.py \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"

python -B run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-chain-preflight-v1 \
  --preflight-only
```

The smallest real baseline → generation → QLoRA → fixed-set regression chain
is then:

```bash
python -B run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-chain-v1
```

Resume the same run, never invent a replacement run ID:

```bash
python -B run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-chain-v1 \
  --resume true
```

An API connectivity probe is optional and always explicit:

```bash
python -B smoke_test.py --api --model deepseek-v4-pro
```

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
| `study` | Sampling policy, named variant, baseline parameters, and component switches. |
| `budget` | Fairness protocol, data/token/API/GPU caps, and optional pricing. |
| `generation` | Allowed difficulty range 2–6 by default, reasoning limit, retries, quota repair. |
| `difficulty` | Observable rubric, dimension weights, target tolerance, relabel/reject policy, and sampler score source. |
| `evaluator_pipeline` | Prompt paths, Python timeout/size limits, retries, semantic-judge threshold. |
| `test_taker_prompt` | No-tool, strict-JSON, reasoning and output-injection limits. |
| `answer_normalization` | Numeric tolerances, symbolic rules, Math-Verify switch. |
| `adaptive_sampling` | Previous-round global bands plus per-subcategory Beta-Binomial sampling. |
| `dataset` | Deduplication and fixed-test similarity thresholds. |
| `training_mix` | Exact 25/75 ratio and `current_cycle` scope. |
| `fixed_test` | Backward-compatible execution switches for the active evaluation set. |
| `evaluation_sets` | Development/official/blind role registry, phase gate, and model-selection prohibition. |
| `retention_test` | Optional instruction, format, simple-task, and non-target forgetting checks. |
| `finetune` | GPU/LoRA settings, template-cluster split, validation selection, early stopping, and token/step caps. |
| `structured_output` | Outlines/Guidance local JSON backends and fail policy. |
| `tracking.wandb` | W&B mode, project, group, tags, and model logging. |

The evaluator and generation difficulty bounds must match. The validated
defaults prohibit competition-level questions.

### Baselines and ablations

The unified policy interface supports `base`, `random`, `uniform`,
`error_only`, `full`, `full_no_hard_pool`, `full_no_error_targeting`,
`full_no_observed_difficulty_sampling`, and `full_no_difficulty_module`.
Study profiles live under `configs/studies/`;
the default remains `full`, so existing resolved configurations keep the
complete adaptive behavior.

```bash
python run_scripts.py math \
  --config configs/studies/uniform.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id uniform-seed-42
```

Definitions, fair-comparison controls, additional commands, and manifest
inspection are documented in
[docs/baselines_and_ablations_zh-CN.md](docs/baselines_and_ablations_zh-CN.md).

Freeze the committed baseline before starting a study. The manifest hashes the
actual generator, test-taker, and semantic-judge prompt files—not a version
label—and records Git, Python/CUDA, package, model, fixed-test, and resolved
configuration fingerprints:

```bash
python freeze_baseline.py \
  --baseline-id baseline-ablation-v1 \
  --config configs/studies/full.yaml \
  --environment configs/environments/volcengine.yaml

git tag -a baseline-ablation-v1 -m "Frozen ablation baseline v1"
```

The worktree must be clean unless `--allow-dirty` is explicitly used for a
provisional, non-paper manifest. The production output defaults to
`/vepfs-mlp2/queue010/20262202597/math_flywheel/baselines/`.

Run the complete first-round smoke matrix with one command:

```bash
python run_study.py \
  --suite configs/study_suites/smoke.yaml \
  --dry-run

python run_study.py \
  --suite configs/study_suites/smoke.yaml
```

Use `--resume` after interruption. A `study_id` maps to one content-addressed
`run_<study-id>_<sha12>/` directory, so a partial experiment reopens the same
cycle records, history, Hard Pool, ledger, and checkpoints. Before mutable
state is opened, resume verifies the configuration, Git commit, prompt bundle,
and base-model hashes. Completed experiments are skipped; only pending,
failed, or partial experiments are continued. `main.yaml` expands
nine methods × three seeds × one model × one total question budget. Every
matrix cell has its own outputs, cache, hard-pool/history state, datasets,
checkpoints, and model directory. The runner rejects dirty code, changed
fingerprints, non-identical base models/fixed tests/prompts, and any
non-strategy configuration difference among the six training methods.

```bash
python run_study.py \
  --suite configs/study_suites/main.yaml

python aggregate_study.py \
  --index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/experiment_index.json
```

A zero process exit code is necessary but not sufficient. The runner marks a
cell completed only when its bound run manifest, summary, finished ledger,
fixed-test snapshot and complete item results, resolved config, trained model
(for non-Base methods), and artifact hashes all agree with the registry. The
registry stores exact artifact paths and hashes; aggregation never searches by
modification time. Formal aggregation fails closed if any preregistered cell is
incomplete. For exploratory diagnostics only, opt in explicitly:

```bash
python aggregate_study.py \
  --index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/experiment_index.json \
  --allow-partial-development-results
```

`budgets` in a suite are total generated-question budgets. They must divide
evenly by `experiment.num_iterations * experiment.max_cycles`; the runner
refuses an inexact split instead of silently changing experimental cost.
`full_no_observed_difficulty` is retained only as a deprecated input alias and
resolves to the canonical `full_no_observed_difficulty_sampling`.

Run `configs/study_suites/ablation_round2.yaml` only after the first round is
stable. Posterior staleness is tested separately with
`configs/study_suites/history_modes.yaml`: `cumulative` uses \(w_i=1\),
`cycle_reset` uses \(w_i=\mathbf{1}[c_i=c_t]\), and the production default
`time_decay` uses \(w_i=\exp[-\lambda\max(0,c_t-c_i)]\) with
`decay_lambda: 0.5`. Run manifests and per-iteration generation plans store
the effective weights and runtime assertions for disabled components.

### Fair budgets and paper tables

Every run writes `budget_ledger.json` from actual generator, SymPy, judge,
deduplication, training, GPU, and timing events. It separately records selected
pool, train, validation, internal-test, and actually trained counts.
`data_matched` accumulates filtered candidates until methods have the same
post-template-split train count and the same configured correct/error ratio;
`generation_token_matched` reserves the estimated prompt plus declared maximum
output before every generator call and stops at the shared generation-token
cap. It is intentionally not called cost-matched: judge tokens, retries,
training tokens, and GPU time are measured but are not one fungible hard
budget. Provider usage is exact when available; fallback counts set
`cost_quality=estimated` or `mixed` and cannot be marked complete. The ledger
also records the last-call cost and any overshoot.

```bash
python -B run_study.py \
  --suite configs/study_suites/fair_budget.yaml

python -B aggregate_study.py \
  --index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/fair_budget_v1/experiment_index.json
```

Aggregation rebuilds `results_long.csv` from raw fixed-item comparisons and
creates main, ablation, category, difficulty, efficiency, and significance
tables without mixing evaluation-set hashes, budget protocols, budgets,
methods, variants, or seeds. Significance tests run only preregistered pairs;
the primary test combines per-seed McNemar results and reports a seed/item
cluster-bootstrap interval. Across-seed summaries use a Student-t 95% interval
(critical value 4.303 for three seeds), not the large-sample 1.96 shortcut.
Pooled McNemar is descriptive only. See
[Fair budgets and statistics](docs/fair_budget_and_statistics_zh-CN.md).

### Observable difficulty definition

`difficulty` is the effective score used for bucketing and adaptive sampling.
It is no longer copied blindly from the generation plan. Every solved question
stores three related values:

- `target_difficulty`: the score requested by the scheduler;
- `observed_difficulty`: the score recomputed after solving;
- `difficulty`: the effective score used by the next sampling round.

The `observable_math_v1` rubric derives the observed 1–10 score from five
stored dimensions:

| Dimension | Default weight | Observable meaning |
| --- | ---: | --- |
| Reasoning steps | 0.30 | Number of dependent verified transformations |
| Operation count | 0.20 | Mathematical operators and functions |
| Constraint count | 0.20 | Equations, inequalities, and domain conditions |
| Symbolic depth | 0.20 | Variables, functions, powers, and nesting |
| Representation load | 0.10 | Translation, units, rates, cases, matrices, or geometry |

The bands have explicit interpretations: 1–2 foundational, 3–4 routine
multi-step, 5–6 integrated, 7–8 advanced, and 9–10 expert. The production
generator remains limited to 2–6. Large literals, verbose stories, obscure
names, and unnecessary arithmetic do not independently increase the score.

`difficulty.mismatch_action: relabel` preserves a mathematically valid,
SymPy-verified question but uses its observed score and records the target gap.
`reject` instead discards questions outside `target_tolerance`. Production and
the 27-question profile reject observed scores outside the configured 2–6
range; the 8-question functional profile records and relabels them without
spending its small repair budget.

`observable_math_v1` is the structural baseline. Estimate and freeze
`calibrated_math_v2` only on a dedicated calibration set, never on a fixed or
blind test:

```bash
python calibrate_difficulty.py prepare-panel \
  --questions /vepfs-mlp2/queue010/20262202597/math_flywheel/difficulty/questions.jsonl \
  --panel-config configs/difficulty_panels/panel_v1.yaml \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/difficulty/panel_tasks.json

python calibrate_difficulty.py calibrate \
  --questions /vepfs-mlp2/queue010/20262202597/math_flywheel/difficulty/questions.jsonl \
  --responses /vepfs-mlp2/queue010/20262202597/math_flywheel/difficulty/panel_responses.jsonl \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/difficulty/calibrated_math_v2.json \
  --minimum-model-coverage 0.80 --minimum-models-per-item 3 \
  --minimum-models-per-tier 1 --maximum-missing-rate 0.20 \
  --freeze
```

The frozen artifact reports panel error rates, Pearson/Spearman/MAE, binned
calibration, model-tier consistency, Rasch/1PL, exploratory 2PL, and calibrated
five-dimension weights. Rasch identification fixes only mean item difficulty
to zero; panel ability is not separately recentered. The artifact includes
optimization loss/convergence and model-cluster-bootstrap item standard errors.
Five difficulty dimensions use frozen raw scales (or validated pre-normalized
values); missing, non-finite, or out-of-range dimensions fail. Reported
calibrated correlations/MAE are out-of-fold, while full-data fit is explicitly
descriptive. Duplicate model×item rows, insufficient model/item coverage, or
fewer than three declared ability tiers fail. Every response binds model and
tokenizer directory SHA, prompt SHA, the complete decoding config plus its SHA,
provider revision, and raw response, so a reused model ID cannot silently
change snapshots. Duplicate panel model IDs fail during schedule creation. 2PL remains
explicitly exploratory when the panel is small. A v2 runtime config must use
the artifact's SHA-256 and exact weights.

### Adaptive difficulty and question allocation

The next round uses the immediately previous test-taker round—not the fixed
holdout score—as a coarse global difficulty guardrail:

| Previous-round overall accuracy | Default next-round bias |
| --- | --- |
| Below `0.30` | Decrease difficulty by 1 |
| `0.30` through `0.60` | Keep difficulty |
| Above `0.60` | Increase difficulty by 1 |

The global rule starts after
`adaptive_sampling.global_accuracy_min_observations: 10`. Configure the two
bounds with `global_accuracy_low` and `global_accuracy_high`, the step with
`global_difficulty_step`, and the safety cap with
`max_difficulty_change_per_iteration`. The next target is always clamped to
`generation.minimum_difficulty` and `generation.maximum_difficulty` (2–6 by
default).

This does not replace AutoBencher's adaptive framework. The global bias is
combined with the original per-subcategory and per-difficulty Beta-Binomial
posterior, coverage quota deficits, uncertainty, persistent errors, retention
probes, and eligible hard-pool variants. The local posterior target remains
separately configurable as `target_accuracy_low/mid/high` (0.10/0.20/0.30 by
default). A subcategory also resumes from its most recently sampled difficulty
instead of resetting every round. The Beta-Binomial state is keyed by the
observed/effective difficulty, so a question requested at 5 but objectively
scored at 3 teaches the sampler about difficulty 3.

## Running

Install the checked-in project-native holdout into VEPFS before the first
server run. This command is offline and does not access Hugging Face datasets:

```bash
export AUTOBENCHER_DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel

python prepare_fixed_math_benchmark.py \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

The installer validates schema, taxonomy coverage, unique IDs, answers, and
question provenance before writing. It rejects GSM8K, Hendrycks MATH, and MMLU
provenance, writes the exact immutable bytes plus a SHA-256 manifest, refuses
an accidental overwrite, and is idempotent when the same file already exists.

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

This profile evaluates the original model on the 81-question fixed set,
generates 27 adaptive training candidates, performs one QLoRA cycle when a
complete 25/75 training block is available, and evaluates the merged model on
the same 81-question fixed set again.

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

Gold answers use disjoint exact and approximate contracts:

- Exact irrational results containing `pi`, roots, logarithms, or similar
  constants are stored as `symbolic_expression` (`symbolic` is accepted as an
  input alias), with no decimal tolerance.
- A question is classified as `decimal` only when it explicitly requests a
  decimal approximation. Its gold expression is evaluated with SymPy before
  persistence and stored as a floating-point string with
  `tolerance: 1.0e-3`.
- A symbolic test-taker response cannot pass through the decimal parser.
  Decimal gold normalization may evaluate a legacy symbolic constant, but the
  predicted decimal must already be numeric.
- Taxonomy v3 adds `numeric_approximation_error` for the evidenced case where
  reasoning retains the correct exact irrational result but the final
  approximation falls outside the grading tolerance.

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

### Evidence-based error attribution

Attribution follows a deterministic evidence ladder: protocol/parse failures,
typed normalization, unit and option checks, TruthSolver substitution,
SymPy validation of constant equalities, answer-transfer consistency, numeric
error signatures, and symbolic equivalence. Each result stores its evidence,
confidence, verification tier, first failing step, and taxonomy version. When
no check isolates a defensible mechanism, the system emits `unknown_error`
instead of inferring a cognitive cause from keywords.

For publication-quality validation, export two independently shuffled blind
packets (default 400 errors). Sampling is stratified by predicted tag,
confidence, verification tier, category, and difficulty; inverse-probability
weights recover population metrics. Annotators see only the question, gold,
raw answer/reasoning, and the frozen codebook. System labels, evidence, and
first-error predictions are stored in an operator-only directory outside the
public packets. Merge completed packets, adjudicate every disagreement, then
score the merged file:

```bash
python evaluate_error_attribution.py export-blinded \
  --input /vepfs-mlp2/queue010/20262202597/math_flywheel/review/errors.jsonl \
  --output-dir /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1 \
  --sealed-output-dir /vepfs-mlp2/queue010/20262202597/math_flywheel/review_private/blind_v1

python evaluate_error_attribution.py merge \
  --system-predictions /vepfs-mlp2/queue010/20262202597/math_flywheel/review_private/blind_v1/system_predictions.csv \
  --annotator-1 /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1/annotator_1.csv \
  --annotator-2 /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1/annotator_2.csv \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1/adjudication.csv

python evaluate_error_attribution.py score \
  --review-csv /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1/adjudication.csv \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/review/blind_v1/metrics.json
```

The report includes coverage, selective accuracy, Macro-F1, Cohen's kappa,
Brier score, a confusion matrix, and the unknown-error ratio. The conservative
readiness gate requires at least 300 resolved samples, at least 50
high-confidence samples, kappa ≥ 0.70, high-confidence accuracy ≥ 0.80,
at least 90% completion, no unresolved adjudication in formal mode, and the
configured minimum count for every observed label. Unknown or misspelled
labels fail against the frozen taxonomy.

## Fixed test and leakage protection

The server environment uses the immutable project-native
`benchmarks/fixed_math_test_set.json` artifact under VEPFS. It is copied from
the checked-in benchmark without any dataset download. GSM8K, Hendrycks MATH,
MMLU, and other Hugging Face-hosted questions are excluded from the active
evaluation chain. The v3 benchmark contains 81 original questions: three for
each of the 27 subcategories, spanning basic, intermediate, and advanced
difficulty intent. GSM8K, MATH, MMLU, and DeepMind Mathematics inform only the
capability distribution; no external question text is copied.

- At startup, the original test taker is evaluated and its baseline accuracy is
  stored.
- After every successful training cycle, the merged model is evaluated on the
  same dataset.
- Each cycle summary records baseline accuracy, current accuracy, and delta.
- Summaries also report per-source and per-difficulty accuracy, observable
  dimension means, and the mean absolute target/observed difficulty gap.
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

The auxiliary retention holdout contains 120 project-authored items across six
dimensions (instruction, language transformation, strict formatting, stable
knowledge, logic, and non-target string reasoning). It is included in every
training-leakage filter and reports Wilson 95% intervals for overall and
per-dimension forgetting. Development, retention, every locally visible
non-blind registry set, and frozen official questions are holdouts; blind text
remains unavailable to the training process.

Official and blind evaluation must be launched through the checkpoint-bound
runner. Direct loading of either YAML fails until source study, method, seed,
run ID, checkpoint path, and checkpoint SHA are supplied and verified against
the completed experiment registry:

```bash
python run_formal_evaluation.py \
  --config configs/experiments/official_fixed_eval.yaml \
  --environment configs/environments/volcengine.yaml \
  --source-index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/experiment_index.json \
  --source-study-id '<registered-study-id>' \
  --source-method full --source-seed 42 \
  --checkpoint-path /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/.../model \
  --checkpoint-sha256 '<directory-sha256>' \
  --run-id official-full-seed42
```

Official assembly requires every item either to pass independent recomputation
or to pass two independent human reviews plus a distinct adjudicator when the
reviews conflict. It also requires zero unresolved conflicts, two distinct
validation methods and actors (including distinct model snapshot hashes),
template/lexical/math-AST/embedding
leakage checks, and an audit manifest binding candidate and training-corpus
hashes, thresholds, algorithm version, and report hash. Blind release also
requires a pre-registered or independently provisioned release-token SHA-256;
an arbitrary non-empty token is never sufficient.

Hard-pool samples now move through `active`, `mastered`, `stale`, and `retired`.
Each new model version retests high-priority active/stale samples. Repeatedly
correct samples become mastered; aged samples become stale/retired; only active
samples may guide prompts or enter legacy hard-pool training export.

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

Template clusters are assigned as indivisible groups with a target-deviation
objective rather than placing the largest clusters into fixed splits. The
25/75 ratio is re-enforced on the actual train split, and manifests report
train/validation/internal-test fractions plus category, source, correctness,
and difficulty distributions. `data_matched` matches the post-split train
sample count and ratio. If the training-token cap truncates data, the trainer
uses deterministic round-robin sampling over category, source, correctness,
difficulty, and template strata instead of retaining a file prefix. Effective
optimizer steps bound eval/save intervals so even a short capped run evaluates
and saves at least once.

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

The configured output root contains one stable, content-addressed
`run_<normalized-run-id>_<sha12>/` directory per `run_id`. Resuming reopens
that exact directory; it never allocates a `test_2` replacement. Study cells
place these bound run directories beneath their isolated suite/method/seed
directories. Server configurations enforce
`/vepfs-mlp2/queue010/20262202597/math_flywheel`; writable output, caches,
temporary files, logs, datasets, checkpoints, and model artifacts outside that
root are rejected. Do not locate research results by modification time; use
the exact run path recorded in `experiment_index.json`.

A completed run has this top-level contract:

```text
run_<id>_<sha12>/
├── run_manifest.json
├── resolved_config.json
├── resolved_config.yaml
├── config_sources.json
├── config_validation.json
├── environment.json
├── budget_ledger.json
├── cycle_record.json
├── experiment_summary.json
├── artifact_manifest.json
├── logs/
├── cycle/
└── fixed_test/
```

`artifact_manifest.json` uses layout `research_run_v2`, lists every regular
file with path, byte size, kind, and SHA-256, and refuses finalization while a
`.tmp` artifact remains. Generated artifact names reject path traversal and
unsafe characters. Binary model payloads are hashed as well as their
checkpoint manifests.

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

The evaluation questions are project-native and do not import questions from external
datasets. Error-attribution contracts follow the declarative separation
encouraged by
[DSPy](https://github.com/stanfordnlp/dspy), while the audit report decomposes
accuracy, coverage, evidence, agreement, and confidence calibration in the
spirit of [Ragas](https://github.com/vibrantlabsai/ragas). DSPy and Ragas are
design references, not runtime dependencies.

See `THIRD_PARTY_NOTICES.md` for the precise reuse boundary and license notes.

## Evaluation hierarchy and training selection

`configs/evaluation_sets.yaml` gives every evaluation set an explicit role:

- `development_regression_v3` is the existing 81-question fast regression set.
  It may be used during development but is not evidence for fine-grained paper
  claims.
- `official_fixed_v1` requires 540 questions (20 for each of 27
  subcategories), two validation sources per item, version metadata, and an
  immutable file hash. It cannot enter training, the hard pool, generator
  prompts, method selection, or threshold tuning.
- `blind_final_v1` has no repository-visible path or hash. It loads only in
  `blind_evaluation` phase with an explicit release token and matching
  out-of-band SHA-256. Fine-tuning is rejected in that phase.

The historical `fixed_math_test_set_v3.json` is not changed or deleted. New
runs install the same 81 immutable questions as `fixed_math_dev_v3.json`, whose
manifest explicitly permits regression/debugging and prohibits training or
gold-prompt use. For a lower-variance development-only evaluation, create the
deterministic 540-item static stress set (20 items per subcategory):

```bash
python -B prepare_large_development_benchmark.py \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_development_static_540_v1.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"

python -B run_scripts.py math \
  --config configs/experiments/development_static_540_eval.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id dev-static-540-v1
```

This set is project-authored and only adopts coverage/task-design ideas from
GSM8K, MATH, and the DeepMind Mathematics Dataset; no external question is
copied. Its manifest marks it unsuitable for blind or paper claims. The formal
540-item set must still pass the independent-solver, reviewer, diversity, and
training-leakage gates in `prepare_evaluation_sets.py assemble-official`.

To construct a separate open-source-derived candidate pool, provision the
pinned upstream test files under
`$AUTOBENCHER_DATA_ROOT/source_datasets` using the layout in
`configs/open_source_evaluation_sources.yaml`, then run:

```bash
python -B prepare_open_source_math_benchmark.py \
  --source-root "$AUTOBENCHER_DATA_ROOT/source_datasets" \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/official_candidates_v1.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

This command performs no download. It deterministically imports only upstream
test splits, removes exact duplicates, caps template clusters, balances all 27
subcategories at 20 items, and records source revisions plus per-file SHA-256.
It fails closed when any subcategory is short. Its taxonomy assignments are
explicitly marked for human review, and every answer initially has only the
upstream source, so the result is not release-ready. Reviewers must confirm the
taxonomy and add independent validation evidence before `assemble-official`.
Using public benchmarks supports cross-benchmark evaluation, but does not prove
pretraining-unseen generalization; that stronger claim remains reserved for the
independently held blind set.

The same open-source candidate format can be distilled offline into a
270-record generation-guidance set (10 records for each of 27 subcategories):

```bash
python -B prepare_generation_guidance.py \
  --source-candidates "$AUTOBENCHER_DATA_ROOT/benchmarks/official_candidates_v1.json" \
  --output "$AUTOBENCHER_DATA_ROOT/guidance/open_source_generation_guidance_v1.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

This artifact is neither an evaluation set nor direct fine-tuning data. The
builder retains only taxonomy, difficulty band, answer contract, reasoning
structure, provenance hashes, and abstract variation instructions; public
question text, answers, and solutions are excluded. Selection rejects exact
normalized matches, shared parameterized templates, and unigram/bigram
near-duplicates above the configured threshold. Each subcategory receives ten
distinct variation axes. `main.yaml`, `fair_budget.yaml`, and
`ablation_round2.yaml` inject at most three abstract records per generation
prompt and preserve their guide IDs on generated questions; `smoke.yaml`
remains disabled. Baseline manifests, Study Runner fairness checks, and resume
identity checks bind the guidance file SHA-256. Public test-derived guidance
does not establish pretraining-unseen generalization; confirmatory claims still
require a source-isolated frozen official set and independently held blind set.

Regrade an old run entirely offline without changing original artifacts:

```bash
python -B regrade_run.py \
  --run-dir "$AUTOBENCHER_DATA_ROOT/run_mini-chain-v2_62e2915f6644" \
  --evaluator-version typed_equivalence_v2 \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

Blind construction consumes a separately curated candidate pool and all known
development/training/generation corpora. The seed path is supplied out of band
and is neither printed nor written to the manifest:

```bash
export AUTOBENCHER_BLIND_SEED_FILE=/secure/blind_seed.bin
python -B prepare_blind_math_benchmark.py \
  --candidates "$AUTOBENCHER_DATA_ROOT/benchmarks/blind_candidates.json" \
  --contamination-data "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json" \
  --contamination-data "$AUTOBENCHER_DATA_ROOT/audits/all_training_and_generation.json" \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/blind/fixed_math_blind_v1.json" \
  --manifest-output "$AUTOBENCHER_DATA_ROOT/benchmarks/blind/fixed_math_blind_v1.manifest.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

`run_formal_evaluation.py` binds the checkpoint hash and creates a private
receipt. The same model hash × blind-set hash pair is rejected on a second run.

Audit the development set and write the report only to VEPFS:

```bash
python prepare_evaluation_sets.py audit-development \
  --config configs/math_flywheel.yaml \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/development_regression_v3.audit.json
```

The command returns exit code 2 while any item needs manual review or
adjudication; it never silently calls a typed gold answer an independent
re-solve. Assemble a curated official set only after every candidate has
`validation.status: verified` and at least two validation sources:

```bash
python prepare_evaluation_sets.py assemble-official \
  --candidates /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/official_candidates_v1.json \
  --training-data /vepfs-mlp2/queue010/20262202597/math_flywheel/audits/all_final_training_questions.json \
  --generation-data /vepfs-mlp2/queue010/20262202597/math_flywheel/audits/all_generated_questions.json \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/official_fixed_v1.json
```

Training data is split 80/10/10 by `template_signature`, never by individual
row. `split_manifest.json` proves zero template overlap. TRL receives
`dataset_train.jsonl` and `dataset_validation.jsonl`; early stopping and best
checkpoint restoration use internal validation loss only.
`dataset_internal_test.jsonl` is evaluated after selection. The training
summary records token/step caps, actual optimizer steps, best validation
metric, and `evaluation_set_used_for_model_selection: false`.

Enable the auxiliary retention set for pilot/main runs with
`retention_test.enabled=true`. Its baseline-correct to final-wrong transitions
are reported as forgetting rate by instruction, format, simple-task, and
non-target dimensions. Detailed Chinese operations are in
[docs/evaluation_and_training_protocol_zh-CN.md](docs/evaluation_and_training_protocol_zh-CN.md).

## Verification

Run the CPU test suite:

```bash
python -m pytest -q
```

Offline repository, syntax, schema, configuration, storage, benchmark, secret,
and CLI smoke checks:

```bash
python verify_project.py
python smoke_test.py
python -m ruff check .
python math_autobencher.py --help
python run_scripts.py --help
python run_study.py --help
python aggregate_study.py --help
python run_formal_evaluation.py --help
```

Real API inference, local model loading, CUDA QLoRA, and remote scheduling
require their respective credentials, model files, and hardware; unit tests do
not claim those external integrations passed.
