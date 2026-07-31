# Third-party references and dependencies

This file records external projects consulted or reused by the maintained math
data flywheel.

## Microsoft ToRA

- Project: https://github.com/microsoft/ToRA
- License: MIT
- Use in this repository: the evaluator strategy in
  `prompts/tora_evaluator_strategy.txt` adapts ToRA's interleaving of natural
  language planning, program-based computation, program output, and a final
  answer. AutoBencher uses a single-round, fail-closed variant: the model emits
  a plan and code, AutoBencher validates and executes the code, and a separate
  model call post-checks the executed result.
- ToRA source files, model weights, and example corpus are not vendored. Its
  unrestricted code-execution path is not used.

## XiangLi1999 AutoBencher

- Project: https://github.com/XiangLi1999/AutoBencher
- Use in this repository: the original benchmark-planning, question-generation,
  test-taker evaluation, comparison, and iterative feedback topology is
  retained and extended by the local math data-flywheel implementation.
- No additional upstream source file was vendored during the 2026-07-29
  evaluator update. The upstream repository did not display a license file
  during that review.

## Hugging Face Math-Verify

- Project: https://github.com/huggingface/Math-Verify
- License: Apache License 2.0
- Version range: `>=0.9,<1`, locked to `0.9.0`
- Use in this repository: optional mathematical parsing and equivalence fallback
  after AutoBencher's typed normalizers and before the mandatory semantic judge.

## SymPy

- Project: https://github.com/sympy/sympy
- License: BSD 3-Clause
- Version range: `>=1.13,<2`, locked to `1.13.1`
- Use in this repository: restricted evaluator-side symbolic computation,
  exact simplification, numeric evaluation, substitution, and deterministic
  equivalence checks. Evaluator-generated code can access SymPy only through
  the validated `sp` namespace inside the isolated runner.

## Hugging Face TRL

- Project: https://github.com/huggingface/trl
- License: Apache License 2.0
- Locked version: `1.8.0`
- Use in this repository: `train_llm.py` uses `SFTConfig` and `SFTTrainer` for
  supervised QLoRA training on the filtered Alpaca JSONL dataset.

## ChenghaoMou text-dedup

- Project: https://github.com/ChenghaoMou/text-dedup
- License: Apache License 2.0
- Use in this repository: `autobencher/similarity.py` contains a small
  clean-room adaptation of the project's deterministic MinHash design for
  lexical near-duplicate screening. No source file is copied and the
  `text-dedup` package is not imported.
- The upstream package currently requires Python 3.12 or newer, while this
  repository supports Python 3.10; keeping the bounded local implementation
  avoids silently narrowing AutoBencher's supported runtime.

## ekzhu datasketch

- Project: https://github.com/ekzhu/datasketch
- License: MIT
- Version range: `>=2.0,<3`, locked to `2.0.0`
- Use in this repository: production MinHash signatures and an in-memory
  MinHashLSH candidate index. LSH candidates are always rechecked with the
  configured similarity score; smoke mode alone may use the clean-room
  fallback when the dependency is unavailable.

## Hugging Face Sentence Transformers

- Project: https://github.com/huggingface/sentence-transformers
- License: Apache License 2.0
- Version range: `>=5.6,<6`, locked to `5.6.0`
- Use in this repository: sentence embeddings and cosine similarity provide a
  semantic leakage guard between generated training questions and the fixed
  holdout, and a second-stage semantic deduplication guard within training data.
- Model name, revision, cache path, device, batch size, offline behavior, and
  fail-closed requirement are all configuration options.

## dottxt-ai Outlines

- Project: https://github.com/dottxt-ai/outlines
- License: Apache License 2.0
- Version range: `>=1.3,<2`, locked to `1.3.2`
- Use in this repository: preferred token-constrained Pydantic/JSON generation
  for evaluator and test-taker calls made directly against a loaded local
  Transformers model. Remote DeepSeek/OpenAI-compatible routes are unchanged.

## guidance-ai Guidance

- Project: https://github.com/guidance-ai/guidance
- License: MIT
- Version range: `>=0.3.1,<0.4`, locked to `0.3.1` without its
  `transformers` extra because that extra pins Transformers below TRL 1.8's
  supported range; AutoBencher supplies its own locked Transformers runtime.
- Use in this repository: configurable fallback JSON-schema constrained
  decoding when the local Outlines backend is unavailable or incompatible.

## Weights & Biases

- Project: https://github.com/wandb/wandb
- License: MIT
- Version range: `>=0.28,<0.29`, locked to `0.28.1`
- Use in this repository: optional Hugging Face Trainer metrics reporting.
  AutoBencher defaults to W&B offline mode; network upload requires explicitly
  selecting online mode in YAML.

## Stanford DSPy

- Project: https://github.com/stanfordnlp/dspy
- License: MIT
- Use in this repository: design reference for separating declarative
  prediction contracts, modules, and measurable optimization objectives.
  Error attribution remains deterministic at runtime; DSPy source code and a
  DSPy runtime dependency are not vendored.

## Ragas

- Project: https://github.com/vibrantlabsai/ragas
- License: Apache License 2.0
- Use in this repository: design reference for decomposing quality into
  independently measurable signals. Attribution audits report accuracy,
  selective coverage, evidence coverage, inter-annotator agreement, macro-F1,
  confidence calibration, and accuracy by verification tier.
