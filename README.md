# AutoBencher

AutoBencher is a configurable benchmark-generation, evaluation, and model-training
pipeline. This repository extends the original AutoBencher workflow with mathematical
answer verification, dataset filtering, resumable execution, and local fine-tuning.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Open-source notice

This project is based on or uses ideas and components from the following open-source
projects:

- [XiangLi1999/AutoBencher](https://github.com/XiangLi1999/AutoBencher)
- [SymPy](https://github.com/sympy/sympy)
- [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify)
- [Microsoft ToRA](https://github.com/microsoft/ToRA)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [dottxt-ai Outlines](https://github.com/dottxt-ai/outlines)
- [Guidance](https://github.com/guidance-ai/guidance)
- [datasketch](https://github.com/ekzhu/datasketch)
- [Sentence Transformers](https://github.com/huggingface/sentence-transformers)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [Ragas](https://github.com/vibrantlabsai/ragas)

The following public datasets or repositories are used only when explicitly prepared
by the user or as design references:

- [OpenAI GSM8K](https://github.com/openai/grade-school-math)
- [Hendrycks MATH](https://github.com/hendrycks/math)
- [Hendrycks MMLU](https://github.com/hendrycks/test)
- [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)

Upstream datasets are not downloaded automatically. Review the license and
redistribution terms of every exact source archive before use. Dependency versions,
licenses, revisions, and the way each project is used are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cp .env.example .env
```

On Windows, activate with `.\.venv\Scripts\Activate.ps1`. Keep `.env` local and do
not commit or print credentials.

Runtime data should be written to the configured data directory, not to the source
repository. Production configurations use:

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel/
```

## Basic checks

These commands are offline and do not start API inference or GPU training:

```bash
python verify_project.py
python smoke_test.py
python -m pytest -q
```

## Basic run

Choose an experiment YAML, environment YAML, and unique run ID:

```bash
python run_scripts.py math \
  --config <experiment.yaml> \
  --environment <environment.yaml> \
  --run-id <run-id>
```

Resume with the same configuration and run ID:

```bash
python run_scripts.py math \
  --config <experiment.yaml> \
  --environment <environment.yaml> \
  --run-id <run-id> \
  --resume true
```

Run or resume a study suite:

```bash
python run_study.py --suite <study-suite.yaml>
python run_study.py --suite <study-suite.yaml> --resume
```

If a changed suite prevents normal resume, restore the Git commit recorded in the
Registry and resume the exact registered run:

```bash
python resume_experiment.py \
  --registry <experiment_index.json> \
  --study-id <study-id>
```

Aggregate a completed study:

```bash
python aggregate_study.py --index <experiment_index.json>
```

List the checkpoint-bound evaluation options:

```bash
python run_formal_evaluation.py --help
```

When assembling an evaluation set, provide both training and generation records for
leakage checks:

```bash
python prepare_evaluation_sets.py assemble-official \
  --candidates <candidates.json> \
  --training-data <training-records.json> \
  --generation-data <generation-records.json> \
  --output <evaluation-set.json>
```
