# AutoBencher

AutoBencher 是一个可配置的基准生成、模型评测与训练流水线。本仓库在原始 AutoBencher
流程上增加了数学答案验证、数据过滤、断点恢复和本地微调支持。

English documentation: [README.md](README.md)

## 开源项目声明

本项目基于或参考了以下开源项目的代码、工具或设计：

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

以下公开数据集或仓库仅在用户显式准备数据时使用，或作为设计参考：

- [OpenAI GSM8K](https://github.com/openai/grade-school-math)
- [Hendrycks MATH](https://github.com/hendrycks/math)
- [Hendrycks MMLU](https://github.com/hendrycks/test)
- [DeepMind Mathematics Dataset](https://github.com/google-deepmind/mathematics_dataset)

项目不会自动下载这些上游数据集。使用前必须检查实际数据归档的许可证与再分发条款。
依赖版本、许可证、固定 revision 及具体使用方式见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
cp .env.example .env
```

Windows 使用 `.\.venv\Scripts\Activate.ps1` 激活环境。`.env` 只能保存在本地，禁止提交
或打印密钥。

运行数据应写入配置的数据目录，不要写入源码仓库。生产配置使用：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel/
```

## 基础检查

以下命令为离线检查，不会启动 API 推理或 GPU 训练：

```bash
python verify_project.py
python smoke_test.py
python -m pytest -q
```

## 基础运行

选择实验 YAML、环境 YAML，并设置唯一的运行 ID：

```bash
python run_scripts.py math \
  --config <experiment.yaml> \
  --environment <environment.yaml> \
  --run-id <run-id>
```

使用完全相同的配置和运行 ID 恢复：

```bash
python run_scripts.py math \
  --config <experiment.yaml> \
  --environment <environment.yaml> \
  --run-id <run-id> \
  --resume true
```

运行或恢复 Study Suite：

```bash
python run_study.py --suite <study-suite.yaml>
python run_study.py --suite <study-suite.yaml> --resume
```

如果 suite 变化导致普通恢复被拒绝，应先恢复 Registry 中登记的 Git commit，再恢复对应记录：

```bash
python resume_experiment.py \
  --registry <experiment_index.json> \
  --study-id <study-id>
```

聚合已完成的 Study：

```bash
python aggregate_study.py --index <experiment_index.json>
```

查看 checkpoint 绑定评测入口的参数：

```bash
python run_formal_evaluation.py --help
```

组装评测集时，应同时提供训练记录与生成记录进行泄漏检查：

```bash
python prepare_evaluation_sets.py assemble-official \
  --candidates <candidates.json> \
  --training-data <training-records.json> \
  --generation-data <generation-records.json> \
  --output <evaluation-set.json>
```
