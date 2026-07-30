# 开源固定测试集与证据化错题归因

本文说明如何构建不可变的开源数学测试集、运行完整数据飞轮，以及如何审计错误归因。所有下载数据、缓存、运行产物、训练集、模型和人工复核文件都必须位于：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel
```

代码可以位于 `/root/code/AutoBencher`，但程序不会把数据写入代码仓库或系统盘。

## 1. 固定测试集的组成

默认清单是 `configs/benchmarks/open_math_fixed_suite.yaml`，名义规模为 595 题：

| 来源 | 固定切分 | 数量 | 用途 |
|---|---:|---:|---|
| GSM8K | test | 100 | 小学算术、多步应用题 |
| MATH | test | 245 | 7 个主题各 35 题 |
| MMLU 数学子集 | test | 250 | 5 个数学任务各 50 题 |
| DeepMind Mathematics | interpolate / extrapolate | 默认关闭 | 可选的分布内/外泛化测试 |

选择过程不是按数据原始顺序截断，而是用“随机种子 + 样本内容 SHA-256”排序后确定性抽样。生成的 JSON 保存数据集配置、切分、原始索引、问题哈希、Hugging Face fingerprint、许可、上游地址、构建清单哈希和最终文件哈希。

构建器有以下硬约束：

- GSM8K、MATH、MMLU 只接受 `test`，填入 `train` 或 `validation` 会在下载前失败。
- DeepMind 本地适配器只接受 `interpolate/` 和 `extrapolate/`，拒绝 `train*`。
- 输出目录、Hugging Face 缓存和 DeepMind 本地数据都必须位于 VEPFS 数据根目录。
- 已生成的固定集默认不可覆盖；清单发生变化时必须换一个带版本号的输出文件。`--allow-overwrite` 只用于明确放弃旧实验的情况。
- 任一来源的可用题数低于清单配额时会失败，不会静默生成缩水测试集。
- GSM8K/MATH 的完整推理过程不会进入固定集和模型提示，仅保存最终答案及源解答哈希。

## 2. 首次构建固定测试集

在服务器执行：

```bash
cd /root/code/AutoBencher

export AUTOBENCHER_DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel
export HF_HOME="$AUTOBENCHER_DATA_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$AUTOBENCHER_DATA_ROOT/cache/huggingface/datasets"
export HUGGINGFACE_HUB_CACHE="$AUTOBENCHER_DATA_ROOT/cache/huggingface/hub"
export TRANSFORMERS_CACHE="$AUTOBENCHER_DATA_ROOT/cache/huggingface/transformers"
export TMPDIR="$AUTOBENCHER_DATA_ROOT/temp"
mkdir -p "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
  "$TRANSFORMERS_CACHE" "$TMPDIR" "$AUTOBENCHER_DATA_ROOT/benchmarks"

python prepare_open_math_benchmark.py \
  --manifest configs/benchmarks/open_math_fixed_suite.yaml \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/open_math_fixed_suite.json" \
  --cache-dir "$HF_DATASETS_CACHE" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

验收构建结果：

```bash
jq '{
  name,
  question_count,
  source_counts,
  subcategory_counts,
  builder_manifest_sha256,
  selection_policy
}' "$AUTOBENCHER_DATA_ROOT/benchmarks/open_math_fixed_suite.json"

sha256sum \
  "$AUTOBENCHER_DATA_ROOT/benchmarks/open_math_fixed_suite.json" \
  "$AUTOBENCHER_DATA_ROOT/benchmarks/open_math_fixed_suite.json.manifest.json"
```

首次构建后，应把两个 SHA-256 写入实验记录。后续实验复用同一个文件，不要重新下载和构建。若确实要改变题量或来源，应生成诸如 `open_math_fixed_suite_v2.json` 的新文件，并把它视为一个新实验。

若服务器不能联网、但缓存已经齐全，可在构建命令后追加 `--local-files-only`。

## 3. 可选的 DeepMind Mathematics

把数据解压到：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel/source_data/deepmind_mathematics
```

然后把清单中的 `deepmind_mathematics_local.enabled` 改为 `true`。不要把训练目录加入 `patterns`。建议将 `interpolate` 作为额外分布内测试，将 `extrapolate` 单独报告为分布外泛化指标，不要把二者与主测试集准确率混为一个数字。

## 4. 完整链路入口

`configs/environments/volcengine.yaml` 已指向上述固定集。先做只读预检：

```bash
python run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --preflight-only
```

预检应确认固定集路径和 SHA-256、SymPy 后端、VEPFS 写入路径、依赖以及 `cuda_available: true`。

执行“少量生成 → 构建训练集 → QLoRA 微调 → 合并模型 → 固定集复测”：

```bash
python run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-open-holdout-e2e
```

8 题配置只验证链路是否真的运行，不适合判断模型效果。链路通过后再运行 27 子类快速实验：

```bash
python run_scripts.py math \
  --config configs/experiments/quick_flywheel_27.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id quick-open-holdout-27
```

最后才运行正式多轮实验：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode data_flywheel \
  --num-iters 5 \
  --max-cycle 3 \
  --run-id production-open-holdout
```

每次固定集评测的 `summary.json` 都包含总准确率、答案类型、子类别以及 `source_statistics` 分来源准确率。论文分析应同时报告 GSM8K、MATH 和 MMLU 数学子集结果，不能只报告混合总分。

## 5. 错误归因的理论与执行顺序

自动归因采用“可观测错误机制”，不猜测模型内部的心理原因。执行顺序是：

1. 协议证据：工具违规、提示词复述、无关输出、JSON 解析失败。
2. 类型证据：答案无法按声明类型标准化、非法选择题标签、单位不一致。
3. 约束证据：把方程组候选解代回原方程，区分部分解和违反约束。
4. 步骤证据：用 SymPy 检验推理摘要中的常数等式，定位第一处错误步骤。
5. 结果传递证据：推理中最后一个已验证结果等于金标，但最终答案字段写成另一个值。
6. 数值签名：仅在存在数学推理时识别符号、倒数、百分比缩放、差一和舍入错误。
7. 符号证据：表达式都能解析但 SymPy/Math-Verify 判定不等价时，标记符号变换错误。
8. 弃权：没有确定性证据时返回 `unknown_error`，置信度为 0，并要求人工复核。

`concept_confusion`、`formula_memory_error` 等认知性标签仍保留在标签表中，但自动规则不会仅凭关键词猜测它们；这些标签必须由人工审计或后续经过标注数据校准的分类器给出。这会降低自动覆盖率，但能提高被输出标签的精确率。

每个归因结果都记录：

- `primary_error_tag`
- `attribution_confidence`
- `verification_tier`
- `first_error_step`
- `evidence[]`，含检查名、响应片段、期望值和观测值
- `needs_review`
- `taxonomy_version`

## 6. 错题池注入与噪声控制

固定测试集的错误不会进入错题池，也不会指导训练题生成。错题池只接收当轮生成题的作答错误，并进行以下控制：

- 仅保留子类别准确率在 0.1–0.4 的边界样本作为 `train_eligible`；低于 0.1 的样本被视为过难/高噪声，高于 0.4 的样本不作为主要修复对象。
- 以规范化问题哈希作为唯一键；恢复同一轮运行不会重复累计。
- 注入时按“确定性证据、归因置信度、重复出现次数、最近出现时间”排序。
- 只有 `verification_tier=deterministic`、有证据且达到配置置信阈值的标签，才能成为 `target_error_type`。
- 低置信标签在提示中退化为 `unknown_error`，不会把错误因果猜测放大到下一轮。
- 生成器只得到匿名错题 ID、子类别、难度、错误机制和证据检查名，不得到参考题的金标答案；提示明确要求更换全部数值、措辞和结构。
- 新生成题仍必须经过精确去重、模板去重、MinHash/LSH 近重复检查和 Sentence-Transformers 语义检查。

## 7. 数据泄露防护

训练候选会与整个固定集逐题比较：

- 规范化后的精确文本；
- 结构模板；
- MinHash 词法相似度；
- Sentence-Transformers 语义相似度。

正式配置的固定集阈值比训练集内部去重更严格；命中任一泄露规则就拒绝样本。固定集只用于 baseline 和每轮微调后的评测，不参与出题、错题池、训练样本选择或提示优化。固定集快照和 SHA-256 会写入每个运行目录，便于追溯。

## 8. 人工复核与归因校准

每轮自动在迭代目录生成 `error_attribution_review.csv`，最多抽取 100 个错误样本。两位标注者分别填写 `human_label_1` 和 `human_label_2`，不要互相查看标签。

也可以手动从任意 VEPFS JSON/JSONL 导出：

```bash
python evaluate_error_attribution.py export \
  --input "$AUTOBENCHER_DATA_ROOT/test_N/cycle/cycle_1/iter_1/inference.json" \
  --output "$AUTOBENCHER_DATA_ROOT/reviews/attribution_round_1.csv" \
  --sample-size 100 \
  --seed 42
```

复核完成后评分：

```bash
python evaluate_error_attribution.py score \
  --review-csv "$AUTOBENCHER_DATA_ROOT/reviews/attribution_round_1.csv" \
  --output "$AUTOBENCHER_DATA_ROOT/reviews/attribution_round_1_metrics.json"
```

重点查看：

- `attribution_accuracy`：全部已复核样本的标签准确率；
- `selective_coverage`：非 `unknown_error` 的比例；
- `selective_accuracy`：系统愿意给出明确标签时的准确率；
- `macro_f1`：避免多数错误类型掩盖少数类型；
- `cohen_kappa`：两位标注者的一致性；
- `evidence_coverage` 与 `verified_evidence_coverage`；
- `confidence_brier_score`：置信度是否校准；
- `accuracy_by_verification_tier`。

推荐先把 `selective_accuracy` 做到稳定，再逐步提高覆盖率。若两位标注者的一致性较低，应先改进标签定义和标注规范，而不是直接训练一个更复杂的归因模型。

## 9. 对 Outlines、DSPy 与 Ragas 的使用边界

- Outlines 已用于本地模型的 Pydantic/JSON 约束生成，减少结构解析噪声；远程 API 路径仍使用严格解析、修复和失败关闭。
- DSPy 的“声明式签名 + 模块 + 可测量优化目标”被用于拆分归因合同、确定性检查和人工校准指标。当前运行时不依赖 DSPy，也不让另一个 LLM 覆盖确定性证据。
- Ragas 的可组合评测思想被用于把归因质量拆成准确率、覆盖率、证据完整性、一致性和置信度校准，避免只用一个总体准确率掩盖问题。

这种设计的核心原则是：结构化生成解决“格式”，SymPy 和类型检查解决“事实”，人工双标解决“不可直接观察的因果标签”，三者不能互相替代。
