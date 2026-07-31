# 项目原生固定测试集与证据化错题归因

本文说明如何安装不可变的项目原生数学测试集、运行完整数据飞轮，以及如何审计错误归因。所有运行产物、训练集、模型和人工复核文件都必须位于：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel
```

代码可以位于 `/root/code/AutoBencher`，但程序不会把数据写入代码仓库或系统盘。

## 1. 开发回归集的组成与正式集边界

仓库内置的 `benchmarks/fixed_math_test_set.json` 现在是开发回归集，不再是唯一
正式论文测试集。它覆盖配置中的 9 个数学大类和 27 个细分题型，
共 81 题，每个细分题型固定 3 题，并覆盖基础、中等和较难层级。题集只保存原创
题目、规范答案、答案类型、难度、构造与校验元数据及项目原生题号。

正式固定集要求每个子类别 20 题并经过双来源验证；最终盲测集只在冻结后释放。
完整协议见
[`evaluation_and_training_protocol_zh-CN.md`](evaluation_and_training_protocol_zh-CN.md)。

GSM8K、Hendrycks MATH、MMLU 等 Hugging Face 托管数据集已经从生效配置、准备命令
和文档入口中移除。加载器与安装器还会按 `source_dataset` 做拒绝检查，防止旧文件
被误接回固定评测链路。

安装器有以下硬约束：

- 不联网，不调用 Hugging Face Dataset API，也不需要数据集缓存。
- 安装前验证 JSON schema、题号/题面唯一性、答案类型、难度范围和 27 子类覆盖。
- 输出必须位于 `allowed_data_root` 下。
- 已存在且 SHA-256 相同则幂等成功；内容不同则默认拒绝覆盖。
- 同时生成 `.manifest.json`，记录题数、覆盖数、来源策略与 SHA-256。

## 2. 首次安装固定测试集

在服务器执行：

```bash
cd /root/code/AutoBencher

export AUTOBENCHER_DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel
export TMPDIR="$AUTOBENCHER_DATA_ROOT/temp"
mkdir -p "$TMPDIR" "$AUTOBENCHER_DATA_ROOT/benchmarks"

python prepare_fixed_math_benchmark.py \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

验收构建结果：

```bash
jq '{
  name,
  question_count: (.questions | length)
}' "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json"

sha256sum \
  "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json" \
  "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_dev_v3.json.manifest.json"
```

首次安装后，应把题集与清单 SHA-256 写入实验记录。后续实验复用同一文件。若确实
修改题目，应使用新文件名并视为一个新基准，不能与旧分数直接比较。

## 3. 外部题集策略

当前 81 题主链路仍不接受 Hugging Face Dataset API 或外部题库。开源正式集候选必须
通过独立的 `prepare_open_source_math_benchmark.py` 离线入口，从 VEPFS 中已准备且固定
版本的上游测试文件导入；输出使用独立文件名，并记录来源提交和逐文件 SHA-256。候选题
仍须完成人工题型复核、独立答案验证和训练泄漏审计，不得直接替换开发集或混合准确率。

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

每次固定集评测的 `summary.json` 都包含总准确率、答案类型、子类别以及
`source_statistics`。当前来源固定为 `project_native`，论文分析应重点报告 27 个
细分题型与难度分层结果，不能只报告混合总分。

## 5. 难度定义及其作用

系统不再把出题计划中的整数直接当成真实难度。每道已通过 SymPy 求解的生成题都会保存：

- `target_difficulty`：调度器要求出题器达到的难度；
- `observed_difficulty`：由 `observable_math_v1` 客观量表重新计算的分数；
- `difficulty`：实际进入统计分桶、错题池和下一轮自适应采样的分数。

量表由推理步骤 30%、运算数量 20%、约束数量 20%、符号深度 20% 和表征负荷 10% 组成。大数字、冗长故事和无意义计算本身不增加难度。正式配置将生成范围限制在 2–6，并拒绝实测难度越界题；目标难度与实测难度偏差较大但仍在范围内时，默认保留题目并按实测分数重标，避免浪费已通过金标验证的数据。

固定测试集保留各开源来源的声明难度作为分层字段，同时计算同一客观画像用于跨来源诊断；它不会影响训练采样。评测摘要中的 `difficulty_statistics` 和 `difficulty_calibration` 用于检查模型提升究竟发生在哪些难度层，以及出题器是否长期偏离目标。

## 6. 错误归因的理论与执行顺序

错误 taxonomy 当前版本为 `math_error_taxonomy_v4`，归因方法为
`evidence_rules_v3`。其中
`numeric_approximation_error` 专门描述：模型推导保留了正确的 `pi`、根号、对数等
精确无理数结果，但写入 final answer 时采用了超出题目容差的粗略小数。该标签必须
同时具备“推理中出现正确精确式、没有更早的可验证算术错误、最终数值位于诊断近似
区间但超出评分容差”三类证据；否则回退到 `rounding_error` 或 `unknown_error`。

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

## 7. 错题池注入与噪声控制

固定测试集的错误不会进入错题池，也不会指导训练题生成。错题池只接收当轮生成题的作答错误，并进行以下控制：

- 仅保留子类别准确率在 0.1–0.4 的边界样本作为 `train_eligible`；低于 0.1 的样本被视为过难/高噪声，高于 0.4 的样本不作为主要修复对象。
- 以规范化问题哈希作为唯一键；恢复同一轮运行不会重复累计。
- 注入时按“确定性证据、归因置信度、重复出现次数、最近出现时间”排序。
- 只有 `verification_tier=deterministic`、有证据且达到配置置信阈值的标签，才能成为 `target_error_type`。
- 低置信标签在提示中退化为 `unknown_error`，不会把错误因果猜测放大到下一轮。
- 生成器只得到匿名错题 ID、子类别、难度、错误机制和证据检查名，不得到参考题的金标答案；提示明确要求更换全部数值、措辞和结构。
- 新生成题仍必须经过精确去重、模板去重、MinHash/LSH 近重复检查和 Sentence-Transformers 语义检查。

## 8. 数据泄露防护

训练候选会与整个固定集逐题比较：

- 规范化后的精确文本；
- 结构模板；
- MinHash 词法相似度；
- Sentence-Transformers 语义相似度。

正式配置的固定集阈值比训练集内部去重更严格；命中任一泄露规则就拒绝样本。固定集只用于 baseline 和每轮微调后的评测，不参与出题、错题池、训练样本选择或提示优化。固定集快照和 SHA-256 会写入每个运行目录，便于追溯。

## 9. 人工复核与归因校准

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

## 10. 对 Outlines、DSPy 与 Ragas 的使用边界

- Outlines 已用于本地模型的 Pydantic/JSON 约束生成，减少结构解析噪声；远程 API 路径仍使用严格解析、修复和失败关闭。
- DSPy 的“声明式签名 + 模块 + 可测量优化目标”被用于拆分归因合同、确定性检查和人工校准指标。当前运行时不依赖 DSPy，也不让另一个 LLM 覆盖确定性证据。
- Ragas 的可组合评测思想被用于把归因质量拆成准确率、覆盖率、证据完整性、一致性和置信度校准，避免只用一个总体准确率掩盖问题。

这种设计的核心原则是：结构化生成解决“格式”，SymPy 和类型检查解决“事实”，人工双标解决“不可直接观察的因果标签”，三者不能互相替代。
