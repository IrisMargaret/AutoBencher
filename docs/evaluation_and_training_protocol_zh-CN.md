# 三级评测集与训练协议

这一版把“开发调试”“论文正式比较”“最终盲测”彻底分开。代码仍兼容旧的
`fixed_test` 开关，但评测集身份、权限和版本由
`configs/evaluation_sets.yaml` 统一登记。

所有运行产生的数据、审核报告、模型、checkpoint 和缓存必须写入：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel
```

## 1. 三类评测集

`development_regression_v3` 是原来的 81 题，27 个子类别各 3 题。它只用于
流程检查、提交回归和发现明显退化，不用于支撑细粒度论文结论。

`official_fixed_v1` 的发布门槛是 27 个子类别各 20 题，共 540 题。加载器会
强制检查题量、schema、`validation.status: verified`、至少两个验证来源、版本
和整体哈希。正式集禁止参与训练、Hard Pool、生成提示、方法选择和阈值调优。

`blind_final_v1` 不在仓库中保存路径或哈希。只有代码、提示词、算法和超参数冻结
后，管理员才通过环境变量释放：

```bash
export AUTOBENCHER_BLIND_TEST_PATH=/vepfs-mlp2/queue010/20262202597/math_flywheel/blind/blind_final_v1.json
export AUTOBENCHER_BLIND_TEST_SHA256='<预登记的 sha256>'
export AUTOBENCHER_BLIND_RELEASE_TOKEN='<一次性发布令牌>'

python run_scripts.py math \
  --config configs/experiments/blind_final.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id blind-final-v1
```

阶段、许可、令牌、路径或哈希任一不满足都会在模型加载前失败。盲测配置禁止微调。

## 2. 审核 81 题

```bash
python prepare_evaluation_sets.py audit-development \
  --config configs/math_flywheel.yaml \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/development_regression_v3.audit.json
```

报告逐题记录类型和格式检查、SymPy 独立重求解、solver 与 gold 等价性、原验证
来源、模板签名、精确重复和训练泄漏标记。若仍有 `manual_review_required` 或
`conflict_requires_adjudication`，命令返回退出码 2，不会把“gold 能解析”冒充
“答案已独立重求解”。正式发布逐题接受两种证据路径：独立求解通过；或两位身份独立
的审题者给出答案，发生分歧时由第三位身份独立的仲裁者裁决。仲裁答案仍须与规范答案
一致，且 `unresolved_conflict_count` 必须为零。

## 3. 组装正式 540 题

开源候选集使用 `configs/open_source_evaluation_sources.yaml` 中固定的上游提交，且只从
VEPFS 中已经准备好的测试源文件离线导入：

```bash
python -B prepare_open_source_math_benchmark.py \
  --source-root /vepfs-mlp2/queue010/20262202597/math_flywheel/source_datasets \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/official_candidates_v1.json \
  --allowed-data-root /vepfs-mlp2/queue010/20262202597/math_flywheel
```

准备器不调用 Hugging Face Dataset API，也不自动下载。它记录上游版本、输入文件哈希、
来源 split 和原始记录 ID，并严格要求 27 个子类别各 20 题。自动分类只用于建立人工复核
队列，不能代替题型审定；上游答案也不能代替第二个独立验证来源。公共测试集可能已进入
模型预训练数据，因此它衡量的是跨公开基准迁移，不等价于未见题泛化。

```bash
python prepare_evaluation_sets.py assemble-official \
  --candidates /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/official_candidates_v1.json \
  --training-data /vepfs-mlp2/queue010/20262202597/math_flywheel/audits/all_final_training_questions.json \
  --generation-data /vepfs-mlp2/queue010/20262202597/math_flywheel/audits/all_generated_questions.json \
  --output /vepfs-mlp2/queue010/20262202597/math_flywheel/benchmarks/official_fixed_v1.json \
  --minimum-per-subcategory 20
```

组装器不会为了凑数降低门槛。某子类不足、缺少双来源验证、存在重复题或目标文件
已存在都会失败。成功后生成 `.manifest.json`，记录版本、题数、覆盖率、训练语料、
生成语料、候选语料、算法版本、阈值、审计报告和整体 SHA-256。正式文件不可覆盖；
修改题目必须发布新版本。

冻结后对选定模型运行正式集：

```bash
python run_formal_evaluation.py \
  --config configs/experiments/official_fixed_eval.yaml \
  --environment configs/environments/volcengine.yaml \
  --source-index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/experiment_index.json \
  --source-study-id '<registered-study-id>' --source-method full --source-seed 42 \
  --checkpoint-path /vepfs-mlp2/queue010/20262202597/math_flywheel/models/selected_model \
  --checkpoint-sha256 '<directory-sha256>' \
  --run-id official-fixed-v1
```

## 4. 模板簇级训练划分

每个合格训练样本先按 `template_signature` 聚类，再用“目标样本比例偏差为主、类别/
training source/正确错题平衡为辅”的确定性贪心算法，把整个簇分配到训练集 80%、
内部验证集 10% 和内部训练测试集 10%。同一模板及参数变体不会跨集合。第二、第三
大簇不会被机械指定给留出集。每轮新增：

```text
dataset_train.jsonl
dataset_validation.jsonl
dataset_internal_test.jsonl
split_manifest.json
```

检查零重叠：

```bash
jq '{strategy, split_record_counts, split_fractions, fraction_deviation,
     split_cluster_counts, category_counts, missing_categories,
     training_source_counts, correctness_counts, warnings,
     template_overlap_count, cluster_assignments_sha256}' \
  "$RUN_DIR/cycle/cycle_1/training/split_manifest.json"
```

不足三个独立模板簇时系统会明确拒绝切分，而不是复制模板制造验证泄漏。

## 5. 验证选模与公平训练预算

`train_llm.py` 会真实传入 `eval_dataset`，按内部 `eval_loss` 早停并恢复最佳权重。
配置统一控制 `eval_steps`、`save_steps`、`early_stopping_patience`、
`max_training_tokens` 和 `max_optimizer_steps`。固定评测集从不参与选模。

```bash
jq '{
  final_training_sample_count,
  validation_sample_count,
  internal_test_sample_count,
  training_token_count,
  optimizer_steps,
  gpu_hours,
  peak_gpu_memory_gb,
  best_checkpoint,
  best_metric,
  checkpoint_selection_source,
  evaluation_set_used_for_model_selection
}' "$RUN_DIR/cycle/cycle_1/training/training_cost_summary.json"
```

`checkpoint_selection_source` 必须是 `internal_validation`，
`evaluation_set_used_for_model_selection` 必须为 `false`。早停可能让实际步数低于
统一上限，因此论文必须报告实际 optimizer steps、训练 token 和 GPU 小时。

## 6. 遗忘评测与完整小链路

辅助集合覆盖基础指令遵循、格式遵循、简单任务和非目标数学能力。正式 pilot/main
运行增加 `retention_test.enabled=true`，系统按“baseline 正确、微调后错误”计算
遗忘率：

```bash
python run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-template-split-e2e \
  --override retention_test.enabled=true
```

结果位于 `retention_test/baseline/summary.json` 与
`retention_test/cycle_1/summary.json`。只有微调状态完成、三个 split 文件存在、
训练摘要显示非零 GPU 用量、最佳 checkpoint 来自内部验证，并且开发回归与遗忘
复测均完成，才算真正走通完整链路。
