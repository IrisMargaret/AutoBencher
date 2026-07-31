# 公平预算、成本账本与论文结果表

本模块解决“计划题数相同，但生成重试、过滤率、训练 Token 和 GPU 时间不同”的实验
混杂。账本以真实运行阶段为准，不用 `questions_per_iteration` 冒充实际成本。

## BudgetLedger

每个 `run_<run-id>_<sha12>/` 首次启动时创建 `budget_ledger.json`，失败路径也会保留；
恢复时重新打开原账本并累计，不会覆盖为零。主要字段：

- `generation`：请求题数、模型原始输出数、SymPy 后合格数、失败数、重试、输入/输出
  Token、API 调用和墙钟时间；
- `validation`：SymPy 次数、Judge 调用、验证失败、难度拒绝、去重拒绝、语义过滤及
  Judge Token/时间；
- `training`：`selected_pool_count`、`train_count`、`validation_count`、
  `internal_test_count`、`actual_trained_count`，以及数据集 Token、按 Epoch 实际处理 Token、Optimizer
  Step、GPU 小时、峰值显存和训练墙钟时间；
- `totals`：总 Token、API 调用和 GPU 小时；
- `efficiency`：每千生成 Token 的合格样本、每千训练 Token 的准确率增益、每 GPU
  小时准确率增益；
- `estimated_cost`：只有配置了 Token 和 GPU 单价才给出金额。单价为 `null` 时金额
  也是 `null`，不会伪造费用。

DeepSeek/OpenAI 兼容接口提供 usage 时使用供应商的精确 Token；供应商不提供 usage、
本地受约束生成或历史缓存时使用固定的 UTF-8 字节估算，并分别记录
`token_count_exact_calls` 与 `token_count_estimated_calls`。训练 Token 由实际 tokenizer
在截断后精确计算。

## 三种协议

`budget.protocol` 支持 `question_matched`、`data_matched` 和
`generation_token_matched`。
`question_matched` 保留原有计划题数比较，只作兼容基线。

### Data-matched

```yaml
budget:
  protocol: data_matched
  data_matched_target_samples: 1000
```

系统跨尚未训练的 Cycle 累积候选，持续经过同一 SymPy、泄漏过滤和去重流程。候选先
按模板簇完整切分，再在实际 train split 中检查目标。达到目标后，按现有 25% 正确
保留样本、75% 错题协议确定性截取 1000 条，只执行一次训练
并结束该运行。目标必须能被 4 整除。声明的问题预算是防止无限生成的安全上限；上限
内仍未达到目标则实验失败并报告 shortfall，不能拿不足样本冒充 data-matched。

### Generation-token-matched

```yaml
budget:
  protocol: generation_token_matched
  max_generation_tokens: 1000000
  max_total_api_calls: null
```

每次新生成调用前预留输入估算 Token 与声明的最大输出 Token；预算无法容纳完整预留
时不会发起请求。账本记录 `budget_cap`、`used_before_last_call`、`last_call_cost`、
`overshoot` 和 `overshoot_ratio`。Judge、验证、训练 Token 与 GPU 时间仍完整计量，
但本协议只匹配生成 Token，所以不能解释为统一 Cost-matched。每次真实 provider 重试
都在底层记账；缺少 provider usage 时标记 `cost_quality=estimated/mixed`，正式成本不
能标为 complete。

同时运行两种协议：

```bash
python -B run_study.py \
  --suite configs/study_suites/fair_budget_smoke.yaml

python -B run_study.py \
  --suite configs/study_suites/fair_budget.yaml \
  --dry-run

python -B run_study.py \
  --suite configs/study_suites/fair_budget.yaml
```

该 suite 展开 `2 protocols × 7 methods × 3 seeds`，每个单元的缓存、Hard Pool、
训练数据、Checkpoint 和模型完全隔离。

## 论文结果表

```bash
python -B aggregate_study.py \
  --index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/fair_budget_v1/experiment_index.json
```

索引同级的 `results/` 包含：

```text
results/
├── results_long.csv
├── run_summary.csv
├── main_results.csv
├── ablation_results.csv
├── category_results.csv
├── difficulty_results.csv
├── efficiency_results.csv
└── significance_tests.csv
```

`results_long.csv` 直接从 Registry 绑定的固定集原始 `fixed_math.compare_answers.json`
重建，不按 mtime 搜索目录；包含 budget、评测集 ID/版本/哈希、checkpoint 哈希、
study/method/variant/seed/cycle、题号、类别、难度、金标、预测、正确性、错误类型、置信度、
延迟和 Token。其余表从该长表、`experiment_summary.json` 和
`budget_ledger.json` 自动生成，不接受手填准确率。

主结果报告跨 Seed 均值、标准差、95% 区间与相对 Base 增量；三个 seed 的区间使用
Student-t 临界值 4.303，不使用 1.96。类别和难度指标先逐
seed 计算再汇总。显著性检验只执行 suite 中的 `comparison_pairs`。论文主 p 值来自
逐 seed 双侧精确 McNemar 的组合，主区间使用 Seed × Item 簇 Bootstrap；直接池化
不同 seed 的 McNemar 和 Item Bootstrap 只标为描述性结果。所有预注册运行、seed 和
题号集合必须完整一致，否则正式聚合失败。只有显式传入
`--allow-partial-development-results` 才允许输出不完整的开发诊断表。
