# 公平预算、成本账本与论文结果表

本模块解决“计划题数相同，但生成重试、过滤率、训练 Token 和 GPU 时间不同”的实验
混杂。账本以真实运行阶段为准，不用 `questions_per_iteration` 冒充实际成本。

## BudgetLedger

每个 `test_<N>/` 启动时立即创建 `budget_ledger.json`，失败路径也会保留。主要字段：

- `generation`：请求题数、模型原始输出数、SymPy 后合格数、失败数、重试、输入/输出
  Token、API 调用和墙钟时间；
- `validation`：SymPy 次数、Judge 调用、验证失败、难度拒绝、去重拒绝、语义过滤及
  Judge Token/时间；
- `training`：最终训练样本、数据集 Token、按 Epoch 实际处理 Token、Optimizer
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

`budget.protocol` 支持 `question_matched`、`data_matched` 和 `cost_matched`。
`question_matched` 保留原有计划题数比较，只作兼容基线。

### Data-matched

```yaml
budget:
  protocol: data_matched
  data_matched_target_samples: 1000
```

系统跨尚未训练的 Cycle 累积候选，持续经过同一 SymPy、泄漏过滤和去重流程。达到
目标后，按现有 25% 正确保留样本、75% 错题协议确定性截取 1000 条，只执行一次训练
并结束该运行。目标必须能被 4 整除。声明的问题预算是防止无限生成的安全上限；上限
内仍未达到目标则实验失败并报告 shortfall，不能拿不足样本冒充 data-matched。

### Cost-matched

```yaml
budget:
  protocol: cost_matched
  max_generation_tokens: 1000000
  max_total_api_calls: null
```

每次新生成调用前检查已消费 Token/API 数；达到上限后不再发起调用，保留已有样本
继续过滤和训练。供应商只能在响应返回后报告输出 Token，因此最后一个在途请求可能
使 Token 数略高于边界，但边界耗尽后不会启动下一请求。账本保留 `exhausted` 和原因。

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

`results_long.csv` 直接从固定集原始 `fixed_math.compare_answers.json` 重建，包含
study/method/seed/cycle、题号、类别、难度、金标、预测、正确性、错误类型、置信度、
延迟和 Token。其余表从该长表、`experiment_summary.json` 和
`budget_ledger.json` 自动生成，不接受手填准确率。

主结果报告跨 Seed 均值、标准差、95% 区间与相对 Base 增量；运行表还报告 Macro
Accuracy、最差子类别、学习曲线 AUC 和遗忘率。显著性表输出双侧精确 McNemar、
配对 Item Bootstrap、Seed × Item 分层 Bootstrap、Holm 校正、风险差和 matched
odds ratio。Bootstrap 使用固定种子，分组和 CSV 行稳定排序，因此重复聚合可复现。
