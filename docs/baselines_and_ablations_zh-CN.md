# 基线与消融实验

本文说明第一版可运行的数学采样策略实验框架。所有方法共享同一套题目生成、
SymPy 金标、质量过滤、训练集构建和固定测试集评测链路；差异仅来自声明的采样策略
及组件开关。

## 七种方法

| 方法 | 分配依据 | 难度选择 | Hard Pool | 错误类型定向 | 是否训练 |
| --- | --- | --- | --- | --- | --- |
| `base` | 不生成训练题，只评测固定集 | 不适用 | 关闭 | 关闭 | 否 |
| `random` | 按 seed 逐题随机选择题型 | 在生成边界内均匀随机 | 关闭 | 关闭 | 是 |
| `uniform` | 27 个细分题型严格均匀 | `study.uniform_difficulty`，默认 4 | 关闭 | 关闭 | 是 |
| `error_only` | `epsilon + (1 - posterior_mean)` | 最近一次有效难度，无历史时用初始难度 | 关闭 | 关闭 | 是 |
| `full` | 覆盖缺口、边界、不确定性、持续错误和 retention | 全局与局部自适应，并使用客观实测难度 | 开启 | 证据门控 | 是 |
| `full_no_hard_pool` | 与 `full` 相同 | 与 `full` 相同 | 关闭，预算重分配 | 关闭 | 是 |
| `full_no_observed_difficulty` | 与 `full` 相同 | 自适应状态使用请求难度 | 开启 | 证据门控 | 是 |

`random` 和 `uniform` 仍会记录客观难度画像，但不使用历史表现进行下一轮分配。
`error_only` 只使用子类别错误率，不叠加覆盖、不确定性、边界或 retention 权重；无
历史数据时退化为均匀分配。`full_no_observed_difficulty` 不删除诊断信息：每道题仍
保留 requested、observed 和 effective difficulty，只是 effective 使用 requested。

## 组件定义

运行清单中的 `component_state` 保存实际生效状态，而不是仅复制 YAML 声明。组件含义
如下：

| 组件 | 含义 |
| --- | --- |
| `adaptive_allocation` | 是否按历史状态动态改变题型预算 |
| `global_difficulty` | 是否使用上一轮整体正确率调整难度 |
| `observed_difficulty` | 后续采样是否使用客观实测难度 |
| `coverage_priority` | 是否提高覆盖不足题型的优先级 |
| `uncertainty_priority` | 是否使用后验不确定性 |
| `persistent_error_priority` | 是否提高持续错误题型的优先级 |
| `retention_priority` | 是否保留已掌握题型的复测预算 |
| `hard_pool_variants` | 是否生成 Hard Pool 结构变式 |
| `error_type_targeting` | 是否注入证据充分的细粒度错误类型 |

命名方法使用固定组件组合。非法策略名、策略与 variant 不匹配、难度越界，或者在
`hard_pool_variants=false` 时启用 `error_type_targeting`，都会在模型加载前失败。

## 执行单个方法

以下命令使用服务器环境配置。所有运行数据继续写入
`/vepfs-mlp2/queue010/20262202597/math_flywheel/`。

均匀基线：

```bash
python run_scripts.py math \
  --config configs/studies/uniform.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id uniform-seed-42
```

完整方法：

```bash
python run_scripts.py math \
  --config configs/studies/full.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id full-seed-42
```

仅按错误率采样：

```bash
python run_scripts.py math \
  --config configs/studies/error_only.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id error-only-seed-42
```

关闭 Hard Pool 的消融：

```bash
python run_scripts.py math \
  --config configs/studies/ablations/full_no_hard_pool.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id full-no-hard-pool-seed-42
```

关闭实测难度反馈的消融：

```bash
python run_scripts.py math \
  --config configs/studies/ablations/full_no_observed_difficulty.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id full-no-observed-difficulty-seed-42
```

固定集参考基线：

```bash
python run_scripts.py math \
  --config configs/studies/base.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id base-seed-42
```

`base` 在固定集基线评测完成后结束，不生成题、不导出训练集，也不调用微调入口。
`random` 的运行方式与上述命令相同，只需改用 `configs/studies/random.yaml`。

## 冻结基准版本

先提交待测代码，再生成 `baseline-ablation-v1` 基准清单：

```bash
python freeze_baseline.py \
  --baseline-id baseline-ablation-v1 \
  --config configs/studies/full.yaml \
  --environment configs/environments/volcengine.yaml

git tag -a baseline-ablation-v1 -m "Frozen ablation baseline v1"
```

清单记录 Git commit 与脏状态、Python/CUDA、Transformers/TRL/PEFT 等包版本、模型
目录内容哈希、固定测试集哈希、解析后配置与所有配置源文件哈希。`prompt_bundle`
分别保存 generator、test-taker、semantic-judge 的真实文件路径和 SHA-256，再由
三个角色、相对路径和内容哈希计算 `combined_sha256`。任意一个 Prompt 文件变化都会
改变组合指纹。

正式冻结和 Study Runner 都默认拒绝脏工作区。`--allow-dirty` 和 suite 的
`allow_dirty_worktree: true` 只供开发测试，不能作为可发表实验。

## 使用统一 Study Runner

四个 suite 位于 `configs/study_suites/`：

| suite | 用途 |
| --- | --- |
| `smoke.yaml` | 七方法、单 seed、单轮小预算链路检查 |
| `pilot.yaml` | 七方法、两个 seed 的先导实验 |
| `main.yaml` | 七方法 × 3 seeds × 1 model × 1350 总题目 |
| `budget_curve.yaml` | 七方法在 135/270/675/1350 总题目下的预算曲线 |

先只展开矩阵并检查公平性，不启动模型：

```bash
python run_study.py \
  --suite configs/study_suites/main.yaml \
  --dry-run
```

正式执行和断点恢复：

```bash
python run_study.py \
  --suite configs/study_suites/main.yaml

python run_study.py \
  --suite configs/study_suites/main.yaml \
  --resume
```

按方法或唯一实验 ID 恢复：

```bash
python run_study.py \
  --suite configs/study_suites/main.yaml \
  --resume \
  --method full

python run_study.py \
  --suite configs/study_suites/main.yaml \
  --resume \
  --study-id '<experiment_index.json 中的 study_id>'
```

每个记录都有 `study_id/method/variant/seed/model/budget/config_hash/`
`git_commit/status`。状态只允许 `pending/running/completed/failed/partial`；启动时发现
遗留的 `running` 会改为 `partial`。已完成记录不会重复执行，失败恢复不会覆盖其他
方法。

suite 的 `budgets` 是实验总生成题目数，不是每轮题数。例如主实验默认 3 个 Cycle、
每个 Cycle 5 个 Iteration，1350 总预算会解析成每轮 90 题。预算无法整除
`Cycle × Iteration` 数时直接失败，避免静默改变实验成本。

结果聚合：

```bash
python aggregate_study.py \
  --index /vepfs-mlp2/queue010/20262202597/math_flywheel/runs/main_v1/experiment_index.json
```

命令生成 `aggregate.json` 和逐实验 CSV，并按 method/model/budget 汇总均值、标准差和
95% 正态近似置信区间。

同一 method、seed、model、budget、resolved config 和 Git commit 产生相同
`study_id` 与调度计划。修改代码、配置、Prompt、模型或固定集后必须使用新的 suite
名称，不能把变化后的实验写进原索引。

## 检查 allocation 和清单

先定位本次输出目录：

```bash
RUN_DIR=$(ls -dt \
  /vepfs-mlp2/queue010/20262202597/math_flywheel/test_* | head -1)
```

查看运行时真正采用的策略，而不是只看配置文件：

```bash
jq '{
  policy_name,
  policy_version,
  variant,
  component_state,
  seed,
  question_budget,
  study_config_snapshot
}' "${RUN_DIR}/run_manifest.json"
```

查看第一轮 allocation、来源预算和诊断：

```bash
jq '{
  policy_name,
  variant,
  source_budget,
  allocations,
  diagnostics
}' "${RUN_DIR}/cycle/cycle_1/iter_1/generation_plan.json"
```

查看 Cycle 与实验汇总：

```bash
jq '{
  policy_name,
  variant,
  component_state,
  seed,
  question_budget,
  finetune_status
}' "${RUN_DIR}/cycle/cycle_1/cycle_manifest.json"

jq '{
  policy_name,
  variant,
  seed,
  question_budget,
  baseline_accuracy,
  final_accuracy,
  accuracy_delta
}' "${RUN_DIR}/experiment_summary.json"
```

`full_no_hard_pool` 的计划应满足
`source_budget.hard_pool_variant == 0`，且日志包含
`hard_pool_disabled_by_ablation`。所有生成方法的 allocation 总和必须等于
`experiment.questions_per_iteration`，每个 allocation 块不得超过
`generation.max_questions_per_prompt`。

## 公平比较要求

除 `base` 仅作为未训练参考点外，其余六种方法必须保持以下条件一致：

- `experiment.questions_per_iteration` 和 Cycle/Iteration 数量；
- 进入训练器的样本数量或预先声明的相同样本上限；
- `finetune.epochs`、batch、LoRA rank、学习率和基础模型；
- 同一组 experiment seed；
- 完全相同且不可变的固定 benchmark；
- 相同的生成模型、test-taker、质量过滤和泄漏防护配置。

否则观察到的准确率差异可能来自训练计算量、数据量或测试集变化，而不是采样策略。
如果某种方法因质量过滤得到更少的合格样本，应在论文结果中单独报告，不得静默补充
其他来源数据。

## 当前范围

当前版本已实现多 seed/多模型/多预算矩阵、实验目录隔离、状态恢复、公平性拒绝、
内容指纹和基础跨 seed 聚合。尚未实现 token/API 美元成本账本、paired bootstrap、
McNemar 显著性检验和自动论文表格；这些能力可在现有 experiment index 上扩展，
不需要改变七种方法定义。
