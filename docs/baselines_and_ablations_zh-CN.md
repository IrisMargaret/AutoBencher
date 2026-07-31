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

## 使用多个随机种子

单次研究至少应为所有非 `base` 方法使用相同的 seed 集合，例如 42、43、44：

```bash
for seed in 42 43 44; do
  python run_scripts.py math \
    --config configs/studies/full.yaml \
    --environment configs/environments/volcengine.yaml \
    --override experiment.seed=${seed} \
    --run-id full-seed-${seed}
done
```

同一 method、seed 和 resolved config 会产生相同的调度与训练数据选择。不要复用
同一个 `run-id` 表示不同配置。

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

## 第一版范围

本版提供单方法运行、确定性策略计划、统一清单和 CPU 单元测试接口，但尚未实现：

- token-level 生成与训练成本匹配；
- API token 或美元成本账本；
- 多 seed 批量 Study Runner；
- paired bootstrap、McNemar 等统计检验；
- 自动论文表格和多模型矩阵调度。

这些能力可在统一策略接口和 manifest 字段之上继续扩展，不需要改变七种方法的定义。
