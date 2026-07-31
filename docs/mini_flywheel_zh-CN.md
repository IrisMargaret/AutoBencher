# 少量题目 → 微调 → 固定集复测操作指南

这条链路用于低成本验证完整数据飞轮是否可运行：

1. 使用 DeepSeek 只生成题面，不接受它生成的答案。
2. 本地 `TruthSolver` 使用 SymPy 独立解析、求解并回代。
3. 从 SymPy 的求解与回代证据生成训练用 `gold_reasoning_summary`。
4. 让原始 test-taker 在生成题上作答并构建小训练集。
5. 执行一轮 QLoRA、合并模型。
6. 使用同一份 81 题开发回归集，分别评测原始模型和微调后模型；它只验证流程和
   明显回归，不作为细粒度论文正式集。

默认的 8 题功能测试配置是
`configs/experiments/mini_flywheel_8.yaml`。它用于验证功能，不用于判断
模型效果；生产实验请改用 `quick_flywheel_27.yaml` 或
`math_flywheel.yaml`。

## 1. 数据目录约束

服务器配置会强制所有可写目录位于：

```text
/vepfs-mlp2/queue010/20262202597/math_flywheel
```

程序启动后会把运行产物、训练集、模型、checkpoint、日志、临时文件、
Hugging Face 缓存、Torch 缓存和 W&B 离线数据都重定向到该目录。若通过
CLI 或 YAML 把任何可写路径改到 `/root/code/`、系统盘或上述根目录之外，
配置校验会直接失败。

安装依赖时的 pip 缓存发生在程序启动之前，因此也应手动指定到 VEPFS：

```bash
export DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel
mkdir -p "$DATA_ROOT/cache/pip" "$DATA_ROOT/temp"
export PIP_CACHE_DIR="$DATA_ROOT/cache/pip"
export TMPDIR="$DATA_ROOT/temp"
python -m pip install -r requirements-lock.txt
```

代码仓库可以位于任意只读或系统目录，但不要在仓库内指定输出目录。

## 2. 准备模型和 API

确认本地 Hugging Face 模型目录存在：

```bash
test -d /vepfs-mlp2/queue010/20262202597/Qwen2.5-7B-Instruct
```

配置 DeepSeek 凭据。不要把密钥提交到 Git：

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_BASE_URL='...'
```

还需要一张支持 bitsandbytes 4-bit QLoRA 的 CUDA GPU。

## 3. 运行前检查

先执行只读/目录准备检查，不生成题目、不启动微调，也不会创建正式运行目录：

```bash
python run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --preflight-only
```

检查通过时会输出：

- `gold_solver_backend: sympy`
- SymPy 自检答案 `4`
- `difficulty_rubric_version: observable_math_v1` 及难度自检画像
- 实际 MinHash 后端；服务器未安装 `datasketch` 时，8 题配置会显示
  `builtin_minhash_exhaustive`
- 固定测试集题数与 SHA-256
- VEPFS 输出和临时目录
- `cuda_available: true`

任一项失败都应先修复，不要直接开始完整运行。

8 题配置的训练集去重会优先使用 `datasketch`。若该可选包不存在，则自动使用
项目内置的确定性 MinHash，并对小数据集执行完整两两比较，不会再因
`MinHashBackendUnavailable` 中断。该回退只用于功能链路；27 题和正式配置仍
要求 `datasketch` 与 Sentence-Transformers，缺失时会失败并提示安装依赖。

## 4. 执行 8 题完整链路

```bash
python run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-sympy-e2e
```

这个命令依次执行：固定集 baseline、8 道题生成与 SymPy gold 求解、
test-taker 作答、训练集导出、1 个 epoch 的 QLoRA、adapter 合并、固定集
复测。

如果上一轮已经在 `training_export` 阶段因缺少 `datasketch` 失败，更新代码后
必须用相同 `run_id` 加 `--resume true` 恢复。程序会重新打开原来的内容寻址运行目录，
并核对配置、代码、Prompt 与模型指纹；不会另建目录从头训练，也不会把其他运行的
半成品混入当前训练集：

```bash
python -B run_scripts.py math \
  --config configs/experiments/mini_flywheel_8.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id mini-sympy-e2e \
  --resume true
```

查找本次运行目录：

```bash
export DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel
export RUN_DIR="$(python -c 'from autobencher.experiment import run_dir_for_id; print(run_dir_for_id("/vepfs-mlp2/queue010/20262202597/math_flywheel", "mini-sympy-e2e"))')"
echo "$RUN_DIR"
tail -f "$RUN_DIR/logs/run.log"
```

## 5. 验收完整链路

运行结束后检查：

```bash
jq '.status, .cycles[0].finetune_status, .cycles[0].training_sample_count' \
  "$RUN_DIR/cycle_record.json"

jq '.accuracy' "$RUN_DIR/fixed_test/baseline/summary.json"
jq '.accuracy, .accuracy_delta' \
  "$RUN_DIR/fixed_test/cycle_1/summary.json"

find "$RUN_DIR/cycle" -path '*/training/dataset_selected.jsonl' -type f -print
find "$RUN_DIR/cycle" -path '*/training/dataset_train.jsonl' -type f -print
find "$RUN_DIR/cycle" -path '*/training/dataset_validation.jsonl' -type f -print
jq '.template_overlap_count, .split_record_counts' \
  "$RUN_DIR/cycle/cycle_1/training/split_manifest.json"
jq '.checkpoint_selection_source, .evaluation_set_used_for_model_selection' \
  "$RUN_DIR/cycle/cycle_1/training/training_cost_summary.json"
find "$RUN_DIR/models" -name config.json -type f -print
```

通过标准：

- 终端最后输出 `[MathFlywheel] run_completed`，其中包含训练样本数、
  baseline accuracy、微调后 accuracy、accuracy delta 和模型目录。
- `cycle_record.json` 的最终状态为 `completed`。
- `training_sample_count` 大于 0。
- `finetune_status` 为 `completed`。
- `split_manifest.json` 的 `template_overlap_count` 为 0，训练、内部验证与内部
  测试三个集合都非空。
- `checkpoint_selection_source` 为 `internal_validation`，且
  `evaluation_set_used_for_model_selection` 为 `false`。
- `models/` 下存在合并模型的 `config.json`、tokenizer 和权重文件。
- baseline 与 `cycle_1` 两份固定集 `summary.json` 都存在。
- 保留的生成/推理记录中，`truth_validation_details.solver_backend` 为
  `sympy`；默认流程不应再产生 `evaluator_code_failure`。
- 生成记录同时包含 `target_difficulty`、`observed_difficulty` 和
  `difficulty_profile.rubric_version=observable_math_v1`。8 题功能配置允许客观难度
  越界后重标，正式 27 题和生产配置则会拒绝越界题。

若 `finetune_status` 是 `no_train_eligible_samples`，查看：

```bash
jq '.rejection_reasons' \
  "$RUN_DIR/cycle/cycle_1/training/dataset_manifest.json"
```

常见原因是与固定集过度相似、答案解析失败或训练推理步骤不合格。
修复后的程序会在 `[BuildDataset]` 日志中直接打印
`rejection_reasons`；当样本少于 `training_mix.minimum_samples` 时会明确失败，
不会把“未执行微调”误报为完整链路成功。旧版本若显示 8 条全部为
`missing_gold_reasoning_steps`，是标准化阶段丢失 SymPy 推理步骤所致，更新
代码并创建一次新的运行即可。

## 6. 扩大到 27 题

8 题链路验收成功后，再运行覆盖全部 27 个细分题型的快速飞轮：

```bash
python run_scripts.py math \
  --config configs/experiments/quick_flywheel_27.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id quick-sympy-27
```

正式多轮实验使用：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode data_flywheel \
  --num-iters 5 \
  --max-cycle 3
```

不要在生产实验中沿用 8 题配置放宽的
`training_mix.strict_correct_incorrect_ratio: false`；正式配置仍使用严格的
正确/错误样本比例。
