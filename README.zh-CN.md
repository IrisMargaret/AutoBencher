# AutoBencher 数学数据飞轮

少量题目生成、SymPy 金标、微调和固定集复测的完整操作步骤见
[`docs/mini_flywheel_zh-CN.md`](docs/mini_flywheel_zh-CN.md)。
七种基线与消融方法、组件矩阵和运行清单检查见
[`docs/baselines_and_ablations_zh-CN.md`](docs/baselines_and_ablations_zh-CN.md)。

简体中文 | [English](README.md)

AutoBencher 是一个自适应数学评测与本地训练数据飞轮。它保留原 AutoBencher 的
“规划、出题、评测、收集错题、训练、再评测”框架，同时对标准答案生成、答案判定、
固定测试集和缓存保留策略进行了可配置、可审计的加强。

当前维护的入口是 `math_autobencher.py`。本仓库现已专注数学数据飞轮，原 Wiki 和
Multilingual 入口已移除。

## 系统能力

- 固定覆盖 9 个数学大类、27 个细分题型。
- 将调度器要求的目标难度与客观、可复算的五维难度画像分开；后续自适应采样使用
  实测难度，而不是未经证实的 LLM 自我评级。
- test-taker 不接收任何工具 schema，也不能调用外部工具，只能依靠自身推理，并按
  严格 JSON 协议作答。
- 默认使用本地 SymPy 作为标准答案唯一来源，执行精确求解和方程/方程组逐式回代；
  LLM 生成求解代码仅作为显式兼容模式保留。
- 对数值、分数、符号表达式、集合、区间、有序元组、矩阵、布尔值和文本答案做
  规范化。
- 使用独立的大模型语义判定两个答案是否表达相同含义，同时保留确定性比较证据和
  语义判定结果，避免仅因格式差异误判。
- 只有解析器、类型检查、SymPy 等式验证、候选解回代或数值错误签名提供可复现证据
  时才归因；证据不足时明确输出 `unknown_error`。
- 每个 Cycle 的训练集只使用本 Cycle 数据，严格由 25% 做对题和 75% 本轮错题组成；
  每条导出样本都包含已验证的正确答案和非空安全解题步骤。
- 启动时用项目原生固定测试集评测原始 test-taker，每轮训练后评测合并后的新模型，
  并报告总分、分题型、分答案类型和分难度结果；Hugging Face 数据集不再提供题目。
- 使用精确/模板检查、参考 Text-Dedup 的 datasketch MinHash/LSH 和
  Sentence-Transformers 语义相似度，拒绝与固定测试集相同或高度相似的训练题。
- 直接加载的本地模型可使用 Outlines 做 token 级 JSON 约束，并以 Guidance
  作为后备；W&B 支持离线、在线和关闭三种可配置模式。
- 每轮迭代结束后只保留目标规划、test-taker 作答和答案比较三类 JSON 缓存。

## 端到端流程

```text
YAML 配置 + 显式 CLI 覆盖
  → 原 AutoBencher Quota / 自适应调度
  → 中等难度 question-only 出题
  → 受控 SymPy 求解子句
  → 确定性解析 + 本地精确执行
  → 方程/方程组逐式回代和 fail-closed 校验
  → 五维客观难度画像和实际生效难度
  → 根据求解证据生成训练推理步骤
  → 接受规范标准答案
  → 无工具 test-taker 独立推理
  → 确定性规范化 / Math-Verify 兜底
  → 隔离的大模型语义等价判定
  → 错误归因 + 错题池 + 覆盖率更新
  → 本 Cycle 25% 正确 / 75% 错题训练集
  → 固定测试集泄漏过滤
  → 本地 4-bit QLoRA、合并权重、切换模型
  → 固定测试集复测并计算正确率增量
```

出题器、语义判定器和 test-taker 使用彼此独立的提示词和模型调用；gold 生成本身
在本地确定性执行。

## 项目结构

```text
AutoBencher/
├── autobencher/
│   ├── config.py
│   ├── coverage.py
│   ├── dataset.py
│   ├── difficulty.py
│   ├── evaluator.py
│   ├── fixed_benchmark.py
│   ├── output_schemas.py
│   ├── policies.py
│   ├── reasoning.py
│   ├── similarity.py
│   ├── structured.py
│   └── truth_solver.py
├── benchmarks/
│   └── fixed_math_test_set.json
├── configs/
│   ├── environments/
│   ├── experiments/
│   ├── studies/
│   └── math_flywheel.yaml
├── docs/
├── prompts/
│   ├── evaluator_python_solver.txt
│   ├── evaluator_independent_solver.txt
│   ├── evaluator_postcheck.txt
│   ├── semantic_answer_judge.txt
│   └── tora_evaluator_strategy.txt
├── tests/
├── evaluate_error_attribution.py
├── prepare_fixed_math_benchmark.py
├── math_autobencher.py
├── run_scripts.py
├── train_llm.py
├── tool_util.py
├── THIRD_PARTY_NOTICES.md
├── requirements.txt
└── requirements-lock.txt
```

## 环境要求

- Python 3.10 或更高版本。
- 出题与评测 Agent 使用 DeepSeek/OpenAI 兼容模型。
- test-taker 可使用本地 Transformers 模型目录、vLLM 服务或 Ollama。
- `data_flywheel` 的 bitsandbytes 4-bit QLoRA 训练需要 NVIDIA CUDA GPU。
- 需要足够持久化空间保存基础模型、Adapter、合并模型和运行产物。

如果 test-taker 由独立服务提供，纯评测可以不使用训练 GPU。`train_llm.py` 会对
CPU 上的 4-bit QLoRA 明确报错。

## 安装

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 用于复现实验环境。只有在明确接受兼容的新版本时才使用
`requirements.txt`。在 CUDA 服务器上修改 PyTorch 前先检查现有环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

从 `.env.example` 创建 `.env`，或在作业平台的安全变量界面配置：

```dotenv
DEEPSEEK_API_KEY=replace-me
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_API_KEY=EMPTY
```

不要提交 `.env`、访问密钥、私有服务地址或 SSH 凭据。

## 配置

所有可调行为都声明在 YAML 中。Python 代码只提供经过验证的安全默认值，不写死
实验路径或密钥。

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/local.yaml
```

临时参数可通过 `--override` 修改，不需要编辑源码：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=43 finetune.gpu=1
```

配置优先级依次为：安全默认值、继承的基础 YAML、环境 YAML、实验 YAML、显式兼容
CLI、`--override`。未知字段和不安全值会在加载模型前直接报错。

`configs/math_flywheel.yaml` 的重要节点：

| 节点 | 用途 |
| --- | --- |
| `study` | 采样策略、命名 variant、基线参数与组件开关。 |
| `generation` | 默认允许难度 2–6、推理步数、重试与 Quota 修复。 |
| `difficulty` | 客观难度量表、维度权重、目标容差、重标/拒绝策略和采样分数来源。 |
| `evaluator_pipeline` | 主求解、盲审、裁决和语义提示词路径，以及 Python 限制与重试。 |
| `test_taker_prompt` | 无工具、严格 JSON、推理和输出注入限制。 |
| `answer_normalization` | 数值误差、符号规则和 Math-Verify 开关。 |
| `adaptive_sampling` | 上一轮整体正确率区间和细分题型 Beta-Binomial 自适应采样。 |
| `dataset` | 去重和固定测试集相似度阈值。 |
| `training_mix` | 严格 25/75 比例与 `current_cycle` 范围。 |
| `fixed_test` | 测试集路径、题型覆盖、基线与训练后复测。 |
| `finetune` | GPU、Epoch、Batch、LoRA Rank、序列长度和学习率。 |
| `structured_output` | Outlines/Guidance 本地 JSON 后端与失败策略。 |
| `tracking.wandb` | W&B 模式、项目、分组、标签与模型记录。 |

evaluator 与出题器的难度边界必须一致；默认配置禁止竞赛级题目。

### 基线与消融实验

统一策略接口支持 `base`、`random`、`uniform`、`error_only`、`full`、
`full_no_hard_pool` 和 `full_no_observed_difficulty`。配置位于
`configs/studies/`；默认仍为 `full`，因此旧配置继续使用完整自适应行为。

```bash
python run_scripts.py math \
  --config configs/studies/uniform.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=42 \
  --run-id uniform-seed-42
```

方法定义、公平比较约束、更多命令和 manifest 检查见
[`docs/baselines_and_ablations_zh-CN.md`](docs/baselines_and_ablations_zh-CN.md)。

### 客观难度定义

`difficulty` 是进入统计分桶和自适应采样的实际生效分数，不再从出题计划原样复制。
每道已求解题目同时保存：

- `target_difficulty`：调度器要求的目标难度；
- `observed_difficulty`：求解后按客观量表重新计算的难度；
- `difficulty`：下一轮采样真正使用的生效难度。

`observable_math_v1` 将 1–10 分难度拆成五个可观测、可复算的维度：

| 维度 | 默认权重 | 可观测含义 |
| --- | ---: | --- |
| 推理步骤 | 0.30 | 相互依赖且经过验证的变换步数 |
| 运算数量 | 0.20 | 数学运算符和函数数量 |
| 约束数量 | 0.20 | 方程、不等式和定义域条件 |
| 符号深度 | 0.20 | 变量、函数、幂和嵌套结构 |
| 表征负荷 | 0.10 | 文字转译、单位、比率、分情况、矩阵或几何表征 |

难度区间定义为：1–2 基础直接题，3–4 常规多步题，5–6 综合题，7–8 高阶题，
9–10 专家题。生产出题仍限制在 2–6。单纯增大数字、拉长题干、使用冷僻名字或加入
无意义计算不会提高难度。

`difficulty.mismatch_action: relabel` 会保留数学上合法且通过 SymPy 验证的题目，
但按实测难度重新标记，并记录目标偏差；设为 `reject` 时，超出
`target_tolerance` 的题目会被拒绝。正式配置和 27 题配置会拒绝实测难度超出 2–6
的题目；8 题功能测试只记录并重标，避免把有限修复预算浪费在难度校准上。

### 自适应难度与题目分配

下一轮首先读取“紧邻上一轮”的 test-taker 整体正确率（不使用固定测试集分数）作为
全局难度保护：

| 上一轮整体正确率 | 默认的下一轮难度偏置 |
| --- | --- |
| 低于 `0.35` | 降低 1 级 |
| `0.35` 到 `0.70`（含边界） | 保持 |
| 高于 `0.70` | 提高 1 级 |

达到 `adaptive_sampling.global_accuracy_min_observations: 10` 个有效作答后才启用。
两个区间边界由 `global_accuracy_low` 和 `global_accuracy_high` 配置，调整步长由
`global_difficulty_step` 配置，单轮安全上限由
`max_difficulty_change_per_iteration` 配置。最终难度始终受
`generation.minimum_difficulty` 与 `generation.maximum_difficulty` 限制，默认是
2–6。

这层规则不会替换 AutoBencher 的自适应框架。全局偏置会与原来的“细分题型 × 难度”
Beta-Binomial 后验、覆盖 Quota 缺口、不确定性、持续错题、保留探测和合格 hard-pool
变式共同决定题型与难度。局部目标区间仍由
`target_accuracy_low/mid/high` 单独配置，默认是 0.10/0.20/0.30。每个细分题型还会
从最近一次实测/生效难度继续调节，不会每轮重置。如果某题目标难度为 5、实测为 3，
Beta-Binomial 后验会把这条观测计入难度 3，而不是错误地计入难度 5。

## 运行

服务器首次运行前，将仓库内置的项目原生固定测试集离线安装到 VEPFS。该命令不会
访问 Hugging Face 数据集：

```bash
export AUTOBENCHER_DATA_ROOT=/vepfs-mlp2/queue010/20262202597/math_flywheel

python prepare_fixed_math_benchmark.py \
  --output "$AUTOBENCHER_DATA_ROOT/benchmarks/fixed_math_test_set_v2.json" \
  --allowed-data-root "$AUTOBENCHER_DATA_ROOT"
```

安装器会检查 schema、27 个子类覆盖、题号唯一性、答案字段和题目来源，明确拒绝
GSM8K、Hendrycks MATH 与 MMLU 来源；写入时同时生成 SHA-256 清单。相同文件重复
执行是幂等的，不同文件默认拒绝覆盖。

只评测、不训练：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode eval \
  --num-iters 2
```

27 道题的完整快速飞轮（1 个 Iteration、1 个 Cycle、1 个 Epoch）：

```bash
python run_scripts.py math \
  --config configs/experiments/quick_flywheel_27.yaml \
  --environment configs/environments/volcengine.yaml \
  --run-id quick-flywheel-27
```

该配置先用 27 道固定测试题评测原始模型，再生成 27 道自适应训练候选；如果能组成
完整的 25/75 训练数据块，就执行一轮 QLoRA 和模型合并，最后再次用固定测试集评测
合并后的模型。

完整数据飞轮：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode data_flywheel \
  --num-iters 5 \
  --max-cycle 3 \
  --finetune-gpu 0 \
  --finetune-epoch 3 \
  --finetune-batch 8 \
  --lora-rank 8
```

`math_autobencher.py` 仍兼容原数学工作流的长参数形式。显式 CLI 值会覆盖 YAML。

## SymPy 标准答案流程

默认 `generation.gold_solver_backend: sympy`。每一道生成题执行以下步骤：

1. DeepSeek 只返回题面，不允许返回候选答案、推理或求解代码。
2. 题面必须包含受控的最终求解子句，例如 `Compute ...`、`Solve for x: ...`、
   方程组、导数、积分、极限或不等式格式。
3. 本地 `TruthSolver` 解析该子句并调用 SymPy 精确求解；方程与方程组还会把结果
   逐式代回，只有所有残差均为 0 才接受。
4. 系统从 SymPy 的求解分支、规范答案和回代证据生成
   `gold_reasoning_summary`，再执行训练推理质量检查。
5. 无法解析、没有有限闭式解、超时或回代失败的题目直接丢弃，由 quota repair
   生成替代题；不再进入 LLM 自我修复标准答案的循环。

旧的 LLM 生成 Python、双路执行和 postcheck 流程只作为显式兼容模式保留。只有把
`generation.gold_solver_backend` 改为 `llm_python` 并启用
`evaluator_pipeline.enabled` 时才会调用；默认生产配置不使用它，因此
`evaluator_code_failure` 不再是 gold 生成路径的失败来源。

规划、出题、gold 求解、复核、语义判定以及 API 型 test-taker 推理都会使用配置中的
`request_timeout_seconds`、`max_retries` 和 evaluator 的
`retry_delay_seconds`。运行日志会输出 `[API] request_start`、
`request_done`、`request_retry`，出题阶段会输出
`[Generate] subcategory_start` 和 `subcategory_done`。因此服务端请求卡住时会按配置
超时并重试，不会让进程在没有任何日志的情况下无限等待。

`evaluator_pipeline.max_parallel_questions` 只对上述非默认
`llm_python` 兼容模式生效。

## test-taker 隔离与判分

test-taker 每次只接收一道题，并且没有任何外部工具。输出必须是单个 JSON 对象：

```json
{
  "reasoning_summary": ["简短推理步骤", "自检步骤"],
  "final_answer": "规范答案文本",
  "answer_type": "integer",
  "confidence": 0.9
}
```

工具痕迹、提示词复述、无关输出、损坏 JSON 或额外字段都会按失败处理。判分前会先
规范化格式，因此等价的分数、小数、符号表达式、集合、区间、元组和矩阵不会仅因
写法不同被误判。

每个格式有效的答案还会经过隔离的语义判定器。它只能看到题目、规范标准答案、
答案类型、规范化结果和 test-taker 答案。最终记录同时保留确定性证据和大模型判断，
包括置信度及两者不一致标记。

### 证据化错误归因

归因按确定性证据阶梯执行：协议/解析失败、答案类型规范化、单位和选择题检查、
TruthSolver 候选解回代、SymPy 常数等式验证、推理结果到最终答案的一致性、数值错误
签名以及符号等价性。每条结果保存证据、置信度、验证层级、首个错误步骤和标签版本。
没有检查能够隔离出可靠机制时，系统输出 `unknown_error`，不会根据关键词猜测模型的
“认知原因”。

每轮会导出双人标注复核 CSV。两位标注者填写人工标签后，使用
`evaluate_error_attribution.py score` 计算归因准确率、选择性覆盖/准确率、Macro-F1、
Cohen's kappa、证据覆盖率、Brier Score 和分验证层级准确率。只有“确定性验证 +
存在证据 + 置信度达标”的标签才能定向指导错题池出题。

## 固定测试集与泄漏防护

服务器环境使用 VEPFS 中不可变的项目原生
`benchmarks/fixed_math_test_set.json`。它由仓库内置基准离线复制得到，不执行数据集
下载；GSM8K、Hendrycks MATH、MMLU 以及其他 Hugging Face 托管题目均不进入当前
评测链路。v2 固定集共 81 道原创题：27 个细分题型各 3 道，分别覆盖基础、中等和
较难层级。GSM8K、MATH、MMLU 与 DeepMind Mathematics 只用于参考能力分布，不复制
任何外部题面。

- 启动时先评测原始 test-taker 并保存基线正确率。
- 每轮训练成功后，用同一测试集评测合并后的新模型。
- 每轮摘要保存基线正确率、当前正确率和增量。
- 摘要还保存分来源和分难度正确率、各客观难度维度均值，以及目标/实测难度平均绝对
  偏差。
- 固定测试题不会进入错题训练候选。
- 训练候选如与测试题文本完全相同、规范模板相同，或者 Token、TF-IDF、
  datasketch MinHash/LSH、Sentence-Transformers 余弦相似度超过对应阈值，
  会被拒绝。

默认语义模型是 `sentence-transformers/all-MiniLM-L6-v2`。模型名、Revision、
设备、缓存目录、Batch Size 和 `local_files_only` 均可在 `dataset` 中配置。
数据飞轮模式默认要求该后端可用并采用 fail-closed：模型加载失败时停止训练数据
导出，而不是静默放过可能泄漏的题目。首次联网运行可能下载该模型；离线部署应提前
准备缓存并启用 `sentence_transformers_local_files_only`。

固定测试集应纳入版本控制。修改它会改变 SHA-256，也就形成了新的测试基准，不能
再把修改前后的分数当作同一测试集结果直接比较。

## 训练集协议

只有本 Cycle 产生的记录才有训练资格。质量过滤、固定测试集过滤和去重后，系统按
四条一组进行选择：

- 1 条做对保留样本（25%）；
- 3 条来自本轮错题池的错题（75%）。

错题部分按 `training_mix` 分配给边界错题、覆盖修复和格式/指令错误。某一错题来源
不足时可以由其他错题补足，但正确题绝不会占用错题位置。不完整的四条数据块不会
导出；微调前 manifest 会记录计数并断言比例严格正确。

每条合格记录还必须包含满足配置步数和单步字符限制的具体
`gold_reasoning_summary`。默认流程根据 SymPy 的精确求解、规范最终答案和回代
证据生成真实变形与验证步骤。`["compute", "solve", "check"]` 这类只有计划而没有推导的列表，
以及缺失步骤、角色注入、工具请求、Markdown 围栏、没有落到最终答案和超长内容，
都会被拒绝。Alpaca 输出固定包含这份复核后的步骤、`final_answer`、
`answer_type` 和置信度，不再用泛化占位句冒充解题过程。

QLoRA 使用 Hugging Face TRL 的 `SFTConfig`/`SFTTrainer`。当
`tracking.wandb.enabled` 为真时，Trainer 向 W&B 记录指标；默认 `offline` 模式只
写入本地数据，不会自动上传，切换到 `online` 必须显式修改配置。

## 输出与缓存保留

每次运行会保留解析后的配置、配置来源、校验结果、配置哈希、环境快照、运行
manifest、日志、全局错题池、训练产物、模型产物和固定测试结果。

配置的输出根目录会为每次启动自动创建一个 `test_<N>` 目录。服务器配置强制
所有运行产物、缓存、临时文件、日志、训练集、checkpoint 和模型位于
`/vepfs-mlp2/queue010/20262202597/math_flywheel`；任何指向系统盘、
`/root/code/` 或该根目录之外的可写路径都会在启动时被拒绝。可执行
`ls -dt /vepfs-mlp2/queue010/20262202597/math_flywheel/test_* | head -1`
找到最新一次运行目录。

当 `experiment.clean_cycle_cache: true` 时，每个迭代目录在 `finally` 清理后严格只
保留以下三个 JSON：

```text
cycle/cycle_<N>/iter_<M>/
├── <prefix>.question_plan_with_aim.json
├── <prefix>.test_taker_inference.json
└── <prefix>.compare_answers.json
```

所有 `subcat*.json`、生成尝试、evaluator Python 缓存、临时判定、空/损坏 JSON 和
冗余迭代摘要都会删除。清理操作可重复执行，并且成功或失败路径都会运行。Cycle
级训练、指标、错题池与固定测试文件属于正式结果，不按迭代缓存清理。

固定测试结果单独保存：

```text
fixed_test/
├── dataset_snapshot.json
├── baseline/
│   ├── fixed_math.test_taker_inference.json
│   ├── fixed_math.compare_answers.json
│   └── summary.json
└── cycle_<N>/
    ├── fixed_math.test_taker_inference.json
    ├── fixed_math.compare_answers.json
    └── summary.json
```

原始 test-taker 首次加载后会立即执行基线测试。每个 Cycle 成功训练并合并模型后，
系统会加载该轮新模型并用同一固定测试集复测。每个阶段不仅保存一个总正确率，还会
保留原始作答、解析后的 `reasoning_summary`、规范化答案、语义判定、按答案类型和
细分题型统计的正确率，以及模型置信度数据。

## 开源项目基础

评测闭环继续保留
[XiangLi1999/AutoBencher](https://github.com/XiangLi1999/AutoBencher) 的规划、出题、
评测和反馈拓扑。evaluator 策略适配了 MIT 许可证的
[Microsoft ToRA](https://github.com/microsoft/ToRA) 工具集成推理方法，但将其自由
工具执行替换为上文所述的受限本地运行时。

项目还使用 Hugging Face
[Math-Verify](https://github.com/huggingface/Math-Verify) 作为数学答案解析和表达式
比较的高质量兜底实现，其许可证为 Apache-2.0。AutoBencher 自己的类型化规范化和
SymPy 检查优先执行，隔离的大模型语义判断默认仍为必需步骤。精确复用边界和许可
说明见 `THIRD_PARTY_NOTICES.md`。

训练集与固定测试集的泄漏防护还使用了参考
[ChenghaoMou/text-dedup](https://github.com/ChenghaoMou/text-dedup) 设计的
Apache-2.0 兼容 MinHash 独立后备实现、MIT 许可的
[datasketch](https://github.com/ekzhu/datasketch) MinHash/LSH，以及 Apache-2.0
许可的 [Hugging Face Sentence Transformers](https://github.com/huggingface/sentence-transformers)。
本地结构化 JSON 使用 [Outlines](https://github.com/dottxt-ai/outlines)，失败时可
回退到 [Guidance](https://github.com/guidance-ai/guidance)；微调使用
[TRL](https://github.com/huggingface/trl)，实验追踪使用可配置的
[Weights & Biases](https://github.com/wandb/wandb)。

固定测试集完全采用项目原生题目，不导入外部数据集题目。错误归因合同借鉴
[DSPy](https://github.com/stanfordnlp/dspy) 的声明式职责拆分；人工审计把准确率、覆盖率、
证据、一致性和置信度校准分开报告，借鉴了
[Ragas](https://github.com/vibrantlabsai/ragas) 的可组合评测思想。DSPy 和 Ragas
只是设计参考，不是运行时依赖。

精确复用边界和许可说明见 `THIRD_PARTY_NOTICES.md`。

## 检查与验证

运行 CPU 测试：

```bash
python -m pytest -q
```

语法与 CLI 检查：

```bash
python -m py_compile math_autobencher.py run_scripts.py tool_util.py autobencher/*.py
python math_autobencher.py --help
python run_scripts.py --help
```

真实 API 推理、本地大模型加载、CUDA QLoRA 和远程调度需要相应密钥、模型文件与
硬件；单元测试不会把这些未执行的外部集成声明为已通过。
