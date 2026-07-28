# AutoBencher

简体中文 | [English](README.md)

AutoBencher 是一个可复现的自适应数学评测与本地模型数据飞轮系统。项目支持
DeepSeek 等在线模型作为出题与评测 Agent，并支持 Qwen2.5 等本地模型通过
Transformers、vLLM 或 Ollama 作为 test-taker。`data_flywheel` 模式能够自动完成
评测、错题筛选、训练集构建、4-bit QLoRA 微调、权重合并和下一 Cycle 模型切换。

Wiki 与 Multilingual 入口继续保持兼容。

## 功能概览

- `eval`：仅生成和评测数学基准，不触发微调。
- `data_flywheel`：评测后构建训练集，调用项目内置 `train_llm.py` 微调并循环。
- 固定 9 个数学大类和 27 个细分题型。
- 多轮累计题型覆盖，不要求单次 Iteration 覆盖全部细分题型。
- 基于 Beta-Binomial 后验的难度与采样优先级调整。
- test-taker 无工具隔离和严格 JSON 输出。
- 数值、分数、方程、不等式、集合、区间、矩阵与单位值规范化。
- hard pool 去重、分级和训练资格筛选。
- 训练数据精确去重、文本近似去重、模板去重和语义去重。
- 统一配置加载、配置来源追踪、配置哈希和原子化输出。
- 每个阶段只显示一个动态进度条。

## 新生成与真值架构

```text
动态 Quota 调度
→ question-only LLM Generator（temperature=0.0, top_p=0.1）
→ 自然语言剥离与表达式标准化
→ TruthSolver 分支求解
→ canonical gold 回代验证
→ Truth 失败样本丢弃、失败计数与子类目冷却
→ 无工具 test-taker 作答
→ 与 canonical gold 确定性比较
→ hard pool、训练集与下一 Cycle
```

## 环境要求

- Python 3.10 或更高版本。
- 在线 Agent 需要 DeepSeek/OpenAI 兼容 API。
- 本地 test-taker 可使用 Transformers 模型目录、vLLM 服务或 Ollama。
- `data_flywheel` 的 bitsandbytes 4-bit QLoRA 训练需要 NVIDIA CUDA GPU。
- 需要足够磁盘空间保存基础模型、LoRA Adapter、合并模型和实验数据。

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

在 CUDA 服务器上安装依赖前，建议先确认现有 PyTorch 与 CUDA 是否匹配：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi
```

## 敏感信息配置

在项目根目录创建 `.env`，或者在作业平台的安全环境变量界面配置：

```dotenv
DEEPSEEK_API_KEY=replace-me
DEEPSEEK_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
VLLM_BASE_URL=
VLLM_API_KEY=EMPTY
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
```

环境变量名称统一声明在配置文件的 `sensitive_environment` 节点。密钥与私有地址
不会写入正式 YAML，也不会明文写入配置快照。请勿提交 `.env`。

## 统一配置系统

数学入口只加载一次配置，并向运行模块传递递归只读的 `ProjectConfig`。运行参数不应
通过修改 Python 源码调整，新实验应复制或继承 YAML 配置。

### 配置目录

```text
configs/
├── math_flywheel.yaml
├── math_flywheel_smoke_test.yaml
├── environments/
│   ├── local.yaml
│   ├── server.yaml
│   └── volcengine.yaml
└── experiments/
    ├── math_flywheel.yaml
    └── smoke_test.yaml
```

- `math_flywheel.yaml`：完整基础配置和唯一默认值来源。
- `environments/*.yaml`：机器、模型路径和持久化目录。
- `experiments/*.yaml`：实验名称、运行模式及实验差异。
- `math_flywheel_smoke_test.yaml`：小规模调度与 CPU Mock 测试配置。

### 合并优先级

```text
代码安全底线
  < Schema 安全默认值
  < 继承的基础 YAML
  < 环境 YAML
  < 实验 YAML
  < 用户显式输入的兼容 CLI
  < --override
  < 敏感环境变量
```

安全限制不能通过配置放宽。普通实验参数不应通过环境变量注入。

### 推荐配置启动方式

本地环境：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/local.yaml
```

火山引擎环境：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml
```

`--experiment` 是 `--config` 的等价别名。

### 临时覆盖

不修改 YAML 即可临时覆盖任意已声明字段：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --override experiment.seed=43 finetune.gpu=1
```

布尔值支持 `true/false`、`yes/no`、`on/off` 和 `1/0`。未知字段、错误类型、
无效比例、非法阈值、可写路径失败或 test-taker 工具开启会在模型加载前报错。

### 兼容 CLI 映射

原有命令仍然支持，显式输入的参数会映射到统一配置：

| 原 CLI | 配置字段 |
| --- | --- |
| `--agent_modelname` | `models.evaluator.model_name` |
| `--test_taker_modelname` | `models.test_taker.model_path` |
| `--num_iters` | `experiment.num_iterations` |
| `--max_cycle` | `experiment.max_cycles` |
| `--acc_target` | `adaptive_sampling.target_accuracy_low/high/mid` |
| `--finetune_gpu` | `finetune.gpu` |
| `--finetune_epoch` | `finetune.epochs` |
| `--finetune_batch` | `finetune.batch_size` |
| `--lora_rank` | `finetune.lora_rank` |
| `--clean_cycle_cache` | `experiment.clean_cycle_cache` |

YAML 中的旧字段会在可迁移时给出 `DeprecationWarning` 并迁移；未知字段不会静默忽略。

### 配置快照与审计

每次运行都会保存：

```text
resolved_config.yaml
resolved_config.json
config_sources.json
config_validation.json
config_hash.txt
```

- `resolved_config.*`：合并后的全部实际生效参数。
- `config_sources.json`：每个叶子字段的最终值和来源。
- `config_validation.json`：启动校验状态。
- `config_hash.txt`：稳定配置哈希，用于缓存兼容和实验比较。

比较两个 `test_<N>/resolved_config.json` 或 `config_hash.txt` 即可定位实验配置差异。
影响结果的配置发生变化后，不会静默复用不兼容缓存。

## 基础评测

下面的兼容命令只评测，不训练：

```bash
python run_scripts.py math \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname qwen2.5:7b-instruct \
  --exp_mode autobencher \
  --use_helm no \
  --num_iters 2 \
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. \
  --acc_target 0.1--0.3 \
  --mode eval
```

本地模型目录也可以直接作为 test-taker：

```bash
python run_scripts.py math \
  --agent_modelname deepseek-v4-pro \
  --test_taker_modelname /models/Qwen2.5-7B-Instruct \
  --num_iters 2 \
  --mode eval
```

## 全自动数据飞轮

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/volcengine.yaml \
  --mode data_flywheel \
  --num_iters 5 \
  --max_cycle 3 \
  --export_interval 1 \
  --finetune_gpu 0 \
  --finetune_epoch 3 \
  --finetune_batch 8 \
  --lora_rank 8
```

闭环流程：

```text
自适应出题
→ evaluator 独立重算并回代验证 gold answer
→ 本地 test-taker 推理
→ 答案规范化与判分
→ hard pool 更新
→ 构建 Alpaca JSONL
→ 调用项目根目录 train_llm.py
→ 4-bit QLoRA 训练
→ 合并完整本地模型
→ 切换 test-taker
→ 下一 Cycle
```

无需提供外部微调脚本路径。

## 每次运行的统一归档目录

每次启动都会在输出根目录扫描已有 `test_<数字>` 文件夹，并以“当前最大数字加 1”
原子创建新目录。无关目录不会参与编号，旧实验不会被覆盖。

```text
<output_root>/
├── test_1/
├── test_2/
└── test_3/
```

一次运行产生的配置、日志、全局状态、迭代结果、训练数据和模型全部归档在同一个
`test_<N>` 中。所有 `cycle_<N>` 统一放入 `cycle/`：

```text
<output_root>/
└── test_3/
    ├── resolved_config.yaml
    ├── resolved_config.json
    ├── config_sources.json
    ├── config_validation.json
    ├── config_hash.txt
    ├── environment.json
    ├── run_manifest.json
    ├── taxonomy_snapshot.json
    ├── experiment_summary.json
    ├── hard_pool.json
    ├── meta_summary.json
    ├── cycle_record.json
    ├── logs/
    │   ├── run.log
    │   └── events.jsonl
    └── cycle/
        ├── cycle_1/
        │   ├── cycle_manifest.json
        │   ├── iter_1/
        │   └── training/
        │       ├── dataset_candidates.json
        │       ├── dataset_selected.jsonl
        │       ├── dataset_manifest.json
        │       ├── dedup_report.json
        │       ├── diversity_report.json
        │       ├── finetune_config.json
        │       ├── finetune_metrics.jsonl
        │       ├── finetune_summary.json
        │       └── checkpoint_manifest.json
        └── cycle_2/
```

训练数据集路径为：

```text
<output_root>/test_<N>/cycle/cycle_<N>/training/dataset_selected.jsonl
```

只有完成相应评测并存在合格训练样本时才会创建 `training/`。失败于 evaluation
阶段的运行不会产生训练集，失败信息保存在：

```text
test_<N>/cycle/cycle_<N>/failure/failure_summary.json
```

## TruthSolver 权威真值

LLM Generator 只能返回 `{"question": "..."}`。答案、候选答案、答案类型、解题过程
和元数据字段全部禁止输出。系统先剥离自然语言包装并将 `x^2` 等语法标准化为
`x**2`，再交给独立的 `TruthSolver`。只有 `TruthSolver` 可以写入
`canonical_answer`、`gold_answer`、`display_answer` 和 `answer_type`。

`TruthSolver` 分别路由一元方程、线性方程组、多项式系统、积分、极限和直接表达式。
解析失败、无闭式解、无穷多解和求解超时的样本会在 test-taker 推理前直接丢弃并记录
结构化 `failure_type`，不会把失败真值送回 LLM 盲目修复。

题目只有同时满足以下条件才会进入 test-taker 推理：

- 独立重算成功；
- 回代和约束检查成功；
- 重算答案与候选 gold answer 通过确定性等价比较；
- validator 返回的答案类型与题目答案类型一致。

对于 `ordered_tuple` 方程组，系统不存在也不信任 evaluator 自报答案。SymPy 会从
完全一致的原始 `question` 字符串解析方程并独立求解，然后把候选元组逐条代入每一个
原始等式。审计记录每条原始方程、代入后的左值、右值、差值和通过状态；题干哈希用于
证明独立求解与回代使用的是同一份原始题干。解析异常、超时、无解、多解和无穷多解均
按验证失败处理。

只有 Generator 输出格式错误时才允许修复，下一次 Prompt 必须携带上一轮失败摘要。
如果 Generator 违规泄漏候选答案，且该答案只满足部分方程，则记录
`partial_solution` 和“通过方程数/总方程数”，并把它作为负面反馈。TruthSolver 失败
不 repair，直接丢弃。

单条样本或单个细分题型修复耗尽时不再抛出 runtime error。系统记录覆盖率缺口后继续
当前 Cycle；即使本轮没有任何合格题目也会输出空迭代统计。连续失败达到配置阈值后，
对应子类目进入临时冷却。Cycle 结束时
`cycle_<N>/metrics/generation_statistics.json` 输出生成总数、有效样本数、各失败计数和
category/subcategory 覆盖率缺口。

固定失败标签：

```text
truth_parse_fail
no_closed_solution
infinite_solutions
solve_timeout
partial_solution
repair_exhausted
generator_format_error
```

## JSON 与答案规范

所有项目 JSON 使用 UTF-8、两空格缩进、换行结尾和原子替换写入。JSONL 每行一个
完整 UTF-8 JSON 对象。

test-taker 的首个完整结构化 JSON 会在生成完成后立即解析。JSON 后出现的
`Human:`、`User:` 等串话会被安全丢弃并留下审计字段。超过配置上限的推理步骤会在
判定副本中截断，不修改原始响应。

数值答案比较会对 gold answer 和 test-taker answer 使用相同的临时清洗副本：

- 去除首尾空白；
- 去除数值答案前的 `x =`、`y=` 或 `z =`；
- 去除数值答案末尾的 `°`。

`raw_response` 和持久化的 `parsed_response.final_answer` 不会被改写。方程、符号、
单位和结构化答案保留原始语义。

## 日志与进度条

日志位于：

```text
test_<N>/logs/run.log
test_<N>/logs/events.jsonl
```

每个阶段只创建一个动态进度条，完成后关闭，不会为每道题反复创建永久进度条。
第三方库进度条默认关闭。

`experiment.clean_cycle_cache=true` 时，每轮迭代都在 `finally` 中清理冗余分片、
attempt 日志、临时推理文件以及空或损坏 JSON。因此生成、推理或评测异常也不会跳过
清理；正式推理结果和 gold answer 校验审计会保留。

失败记录包含：

```json
{
  "stage": "evaluation",
  "failure_type": "runtime_error",
  "error": "AssertionError()",
  "exception_type": "AssertionError",
  "traceback": "..."
}
```

即使异常消息为空，也会保存异常类型和 traceback。手动中断记录为 `interrupted`。

## 常见故障

### 没有生成训练数据集

检查 `cycle_record.json`。如果 `iterations_completed=0`、`training_export=null`，
说明评测尚未完成或已失败。训练目录只在达到导出条件且存在合格样本后创建。

### 本地模型加载失败

- 确认模型目录存在并包含 Transformers 配置、Tokenizer 和权重文件。
- 检查 CUDA、PyTorch 与 transformers 版本。
- 确认远程作业能够访问持久化模型目录。

### 微调 OOM

- 降低 `finetune.batch_size`。
- 降低 `finetune.max_seq_length`。
- 增大梯度累积并保持实际批次。
- 确认其他进程没有占用 GPU。

### API 超时

- 检查 API 地址与凭证。
- 查看 `logs/events.jsonl` 中的 stage 和 traceback。
- 在配置中调整已声明的请求超时与重试参数。

### JSON 解析失败

- 查看 `raw_response`、`parse_status` 和 repair 审计字段。
- 确认模型输出包含一个完整 JSON 对象。
- 续跑时，旧失败响应会由当前解析器重新验证。

### 配置字段报错

错误会包含完整 dotted path，例如：

```text
ConfigurationError: dataset.near_duplicate_threshold: threshold must not exceed 1
```

修正对应 YAML 或 `--override`。不要在 Python 文件中添加备用默认值。

## 测试

无需 API 或 GPU的完整回归测试：

```bash
python -m pytest -q
```

可选 GPU/API 测试默认跳过，只有在配置好真实环境后才应显式启用。未执行的真实 API
或远程 GPU 测试不应声明为已通过。
