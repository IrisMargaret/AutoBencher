# AutoBencher 数学数据飞轮

简体中文 | [English](README.md)

AutoBencher 是一个自适应数学评测与本地训练数据飞轮。它保留原 AutoBencher 的
“规划、出题、评测、收集错题、训练、再评测”框架，同时对标准答案生成、答案判定、
固定测试集和缓存保留策略进行了可配置、可审计的加强。

当前维护的入口是 `math_autobencher.py`。本仓库现已专注数学数据飞轮，原 Wiki 和
Multilingual 入口已移除。

## 系统能力

- 固定覆盖 9 个数学大类、27 个细分题型。
- 在可配置难度范围内生成中等难度题目。
- test-taker 不接收任何工具 schema，也不能调用外部工具，只能依靠自身推理，并按
  严格 JSON 协议作答。
- 主 evaluator 与盲审独立 evaluator 分别分析题目并生成专题 Python 代码；系统
  隔离执行两份代码，要求答案一致，再由全新裁决上下文重新计算或回代。
- 对数值、分数、符号表达式、集合、区间、有序元组、矩阵、布尔值和文本答案做
  规范化。
- 使用独立的大模型语义判定两个答案是否表达相同含义，同时保留确定性比较证据和
  语义判定结果，避免仅因格式差异误判。
- 每个 Cycle 的训练集只使用本 Cycle 数据，严格由 25% 做对题和 75% 本轮错题组成；
  每条导出样本都包含已验证的正确答案和非空安全解题步骤。
- 启动时用固定测试集评测原始 test-taker，每轮训练后评测合并后的新模型，并报告
  正确率变化。
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
  → evaluator 求解提示词（题目仅作为不可信数据）
  → 主求解器 + 盲审独立求解器分别生成 Python
  → AST 安全检查 + 两次隔离 Python 执行
  → 双路答案一致性、运行时验证与答案回代
  → 全新上下文 evaluator 裁决
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

出题器、evaluator 求解器、evaluator 二次复核、语义判定器和 test-taker 使用彼此
独立的提示词和模型调用。题目被序列化到明确命名的 JSON 数据块中，不会被拼接成
系统指令，从而限制跨题上下文污染和提示词注入。

## 项目结构

```text
AutoBencher/
├── autobencher/
│   ├── config.py
│   ├── coverage.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── fixed_benchmark.py
│   ├── output_schemas.py
│   ├── reasoning.py
│   ├── similarity.py
│   ├── structured.py
│   └── truth_solver.py
├── benchmarks/
│   └── fixed_math_test_set.json
├── configs/
│   ├── math_flywheel.yaml
│   ├── environments/
│   └── experiments/
├── prompts/
│   ├── evaluator_python_solver.txt
│   ├── evaluator_independent_solver.txt
│   ├── evaluator_postcheck.txt
│   ├── semantic_answer_judge.txt
│   └── tora_evaluator_strategy.txt
├── tests/
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
| `generation` | 默认难度 2–6、推理步数、重试与 Quota 修复。 |
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
从最近一次实际采样难度继续调节，不会每轮重置。

## 运行

只评测、不训练：

```bash
python run_scripts.py math \
  --config configs/experiments/math_flywheel.yaml \
  --environment configs/environments/local.yaml \
  --mode eval \
  --num-iters 2
```

27 道题的完整快速飞轮（1 个 Iteration、1 个 Cycle、1 个 Epoch）：

```bash
python run_scripts.py math \
  --config configs/experiments/quick_flywheel_27.yaml \
  --environment configs/environments/server.private.yaml \
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

## evaluator 标准答案流程

每一道生成题都执行以下步骤：

1. `evaluator_python_solver.txt` 要求主 evaluator 分析题目，并且只返回
   `analysis_summary` 和本题专用的 `python_code`。可配置的
   `tora_evaluator_strategy.txt` 将 ToRA 的“规划—程序—输出—答案”方法适配为一轮
   fail-closed 求解。
2. `evaluator_independent_solver.txt` 只把原题交给第二个盲审求解器，它看不到
   第一条路径的推理、代码或答案。
3. 系统使用 `ast` 检查两份代码；导入、文件/网络访问、动态执行、私有属性、函数定义及
   未批准调用会被拒绝。
4. 两份通过检查的代码分别使用 `python -I` 在临时工作目录、最小环境变量和受限
   built-in 下执行，同时限制输出大小和执行时间。代码必须返回规范答案，通过自身
   验证和回代检查，并且两条运行时答案必须等价。
5. `evaluator_postcheck.txt` 在一个全新模型调用中运行，只看到题目和两条已验证运行
   结果，独立裁决答案并拒绝超出目标难度的题目。
6. 两条运行时答案与裁决答案必须确定性等价，答案类型必须兼容。

原有确定性 `TruthSolver` 没有被丢弃：对它支持的题目继续做独立交叉验证。两条求解
路径不一致时直接丢弃题目，不会静默选择其中一个答案。

生成的 Python 和原始分析只存在于可清理 evaluator 缓存中，不会进入 test-taker
提示词。训练数据只保留经过长度、角色注入和无关内容检查的 `analysis_summary`
作为标准解题步骤；代码、提示词片段和盲审求解器原始记录不会进入训练输入。

DeepSeek 兼容请求不会设置 `max_tokens`，因此 evaluator 不会被应用层输出 token
上限截断；本地模型仍保留进程安全所需的生成长度限制。

规划、出题、gold 求解、复核、语义判定以及 API 型 test-taker 推理都会使用配置中的
`request_timeout_seconds`、`max_retries` 和 evaluator 的
`retry_delay_seconds`。运行日志会输出 `[API] request_start`、
`request_done`、`request_retry`，出题阶段会输出
`[Generate] subcategory_start` 和 `subcategory_done`。因此服务端请求卡住时会按配置
超时并重试，不会让进程在没有任何日志的情况下无限等待。

gold 校验仍完整保留三阶段 evaluator 链，但对彼此独立的题目使用 OpenAI 兼容 API
并发处理。`evaluator_pipeline.max_parallel_questions` 控制并发数，默认是 `4`；
本地 Hugging Face 和 Ollama evaluator 保持串行，避免不安全地共享模型状态。如果
API 账号限流严格，可调低该值。

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

## 固定测试集与泄漏防护

`benchmarks/fixed_math_test_set.json` 是训练循环的不可变输入，包含每个配置细分题型
各一道中等难度题目。

- 启动时先评测原始 test-taker 并保存基线正确率。
- 每轮训练成功后，用同一测试集评测合并后的新模型。
- 每轮摘要保存基线正确率、当前正确率和增量。
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
`gold_reasoning_summary`。两路 Python 求解运行并达成一致后，最终隔离裁决器会
重新推导题目，记录按顺序发生的真实变形、中间数值、规范最终答案，以及代回原题
或独立重算步骤。`["compute", "solve", "check"]` 这类只有计划而没有推导的列表，
以及缺失步骤、角色注入、工具请求、Markdown 围栏、没有落到最终答案和超长内容，
都会被拒绝。Alpaca 输出固定包含这份复核后的步骤、`final_answer`、
`answer_type` 和置信度，不再用泛化占位句冒充解题过程。

QLoRA 使用 Hugging Face TRL 的 `SFTConfig`/`SFTTrainer`。当
`tracking.wandb.enabled` 为真时，Trainer 向 W&B 记录指标；默认 `offline` 模式只
写入本地数据，不会自动上传，切换到 `online` 必须显式修改配置。

## 输出与缓存保留

每次运行会保留解析后的配置、配置来源、校验结果、配置哈希、环境快照、运行
manifest、日志、全局错题池、训练产物、模型产物和固定测试结果。

配置的输出根目录会为每次启动自动创建一个 `test_<N>` 目录。使用
`configs/environments/server.yaml` 且仓库位于 `/root/code/AutoBencher` 时，实际
路径是 `/root/code/AutoBencher/output/math_flywheel/test_<N>/`。可执行
`ls -dt output/math_flywheel/test_* | head -1` 找到最新一次运行目录。

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
