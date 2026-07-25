# AutoBencher

简体中文 | [English](README.md)

AutoBencher 用于自动生成并评测基准问题。当前版本已适配 OpenAI 兼容的
DeepSeek API，可在 Windows、macOS 和 Linux 上运行，要求 Python 3.10 或更高版本。

## 1. 创建并激活虚拟环境

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 阻止运行本地激活脚本，可先在当前终端执行一次以下命令，再重新激活：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows 命令提示符（CMD）

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

激活后，下文中的 `python` 和 `pip` 均指向虚拟环境，无需反复输入解释器完整路径。

## 2. 安装依赖

如需复现已测试的依赖环境，请执行：

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 固定了经过测试的准确版本。如果您明确希望安装兼容范围内的
较新依赖，才使用：

```shell
python -m pip install -r requirements.txt
```

## 3. 配置 API

在项目根目录新建 `.env` 文件：

```dotenv
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

请勿提交 `.env`，也不要向他人透露 API 密钥。

## 4. 检查安装

冒烟测试会检查项目导入、JSON 解析、API 配置，并发送一次低成本 API 请求：

```shell
python smoke_test.py
```

另外可运行完全不联网的回归测试：

```shell
python -m unittest discover -s tests -v
```

## 5. 运行基准测试

虚拟环境激活后，运行命令可简化为：

```shell
python run_scripts.py math
python run_scripts.py wiki
python run_scripts.py multilingual
```

建议先用一轮验证完整流程，这样耗时和 API 成本较低：

```shell
python run_scripts.py math --num-iters 1
```

如果 API 账户开放了其他模型，可通过参数指定：

```shell
python run_scripts.py math --model deepseek-v4-pro --num-iters 1
```

agent 和 test-taker 可以使用不同的模型服务。带 Ollama 标签的模型名（例如
`qwen2.5:7b-instruct`）会自动通过 Ollama 原生本地 API 调用：

```powershell
python run_scripts.py math `
  --agent_modelname deepseek-v4-pro `
  --test_taker_modelname qwen2.5:7b-instruct `
  --exp_mode autobencher `
  --use_helm no `
  --num_iters 2 `
  --outfile_prefix1 math_test/qwen7b_dsagent.0.3. `
  --acc_target 0.1--0.3
```

Ollama 默认地址为 `http://localhost:11434`。程序会在第一道题之前预热模型，并让模型
保持加载 30 分钟；模型进程临时异常或 HTTP 502 最多自动重试 10 次。Ollama 模式不需要
安装 `torch`、`transformers` 或 `accelerate`。

可以在 `.env` 中使用以下可选配置：

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=30m
OLLAMA_MAX_RETRIES=10
OLLAMA_RETRY_DELAY_SECONDS=5
OLLAMA_REQUEST_TIMEOUT=300
```

不激活虚拟环境也可以直接调用其中的解释器。例如 Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe run_scripts.py math
```

## 输出、缓存与断点续跑

- 数学任务输出到 `math_v5/`。
- Wiki 任务输出到 `KI/`。
- 多语言任务输出到 `multilingual/`。
- 已完成步骤会保存缓存。运行中断后，使用完全相同的命令即可复用已有结果并继续。
- 部分写入的推理缓存会逐条校验；重新运行时会从第一道缺失题继续，而不会把半成品误判为
  完整缓存。
- 完整任务会生成数百道问题，可能运行很久并消耗较多 API Token。

模型偶尔不会严格按要求输出 JSON。数学任务的 JSON 生成步骤现在会对无效结果自动重试
最多三次，并且只在结构校验通过后写入缓存。被拒绝的原始响应会保存为
`*.attemptN.txt`，便于排查 API 返回内容。
数学规划固定使用提示要求的前 5 个子类别，避免模型返回过多规划时意外成倍扩大任务量。

## 常见问题

### 提示 `DEEPSEEK_API_KEY is not configured`

确认 `.env` 位于项目根目录，且变量名准确写为 `DEEPSEEK_API_KEY`，然后重新运行
`python smoke_test.py`。

### 提示 `Model response did not contain a JSON block`

使用原命令重新运行。已完成的结果会直接复用，失败的 JSON 步骤会自动重试。如果三次
重试仍失败，请检查对应的 `*.attemptN.txt` 文件。

### Gold answer 与 inference 数量不一致

使用完全相同的命令重新运行。程序会将推理缓存与当前问题逐条核对，丢弃无效的尾部内容，
并从第一条缺失记录继续生成。

### Ollama 返回 HTTP 502

程序现在会通过 `/api/generate` 预热 Ollama 模型，使用原生 `/api/chat` 接口推理，
保持模型常驻，并自动重试临时错误。更新代码后，使用完全相同的命令重新运行即可；已经
生成的问题和完成的推理记录都会复用。

可以通过以下命令确认 Ollama 正在运行，并检查所需模型是否已经安装：

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

如果重试后仍然失败，请释放足够的内存或显存，或者改用更小的 Ollama 模型。最终错误信息
现在会包含 Ollama 返回的详细内容以及实际使用的服务地址。

### PowerShell 无法激活 `.venv`

执行第 1 节中的当前进程执行策略命令；或者不激活环境，直接使用
`.\.venv\Scripts\python.exe` 运行。

### 使用本地 Hugging Face 模型或 HELM

默认依赖专注于 API 模式。本地模型还需安装 `torch`、`transformers` 和
`accelerate`；HELM 与 Anthropic 模式也需要各自的可选 SDK 和凭据。

## 退出虚拟环境

```shell
deactivate
```
