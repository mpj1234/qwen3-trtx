# qwen3-trtx

[English Version](README_EN.md)

基于 TensorRT / TensorRTX 方式部署 Qwen3 模型。

## 功能

- 支持从导出的 Qwen3 `.wts` 权重构建 TensorRT engine。
- 支持从序列化后的 `.trt` engine 进入对话推理模式。
- 支持自定义 `apply_rotary_pos_emb`、`repeat_kv` TensorRT 插件。
- 支持 Windows 控制台 UTF-8 中文输入与 UTF-8 对话日志输出。
- 支持过滤模型输出中的 `<think>...</think>` 内容。

## 项目初始化

首次克隆项目后，需要同步并初始化子模块：

```bash
git clone https://github.com/mpj1234/qwen3-trtx.git
cd qwen3-trtx
git submodule update --init --recursive
```

如果仓库已经拉取过，后续想更新子模块，可执行：

```bash
git submodule update --init --recursive
```

## 编译

使用 CMake 进行配置和编译，例如：

```bash
cmake -S . -B cmake-build-release
cmake --build cmake-build-release --config Release
```

## 用法

程序支持两种启动方式：

```bash
# 生成trt模型
qwen3_trtx.exe -s [qwen3_wts_dir] [.trt]
# qwen3推理
qwen3_trtx.exe -d [.trt] [tokenizer_dir]
```

### 0. 导出 Qwen3 wts 文件

下载Transformers库并导出Qwen3的`.wts`文件。

修改 [qwen3_save_wts.py](qwen3_save_wts.py) 脚本中的 model_name 变量为 Qwen3 模型名称或者路径。

修改 [qwen3_save_wts.py](qwen3_save_wts.py) 脚本中的 wts_file 变量为 Qwen3 wts 文件保存路径。

```bash
python qwen3_save_wts.py
```

### 1. 序列化 engine

从 Qwen3 的 `.wts` 目录构建 TensorRT engine：

```bash
qwen3_trtx.exe -s F:/LLM/transformers-4.57.6/wts ../models/qwen3.bf16.trt
```

参数说明：

- `qwen3_wts_dir`：导出的 Qwen3 `.wts` 文件目录
- `.trt`：输出的 TensorRT engine 路径

### 2. 对话推理

加载 TensorRT engine 并进入交互式对话模式：

```bash
qwen3_trtx.exe -d ../models/qwen3.bf16.trt F:/LLM/modelscope/hub/models/Qwen/Qwen3-0.6B
```

参数说明：

- `.trt`：TensorRT engine 路径
- `tokenizer_dir`：`tokenizer.cpp` 使用的 tokenizer / 模型目录，必须传入

启动后：

- 在控制台输入对话内容
- 输入 `exit` 或 `quit` 退出

### 3. 对话示例

示例输入输出如下：

````text
[INFO] dialog mode ready, type `exit` to quit.

user> 你是谁，有什么能力
assistant> <think>
好的，用户问我是谁以及有什么能力。作为一个AI助手，我需要明确回答。首先，我应该确认用户的需求，可能是想了解我的身份或功能。接下来，我需要简洁明了地介绍自己，同时说明我的能力范围。用户可能希望得到一个全面而友好的回应，所以应该涵盖主要功能，比如知识库、语言理解、问题解答等。同时，要保持语气友好，让用户感到被重视。最后，确保回答符合中文表达习惯，避免使用复杂术语，让用户容易理解。
</think>

我是AI助手，我能够帮助您完成各种任务，包括回答问题、提供信息、协助写作等。我的能力涵盖自然语言处理、知识库检索、多语言支持以及问题解答等。如果您有任何问题或需要帮助，请随时告诉我！

user> 写一个python脚本实现累加和
assistant> <think>
好的，用户让我写一个Python脚本来实现累加和。首先，我需要确定用户的需求是什么。他们可能想要一个简单的程序来计算连续数字的总和，比如从1开始，直到某个数。接下来，我应该考虑如何设计这个脚本。

用户可能没有明确说明具体的数字范围，所以需要给出一个通用的示例，这样他们可以根据自己的需要调整。然后，我需要确保脚本的结构清晰，包括导入必要的模块，比如sys，然后初始化变量sum和current_sum。接着，循环迭 代，每次将current_sum加上当前数字，并更新sum。最后，打印结果。

还要注意代码的可读性和易用性，避免使用复杂的结构。可能需要提醒用户在运行时设置具体的数字范围，以确保脚本能正确工作。此外，测试一下脚本是否能处理不同长度的数字，比如包含零的情况。这样用户就能更好地理解和使用这个脚本了。最后，保持回答简洁，直接给出结果，不需要额外解释。
</think>

以下是一个简单的Python脚本，用于实现累加和：

```python
# 初始化总和和当前值
sum_total = 0
current_sum = 0

# 循环从1到100（可替换为需要累加的数字范围）
for i in range(1, 101):
    current_sum += i
    sum_total += current_sum

# 输出结果
print("累加和为:", sum_total)
```

这个脚本会从1开始累加到100，输出总和。您可以根据需要调整循环范围（例如从1到n），并修改循环体中的数字。

user> exit
````

## 对话行为

- 系统提示词固定为中文助手模式。
- 助手输出在显示和写日志前会进行清理。
- 对话历史会以 UTF-8 格式追加写入 `models/dialog_utf8.txt`。

## 说明

- `models/` 默认已被 git 忽略，因此生成的 engine 和日志不会被提交。
- 在 Windows 下，控制台输入做了 UTF-8 兼容处理，适合中文对话输入。
