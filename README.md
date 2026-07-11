# qwen3-trtx

[English Version](README_EN.md)

基于 TensorRT / TensorRTX 方式部署 Qwen3 模型。

## 功能

- 支持从导出的 Qwen3 `.wts` 权重构建 TensorRT engine。
- 支持从序列化后的 `.trt` engine 进入对话推理模式。
- 支持自定义 `apply_rotary_pos_emb`、`repeat_kv` TensorRT 插件。
- 支持 Windows 控制台 UTF-8 中文输入与 UTF-8 对话日志输出。
- 支持过滤模型输出中的 `<think>...</think>` 内容。

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

## 对话行为

- 系统提示词固定为中文助手模式。
- 助手输出在显示和写日志前会进行清理。
- 对话历史会以 UTF-8 格式追加写入 `models/dialog_utf8.txt`。

## 说明

- `models/` 默认已被 git 忽略，因此生成的 engine 和日志不会被提交。
- 在 Windows 下，控制台输入做了 UTF-8 兼容处理，适合中文对话输入。
