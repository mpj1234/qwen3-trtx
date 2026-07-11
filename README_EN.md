# qwen3-trtx

[中文版本](README.md)

Deploy the Qwen3 model with TensorRT / TensorRTX style inference.

## Features

- Build a TensorRT engine from exported Qwen3 `.wts` weights.
- Run interactive dialog inference from a serialized `.trt` engine.
- Support a custom `apply_rotary_pos_emb` TensorRT plugin.
- Support UTF-8 Chinese console input and UTF-8 dialog log output on Windows.
- Filter `<think>...</think>` content from model output.

## Build

Use CMake to configure and build the project, for example:

```bash
cmake -S . -B cmake-build-release
cmake --build cmake-build-release --config Release
```

## Usage

The executable supports two startup modes:

```bash
qwen3_trtx.exe -s [qwen3_wts_dir] [.trt]
qwen3_trtx.exe -d [.trt] [tokenizer_dir]
```

### 1. Serialize engine

Build a TensorRT engine from the Qwen3 `.wts` directory:

```bash
qwen3_trtx.exe -s F:\LLM\transformers-4.57.6\wts ..\models\test.engine
```

Arguments:

- `qwen3_wts_dir`: directory containing exported Qwen3 `.wts` files
- `.trt`: output TensorRT engine path

### 2. Dialog inference

Load a TensorRT engine and enter interactive dialog mode:

```bash
qwen3_trtx.exe -d ..\models\test.engine F:\LLM\modelscope\hub\models\Qwen\Qwen3-0.6B
```

Arguments:

- `.trt`: TensorRT engine path
- `tokenizer_dir`: tokenizer / model directory used by `tokenizer.cpp`; this argument is required

After startup:

- enter dialog text in the console
- type `exit` or `quit` to leave

## Dialog Behavior

- The system prompt is fixed to Chinese assistant mode.
- Assistant output is cleaned before display and logging.
- Dialog history is appended to `models/dialog_utf8.txt` in UTF-8 format.

## Notes

- `models/` is ignored by git by default, so generated engines and logs are not committed.
- On Windows, console input is handled in a UTF-8 compatible way for Chinese dialog input.
