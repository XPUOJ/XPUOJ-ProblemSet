# 本地 CUDA Runner 使用说明

`tools/local_cuda_runner.py` 用于在本机模拟 OJ 的核心流程：

1. 用题目的 `testcase_config.py` 生成输入。
2. 编译 CUDA 源文件为 `.so`。
3. 通过 `ctypes` 调用导出的 `extern "C" run_kernel(...)`。
4. 调用题目的 `baseline()` 和 `check()` 做正确性验证。
5. 用 CUDA event 按题目给出的 `warmup/iters` 计时。
6. 根据 `getWorkload()` 输出 TFLOP/s 或 GB/s。
7. 可选计时 PyTorch `baseline()`，输出本地 speedup。

## 环境

当前机器可使用 `matris311` 环境：

```bash
conda run -n matris311 python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

runner 会优先使用完整系统 CUDA toolkit：

```text
/usr/local/cuda-13.0/bin/nvcc
```

PyTorch 数据生成和校验仍使用 `matris311` 环境。

## 示例

运行 `problem_1` 示例 kernel：

```bash
conda run -n matris311 python tools/local_cuda_runner.py problem_1 examples/problem_1_add.cu --case 1 --gpu 0
```

运行某题全部测试点：

```bash
conda run -n matris311 python tools/local_cuda_runner.py problem_13 path/to/solution.cu --case all --gpu 0
```

只运行某个测试点：

```bash
conda run -n matris311 python tools/local_cuda_runner.py problem_13 path/to/solution.cu --case 2 --gpu 0
```

给 nvcc 追加参数：

```bash
conda run -n matris311 python tools/local_cuda_runner.py problem_13 path/to/solution.cu --extra-nvcc=-lineinfo
```

同时计时 PyTorch baseline 并输出加速比：

```bash
conda run -n matris311 python tools/local_cuda_runner.py problem_13 path/to/solution.cu --case 1 --time-baseline
```

## CUDA 源文件要求

CUDA 源文件必须导出题面要求的 C 符号：

```cpp
extern "C" void run_kernel(...);
```

参数顺序和类型必须与对应题目的 `zh_CN/01_接口约定.cuda.md` 完全一致。

## 注意

- 这个工具模拟的是 OJ 的核心正确性和计时流程，不保证与正式 OJ 外层 runner 完全一致。
- `--time-baseline` 比较的是题目 `testcase_config.py` 里的 PyTorch baseline，适合看本地加速比；正式 OJ 可能按 kernel 耗时、吞吐或平台规则评分。
- 不要在 `run_kernel` 内部主动 `cudaDeviceSynchronize()`，否则会影响计时。
- 编译产物放在 `.local_build/`，已加入 `.gitignore`。
- A800 架构为 `sm_80`，runner 默认使用 `--arch sm_80`。
