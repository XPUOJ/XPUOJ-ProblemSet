# XPUOJ 题目编写规范

本文档用于约束后续新增题目的目录结构、题面内容、评测配置和可变项。目标是让新题与本仓库已有 `problem_*` 题目保持同一套框架，导入 OJ 后可以直接展示题面、生成测试数据、调用参赛者提交的 `run_kernel` 并完成正确性检查与性能统计。

## 1. 目录结构

每道题必须独立放在仓库根目录的一个目录中：

```text
problem_<displayId>/
  meta.json
  testcase_config.py
  zh_CN/
    _title.txt
    00_题目描述.md
    01_接口约定.<backend>.md
    02_输入格式.md
    03_输出格式.md
    04_样例.md
    05_数据范围与提示.md
    06_pytorch_参考实现.md
```

约束：

- `problem_<displayId>` 中的 `<displayId>` 必须与 `meta.json` 的 `displayId` 一致。
- 新题默认只提供 `zh_CN` 语言目录，`meta.json.locales` 写 `["zh_CN"]`。
- `_title.txt` 只放题目标题，一行即可，建议包含算子名称和主要 dtype，例如 `Dense GeMM (bf16)`。
- Markdown 文件名使用数字前缀控制展示顺序，front matter 必须放在文件开头。
- 标准题应保留 `00` 到 `06` 这些章节。确需增加章节时，使用新的数字前缀并保持顺序清晰。
- 已有历史题中存在 `02_参数说明.md`、重复 `数据范围与提示` 等例外；新题不要复制这些例外，除非平台侧有明确需求。

## 2. meta.json

`meta.json` 描述 OJ 题目的基础元信息。标准模板：

```json
{
  "id": 51,
  "displayId": 34,
  "type": "Traditional",
  "isPublic": true,
  "locales": [
    "zh_CN"
  ],
  "samples": [
    {
      "inputData": "",
      "outputData": ""
    }
  ],
  "problemTagIds": []
}
```

字段要求：

- `id`：平台内部题目 ID，必须全仓库唯一。由平台或维护者分配，不要随意复用。
- `displayId`：用户可见题号，必须全仓库唯一，并与目录名 `problem_<displayId>` 一致。
- `type`：固定为 `"Traditional"`。
- `isPublic`：公开题写 `true`，内部或待发布题写 `false`。
- `locales`：当前固定为 `["zh_CN"]`。
- `samples`：至少保留一个对象，包含 `inputData` 与 `outputData`。GPU kernel 题通常可为空字符串；若题面需要展示可理解样例，可以写人类可读的输入输出说明。
- `problemTagIds`：没有平台标签时保留空数组。

开放项：

- `id`、`displayId`、`isPublic`、`samples` 内容可按题目和发布状态变化。
- 不要新增平台未识别字段，除非同步更新导入工具或 OJ 平台。

## 3. 题面 Markdown 规范

每个 `zh_CN/*.md` 文件开头必须包含 front matter：

```markdown
---
sectionTitle: "题目描述"
type: "Text"
---
正文内容
```

代码接口章节使用：

```markdown
---
sectionTitle: "接口约定"
type: "codeSample"
lang: "cuda"
---
```

常用章节要求：

- `00_题目描述.md`：说明要实现的算子、数学定义、调用方式。必须明确评测程序会调用 `run_kernel`。结尾保留评测指南链接：`如何提交代码详见[评测指南](/d/2)。`
- `01_接口约定.<backend>.md`：给出参赛者必须实现的 `run_kernel` 签名。函数名、参数类型、参数顺序必须与 `testcase_config.py` 的返回参数完全一致。
- `02_输入格式.md`：说明评测程序会在 GPU 上构造输入并按顺序传入 `run_kernel`，列出每个参数的 dtype、shape、读写属性、连续存储要求和标量含义。
- `03_输出格式.md`：说明需要写入哪个输出 tensor 或哪些 INOUT tensor，给出数学公式或逐元素定义。
- `04_样例.md`：给小规模概念样例，帮助理解计算流程。样例不必与真实测试点尺寸一致。
- `05_数据范围与提示.md`：列出 dtype、shape、contiguous、边界条件、正确性要求、是否允许修改输入、真实测试点尺寸表。
- `06_pytorch_参考实现.md`：给出与 `testcase_config.py` 中 `baseline` 逻辑一致的 PyTorch 参考实现。

Markdown 内容约束：

- 参数名称在所有文件中必须一致，包括大小写。
- 数学公式、shape 和测试配置必须与 `testcase_config.py` 保持一致。
- 若某个维度可能不是 block size 的倍数，题面必须明确要求处理边界。
- 输出 tensor 初值通常视为未定义，题面应要求完整覆盖输出。
- 对只读输入必须明确“不允许修改输入 tensor”；对原地题必须明确哪个参数是 INOUT。

## 4. 接口约定文件

本仓库现有公开题通常同时提供 CUDA、Triton、TileLang 三种接口：

```text
01_接口约定.cuda.md
01_接口约定.triton.md
01_接口约定.tilelang.md
```

新题默认也应提供三种接口，除非题目明确只开放某个后端。仅开放某后端时，只保留对应接口文件，并在题面中避免暗示其他后端可提交。

CUDA 接口要求：

- 必须声明 `extern "C" void run_kernel(...)`，避免 name mangling。
- 指针参数使用与 dtype 对应的 C/CUDA 类型，例如 `__half*`、`__nv_bfloat16*`、`float*`、`int32_t*`。
- 标量尺寸参数统一使用 `int64_t`。
- 输出参数用非 const 指针；只读输入用 `const` 指针。
- 文档中说明 `run_kernel` 内部需要自行计算 grid/block 并 launch kernel。

Triton/TileLang 接口要求：

- 必须提供 Python 层 `run_kernel(...)` 函数。
- 参数顺序必须与 CUDA 接口、题面输入格式、`genTestCase` 返回列表、`INPUT_CLASS` 完全一致。
- TileLang 题可以在 `run_kernel` 内缓存编译后的 kernel，例如使用全局 `real_kernel`。

开放项：

- 可按题目选择支持的后端集合。
- 可使用 workspace 参数，但必须在输入格式、输出格式、`INPUT_CLASS` 和 check 逻辑中说明其角色。

## 5. testcase_config.py 必备接口

`testcase_config.py` 是评测生成数据、运行基准和检查答案的核心文件。新题必须提供以下对象或函数：

```python
from __future__ import annotations

from typing import List, Tuple, Union
import torch

KernelArg = Union[torch.Tensor, int, float]

TESTCASES = [
    # 每项建议包含题目维度、warmup、iters
]


def getNumOfTestcases() -> int:
    return len(TESTCASES)


def getTestCaseSize():
    testcase_id = int(input())
    ...
    return [
        # tensor 参数返回 shape tuple
        # scalar 参数返回 ()
    ], (warmup, iters)


def genTestCase(testcase_sizes, device: str = "cuda") -> List[KernelArg]:
    ...
    return [arg0, arg1, ...]


def baseline(*args):
    ...
    return [arg0, arg1, ...]


def check(testcase_sizes, original_input_tensors, target_kernel_input_tensors, baseline_input_tensors, rtol=1e-2, atol=1e-2) -> bool:
    ...
    return True


INPUT_CLASS = ["INPUT", "OUTPUT", ...]


def getWorkload(testcase_sizes) -> dict:
    return {
        "flops": ...,
        "memory_bytes": ...,
        "dtype": "fp16",
    }


DESIGNED_VRAM_SIZE = 48
```

函数约束：

- `getNumOfTestcases()`：返回真实测试点数量，必须与 `TESTCASES` 一致。
- `getTestCaseSize()`：从标准输入读取测试点编号，返回参数 shape 列表和 `(warmup, iters)`。tensor 参数用 shape tuple，标量参数用 `()`。
- `genTestCase(testcase_sizes, device="cuda")`：按 shape 构造 CUDA tensor 和 Python 标量，返回顺序必须等于 `run_kernel` 参数顺序。
- `baseline(...)`：使用 PyTorch 实现正确结果，并把结果写入对应输出或 INOUT tensor，返回完整参数列表。
- `check(...)`：比较用户 kernel 后的参数列表与 baseline 后的参数列表，失败时打印清晰原因并返回 `False`。
- `INPUT_CLASS`：长度必须等于参数个数，顺序必须与 `run_kernel` 参数一致。
- `getWorkload(testcase_sizes)`：返回性能统计口径，至少包含 `flops`、`memory_bytes`、`dtype`。
- `DESIGNED_VRAM_SIZE`：题目设计显存，默认 `48`，需要更大显存时可以提高，例如已有题使用 `256`。

兼容建议：

- 可以把重依赖导入和函数定义放在 `try: ... except: pass` 中，保持平台加载元信息时更稳健。但新题仍应确保完整运行时这些函数存在。
- 推荐提供容错的测试点编号读取函数：空输入、EOF 或非法编号时回退到第一个测试点。
- 如需自动调参工具编辑测试点，`TESTCASES` 使用简单列表结构，不要把核心尺寸藏在复杂逻辑中；确需生成大量 case 时可用 `_build_cases()`。

## 6. INPUT_CLASS 语义

`INPUT_CLASS` 告诉评测器每个参数的读写角色，必须与 kernel 实际行为一致：

- `"INPUT"`：只读输入或标量。用户 kernel 不应修改对应 tensor。
- `"OUTPUT"`：输出缓冲区。初值不应被依赖，kernel 必须写入结果。
- `"INOUT"`：原地输入输出。baseline 和用户 kernel 都会修改该 tensor，check 应比较修改后的结果。

约束：

- 输出 tensor 和 workspace tensor 通常标记为 `"OUTPUT"`。
- 标量维度参数标记为 `"INPUT"`。
- 只读 tensor 在 CUDA 接口中应使用 `const` 指针，在题面中明确“不允许修改”。
- `baseline` 返回的参数列表必须保持与 `genTestCase` 相同顺序，方便 check 按索引比较。

## 7. 测试点设计

每个测试点应同时包含计算尺寸和计时参数：

```python
# 示例: (M, K, N, warmup, iters)
TESTCASES = [
    (4096, 4096, 4096, 5, 100),
    (8192, 2048, 4096, 5, 100),
]
```

设计要求：

- 测试点数量通常为 3 到 6 个；复杂覆盖型题可以更多。
- 每个测试点都要在 `05_数据范围与提示.md` 中列出可见尺寸。
- 覆盖典型大尺寸、边界尺寸和非整除尺寸，尤其是 block size、tile size、heads/group 等边界。
- `warmup` 和 `iters` 应根据算子耗时设置，避免过短导致计时抖动，避免过长导致评测超时。
- 数据规模不得超过 `DESIGNED_VRAM_SIZE` 所声明的目标显存。
- 随机数据应控制数值范围，避免参考实现溢出、NaN 或容差不稳定。
- 如果 baseline 很慢，降低大测试点的 `iters`，但不要降低正确性覆盖。

索引约束：

- 公开题已有代码多数使用 1-based 测试点编号；部分内部题使用 0-based。新题推荐使用 1-based，并在非法输入时回退到 1。
- `meta.json.samples` 中的输入输出只是展示用，不应依赖它驱动真实测试。真实测试点由 `testcase_config.py` 控制。

## 8. 正确性检查

`check` 必须验证：

- 输出 tensor shape 一致。
- 输出 tensor dtype 一致。
- 数值满足题目容差，通常使用 `torch.allclose`。
- 失败时打印 `shape mismatch`、`dtype mismatch`、`max_abs_diff`、`mean_abs_diff` 等信息，便于定位。

容差建议：

- fp32 简单算子可使用更严格容差。
- fp16/bf16 常用 `rtol=1e-2, atol=1e-2`。
- fp8、复杂融合、累加顺序差异较大的题可适当放宽，但必须在题面和 check 中保持一致。
- 对 softmax、attention、MoE 等数值敏感题，先用多组随机数据验证容差稳定后再发布。

禁止项：

- 不要只检查某一个元素或只检查 shape。
- 不要在 `check` 中修改目标输出后再比较。
- 不要让 baseline 依赖用户 kernel 修改后的数据。

## 9. workload 统计

`getWorkload(testcase_sizes)` 用于给平台展示或计算性能指标。返回格式：

```python
{
    "flops": 2 * M * N * K,
    "memory_bytes": ...,
    "dtype": "bf16",
}
```

要求：

- `flops` 写主要计算量口径，并在函数注释中解释公式。
- `memory_bytes` 写主要读写访存口径，按 dtype 字节数计算。
- `dtype` 使用主要计算或输入输出 dtype，例如 `"fp16"`、`"bf16"`、`"fp8"`。
- 对融合算子，应把所有主要子算子的 FLOPs 纳入同一口径。
- 题面 `05_数据范围与提示.md` 中的描述应与 workload 口径不矛盾。

## 10. 开放项与不可变项

不可变项：

- 目录必须是 `problem_<displayId>`。
- 必须有 `meta.json`、`testcase_config.py`、`zh_CN/_title.txt`。
- 必须有题目描述、接口约定、输入格式、输出格式、样例、数据范围与提示、PyTorch 参考实现。
- `run_kernel` 函数名不可变。
- 参数顺序在接口文档、题面、`genTestCase`、`baseline`、`check`、`INPUT_CLASS` 中必须完全一致。
- 真实测试数据必须由 `testcase_config.py` 生成。
- 输出正确性必须由 PyTorch baseline 和 check 验证。

开放项：

- 题目算子、数学定义、dtype、shape、测试点尺寸。
- 支持 CUDA/Triton/TileLang 的一种或多种后端。
- 公开或私有发布状态。
- 样例是否使用真实数值，或仅作概念说明。
- 容差、warmup/iters、目标显存。
- 是否提供 workspace 参数。

## 11. 新题提交前自检清单

- 目录名、`displayId`、`_title.txt` 已确认。
- `meta.json` 是合法 JSON，`id` 与 `displayId` 未重复。
- `zh_CN` 下章节齐全，front matter 正确。
- 所有接口文件中的 `run_kernel` 签名一致。
- `testcase_config.py` 可以被 Python 导入。
- `getNumOfTestcases()` 与 `TESTCASES` 数量一致。
- 每个测试点调用 `getTestCaseSize()` 后，`genTestCase()` 能生成参数。
- `baseline()` 能写出正确输出，并返回完整参数列表。
- `INPUT_CLASS` 长度等于参数数量。
- `check()` 能通过 baseline 自测，失败时有明确错误信息。
- `getWorkload()` 返回 `flops`、`memory_bytes`、`dtype`。
- `05_数据范围与提示.md` 的测试点表与 `TESTCASES` 完全一致。
- 输出 tensor 初值不被依赖，输入 tensor 的可修改性与 `INPUT_CLASS` 一致。

## 12. 推荐最小模板

新增题目时可以先复制一个结构相近的已有题，再按以下顺序修改：

1. 修改目录名、`meta.json`、`_title.txt`。
2. 修改 `testcase_config.py` 的 `TESTCASES`、参数列表、baseline、check、workload。
3. 修改 `01_接口约定.*.md` 的 `run_kernel` 签名和参数说明。
4. 修改 `00` 到 `06` 题面章节。
5. 对照自检清单逐项校验。

优先参考：

- 简单原地算子：`problem_1`
- GEMM 类题：`problem_2`、`problem_10`
- 融合激活/逐元素算子：`problem_24`
- Attention/MoE/复杂 metadata 题：`problem_10001`、`problem_10002`、`problem_10003`

