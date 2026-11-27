# Fused MOE 验证记录

验证日期：2026-11-26。全部用例使用本目录的 `testcase_config.py` 和 `reference/tilelang.py`。

## 环境

* MTT S5000，MUSA SDK / MTCC 5.3.0。
* TileLang-MUSA `0.1.12+musa.5.gita5fc8648`。
* PyTorch / Torch-MUSA `2.13.0.post1+musa20261110`，Python 3.12.13。
* 原生 Judge 沙箱，UID 188001，禁用网络，采用评测端 CPU affinity。

## 正确性与计时

12 个用例，每个使用 2 个随机输入，全部通过 PyTorch 参考校验。校验输入保护、输出 shape/dtype/device、原地写入及返回同一输出存储。重复执行后再次校验结果。

所有用例均使用 MUSA events 预热 20 次、计时 50 次，首次编译在计时前完成。下表为参考 kernel 的实测耗时，单位 ms。

| 用例 | 名称 | 中位数 | 最小值 | 最大值 |
| --- | --- | --- | --- | --- |
| 1 | ds-v4-fp16 | 0.1562 | 0.1331 | 0.1634 |
| 2 | ds-v4-bf16 | 0.1828 | 0.1586 | 0.1938 |
| 3 | ds-r1-fp16 | 0.1992 | 0.1790 | 0.2108 |
| 4 | ds-r1-bf16 | 0.2721 | 0.2478 | 0.2996 |
| 5 | qwen-fp16 | 0.2462 | 0.2301 | 0.2579 |
| 6 | qwen-bf16 | 0.3705 | 0.3596 | 0.3820 |
| 7 | r1-fp16-decode-4k | 5.2798 | 5.0091 | 5.3800 |
| 8 | r1-bf16-m16-4k | 66.7084 | 63.2053 | 69.6940 |
| 9 | v4-fp16-m8-e256 | 4.8525 | 4.7858 | 5.1797 |
| 10 | v4-bf16-m64-e256 | 107.6039 | 106.0223 | 110.4130 |
| 11 | qwen-fp16-m16-e256-4k | 82.7081 | 80.2946 | 84.8300 |
| 12 | qwen-bf16-m256-e256 | 352.9149 | 348.3777 | 357.4470 |

用例 7–12 按题面权重聚合，参考 kernel 的 `weighted_geomean_ms` 为 **34.9242 ms**。参赛提交的性能由统一平台按相同规则计量。

## Runner 与错误结果检查

实际 `/usr/local/bin/tilelang_runner.py` 执行用例 1, 8, 10, 11, 12，日志均为 `warmup=20, iters=50`，签名校验通过且 `pass=true`。Runner 对 no-op 提交输出 `pass=false`。

独立检查验证以下错误或边界处理：

* `no_write`：通过。
* `wrong_dtype`：通过。
* `wrong_shape`：通过。
* `modified_input`：通过。
* `nan`：通过。
* `inf`：通过。
* `wrong_value`：通过。
* `ignoring_routing_bias`：通过。
* `none_return`：通过。
* `different_output_storage`：通过。

## 资源记录

沙箱报告的最大累计 CPU 时间为 157.26 s，进程内存峰值为 3775 MiB。独立验证的最大 GPU allocated 为 20.003 GiB，包含原输入、提交输入和参考输入等验证副本，不等于 kernel 自身的 workspace。

## 文件校验

```text
testcase_config.py  bd80d9dfe1fcaeee186071ac4f2a1bf4daba7b2607d52c36bcd253bf8cd8226c
reference/tilelang.py  013ebd10a09cf803d86d8f02f00055ce18d2d573c2b4923a8214884b5148dd13
```
