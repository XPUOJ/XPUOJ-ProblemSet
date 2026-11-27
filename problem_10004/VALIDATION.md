# Indexer TopK Transform 验证记录

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
| 1 | dsv4-paged-boundaries | 0.2968 | 0.2680 | 0.3045 |
| 2 | dsv4-paged-selection | 0.5214 | 0.4932 | 0.5297 |
| 3 | nsa-paged | 1.0040 | 0.9748 | 1.0284 |
| 4 | nsa-ragged | 1.0066 | 0.9768 | 1.0254 |
| 5 | tie-membership | 0.0650 | 0.0454 | 0.0740 |
| 6 | row-stride-and-offset | 0.0742 | 0.0457 | 0.0822 |
| 7 | paged512-16k | 5.7061 | 5.6910 | 5.8070 |
| 8 | paged512-b8-32k | 27.4698 | 27.4606 | 27.5615 |
| 9 | paged2048-32k | 11.4419 | 11.4260 | 11.5216 |
| 10 | paged2048-b4-64k | 56.6143 | 56.5894 | 57.7256 |
| 11 | ragged2048-64k | 23.5743 | 23.5595 | 23.6740 |
| 12 | ragged2048-b8-32k-strided | 27.4754 | 27.4240 | 28.4430 |

用例 7–12 按题面权重聚合，参考 kernel 的 `weighted_geomean_ms` 为 **18.4951 ms**。参赛提交的性能由统一平台按相同规则计量。

## Runner 与错误结果检查

实际 `/usr/local/bin/tilelang_runner.py` 执行用例 1, 7, 10, 12，日志均为 `warmup=20, iters=50`，签名校验通过且 `pass=true`。Runner 对 no-op 提交输出 `pass=false`。

独立检查验证以下错误或边界处理：

* `no_write`：通过。
* `wrong_dtype`：通过。
* `wrong_shape`：通过。
* `modified_input`：通过。
* `duplicate_member`：通过。
* `invalid_transform`：通过。
* `non_topk_member`：通过。
* `bad_padding`：通过。
* `alternative_tie_set_accepted`：通过。
* `unordered_set_accepted`：通过。
* `none_return`：通过。
* `different_output_storage`：通过。

非连续行步长在保留原始 stride 的沙箱调用中校验；平台导入操作按 README 保留该布局。

## 资源记录

沙箱报告的最大累计 CPU 时间为 26.83 s，进程内存峰值为 3126 MiB。独立验证的最大 GPU allocated 为 0.009 GiB，包含原输入、提交输入和参考输入等验证副本，不等于 kernel 自身的 workspace。

## 文件校验

```text
testcase_config.py  4da84edc6cf5ab04f3da19e6d1b0e6e70d6e58fd768a55497cbcb28e0ba32bfd
reference/tilelang.py  2b417fbae1dd724a4eb984e67f3956891b6bfdb4effd7ed0aa44fe4de4a10582
```
