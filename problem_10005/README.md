# Fused MOE

MTT S5000 / TileLang-MUSA 正式赛题包。赛题语义及计分规则依据[摩尔学院 S3 赛季](https://academy.mthreads.com/preview/contest/1/race/3#question)。

## 题包内容

* `meta.json`：导入元数据与样例。
* `zh_CN/`：参赛题面，包含接口、数据范围、参考语义及评分规则。
* `testcase_config.py`：12 个用例的数据生成、PyTorch 参考与正确性检查。
* `reference/tilelang.py`：可执行的 TileLang-MUSA 参考 kernel。
* [验证记录](VALIDATION.md)：S5000 原生 Judge 沙箱执行结果与计时。

## 评测配置

用例 1–6 检查边界正确性，性能权重为 0。用例 7–12 同时检查正确性和性能；各业务组包含 2 个等权性能用例，业务组权重为 R1/V3.2 40%、V4 35%、Qwen 25%。全部 12 个用例均须通过正确性校验。

`getTestCaseSize()` 返回 `[case_id], (20, 50)`：预热 20 次，计时 50 次。每次调用按本次输入重新计算结果。正式性能指标使用 MUSA events 中位耗时及加权几何平均；具体公式和逐用例权重见题面评分章节。

`CASES` 中的 `score_weight` 是用例权重元数据。通用 Runner 不读取此字段，平台计分层须按该字段聚合 `median_ms`，跳过权重为 0 的性能数据，同时保留全部用例的正确性门槛。

## 平台导入与发布

1. 导入 12 个用例及对应题面，保持题面中的尺寸、权重与 `CASES` 一致；样例使用用例 1。
2. 执行语言路由使用 `tilelang.musa-S5000`。`meta.json` 中的目录编号用于题库导入，平台沿用已经分配的题目 ID；按赛事开放时间发布。
3. 计时参数从配置读取，首次 import/JIT/扩展初始化在计时前完成。计时采集中记录 median/min/max，排名采用中位数加权几何平均。
4. 调用层校验返回 Tensor 非 None，shape/dtype/device 正确，并引用传入 `out` 的同一输出存储。题目 `check()` 校验写入的结果与输入保护；返回对象由调用层验收。
5. 单测试用例执行时限设为 180000 ms，进程内存上限设为 8192 MiB；包含冷启动、编译、正确性及重复执行。显存独立计量，设备为 80 GiB 规格的 S5000。
