# SCC AKO4ALL Optimization Goal

## Scoring Interpretation

Based on the OJ discussion, scoring is best understood as a normalized position between two timing anchors per test case:

- Baseline anchor: the baseline implementation time, mapped to a configurable base score such as 50 or 30.
- Hardware-limit anchor: the target best-achievable time for that test case, mapped to 100.

This means the platform does not need one global full-score speedup such as 5x for every problem. Different cases can have different baseline times and hardware-limit times:

```text
case A: baseline 10 ms, hardware limit 2 ms -> full-score speedup is 5x
case B: baseline  9 ms, hardware limit 3 ms -> full-score speedup is 3x
```

Both can still map to 100 points because the full score is tied to that case's configured hardware-limit anchor, not a global speedup ratio.

The discussion also says case weights can be configured, so the final score should be treated as either an equal-weighted or weighted aggregation of per-case normalized scores. The exact formula and field names must still be confirmed from the OJ documentation or maintainers.

## Our Target

For this workspace, the concrete target is:

1. Use AKO4ALL to optimize the three SCC operators on NVIDIA A800.
2. Run correctness and timing through Slurm on `a800/g07`.
3. For each problem and each test case, record:
   - baseline runtime
   - best optimized runtime
   - speedup
   - measurement stability, such as min/mean/std if the harness exposes them
4. Use the best stable optimized runtime as the starting point for the OJ hardware-limit time.
5. Add a conservative margin only after repeated full measurements, so the configured full-score target is difficult but reachable by a strong implementation.

## Problems

### A: Fused SwiGLU Up Projection

Path: `USTB-SCC-A-fused-swiglu-up-projection`

Interface: Triton `run_kernel(x, w_gate, w_up, b_gate, b_up, y, M, K, N)`

Test cases: 6 GEMM-heavy BF16 cases in `testcase_config.py`.

Optimization objective: fuse two shared-input GEMMs with bias, SiLU, and output writeback; minimize memory traffic and maximize Tensor Core utilization within Triton.

### B: Online Softmax GEMM

Path: `USTB-SCC-B-online-softmax-gemm`

Interface: CUDA `run_kernel(Q, K, V, mask, B, H, S, D, alpha, O)`

Test cases: 6 BF16 attention cases in `testcase_config.py`.

Optimization objective: implement a FlashAttention-style online softmax kernel that avoids writing the full attention matrix to HBM.

### C: Mixed-Precision Grouped GEMM

Path: `USTB-SCC-C-mixed-precision-grouped-gemm`

Interface: CUDA `run_kernel(A, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k)`

Test cases: 8 irregular FP8/INT4 grouped GEMM cases in `testcase_config.py`.

Optimization objective: optimize dequantization, INT4 unpacking, grouped row scheduling, padding-row handling, and GEMM tiling on A800.

## Open Platform Questions

Before writing final OJ metadata, confirm:

- What exact config field names does XPUOJ use for baseline time, hardware-limit time, base score, and case weight?
- Is hardware-limit time supplied per case by the problem author, or stored elsewhere during deployment?
- Does the normalization use time linearly, speedup linearly, log-speedup, or another curve?
- How are results faster than the hardware-limit anchor clamped?

Until those are confirmed, we can still optimize kernels and produce the measured timing table that will determine sensible hardware-limit anchors.
