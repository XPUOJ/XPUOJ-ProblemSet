---
sectionTitle: "PyTorch 参考实现"
type: "codeSample"
lang: "python"
---

以下代码展示与 `testcase_config.py` 中 `baseline` 逻辑一致的 PyTorch 参考实现。该实现用于说明目标计算语义和生成正确性参考结果，不是参赛提交方式。

```python
import torch

SCALE_BLOCK_A = 128


def unpack_signed_int4(packed: torch.Tensor, K: int) -> torch.Tensor:
    """将每个 uint8 中的两个 signed INT4 解包为 int8，低 4 位在前。"""
    bytes_i16 = packed.to(torch.int16)
    low = bytes_i16 & 0xF
    high = (bytes_i16 >> 4) & 0xF
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    out = torch.empty(
        *packed.shape[:-1],
        packed.shape[-1] * 2,
        device=packed.device,
        dtype=torch.int8,
    )
    out[..., 0::2] = low.to(torch.int8)
    out[..., 1::2] = high.to(torch.int8)
    return out[..., :K].contiguous()


def expand_scale(scale: torch.Tensor, block: int, K: int) -> torch.Tensor:
    return scale.repeat_interleave(block, dim=-1)[..., :K].contiguous()


def reference(
    A_fp8,
    A_scale,
    B_packed,
    B_scale,
    m_indices,
    D,
    M_total: int,
    K: int,
    N: int,
    num_groups: int,
    group_k: int,
):
    """
    参考语义：
      D[i, :] = dequant_fp8(A[i, :]) @ dequant_int4(B[m_indices[i]]).T

    当 m_indices[i] == -1 时，该行是 padding 行，输出为 0。
    """
    A_real = A_fp8.float() * expand_scale(A_scale, SCALE_BLOCK_A, K)

    B_q = unpack_signed_int4(B_packed, K).float()
    B_real = B_q * expand_scale(B_scale, group_k, K)

    D.zero_()
    for g in range(num_groups):
        rows = torch.nonzero(m_indices == g, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        out = A_real.index_select(0, rows) @ B_real[g].transpose(0, 1)
        D.index_copy_(0, rows, out.to(torch.bfloat16))

    return [A_fp8, A_scale, B_packed, B_scale, m_indices, D,
            M_total, K, N, num_groups, group_k]
```

**说明**：

- 参考实现为 PyTorch 语义实现，用于说明正确结果；正式计时只统计参赛者提交的 `run_kernel`。
- 参赛者不能直接调用 PyTorch 参考实现，需要在 CUDA kernel 中完成 FP8 反量化、INT4 unpack/反量化、grouped GEMM 和 padding 行处理。
- 参考实现会先完整展开 `A_real` 和 `B_real`，这便于表达语义，但不是高性能实现方式。优化实现应尽量将 unpack、scale 乘法和矩阵乘融合，避免把完整反量化后的 `B_real` 写入全局内存。
- 正确性检查会将用户 kernel 的输出与参考实现进行 `torch.allclose(rtol=2e-2, atol=2e-2)` 比较。
