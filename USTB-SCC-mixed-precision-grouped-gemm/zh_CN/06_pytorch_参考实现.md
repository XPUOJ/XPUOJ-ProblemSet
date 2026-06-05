---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A_fp8, A_scale, B_packed, B_scale, m_indices, D, M_total, K, N, num_groups, group_k):
    A_real = A_fp8.float() * repeat_interleave(A_scale, 128, dim=-1)[..., :K]
    B_q = unpack_signed_int4(B_packed, K).float()
    B_real = B_q * repeat_interleave(B_scale, group_k, dim=-1)[..., :K]
    D.zero_()
    for g in range(num_groups):
        rows = where(m_indices == g)
        if rows.numel() > 0:
            D[rows] = A_real[rows] @ B_real[g].T
```

