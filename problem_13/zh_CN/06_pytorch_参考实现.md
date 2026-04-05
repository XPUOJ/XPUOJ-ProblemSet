---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
BLOCK_SIZE = 128

def _expand_a_sf(a_sf, k):
    """将 a_sf [..., k_blocks] 沿最后一维扩展到 k。"""
    ex = a_sf.repeat_interleave(BLOCK_SIZE, dim=-1)
    if ex.shape[-1] >= k:
        return ex[..., :k].contiguous()
    return torch.cat([ex, ex[..., -1:].expand(*ex.shape[:-1], k - ex.shape[-1])], dim=-1)

def baseline(a_data, a_sf, b_data, b_sf, m_indices, D, M_total, K, N, num_groups):
    """
    PyTorch 参考实现：M-grouped FP8 GEMM NT contiguous
    
    参数说明：
    * a_data: 输入张量（FP8E4M3），shape (M_total, K)，行优先连续存储，只读
    * a_sf: 输入张量（float32），shape (M_total, ceil(K/128))，per-token 缩放因子，只读
    * b_data: 输入张量（FP8E4M3），shape (num_groups, N, K)，每个组的 FP8 权重矩阵，只读
    * b_sf: 输入张量（float32），shape (num_groups, ceil(N/128), ceil(K/128))，per-block 缩放因子，只读
    * m_indices: 输入张量（int32），shape (M_total,)，每行的组索引，-1 表示填充行，只读
    * D: 输出张量（bfloat16），shape (M_total, N)，行优先连续存储，需写入结果
    * M_total: A 和 D 的第一维大小（int64）
    * K: A 的最后一维及 B 的最后一维大小（int64）
    * N: B 的第二维及 D 的第二维大小（int64）
    * num_groups: 分组数量（int64）
    """
    # a 反量化
    a_sf_exp = _expand_a_sf(a_sf, K)  # (M_total, K)
    a_fp32 = a_data.float() * a_sf_exp
    
    # b 反量化
    b_sf_exp = b_sf.repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)
    b_sf_exp = b_sf_exp[:, :N, :K]
    b_fp32 = b_data.float() * b_sf_exp  # (G, N, K)

    D.zero_()
    
    # Gather B matrices
    valid_mask = (m_indices >= 0) & (m_indices < num_groups)
    safe_indices = m_indices.clone()
    safe_indices[~valid_mask] = 0
    
    B_gathered = b_fp32[safe_indices.long()]  # (M, N, K)
    
    # Compute D = A @ B.T
    res = torch.matmul(a_fp32.unsqueeze(1), B_gathered.transpose(1, 2)).squeeze(1)
    
    # Apply mask
    res = res * valid_mask.unsqueeze(1).float()
    
    D.copy_(res.to(torch.bfloat16))
```
