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

def baseline(a_data, a_sf, b_data, b_sf, masked_m, d, num_groups, max_m, n, k):
    """
    PyTorch 参考实现：M-grouped FP8 GEMM NT masked
    
    参数说明：
    * a_data: 输入张量（FP8E4M3），shape (num_groups, max_m, k)，行优先连续存储，只读
    * a_sf: 输入张量（float32），shape (num_groups, max_m, ceil(K/128))，per-token 缩放因子，只读
    * b_data: 输入张量（FP8E4M3），shape (num_groups, n, k)，每个组的 FP8 权重矩阵，只读
    * b_sf: 输入张量（float32），shape (num_groups, ceil(N/128), ceil(K/128))，per-block 缩放因子，只读
    * masked_m: 输入张量（int32），shape (num_groups,)，每个组的实际处理行数，只读
    * d: 输出张量（bfloat16），shape (num_groups, max_m, n)，行优先连续存储，需写入结果
    * num_groups: 分组数量（int64）
    * max_m: 每个组的最大行数（int64）
    * n: b 的第二维大小，d 的第三维大小（int64）
    * k: a 和 b 的最后一维大小（int64）
    """
    # a 反量化
    a_sf_exp = _expand_a_sf(a_sf, k)  # (num_groups, max_m, k)
    a_fp32 = a_data.float() * a_sf_exp
    
    # b 反量化
    b_sf_exp = b_sf.repeat_interleave(BLOCK_SIZE, dim=1).repeat_interleave(BLOCK_SIZE, dim=2)
    b_sf_exp = b_sf_exp[:, :n, :k]
    b_fp32 = b_data.float() * b_sf_exp

    d.zero_()
    
    # res = a @ b.T
    res = torch.matmul(a_fp32, b_fp32.transpose(1, 2))
    
    # Masking
    m_indices = torch.arange(max_m, device=a_data.device).unsqueeze(0)  # (1, M)
    mask = m_indices < masked_m.unsqueeze(1)  # (G, M)
    
    res = res * mask.unsqueeze(2).float()
    d.copy_(res.to(torch.bfloat16))
```
