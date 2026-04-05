---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A, B, m_indices, D, M_total, K, N, num_groups):
    """
    PyTorch 参考实现：M-grouped BF16 GEMM NT contiguous
    
    参数说明：
    * A: 输入张量（bfloat16），shape (M_total, K)，行优先连续存储，只读
    * B: 输入张量（bfloat16），shape (num_groups, N, K)，每个组的权重矩阵，只读
    * m_indices: 输入张量（int32），shape (M_total,)，每行的组索引，-1 表示填充行，只读
    * D: 输出张量（bfloat16），shape (M_total, N)，行优先连续存储，需写入结果
    * M_total: A 和 D 的第一维大小（int64）
    * K: A 和 B 的最后一维大小（int64）
    * N: B 的第二维大小，D 的第二维大小（int64）
    * num_groups: 分组数量（int64）
    """
    D.zero_()
    
    # Gather B matrices
    valid_mask = (m_indices >= 0) & (m_indices < num_groups)
    safe_indices = m_indices.clone()
    safe_indices[~valid_mask] = 0
    
    B_gathered = B[safe_indices.long()]  # (M, N, K)
    
    # Compute D = A @ B.T
    res = torch.matmul(A.unsqueeze(1).float(), B_gathered.transpose(1, 2).float()).squeeze(1)
    
    # Apply mask
    res = res * valid_mask.unsqueeze(1).float()
    
    D.copy_(res.to(torch.bfloat16))
```
