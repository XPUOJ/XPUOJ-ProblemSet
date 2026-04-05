---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(a, b, masked_m, d, num_groups, max_m, n, k):
    """
    PyTorch 参考实现：M-grouped BF16 GEMM NT with masked M
    
    参数说明：
    * a: 输入张量（bfloat16），shape (num_groups, max_m, k)，行优先连续存储，只读
    * b: 输入张量（bfloat16），shape (num_groups, n, k)，每个组的权重矩阵，只读
    * masked_m: 输入张量（int32），shape (num_groups,)，每个组的实际处理行数，只读
    * d: 输出张量（bfloat16），shape (num_groups, max_m, n)，行优先连续存储，需写入结果
    * num_groups: 分组数量（int64）
    * max_m: 每个组的最大行数（int64）
    * n: b 的第二维大小，d 的第三维大小（int64）
    * k: a 和 b 的最后一维大小（int64）
    """
    d.zero_()
    
    # d = (a @ b.T) * mask
    res = torch.matmul(a.float(), b.transpose(1, 2).float())
    
    # Create mask (G, M)
    m_indices = torch.arange(max_m, device=a.device).unsqueeze(0)  # (1, M)
    mask = m_indices < masked_m.unsqueeze(1)  # (G, M)
    
    # Apply mask (expand to N)
    res = res * mask.unsqueeze(2).float()
    d.copy_(res.to(torch.bfloat16))
```
