---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(probs, top_p, renorm_probs, batch_size, num_classes):
    """
    PyTorch 参考实现：Top-p 重归一化
    
    参数说明：
    * probs: 输入张量（float32），shape (batch_size, num_classes)，行优先连续存储，只读
    * top_p: 输入张量（float32），shape (batch_size,)，连续存储，每个样本的 top-p 阈值
    * renorm_probs: 输出张量（float32），shape (batch_size, num_classes)，行优先连续存储，需写入结果
    * batch_size: 批次大小（int64）
    * num_classes: 类别数量（int64）
    """
    B, C = probs.shape
    p = top_p.reshape(B, 1).to(dtype=probs.dtype, device=probs.device)
    
    # 每行降序排序
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    
    # 累积和
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # nucleus mask: cumulative <= top_p
    mask = cumsum <= p
    
    # 保证每行至少保留 1 个
    mask[:, 0] = True
    
    # 过滤：不在 top-p 里的置零
    filtered_sorted = sorted_probs * mask.to(sorted_probs.dtype)
    
    # 计算每行归一化系数
    denom = filtered_sorted.sum(dim=-1, keepdim=True)
    
    # 避免除 0
    renorm_sorted = torch.where(
        denom > 0,
        filtered_sorted / denom,
        torch.zeros_like(filtered_sorted)
    )
    
    # scatter 回原顺序
    renorm_probs.zero_()
    renorm_probs.scatter_(dim=-1, index=sorted_idx, src=renorm_sorted)
```
