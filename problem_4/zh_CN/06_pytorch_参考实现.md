---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(probs, top_k, renorm_probs, batch_size, num_classes):
    """
    PyTorch 参考实现：Top-k 重归一化
    
    参数说明：
    * probs: 输入张量（float32），shape (batch_size, num_classes)，行优先连续存储，只读
    * top_k: 输入张量（int32），shape (batch_size,)，连续存储，每个样本的 top-k 值
    * renorm_probs: 输出张量（float32），shape (batch_size, num_classes)，行优先连续存储，需写入结果
    * batch_size: 批次大小（int64）
    * num_classes: 类别数量（int64）
    """
    B, C = probs.shape
    k = top_k.reshape(B, 1).to(device=probs.device)
    k = k.clamp(min=0, max=C).to(dtype=torch.long)
    
    Kmax = int(k.max().item())
    renorm_probs.zero_()
    if Kmax == 0:
        return
    
    # 取每行 top-Kmax
    topv, topi = torch.topk(probs, k=Kmax, dim=-1, largest=True, sorted=True)
    
    # 构造 mask：只保留前 k[i] 个
    arange = torch.arange(Kmax, device=probs.device).view(1, Kmax)
    mask = arange < k
    
    # 过滤并计算归一化分母
    topv_masked = topv * mask.to(topv.dtype)
    denom = topv_masked.sum(dim=-1, keepdim=True)
    
    # 重归一化
    topv_renorm = torch.where(
        denom > 0,
        topv_masked / denom,
        torch.zeros_like(topv_masked)
    )
    
    # scatter 回原位置
    renorm_probs.scatter_(dim=-1, index=topi, src=topv_renorm)
```
