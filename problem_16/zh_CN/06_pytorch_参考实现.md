---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, w1, w3, group_sizes, out, total_tokens, in_features, out_features, num_groups):
    """
    PyTorch 参考实现：Grouped Linear Variadic Weights
    
    参数说明：
    * x: 输入张量（bfloat16），shape (total_tokens, in_features)，行优先连续存储，只读
    * w1: 输入张量（bfloat16），shape (num_groups, out_features, in_features)，只读
    * w3: 输入张量（bfloat16），shape (num_groups, out_features, in_features)，只读
    * group_sizes: 输入张量（int32），shape (num_groups,)，每组 token 数，只读
    * out: 输出张量（bfloat16），shape (total_tokens, out_features * 2)，行优先连续存储，需写入结果
    * total_tokens: 总 token 数（int64）
    * in_features: 输入特征数（int64）
    * out_features: 单权输出的特征数（int64）
    * num_groups: 组数（int64）
    """
    # Create group indices for each token
    group_indices = torch.repeat_interleave(
        torch.arange(num_groups, device=x.device),
        group_sizes.long()
    )  # (total_tokens,)
    
    # Concatenate weights
    w_concat = torch.cat([w1, w3], dim=1)  # (G, 2*Out, In)
    
    # Gather weights for each token
    w_expanded = w_concat[group_indices]  # (total_tokens, 2*Out, In)
    
    # Compute out = x @ w.T
    res = torch.matmul(x.unsqueeze(1).float(), w_expanded.transpose(1, 2).float()).squeeze(1)
    out.copy_(res.to(torch.bfloat16))
```
