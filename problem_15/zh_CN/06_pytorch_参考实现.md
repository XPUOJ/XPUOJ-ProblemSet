---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, w_all, out, out_dims, batch_size, in_features, K):
    """
    PyTorch 参考实现：Linear Variadic Weights
    
    参数说明：
    * x: 输入张量（bfloat16），shape (batch_size, in_features)，行优先连续存储，只读
    * w_all: 输入张量（bfloat16），shape (total_out, in_features)，只读
    * out: 输出张量（bfloat16），shape (batch_size, total_out)，行优先连续存储，需写入结果
    * out_dims: 输入张量（int64），shape (K,)，每路输出的维度，只读
    * batch_size: 批大小（int64）
    * in_features: 输入特征数（int64）
    * K: 输出路数（int64）
    """
    out.copy_(torch.matmul(x, w_all.t()).to(torch.bfloat16))
```
