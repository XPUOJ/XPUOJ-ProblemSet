---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
import torch.nn.functional as F

def baseline(x, y, T, H):
    """
    PyTorch 参考实现：SwiGLU
    
    参数说明：
    * x: 输入张量（bfloat16），shape (T, 2H)，行优先连续存储，只读
    * y: 输出张量（bfloat16），shape (T, H)，行优先连续存储，需写入结果
    * T: 序列长度（int64）
    * H: 隐藏层维度（int64）
    """
    x_1 = x[:, :H]
    x_2 = x[:, H:]
    silu_x1 = F.silu(x_1)
    y_ref = silu_x1 * x_2
    y.copy_(y_ref)
```
