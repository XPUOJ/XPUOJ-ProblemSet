---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, y, T, H):
    """
    PyTorch 参考实现：SiLU
    
    参数说明：
    * x: 输入张量（bfloat16），shape (T, H)，行优先连续存储，只读
    * y: 输出张量（bfloat16），shape (T, H)，行优先连续存储，需写入结果
    * T: 第一维大小（int64）
    * H: 第二维大小（int64）
    """
    y.copy_(torch.nn.functional.silu(x))
```
