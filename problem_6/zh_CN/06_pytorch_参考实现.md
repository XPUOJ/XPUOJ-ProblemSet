---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, y, T, H):
    """
    PyTorch 参考实现：GeGLU
    
    参数说明：
    * x: 输入张量（bfloat16），shape (T, 2H)，行优先连续存储，只读
    * y: 输出张量（bfloat16），shape (T, H)，行优先连续存储，需写入结果
    * T: 序列长度（int64）
    * H: 隐藏层维度（int64）
    """
    x_1 = x[:, :H]
    x_2 = x[:, H:]
    
    # GELU 使用 tanh 近似
    gelu_x1 = torch.nn.functional.gelu(x_1, approximate="tanh")
    
    y.copy_(gelu_x1 * x_2)
```
