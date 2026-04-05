---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(input, out, batch_size, hidden_size):
    """
    PyTorch 参考实现：Fused SiLU and Mul
    
    参数说明：
    * input: 输入张量（bfloat16），shape (batch_size, 2*hidden_size)，行优先连续存储，只读
    * out: 输出张量（bfloat16），shape (batch_size, hidden_size)，行优先连续存储，需写入结果
    * batch_size: 第一维大小（int64）
    * hidden_size: 隐藏层大小（int64）
    """
    first_half = input[..., :hidden_size]
    second_half = input[..., hidden_size:]
    
    silu_result = torch.nn.functional.silu(first_half)
    result = silu_result * second_half
    
    out.copy_(result)
```
