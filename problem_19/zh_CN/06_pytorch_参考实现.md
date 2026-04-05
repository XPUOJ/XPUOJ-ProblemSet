---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
import torch.nn.functional as F

def baseline(x, weight, bias, conv_states, output, batch_size, dim, seq_len, kernel_size, has_initial_state, silu_activation):
    """
    PyTorch 参考实现：Causal Conv1d Fwd
    
    参数说明：
    * x: 输入张量（float32），shape (batch_size, dim, seq_len)，只读
    * weight: 权重张量（float32），shape (dim, kernel_size)，只读
    * bias: 偏置张量（float32），shape (dim,)，只读
    * conv_states: 初始状态张量（float32），shape (batch_size, dim, kernel_size-1)，只读
    * output: 输出张量（float32），shape (batch_size, dim, seq_len)，需写入结果
    * batch_size: batch 大小（int64）
    * dim: 通道数（int64）
    * seq_len: 序列长度（int64）
    * kernel_size: 卷积核宽度（int64）
    * has_initial_state: 是否使用初始状态（0 或 1）
    * silu_activation: 是否应用 SiLU 激活（0 或 1）
    """
    weight_reshaped = weight.unsqueeze(1)
    if has_initial_state:
        x_with_states = torch.cat([conv_states, x], dim=-1)
        ref = F.conv1d(x_with_states, weight_reshaped, bias, groups=dim, padding=0)
        ref = ref[..., :seq_len]
    else:
        ref = F.conv1d(x, weight_reshaped, bias, groups=dim, padding=kernel_size - 1)
        ref = ref[..., :seq_len]
    
    if silu_activation:
        ref = F.silu(ref)
    
    output.copy_(ref)
```
