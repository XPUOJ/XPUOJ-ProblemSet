---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
以下代码展示与评测 baseline 一致的 PyTorch 参考逻辑：

```python
import torch
import torch.nn.functional as F

def reference(x, w_gate, w_up, b_gate, b_up, y, M: int, K: int, N: int):
    old_precision = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    try:
        x_fp32 = x.to(torch.float32)
        gate = x_fp32 @ w_gate.to(torch.float32)
        gate = gate + b_gate
        up = x_fp32 @ w_up.to(torch.float32)
        up = up + b_up
        y.copy_((F.silu(gate) * up).to(torch.bfloat16))
    finally:
        torch.backends.cuda.matmul.fp32_precision = old_precision
    return y
```

参赛提交中不能直接调用该参考实现，需要在 `run_kernel` 中实现对应计算并写入 `y`。
