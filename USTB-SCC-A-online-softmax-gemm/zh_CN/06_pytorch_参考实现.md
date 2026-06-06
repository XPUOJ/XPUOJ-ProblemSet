---
sectionTitle: "PyTorch 参考实现"
type: "codeSample"
lang: "python"
---

以下是与 `testcase_config.py` 中 `baseline` 逻辑一致的 PyTorch 参考实现：

```python
import torch

def reference(Q, K, V, mask, B, H, S, D, alpha, O):
    """
    PyTorch 参考实现：O = softmax(α · Q @ K^T + mask) @ V

    参数顺序和 names 与 run_kernel 完全一致。
    O 的初值被完整覆盖，不依赖其原有内容。
    """
    # 重塑为便于计算的形状 [B, H, S, D]
    Q_t = Q.view(B, H, S, D).float()
    K_t = K.view(B, H, S, D).float()
    V_t = V.view(B, H, S, D).float()

    # 第 1 步：S = α × Q @ K^T
    S = alpha * torch.matmul(Q_t, K_t.transpose(-2, -1))

    # 第 2 步：加 mask（如果提供）
    if mask is not None:
        mask_t = mask.view(B, 1, S, S).float()
        S = S + mask_t

    # 第 3 步：P = softmax(S)（沿最后一维）
    P = torch.softmax(S, dim=-1)

    # 第 4 步：O = P @ V
    O_t = torch.matmul(P, V_t)

    # 写回 bf16 输出
    O.copy_(O_t.to(torch.bfloat16))

    return [Q, K, V, mask, B, H, S, D, alpha, O]
```

**说明**：

- 参考实现为纯 PyTorch，未做任何 kernel 融合——它展示了正确的数学结果，而非正确的实现方式
- 参赛者的实现应**避免**这种"先算完整 $S$、再 softmax、再算 $O$"的模式，而是使用 online softmax 融合策略
- 正确性检查会将用户 kernel 的输出与参考实现进行 `torch.allclose(rtol=1e-2, atol=1e-2)` 比较
