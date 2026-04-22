---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
# PyTorch 参考实现

```python
def baseline(a, b, output, G, M, N, K):
    """
    PyTorch 参考实现：INT4-FP8 grouped GEMM
    """
    output.copy_(torch.matmul(a.float(), b.float()))
```