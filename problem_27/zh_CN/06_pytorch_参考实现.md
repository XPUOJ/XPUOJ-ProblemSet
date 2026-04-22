---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(a, b, output, M, N, K):
    """
    PyTorch 参考实现：Sparse GEMM S8-S32
    """
    output.copy_(torch.matmul(a.to(torch.int32), b.to(torch.int32)))
```