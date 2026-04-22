---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(a, b, output, alpha, upper, M, N):
    if upper:
        a_tri = torch.triu(a.double())
    else:
        a_tri = torch.tril(a.double())
    output.copy_(alpha * torch.matmul(a_tri, b.double()))
```