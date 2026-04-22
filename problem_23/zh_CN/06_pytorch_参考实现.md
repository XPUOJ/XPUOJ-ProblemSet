---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(input, indices, output, N, M, H):
    output.copy_(input[indices])
```