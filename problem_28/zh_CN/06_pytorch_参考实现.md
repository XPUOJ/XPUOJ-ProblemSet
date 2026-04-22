---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(a, b, output, pattern_id, B, M, N, K):
    if pattern_id == 1:
        output.copy_(torch.einsum("bmk,bkn->bmn", a, b))
    elif pattern_id == 2:
        output.copy_(torch.einsum("bkm,bkn->bmn", a, b))
    else:
        output.copy_(torch.einsum("bmk,bnk->bmn", a, b))
```