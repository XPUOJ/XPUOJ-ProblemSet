---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A, A_scale, B, B_scale, output, M, N, K, BLOCK_SIZE):
    K_blocks = (K + BLOCK_SIZE - 1) // BLOCK_SIZE
    N_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    A_scaled = A.float() * A_scale.repeat_interleave(BLOCK_SIZE, dim=1)[:, :K]
    B_scaled = B.float() * B_scale.repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)[:N, :K]
    output.copy_(torch.matmul(A_scaled, B_scaled.t()).to(output.dtype))
```
