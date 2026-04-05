---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A, B, C, M, N, K):
    """
    PyTorch 参考实现：C = A @ B.T
    
    参数说明：
    * A: 输入张量（bfloat16），shape (M, K)，行优先连续存储，只读
    * B: 输入张量（bfloat16），shape (N, K)，行优先连续存储，只读
    * C: 输出张量（bfloat16），shape (M, N)，行优先连续存储，需写入结果
    * M: A 的行数，也是 C 的行数（int64）
    * N: B 的行数，也是 C 的列数（int64）
    * K: A 的列数，也是 B 的列数（int64）
    """
    C_ref = torch.matmul(A, B.T)
    C.copy_(C_ref)
```
