---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(A, B, numel):
    """
    PyTorch 参考实现：A += B
    
    参数说明：
    * A: 输入/输出张量（fp16），会被原地修改
    * B: 输入张量（fp16），只读
    * numel: 元素总数（int64）
    """
    A.add_(B)
```
