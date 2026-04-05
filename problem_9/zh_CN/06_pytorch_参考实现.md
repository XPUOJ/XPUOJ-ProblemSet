---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(logits, labels, loss, M, N):
    """
    PyTorch 参考实现：Cross Entropy
    
    参数说明：
    * logits: 输入张量（float32），shape (M, N)，行优先连续存储，只读
    * labels: 输入张量（int64），shape (M,)，连续存储，每个元素表示对应样本的真实类别
    * loss: 输出张量（float32），shape (M,)，连续存储，需写入每个样本的交叉熵损失
    * M: 样本数量（int64）
    * N: 类别数量（int64）
    """
    loss.copy_(torch.nn.functional.cross_entropy(logits, labels, reduction='none'))
```
