---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(values, indices, output, first_axis_dim, M, H, dtype_code):
    """
    PyTorch 参考实现：Index Put First Axis

    参数说明：
    * values: 输入张量，shape (M, H)
    * indices: 输入张量，shape (M,)，dtype 为 torch.int64
    * output: 输出张量，shape (first_axis_dim, H)
    * first_axis_dim: 输出第 0 维大小
    * M: 待写入的行数
    * H: 每行元素数
    * dtype_code: 数据类型编码
    """
    output.zero_()
    output[indices] = values
```
