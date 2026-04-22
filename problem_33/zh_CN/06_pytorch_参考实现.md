---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, weight, bias, broadcast, output, N, C_in, C_out, H, W, R, S, stride, padding):
    temp = torch.nn.functional.conv2d(x, weight, bias, stride, padding)
    output.copy_(temp + broadcast)
```