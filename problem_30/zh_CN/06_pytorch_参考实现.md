---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, weight, bias, output, N, C_in, C_out, D, H, W, T, R, S, stride, padding):
    output.copy_(torch.nn.functional.conv3d(x.float(), weight.float(), bias.float(), stride, padding))
```