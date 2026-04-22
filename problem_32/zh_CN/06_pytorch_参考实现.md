---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(x, weight, bias, output, N, C_in, C_out, H, W, R, S, stride, padding):
    output.copy_(torch.nn.functional.conv2d(x.float(), weight.float(), bias.float(), stride, padding).to(output.dtype))
```