---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(input, output_q, scale, M, N, dtype_code):
    """
    PyTorch 参考实现：Per-token FP8 量化

    参数说明：
    * input: 输入张量，shape (M, N)，dtype 可能为 bfloat16 / float16
    * output_q: 输出张量，shape (M, N)，dtype 为 torch.float8_e4m3fn
    * scale: 输出张量，shape (M,)，dtype 为 torch.float32
    * M: 输入行数
    * N: 输入列数
    * dtype_code: 输入类型编码
    """
    amax = input.abs().float().amax(dim=1)
    scale_ref = torch.clamp(amax / 448.0, min=1e-12)
    output_fp32 = input.float() * (1.0 / scale_ref).unsqueeze(-1)
    output_q.copy_(torch.clamp(output_fp32, -448.0, 448.0).to(torch.float8_e4m3fn))
    scale.copy_(scale_ref)
```
