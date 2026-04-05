---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(input, output_q, scale, is_static, M, N):
    """
    PyTorch 参考实现：Per Tensor Quant fp8
    
    参数说明：
    * input: 输入张量（bfloat16/float16/float32），shape (M, N)，行优先连续存储，只读
    * output_q: 输出张量（FP8 E4M3FN），shape (M, N)，行优先连续存储，需写入量化结果
    * scale: 输出张量（float32 标量），需写入计算的缩放因子
    * is_static: int64 标量（0/1），本题固定为 0
    * M: 输入行数（int64）
    * N: 输入列数（int64）
    """
    fp8_max = 448.0
    amax = torch.max(torch.abs(input)).float()
    scale_ref = amax / fp8_max
    scale_ref = torch.maximum(scale_ref, torch.tensor(1e-12, device=input.device))
    
    scale.copy_(scale_ref)
    output_fp32 = input.float() * (1.0 / scale_ref)
    output_clamped = torch.clamp(output_fp32, -fp8_max, fp8_max)
    output_q.copy_(output_clamped.to(torch.float8_e4m3fn))
```
