---
sectionTitle: "PyTorch 参考实现"
type: "Text"
---
```python
def baseline(q, kv, kv_sf, weights, cu_seqlen_ks, cu_seqlen_ke, logits, M, seq_len_kv, num_heads, head_dim):
    """
    PyTorch 参考实现：FP8 MQA logits

    参数说明：
    * q: 输入张量，shape (M, num_heads, head_dim)，dtype 为 torch.float8_e4m3fn
    * kv: 输入张量，shape (seq_len_kv, head_dim)，dtype 为 torch.float8_e4m3fn
    * kv_sf: 输入张量，shape (seq_len_kv,)，dtype 为 torch.float32
    * weights: 输入张量，shape (M, num_heads)，dtype 为 torch.float32
    * cu_seqlen_ks: 输入张量，shape (M,)，dtype 为 torch.int32
    * cu_seqlen_ke: 输入张量，shape (M,)，dtype 为 torch.int32
    * logits: 输出张量，shape (M, seq_len_kv)，dtype 为 torch.float32
    """
    q_fp32 = q.float()
    kv_fp32 = kv.float() * kv_sf.unsqueeze(-1)
    out = torch.zeros(M, seq_len_kv, dtype=torch.float32, device=q.device)
    for h in range(num_heads):
        score_h = torch.matmul(q_fp32[:, h, :], kv_fp32.t()).relu()
        out = out + score_h * weights[:, h:h + 1]
    ar = torch.arange(0, seq_len_kv, device=q.device, dtype=torch.int32)[None, :]
    mask = (ar >= cu_seqlen_ks[:, None]) & (ar < cu_seqlen_ke[:, None])
    logits.copy_(out.masked_fill(~mask, float("-inf")))
```
